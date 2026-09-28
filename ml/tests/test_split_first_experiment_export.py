"""Invented M5.9 candidate isolation; no provider data or held-out fitting."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from raw_interaction_split import build_split_manifest, parse_raw_snapshot  # noqa: E402
from split_first_baseline_export import normalized_file_sha  # noqa: E402
from split_first_experiment_export import (  # noqa: E402
    build_experiment_bundle, parse_content_metadata, parse_experiment_spec,
)


class ExperimentExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = ROOT / "fixtures"
        cls.paths = {
            "raw": fixture / "synthetic-new-user-fit.json",
            "manifest": fixture / "synthetic-new-user-fit-manifest.json",
            "metadata": fixture / "synthetic-new-user-anime-metadata.json",
            "validation": fixture / "synthetic-new-user-validation.json",
            "candidates": fixture / "synthetic-mf-candidates.json",
            "baseline_spec": fixture / "synthetic-baseline-ablation-spec.json",
            "content": fixture / "synthetic-experiment-content-metadata.json",
        }
        cls.spec = json.loads((fixture / "synthetic-experiment-spec.json").read_text())
        cls.baseline_spec = json.loads(cls.paths["baseline_spec"].read_text())
        cls.raw = json.loads(cls.paths["raw"].read_text())
        cls.manifest = json.loads(cls.paths["manifest"].read_text())
        cls.bundle = build_experiment_bundle(cls.paths, cls.spec, cls.baseline_spec)

    def changed_bundle(self, *, raw: dict | None = None, content: dict | None = None,
                       stale_manifest: bool = False) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = dict(self.paths)
            baseline_spec = copy.deepcopy(self.baseline_spec)
            spec = copy.deepcopy(self.spec)
            if raw is not None:
                paths["raw"] = folder / "raw.json"
                paths["raw"].write_text(json.dumps(raw), encoding="utf-8")
                paths["manifest"] = folder / "manifest.json"
                refreshed = self.manifest if stale_manifest else build_split_manifest(
                    parse_raw_snapshot(raw))
                if not stale_manifest:
                    self.assertEqual(refreshed["trainIds"], self.manifest["trainIds"])
                paths["manifest"].write_text(json.dumps(refreshed), encoding="utf-8")
                baseline_spec["fitSnapshotSha256"] = normalized_file_sha(paths["raw"])
                baseline_spec["fitManifestSha256"] = normalized_file_sha(paths["manifest"])
            paths["baseline_spec"] = folder / "baseline-spec.json"
            paths["baseline_spec"].write_text(json.dumps(baseline_spec), encoding="utf-8")
            spec["baselineSpecSha256"] = normalized_file_sha(paths["baseline_spec"])
            if content is not None:
                paths["content"] = folder / "content.json"
                paths["content"].write_text(json.dumps(content), encoding="utf-8")
                spec["contentMetadataSha256"] = normalized_file_sha(paths["content"])
            return build_experiment_bundle(paths, spec, baseline_spec)

    def test_train_only_lightgcn_and_rating_free_content_contract(self) -> None:
        bundle = self.bundle
        self.assertEqual(bundle["format"], "split-first-experiment-bundle-v1")
        self.assertEqual(bundle["trainRowCount"], 72)
        self.assertEqual(bundle["positiveTrainInteractions"], 35)
        self.assertEqual(bundle["positivePairEdges"], 91)
        self.assertEqual(bundle["contentVocabulary"], [
            "genre:Adventure", "genre:Mystery", "studio:Studio A",
            "studio:Studio B", "studio:Studio C",
        ])
        self.assertEqual(set(bundle["models"]), {"lightgcn-bpr", "content-multihot"})
        for item in bundle["models"].values():
            self.assertEqual(len(item["model"]["animeIds"]), 24)
            self.assertEqual(item["model"]["globalMean"], 0)
            self.assertEqual(set(item["model"]["biases"]), {0})
        serialized = json.dumps(bundle)
        self.assertNotIn("invented-fit-", serialized)
        self.assertNotIn('"validation"', serialized)
        self.assertNotIn('"test"', serialized)
        self.assertNotIn('"P"', serialized)

    def test_fixed_membership_heldout_scores_cannot_change_either_model(self) -> None:
        changed = copy.deepcopy(self.raw)
        for user, anime, score in (
            ("invented-fit-08", 213, 10), ("invented-fit-02", 220, 1),
        ):
            rows = [row for row in changed["interactions"]
                    if row["userId"] == user and row["animeId"] == anime]
            self.assertEqual(len(rows), 1)
            rows[0]["rawScore"] = score
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.changed_bundle(raw=changed, stale_manifest=True)
        variant = self.changed_bundle(raw=changed)
        for field in ("trainSha256", "fitSha256", "trainRowCount",
                      "positiveTrainInteractions", "positivePairEdges", "models"):
            self.assertEqual(variant[field], self.bundle[field])

    def test_training_score_changes_lightgcn_but_not_content(self) -> None:
        changed = copy.deepcopy(self.raw)
        row = next(row for row in changed["interactions"]
                   if row["userId"] == "invented-fit-04" and row["animeId"] == 213)
        row["rawScore"] = 1
        variant = self.changed_bundle(raw=changed)
        self.assertNotEqual(variant["trainSha256"], self.bundle["trainSha256"])
        self.assertNotEqual(variant["fitSha256"], self.bundle["fitSha256"])
        self.assertNotEqual(variant["models"]["lightgcn-bpr"],
                            self.bundle["models"]["lightgcn-bpr"])
        self.assertEqual(variant["models"]["content-multihot"],
                         self.bundle["models"]["content-multihot"])

    def test_fixed_content_edit_changes_only_content_model_and_rejects_rating_field(self) -> None:
        content = json.loads(self.paths["content"].read_text())
        content["anime"][0]["studios"] = ["Studio D"]
        variant = self.changed_bundle(content=content)
        self.assertEqual(variant["models"]["lightgcn-bpr"],
                         self.bundle["models"]["lightgcn-bpr"])
        self.assertNotEqual(variant["models"]["content-multihot"],
                            self.bundle["models"]["content-multihot"])
        content["anime"][0]["ratingCount"] = 99
        with self.assertRaisesRegex(ValueError, "fields"):
            self.changed_bundle(content=content)
        del content["anime"][0]["ratingCount"]
        with self.assertRaisesRegex(ValueError, "IDs"):
            parse_content_metadata({"format": "experiment-content-metadata-v1",
                                    "source": "invented-fixture", "anime": content["anime"][:-1]},
                                   list(range(201, 225)))

    def test_spec_and_file_backed_cli_refuse_stale_or_changed_protocol(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["lightgcn"]["epochs"] = 3
        with self.assertRaisesRegex(ValueError, "decision 0025"):
            parse_experiment_spec(changed)
        changed = copy.deepcopy(self.spec)
        changed["contentMetadataSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stale"):
            build_experiment_bundle(self.paths, changed, self.baseline_spec)
        completed = subprocess.run(
            [sys.executable, str(ROOT / "ml" / "split_first_experiment_export.py")],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(json.loads(completed.stdout)["models"], self.bundle["models"])
        self.assertNotIn("invented-fit-", completed.stdout + completed.stderr)
        self.assertNotIn("Epoch", completed.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            other = Path(temporary) / "raw.json"
            other.write_text(json.dumps(self.raw), encoding="utf-8")
            refused = subprocess.run(
                [sys.executable, str(ROOT / "ml" / "split_first_experiment_export.py"),
                 "--raw", str(other)], cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("source/use approval", refused.stderr)


if __name__ == "__main__":
    unittest.main()
