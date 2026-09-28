"""Local item-model adapter checks using invented split-first inputs only."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from raw_interaction_split import build_split_manifest, interaction_id, load_raw_snapshot  # noqa: E402
from split_first_new_user_export import build_eval_bundle  # noqa: E402


RAW = ROOT / "fixtures" / "synthetic-new-user-fit.json"
MANIFEST = ROOT / "fixtures" / "synthetic-new-user-fit-manifest.json"
METADATA = ROOT / "fixtures" / "synthetic-new-user-anime-metadata.json"
CANDIDATES = ROOT / "fixtures" / "synthetic-mf-candidates.json"
CANDIDATE_ID = "graph-two-epochs"


class NewUserExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = build_eval_bundle(RAW, MANIFEST, METADATA, CANDIDATES, CANDIDATE_ID)

    def test_train_only_bundle_excludes_user_rows_and_factors(self) -> None:
        bundle = self.bundle
        self.assertEqual(bundle["format"], "split-first-new-user-bundle-v1")
        self.assertEqual(bundle["fitUserCount"], 8)
        self.assertEqual(bundle["trainRowCount"], 72)
        self.assertEqual(len(bundle["catalog"]), 24)
        self.assertEqual(len(bundle["model"]["animeIds"]), 24)
        self.assertTrue(all(pair["weight"] > 0 and pair["support"] > 0
                            for pair in bundle["positivePairs"]))
        self.assertEqual(bundle["warmValidation"]["eligibleUsers"], 6)
        serialized = json.dumps(bundle)
        self.assertNotIn("invented-fit-", serialized)
        self.assertNotIn('"P"', serialized)
        self.assertNotIn('"bu"', serialized)
        self.assertNotIn('"trainUserItems"', serialized)
        self.assertNotIn("test", serialized.lower())

    def test_stale_manifest_fails_and_changed_training_score_changes_fit(self) -> None:
        raw = json.loads(RAW.read_text(encoding="utf-8"))
        pinned = json.loads(MANIFEST.read_text(encoding="utf-8"))
        training_ids = set(pinned["trainIds"])
        chosen = next(row for row in raw["interactions"]
                      if interaction_id(row["userId"], row["animeId"]) in training_ids)
        chosen["rawScore"] = 1 if chosen["rawScore"] != 1 else 10
        with tempfile.TemporaryDirectory() as temporary:
            raw_path = Path(temporary) / "fit.json"
            manifest_path = Path(temporary) / "manifest.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_eval_bundle(raw_path, MANIFEST, METADATA, CANDIDATES, CANDIDATE_ID)
            refreshed = build_split_manifest(load_raw_snapshot(raw_path), seed=42)
            self.assertEqual(refreshed["trainIds"], pinned["trainIds"])
            manifest_path.write_text(json.dumps(refreshed), encoding="utf-8")
            changed = build_eval_bundle(raw_path, manifest_path, METADATA, CANDIDATES, CANDIDATE_ID)
        self.assertNotEqual(changed["trainSha256"], self.bundle["trainSha256"])
        self.assertNotEqual(changed["fitSha256"], self.bundle["fitSha256"])
        self.assertNotEqual(changed["modelSha256"], self.bundle["modelSha256"])

    def test_metadata_cannot_smuggle_rating_derived_fields(self) -> None:
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        metadata["anime"][0]["ratingCount"] = 99
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only animeId and title"):
                build_eval_bundle(RAW, MANIFEST, path, CANDIDATES, CANDIDATE_ID)

    def test_cli_stdout_is_one_json_record_without_epoch_or_identity_log(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "ml" / "split_first_new_user_export.py"),
             "--raw-ratings", str(RAW), "--split-manifest", str(MANIFEST),
             "--metadata", str(METADATA), "--candidates", str(CANDIDATES),
             "--candidate-id", CANDIDATE_ID],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        self.assertEqual(json.loads(completed.stdout)["modelSha256"], self.bundle["modelSha256"])
        self.assertNotIn("invented-fit-", completed.stdout + completed.stderr)
        self.assertNotIn("Epoch", completed.stdout)


if __name__ == "__main__":
    unittest.main()
