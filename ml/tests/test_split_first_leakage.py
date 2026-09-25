"""M5.3: holdout perturbations cannot change train-only graph or MF state."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from raw_interaction_split import (  # noqa: E402
    build_split_manifest, parse_raw_snapshot, partition_snapshot,
)
from split_first_graph_mf import fit_split_first, model_fingerprint  # noqa: E402
from train_only_preprocessing import load_metadata_snapshot  # noqa: E402


PARAMETERS = dict(factors=4, epochs=2, lr=0.02, reg=0.01, reg_bias=0.005,
                  graph_lambda=0.01, graph_min_weight=0.0, graph_sample_rate=1.0, seed=42)


def change_one(raw: dict, user_id: str, anime_id: int, field: str, value: int) -> dict:
    changed = copy.deepcopy(raw)
    matches = [row for row in changed["interactions"]
               if row["userId"] == user_id and row["animeId"] == anime_id]
    if len(matches) != 1:
        raise AssertionError("The predeclared invented interaction is missing or duplicated.")
    matches[0][field] = value
    return changed


class SplitFirstLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_path = ROOT / "fixtures" / "synthetic-split-input.json"
        cls.manifest_path = ROOT / "fixtures" / "synthetic-split-manifest.json"
        cls.metadata_path = ROOT / "fixtures" / "synthetic-anime-metadata.json"
        cls.raw = json.loads(cls.raw_path.read_text(encoding="utf-8"))
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.metadata = load_metadata_snapshot(cls.metadata_path)
        cls.base_snapshot = parse_raw_snapshot(cls.raw)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.base = fit_split_first(cls.base_snapshot, cls.manifest, cls.metadata, **PARAMETERS)

    def run_changed(self, raw: dict):
        snapshot = parse_raw_snapshot(raw)
        with self.assertRaisesRegex(ValueError, "does not match"):
            partition_snapshot(snapshot, self.manifest)
        refreshed = build_split_manifest(snapshot)
        self.assertEqual(refreshed["trainIds"], self.manifest["trainIds"],
                         "This comparison requires immutable training membership.")
        self.assertEqual(partition_snapshot(snapshot, refreshed).train,
                         partition_snapshot(self.base_snapshot, self.manifest).train)
        with contextlib.redirect_stdout(io.StringIO()):
            result = fit_split_first(snapshot, refreshed, self.metadata, **PARAMETERS)
        return refreshed, result

    def assert_training_unchanged(self, changed):
        baseline = self.base
        self.assertEqual(changed.fit.metadata, baseline.fit.metadata)
        self.assertEqual(changed.fit.user_baselines, baseline.fit.user_baselines)
        self.assertEqual(changed.fit.rows, baseline.fit.rows)
        self.assertEqual(changed.fit.popularity, baseline.fit.popularity)
        self.assertEqual(changed.fit.pairs, baseline.fit.pairs)
        self.assertEqual(changed.fit.pair_stats, baseline.fit.pair_stats)
        self.assertEqual(changed.fit.pair_config, baseline.fit.pair_config)
        self.assertEqual(changed.fit.train_sha256, baseline.fit.train_sha256)
        self.assertEqual(changed.fit.fit_sha256, baseline.fit.fit_sha256)
        self.assertEqual(changed.dataset.user_ids, baseline.dataset.user_ids)
        self.assertEqual(changed.dataset.anime_ids, baseline.dataset.anime_ids)
        self.assertEqual(changed.dataset.anime_titles, baseline.dataset.anime_titles)
        self.assertEqual(changed.dataset.interactions, baseline.dataset.interactions)
        for name in ("train_u", "train_i", "train_r"):
            np.testing.assert_array_equal(getattr(changed.split, name), getattr(baseline.split, name))
        self.assertEqual(changed.split.train_user_items, baseline.split.train_user_items)
        self.assertEqual(changed.split.test_pos_items, baseline.split.test_pos_items)
        np.testing.assert_array_equal(changed.graph_edges, baseline.graph_edges)
        for name in ("P", "Q", "bu", "bi", "global_mean"):
            np.testing.assert_array_equal(np.asarray(changed.model[name]), np.asarray(baseline.model[name]))
        self.assertEqual(model_fingerprint(changed.model), model_fingerprint(baseline.model))

    def test_validation_and_test_score_edits_leave_every_train_value_unchanged(self):
        changed = change_one(self.raw, "invented-b", 101, "rawScore", 1)
        changed = change_one(changed, "invented-c", 103, "rawScore", 10)
        refreshed, result = self.run_changed(changed)
        for partition in ("trainIds", "validationIds", "testIds"):
            self.assertEqual(refreshed[partition], self.manifest[partition])
        self.assertEqual(refreshed["identitySha256"], self.manifest["identitySha256"])
        self.assertNotEqual(refreshed["rawContentSha256"], self.manifest["rawContentSha256"])
        self.assert_training_unchanged(result)

    def test_changed_hidden_interaction_ids_leave_train_graph_and_model_unchanged(self):
        for user_id, old_anime, new_anime, changed_partition, stable_partition in (
            ("invented-b", 101, 108, "validationIds", "testIds"),
            ("invented-c", 103, 104, "testIds", "validationIds"),
        ):
            with self.subTest(user_id=user_id, old_anime=old_anime, new_anime=new_anime):
                refreshed, result = self.run_changed(
                    change_one(self.raw, user_id, old_anime, "animeId", new_anime))
                self.assertNotEqual(refreshed[changed_partition], self.manifest[changed_partition])
                self.assertEqual(refreshed[stable_partition], self.manifest[stable_partition])
                self.assertNotEqual(refreshed["identitySha256"], self.manifest["identitySha256"])
                self.assertNotEqual(refreshed["rawContentSha256"], self.manifest["rawContentSha256"])
                self.assert_training_unchanged(result)

    def test_changed_training_score_changes_fit_and_model_fingerprints(self):
        raw = change_one(self.raw, "invented-b", 102, "rawScore", 1)
        snapshot = parse_raw_snapshot(raw)
        refreshed = build_split_manifest(snapshot)
        self.assertEqual(refreshed["trainIds"], self.manifest["trainIds"])
        with contextlib.redirect_stdout(io.StringIO()):
            changed = fit_split_first(snapshot, refreshed, self.metadata, **PARAMETERS)
        self.assertNotEqual(changed.fit.train_sha256, self.base.fit.train_sha256)
        self.assertNotEqual(changed.fit.fit_sha256, self.base.fit.fit_sha256)
        self.assertNotEqual(model_fingerprint(changed.model), model_fingerprint(self.base.model))

    def test_added_interaction_rejects_old_manifest_and_changes_train_membership(self):
        raw = copy.deepcopy(self.raw)
        raw["interactions"].append({"userId": "invented-a", "animeId": 106, "rawScore": 5})
        snapshot = parse_raw_snapshot(raw)
        with self.assertRaisesRegex(ValueError, "does not match"):
            partition_snapshot(snapshot, self.manifest)
        refreshed = build_split_manifest(snapshot)
        self.assertNotEqual(refreshed["trainIds"], self.manifest["trainIds"])

    def test_file_backed_cli_reports_unchanged_train_fit_and_model_hashes(self):
        changed = change_one(self.raw, "invented-b", 101, "rawScore", 1)
        changed = change_one(changed, "invented-c", 103, "rawScore", 10)
        refreshed = build_split_manifest(parse_raw_snapshot(changed))
        script = ROOT / "ml" / "split_first_graph_mf.py"

        def run(raw_path: Path, manifest_path: Path) -> dict:
            command = [sys.executable, str(script), "--raw-ratings", str(raw_path),
                       "--split-manifest", str(manifest_path), "--metadata", str(self.metadata_path),
                       "--factors", "4", "--epochs", "2"]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            return json.loads(completed.stdout.splitlines()[-1])

        baseline = run(self.raw_path, self.manifest_path)
        with tempfile.TemporaryDirectory() as temp:
            raw_path = Path(temp) / "invented-perturbed-raw.json"
            manifest_path = Path(temp) / "invented-perturbed-split.json"
            raw_path.write_text(json.dumps(changed), encoding="utf-8")
            manifest_path.write_text(json.dumps(refreshed), encoding="utf-8")
            variant = run(raw_path, manifest_path)
            self.assertEqual({key: variant[key] for key in (
                "trainSha256", "fitSha256", "modelSha256", "metadataSha256",
                "trainPairCandidates", "trainPairEdges", "positiveRegularizationEdges",
            )}, {key: baseline[key] for key in (
                "trainSha256", "fitSha256", "modelSha256", "metadataSha256",
                "trainPairCandidates", "trainPairEdges", "positiveRegularizationEdges",
            )})
            stale = subprocess.run([
                sys.executable, str(script), "--raw-ratings", str(raw_path),
                "--split-manifest", str(self.manifest_path), "--metadata", str(self.metadata_path),
                "--factors", "4", "--epochs", "2",
            ], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("does not match", stale.stderr)
            self.assertNotIn("modelSha256", stale.stdout)


if __name__ == "__main__":
    unittest.main()
