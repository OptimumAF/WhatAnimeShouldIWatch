"""M5.6 invented train-only baseline and graph-sign checks."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from raw_interaction_split import (  # noqa: E402
    build_split_manifest, load_raw_snapshot, parse_raw_snapshot, partition_snapshot,
)
from split_first_baseline_export import (  # noqa: E402
    build_ablation_bundle, normalized_file_sha, parse_spec, similarity_from_train,
)
from train_graph_mf import (  # noqa: E402
    load_anime_graph_edges_compact, load_anime_graph_edges_legacy, train_graph_mf,
)
from train_only_preprocessing import CenteredTrainRow  # noqa: E402
from split_first_graph_mf import model_fingerprint  # noqa: E402


class BaselineExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = {
            "raw": ROOT / "fixtures" / "synthetic-new-user-fit.json",
            "manifest": ROOT / "fixtures" / "synthetic-new-user-fit-manifest.json",
            "metadata": ROOT / "fixtures" / "synthetic-new-user-anime-metadata.json",
            "validation": ROOT / "fixtures" / "synthetic-new-user-validation.json",
            "candidates": ROOT / "fixtures" / "synthetic-mf-candidates.json",
        }
        cls.spec = parse_spec(json.loads(
            (ROOT / "fixtures" / "synthetic-baseline-ablation-spec.json").read_text(encoding="utf-8")
        ))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.bundle = build_ablation_bundle(cls.paths, cls.spec)

    def test_train_only_bundle_matches_existing_model_and_pinned_spec(self):
        base = self.bundle["baseBundle"]
        self.assertEqual(base["trainRowCount"], 72)
        self.assertEqual(len(base["catalog"]), 24)
        self.assertEqual(sum(item["count"] for item in self.bundle["trainCounts"]), 72)
        self.assertEqual(self.bundle["audit"]["positivePairEdges"], len(base["positivePairs"]))
        self.assertGreater(self.bundle["audit"]["nonpositivePairEdgesExcluded"], 0)
        self.assertEqual(self.bundle["audit"]["similarity"]["positiveSupportedPairs"],
                         len(self.bundle["similarityPairs"]))
        self.assertEqual(len({base["modelSha256"], *(
            item["modelSha256"] for item in self.bundle["models"].values())}), 4)
        for key, path in (
            ("fitSnapshotSha256", "raw"), ("fitManifestSha256", "manifest"),
            ("metadataSha256", "metadata"), ("validationSha256", "validation"),
            ("mfCandidatesSha256", "candidates"),
        ):
            self.assertEqual(self.spec[key], normalized_file_sha(self.paths[path]))
        altered = copy.deepcopy(self.spec)
        altered["hybridModelWeight"] = 0.75
        with self.assertRaisesRegex(ValueError, "protocol"):
            parse_spec(altered)
        altered = copy.deepcopy(self.spec)
        altered["fitSnapshotSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fitSnapshotSha256"):
            build_ablation_bundle(self.paths, altered)

    def test_supported_adjusted_cosine_uses_sign_support_and_shrinkage(self):
        rows = (
            CenteredTrainRow("a", 1, 8, 1), CenteredTrainRow("a", 2, 8, 1),
            CenteredTrainRow("a", 3, 2, -1), CenteredTrainRow("a", 4, 8, 1),
            CenteredTrainRow("b", 1, 9, 2), CenteredTrainRow("b", 2, 9, 2),
            CenteredTrainRow("b", 3, 1, -2),
        )
        pairs, stats = similarity_from_train(rows, 2, 2)
        aligned = next(pair for pair in pairs
                       if (pair["leftAnimeId"], pair["rightAnimeId"]) == (1, 2))
        self.assertEqual(aligned["support"], 2)
        self.assertAlmostEqual(aligned["adjustedCosine"], 1)
        self.assertAlmostEqual(aligned["weight"], 0.5)
        self.assertNotIn((1, 3), [(p["leftAnimeId"], p["rightAnimeId"]) for p in pairs])
        self.assertNotIn((1, 4), [(p["leftAnimeId"], p["rightAnimeId"]) for p in pairs])
        self.assertGreater(stats["nonpositiveSupportedPairs"], 0)
        self.assertGreater(stats["lowSupportPairs"], 0)

    def test_fixed_membership_heldout_score_edits_leave_every_train_output_unchanged(self):
        raw = json.loads(self.paths["raw"].read_text(encoding="utf-8"))
        original_manifest = json.loads(self.paths["manifest"].read_text(encoding="utf-8"))
        changed = copy.deepcopy(raw)
        for user, anime, score in (
            ("invented-fit-08", 213, 10), ("invented-fit-02", 220, 1),
        ):
            match = [row for row in changed["interactions"]
                     if row["userId"] == user and row["animeId"] == anime]
            self.assertEqual(len(match), 1)
            match[0]["rawScore"] = score
        snapshot = parse_raw_snapshot(changed)
        with self.assertRaisesRegex(ValueError, "does not match"):
            partition_snapshot(snapshot, original_manifest)
        refreshed = build_split_manifest(snapshot)
        self.assertEqual(refreshed["trainIds"], original_manifest["trainIds"])
        self.assertEqual(partition_snapshot(snapshot, refreshed).train,
                         partition_snapshot(load_raw_snapshot(self.paths["raw"]),
                                            original_manifest).train)
        with tempfile.TemporaryDirectory() as temp:
            raw_path = Path(temp) / "raw.json"
            manifest_path = Path(temp) / "manifest.json"
            raw_path.write_text(json.dumps(changed), encoding="utf-8")
            manifest_path.write_text(json.dumps(refreshed), encoding="utf-8")
            altered_paths = {**self.paths, "raw": raw_path, "manifest": manifest_path}
            altered_spec = {**self.spec,
                            "fitSnapshotSha256": normalized_file_sha(raw_path),
                            "fitManifestSha256": normalized_file_sha(manifest_path)}
            with contextlib.redirect_stdout(io.StringIO()):
                variant = build_ablation_bundle(altered_paths, altered_spec)
        for field in ("trainSha256", "fitSha256", "modelSha256", "catalog", "positivePairs", "model"):
            self.assertEqual(variant["baseBundle"][field], self.bundle["baseBundle"][field])
        for field in ("models", "trainCounts", "similarityPairs", "audit"):
            self.assertEqual(variant[field], self.bundle[field])

    def test_training_score_change_updates_fit_and_all_four_models(self):
        raw = json.loads(self.paths["raw"].read_text(encoding="utf-8"))
        changed = copy.deepcopy(raw)
        match = [row for row in changed["interactions"]
                 if row["userId"] == "invented-fit-04" and row["animeId"] == 213]
        self.assertEqual(len(match), 1)
        match[0]["rawScore"] = 1
        refreshed = build_split_manifest(parse_raw_snapshot(changed))
        original = json.loads(self.paths["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(refreshed["trainIds"], original["trainIds"])
        with tempfile.TemporaryDirectory() as temp:
            raw_path = Path(temp) / "raw.json"
            manifest_path = Path(temp) / "manifest.json"
            raw_path.write_text(json.dumps(changed), encoding="utf-8")
            manifest_path.write_text(json.dumps(refreshed), encoding="utf-8")
            altered_spec = {**self.spec,
                            "fitSnapshotSha256": normalized_file_sha(raw_path),
                            "fitManifestSha256": normalized_file_sha(manifest_path)}
            with contextlib.redirect_stdout(io.StringIO()):
                variant = build_ablation_bundle(
                    {**self.paths, "raw": raw_path, "manifest": manifest_path},
                    altered_spec,
                )
        self.assertNotEqual(variant["baseBundle"]["fitSha256"],
                            self.bundle["baseBundle"]["fitSha256"])
        self.assertNotEqual(variant["baseBundle"]["modelSha256"],
                            self.bundle["baseBundle"]["modelSha256"])
        for name in self.bundle["models"]:
            self.assertNotEqual(variant["models"][name]["modelSha256"],
                                self.bundle["models"][name]["modelSha256"])

    def test_legacy_and_compact_loaders_and_trainer_exclude_negative_attraction(self):
        ids = {201: 0, 202: 1, 203: 2}
        legacy = {"edges": [
            {"edgeType": "anime-anime", "source": "anime:201", "target": "anime:202", "weight": -4},
            {"edgeType": "anime-anime", "source": "anime:201", "target": "anime:203", "weight": 2},
        ]}
        compact = {"anime": [[201], [202], [203]],
                   "aa": [[0, 1, -4, 2], [0, 2, 2, 2]]}
        for edges in (
            load_anime_graph_edges_legacy(legacy, ids, 0),
            load_anime_graph_edges_compact(compact, ids, 0),
        ):
            self.assertEqual(edges.shape, (1, 3))
            self.assertEqual(edges[0].tolist(), [0, 2, 2])
        from split_first_graph_mf import training_arrays
        from train_only_preprocessing import load_metadata_snapshot, fit_training_partition
        fit = fit_training_partition(
            partition_snapshot(load_raw_snapshot(self.paths["raw"]),
                               json.loads(self.paths["manifest"].read_text())).train,
            load_metadata_snapshot(self.paths["metadata"]),
        )
        dataset, split, _ = training_arrays(fit, 0)
        negatives = np.asarray([[0, 1, -4]], dtype=np.float32)
        empty = np.zeros((0, 3), dtype=np.float32)
        def train(edges):
            with contextlib.redirect_stdout(io.StringIO()):
                return train_graph_mf(split, len(dataset.user_ids), len(dataset.anime_ids),
                                      edges, 4, 2, 0.02, 0.01, 0.005, 0.01, 1, 42)
        self.assertEqual(model_fingerprint(train(negatives)), model_fingerprint(train(empty)))


if __name__ == "__main__":
    unittest.main()
