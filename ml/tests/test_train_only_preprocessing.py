import copy
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from raw_interaction_split import (  # noqa: E402
    RawInteraction,
    build_split_manifest,
    interaction_id,
    parse_raw_snapshot,
    partition_snapshot,
)
from split_first_graph_mf import training_arrays  # noqa: E402
from train_only_preprocessing import (  # noqa: E402
    fit_training_partition,
    load_metadata_snapshot,
    parse_metadata_snapshot,
)


class TrainOnlyPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_path = ROOT / "fixtures" / "synthetic-split-input.json"
        cls.manifest_path = ROOT / "fixtures" / "synthetic-split-manifest.json"
        cls.metadata_path = ROOT / "fixtures" / "synthetic-anime-metadata.json"
        cls.raw = json.loads(cls.raw_path.read_text(encoding="utf-8"))
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.metadata = load_metadata_snapshot(cls.metadata_path)
        cls.train = partition_snapshot(parse_raw_snapshot(cls.raw), cls.manifest).train

    def test_exact_pinned_training_fit_and_positive_regularization(self):
        fit = fit_training_partition(self.train, self.metadata)
        self.assertEqual(fit.user_baselines, (
            ("invented-a", 7.0), ("invented-b", 7.0),
            ("invented-c", 8.0), ("invented-sparse", 10.0),
        ))
        self.assertEqual([(row.user_id, row.anime_id, row.raw_score, row.normalized_score)
                          for row in fit.rows], [
            ("invented-a", 102, 8.0, 1.0),
            ("invented-a", 103, 3.0, -4.0),
            ("invented-a", 105, 10.0, 3.0),
            ("invented-b", 102, 9.0, 2.0),
            ("invented-b", 107, 5.0, -2.0),
            ("invented-c", 105, 8.0, 0.0),
            ("invented-sparse", 109, 10.0, 0.0),
        ])
        self.assertEqual(fit.popularity, (
            (101, 0, 0.0), (102, 2, 17.0), (103, 1, 3.0),
            (104, 0, 0.0), (105, 2, 18.0), (106, 0, 0.0),
            (107, 1, 5.0), (108, 0, 0.0), (109, 1, 10.0),
        ))
        self.assertEqual(fit.pairs, (
            (102, 103, -1.5, 1), (102, 105, 2.0, 1),
            (102, 107, 0.0, 1), (103, 105, -0.5, 1),
        ))
        self.assertEqual(fit.pair_stats["pairVisits"], 4)
        self.assertEqual(fit.pair_stats["candidatePairs"], 4)
        self.assertEqual(fit.fit_sha256, "c4763cfab8976cf1cd1b70d493d76a28a811f75e4d4bf8e2ab410915fc5c42b1")
        dataset, split, edges = training_arrays(fit)
        self.assertEqual(len(dataset.anime_ids), 9)
        self.assertEqual(len(split.train_r), 7)
        self.assertEqual(split.test_pos_items, {})
        self.assertEqual(edges.shape, (1, 3))
        self.assertEqual(edges.tolist(), [[1.0, 4.0, 2.0]])

    def test_fixed_metadata_is_separate_and_rejects_rating_derived_fields(self):
        original = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        changed = copy.deepcopy(original)
        changed["anime"][0]["popularity"] = 100
        with self.assertRaisesRegex(ValueError, "only animeId and title"):
            parse_metadata_snapshot(changed)
        changed = copy.deepcopy(original)
        changed["userIds"] = ["invented-a"]
        with self.assertRaisesRegex(ValueError, "only format"):
            parse_metadata_snapshot(changed)
        changed = copy.deepcopy(original)
        changed["anime"] = [row for row in changed["anime"] if row["animeId"] != 102]
        with self.assertRaisesRegex(ValueError, "absent from the fixed metadata"):
            fit_training_partition(self.train, parse_metadata_snapshot(changed))

    def test_cross_language_pair_support_and_order_are_exact(self):
        metadata = parse_metadata_snapshot({
            "format": "anime-metadata-snapshot-v1", "source": "invented-fixture",
            "snapshotAt": "2026-01-01T00:00:00Z",
            "anime": [{"animeId": anime_id, "title": f"Invented {anime_id}"}
                      for anime_id in (101, 102, 103)],
        })
        train = tuple(RawInteraction(user, anime, score, None) for user, anime, score in (
            ("invented-a", 101, 10.0), ("invented-a", 102, 7.0), ("invented-a", 103, 1.0),
            ("invented-b", 101, 9.0), ("invented-b", 102, 8.0), ("invented-b", 103, 1.0),
        ))
        fit = fit_training_partition(train, metadata)
        self.assertEqual(fit.pairs, ((101, 102, 2.5, 2), (101, 103, -0.75, 2),
                                     (102, 103, -1.75, 2)))
        self.assertEqual(fit.fit_sha256, fit_training_partition(tuple(reversed(train)), metadata).fit_sha256)

    def test_holdout_score_does_not_enter_fit_and_manifest_must_be_refreshed(self):
        changed = copy.deepcopy(self.raw)
        heldout = set(self.manifest["validationIds"] + self.manifest["testIds"])
        target = next(row for row in changed["interactions"]
                      if interaction_id(row["userId"], row["animeId"]) in heldout
                      and (row["userId"], row["animeId"]) != ("invented-a", 101))
        target["rawScore"] += 1
        changed_snapshot = parse_raw_snapshot(changed)
        with self.assertRaisesRegex(ValueError, "does not match"):
            partition_snapshot(changed_snapshot, self.manifest)
        refreshed = build_split_manifest(changed_snapshot)
        self.assertEqual(refreshed["trainIds"], self.manifest["trainIds"])
        self.assertEqual(refreshed["validationIds"], self.manifest["validationIds"])
        self.assertEqual(refreshed["testIds"], self.manifest["testIds"])
        self.assertNotEqual(refreshed["rawContentSha256"], self.manifest["rawContentSha256"])
        changed_train = partition_snapshot(changed_snapshot, refreshed).train
        fit = fit_training_partition(changed_train, self.metadata)
        self.assertEqual(fit.fit_sha256, fit_training_partition(self.train, self.metadata).fit_sha256)

    def test_pair_budget_failure_cannot_yield_partial_fit(self):
        anime = [{"animeId": index, "title": f"Invented {index}"} for index in range(1, 6401)]
        metadata = parse_metadata_snapshot({"format": "anime-metadata-snapshot-v1",
                                           "source": "invented-fixture", "snapshotAt": "2026-01-01T00:00:00Z",
                                           "anime": anime})
        train = tuple(RawInteraction("invented-dense", index, 5.0, None) for index in range(1, 6401))
        with self.assertRaisesRegex(ValueError, "Pair-visit budget exceeded"):
            fit_training_partition(train, metadata)

    def test_cli_model_fit_is_deterministic_and_reports_no_holdout_metric(self):
        command = [sys.executable, str(ROOT / "ml" / "split_first_graph_mf.py"),
                   "--raw-ratings", str(self.raw_path),
                   "--split-manifest", str(self.manifest_path),
                   "--metadata", str(self.metadata_path), "--factors", "4", "--epochs", "2"]
        first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        a, b = [json.loads(result.stdout.splitlines()[-1]) for result in (first, second)]
        self.assertEqual(a, b)
        self.assertRegex(a["modelSha256"], re.compile(r"^[0-9a-f]{64}$"))
        self.assertEqual(a["evaluation"], "none; holdout labels were not read by preprocessing or training")
        self.assertNotIn("precision", str(a).lower())


if __name__ == "__main__":
    unittest.main()
