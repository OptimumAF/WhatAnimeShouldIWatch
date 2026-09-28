import contextlib
import copy
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import split_first_selection as selection  # noqa: E402
from raw_interaction_split import RawInteraction, build_split_manifest, parse_raw_snapshot, partition_snapshot  # noqa: E402
from split_first_graph_mf import SplitFirstTraining  # noqa: E402
from train_graph_mf import Dataset, Split  # noqa: E402
from train_only_preprocessing import load_metadata_snapshot  # noqa: E402


def change_one(raw: dict, user_id: str, anime_id: int, score: int) -> dict:
    changed = copy.deepcopy(raw)
    matches = [row for row in changed["interactions"]
               if row["userId"] == user_id and row["animeId"] == anime_id]
    if len(matches) != 1:
        raise AssertionError("Predeclared invented row is missing or duplicated.")
    matches[0]["rawScore"] = score
    return changed


class SplitFirstSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_path = ROOT / "fixtures" / "synthetic-split-input.json"
        cls.manifest_path = ROOT / "fixtures" / "synthetic-split-manifest.json"
        cls.metadata_path = ROOT / "fixtures" / "synthetic-anime-metadata.json"
        cls.candidates_path = ROOT / "fixtures" / "synthetic-mf-candidates.json"
        cls.raw = json.loads(cls.raw_path.read_text(encoding="utf-8"))
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.metadata = load_metadata_snapshot(cls.metadata_path)
        cls.spec = selection.parse_selection_spec(json.loads(cls.candidates_path.read_text(encoding="utf-8")))

    def select(self, raw: dict, selection_path: Path, report_path: Path):
        snapshot = parse_raw_snapshot(raw)
        manifest = build_split_manifest(snapshot)
        with contextlib.redirect_stdout(io.StringIO()):
            record = selection.select_on_validation(snapshot, manifest, self.metadata,
                                                     self.spec, selection_path, report_path)
        return snapshot, manifest, record

    def test_metric_is_hand_computed_with_mask_tie_and_score_floor(self):
        dataset = Dataset(["invented-u"], [1, 2, 3, 4], ["A", "B", "C", "D"], [(0, 0, 1.0)])
        split = Split(np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
                      np.array([1.0], dtype=np.float32), [{0}], {}, 0)
        model = {"P": np.array([[1.0]], dtype=np.float32),
                 "Q": np.zeros((4, 1), dtype=np.float32),
                 "bu": np.zeros(1, dtype=np.float32),
                 "bi": np.array([9.0, 0.9, 0.9, 0.7], dtype=np.float32),
                 "global_mean": 0.0}
        trained = SplitFirstTraining(None, dataset, split, np.zeros((0, 3), dtype=np.float32), model)
        metric = selection.score_holdout(trained, (RawInteraction("invented-u", 3, 8.0, None),),
                                         top_k=2, positive_raw_score_min=7,
                                         model_score_floor=0.8)
        self.assertEqual(metric["eligibleUsers"], 1)
        self.assertEqual(metric["hitsAtK"], 1)
        self.assertAlmostEqual(metric["ndcgAtK"], 1 / math.log2(3))
        self.assertEqual(metric["recallAtK"], 1.0)
        with self.assertRaisesRegex(ValueError, "No warm user"):
            selection.score_holdout(trained, (RawInteraction("invented-u", 3, 6.0, None),),
                                    top_k=2, positive_raw_score_min=7, model_score_floor=None)

    def test_test_score_edits_cannot_change_validation_choice(self):
        selection_path, report_path = Path("data/choice.json"), Path("data/report.json")
        _, _, baseline = self.select(self.raw, selection_path, report_path)
        changed = change_one(self.raw, "invented-c", 103, 1)
        _, changed_manifest, variant = self.select(changed, selection_path, report_path)
        self.assertEqual(changed_manifest["trainIds"], self.manifest["trainIds"])
        self.assertEqual(changed_manifest["validationIds"], self.manifest["validationIds"])
        self.assertEqual(changed_manifest["testIds"], self.manifest["testIds"])
        self.assertEqual(variant["selectedCandidate"], baseline["selectedCandidate"])
        self.assertEqual(variant["validationTrials"], baseline["validationTrials"])
        self.assertEqual(variant["selectedValidation"], baseline["selectedValidation"])
        self.assertEqual(variant["trainSha256"], baseline["trainSha256"])
        self.assertEqual(variant["fitSha256"], baseline["fitSha256"])
        self.assertNotEqual(variant["rawContentSha256"], baseline["rawContentSha256"])
        self.assertEqual(baseline["selectedCandidate"]["id"], "graph-two-epochs")
        self.assertEqual(baseline["selectedValidation"]["eligibleUsers"], 1)
        self.assertFalse(any(user_id in json.dumps(baseline) for user_id in (
            "invented-a", "invented-b", "invented-c", "invented-sparse")))

    def test_validation_label_edit_changes_only_validation_results(self):
        path, report = Path("data/choice.json"), Path("data/report.json")
        _, _, baseline = self.select(self.raw, path, report)
        changed = change_one(self.raw, "invented-b", 101, 9)
        _, manifest, variant = self.select(changed, path, report)
        self.assertEqual(manifest["trainIds"], self.manifest["trainIds"])
        self.assertEqual(manifest["testIds"], self.manifest["testIds"])
        self.assertNotEqual(variant["validationTrials"], baseline["validationTrials"])
        self.assertEqual(variant["selectedValidation"]["eligibleUsers"], 2)
        self.assertEqual(variant["trainSha256"], baseline["trainSha256"])
        self.assertEqual(variant["fitSha256"], baseline["fitSha256"])

    def test_frozen_report_scores_test_once_after_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            selection_path, report_path = Path(temp) / "selection.json", Path(temp) / "test.json"
            snapshot, manifest, record = self.select(self.raw, selection_path, report_path)
            selection._write_new(selection_path, record)
            expected_test = partition_snapshot(snapshot, manifest).test
            original_score = selection.score_holdout
            calls = []

            def guarded_score(trained, rows, **kwargs):
                self.assertEqual(rows, expected_test)
                self.assertTrue(Path(str(selection_path) + ".test-used").exists())
                self.assertFalse(report_path.exists())
                calls.append(True)
                return original_score(trained, rows, **kwargs)

            with mock.patch.object(selection, "score_holdout", side_effect=guarded_score):
                with contextlib.redirect_stdout(io.StringIO()):
                    report = selection.report_frozen_test(snapshot, manifest, self.metadata,
                                                          selection_path)
            self.assertEqual(calls, [True])
            self.assertEqual(report["selectedCandidateId"], record["selectedCandidate"]["id"])
            self.assertEqual(report["test"]["eligibleUsers"], 2)
            self.assertEqual(report["test"]["positiveLabels"], 2)
            self.assertEqual(report["test"]["hitsAtK"], 1)
            self.assertEqual(report["test"]["recallAtK"], 0.5)
            self.assertFalse(any(user_id in json.dumps(report) for user_id in (
                "invented-a", "invented-b", "invented-c", "invented-sparse")))
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)
            with self.assertRaisesRegex(ValueError, "already has a test report or one-use marker"):
                selection.report_frozen_test(snapshot, manifest, self.metadata, selection_path)

    def test_stale_snapshot_tampering_copy_and_public_output_fail_before_test(self):
        with tempfile.TemporaryDirectory() as temp:
            selection_path, report_path = Path(temp) / "selection.json", Path(temp) / "test.json"
            snapshot, manifest, record = self.select(self.raw, selection_path, report_path)
            selection._write_new(selection_path, record)
            changed = change_one(self.raw, "invented-c", 103, 1)
            changed_snapshot = parse_raw_snapshot(changed)
            changed_manifest = build_split_manifest(changed_snapshot)
            with self.assertRaisesRegex(ValueError, "does not match the raw snapshot"):
                selection.report_frozen_test(changed_snapshot, changed_manifest, self.metadata,
                                             selection_path)
            tampered = copy.deepcopy(record)
            tampered["selectedCandidate"]["id"] = "plain-one-epoch"
            selection_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity digest"):
                selection.report_frozen_test(snapshot, manifest, self.metadata, selection_path)
            selection_path.write_text(json.dumps(record), encoding="utf-8")
            copied_path = Path(temp) / "copied.json"
            copied_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection path"):
                selection.report_frozen_test(snapshot, manifest, self.metadata, copied_path)
            self.assertFalse(report_path.exists())
            self.assertFalse(Path(str(selection_path) + ".test-used").exists())
            with self.assertRaisesRegex(ValueError, "public or release assets"):
                selection._private_output(ROOT / "web" / "public" / "data" / "invented-selection.json")

    def test_failure_after_test_marker_cannot_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            selection_path, report_path = Path(temp) / "selection.json", Path(temp) / "test.json"
            snapshot, manifest, record = self.select(self.raw, selection_path, report_path)
            selection._write_new(selection_path, record)
            with mock.patch.object(selection, "score_holdout", side_effect=ValueError("invented scoring failure")):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "invented scoring failure"):
                        selection.report_frozen_test(snapshot, manifest, self.metadata, selection_path)
            self.assertTrue(Path(str(selection_path) + ".test-used").exists())
            self.assertFalse(report_path.exists())
            with self.assertRaisesRegex(ValueError, "already has a test report or one-use marker"):
                selection.report_frozen_test(snapshot, manifest, self.metadata, selection_path)

    def test_model_drift_refuses_test_before_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            selection_path, report_path = Path(temp) / "selection.json", Path(temp) / "test.json"
            snapshot, manifest, record = self.select(self.raw, selection_path, report_path)
            selection._write_new(selection_path, record)
            chosen = selection.parse_candidate(record["selectedCandidate"])
            with contextlib.redirect_stdout(io.StringIO()):
                trained = chosen.train(snapshot, manifest, self.metadata, self.spec.model_seed)
            changed_model = dict(trained.model)
            changed_model["P"] = np.asarray(trained.model["P"]).copy()
            changed_model["P"][0, 0] += 0.25
            drifted = SplitFirstTraining(trained.fit, trained.dataset, trained.split,
                                         trained.graph_edges, changed_model)
            with mock.patch.object(selection.Candidate, "train", return_value=drifted):
                with self.assertRaisesRegex(ValueError, "model parameters no longer match"):
                    selection.report_frozen_test(snapshot, manifest, self.metadata, selection_path)
            self.assertFalse(Path(str(selection_path) + ".test-used").exists())
            self.assertFalse(report_path.exists())

    def test_candidate_spec_rejects_invalid_or_undeclared_choice(self):
        raw = json.loads(self.candidates_path.read_text(encoding="utf-8"))
        extra = copy.deepcopy(raw)
        extra["candidates"][0]["blendWeight"] = 0.5
        with self.assertRaisesRegex(ValueError, "exactly the declared MF fields"):
            selection.parse_selection_spec(extra)
        duplicate = copy.deepcopy(raw)
        duplicate["candidates"][1]["id"] = duplicate["candidates"][0]["id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            selection.parse_selection_spec(duplicate)
        nonfinite = copy.deepcopy(raw)
        nonfinite["candidates"][0]["lr"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            selection.parse_selection_spec(nonfinite)

    def test_file_backed_cli_freezes_before_one_final_report(self):
        with tempfile.TemporaryDirectory() as temp:
            selection_path, report_path = Path(temp) / "selection.json", Path(temp) / "test.json"
            shared = ["--raw-ratings", str(self.raw_path),
                      "--split-manifest", str(self.manifest_path),
                      "--metadata", str(self.metadata_path)]
            script = str(ROOT / "ml" / "split_first_selection.py")
            chosen = subprocess.run([
                sys.executable, script, "select", *shared,
                "--candidates", str(self.candidates_path),
                "--out-selection", str(selection_path),
                "--out-test-report", str(report_path),
            ], cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertIn("test labels were not scored", chosen.stdout)
            self.assertTrue(selection_path.exists())
            self.assertFalse(report_path.exists())
            final = subprocess.run([
                sys.executable, script, "report-test", *shared,
                "--selection", str(selection_path),
            ], cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertIn("one frozen synthetic test report", final.stdout)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["format"], "split-first-final-test-v1")
            self.assertEqual(report["selectedCandidateId"], "graph-two-epochs")
            again = subprocess.run([
                sys.executable, script, "report-test", *shared,
                "--selection", str(selection_path),
            ], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("already has a test report", again.stderr)


if __name__ == "__main__":
    unittest.main()
