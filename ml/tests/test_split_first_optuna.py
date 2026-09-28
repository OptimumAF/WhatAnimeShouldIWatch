from __future__ import annotations

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import search_graph_mf_optuna as search  # noqa: E402
import split_first_selection as boundary  # noqa: E402
from raw_interaction_split import build_split_manifest, parse_raw_snapshot, partition_snapshot  # noqa: E402
from train_only_preprocessing import load_metadata_snapshot  # noqa: E402


def changed_score(raw: dict, user_id: str, anime_id: int, score: int) -> dict:
    updated = copy.deepcopy(raw)
    rows = [row for row in updated["interactions"]
            if row["userId"] == user_id and row["animeId"] == anime_id]
    if len(rows) != 1:
        raise AssertionError("Invented row missing from fixed fixture.")
    rows[0]["rawScore"] = score
    return updated


class SplitFirstOptunaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((ROOT / "fixtures" / "synthetic-split-input.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((ROOT / "fixtures" / "synthetic-split-manifest.json").read_text(encoding="utf-8"))
        cls.metadata = load_metadata_snapshot(ROOT / "fixtures" / "synthetic-anime-metadata.json")
        cls.spec = boundary.parse_selection_spec(json.loads(
            (ROOT / "fixtures" / "synthetic-mf-candidates.json").read_text(encoding="utf-8")))

    def run_search(self, raw: dict, manifest: object | None = None):
        snapshot = parse_raw_snapshot(raw)
        current_manifest = manifest if manifest is not None else build_split_manifest(snapshot)
        selection_path = Path("data/invented-optuna-selection.json")
        report_path = Path("data/invented-optuna-final.json")
        with contextlib.redirect_stdout(io.StringIO()):
            result = search.search_on_validation(snapshot, current_manifest, self.metadata,
                                                 self.spec, selection_path, report_path)
        return snapshot, current_manifest, result

    def test_real_optuna_grid_matches_existing_validation_boundary(self):
        snapshot, manifest, result = self.run_search(self.raw)
        expected = boundary.select_on_validation(
            snapshot, manifest, self.metadata, self.spec,
            Path("data/invented-optuna-selection.json"), Path("data/invented-optuna-final.json"))
        self.assertEqual(result, expected)
        self.assertEqual(result["selectedCandidate"]["id"], "graph-two-epochs")
        self.assertEqual([trial["candidateId"] for trial in result["validationTrials"]],
                         [candidate.candidate_id for candidate in self.spec.candidates])
        self.assertEqual(len({trial["modelSha256"] for trial in result["validationTrials"]}), 3)

    def test_test_label_cannot_change_trial_metric_choice_or_model(self):
        _, _, baseline = self.run_search(self.raw)
        changed = changed_score(self.raw, "invented-c", 103, 1)
        snapshot = parse_raw_snapshot(changed)
        refreshed = build_split_manifest(snapshot)
        self.assertEqual(refreshed["trainIds"], self.manifest["trainIds"])
        self.assertEqual(refreshed["validationIds"], self.manifest["validationIds"])
        self.assertEqual(refreshed["testIds"], self.manifest["testIds"])
        _, _, variant = self.run_search(changed, refreshed)
        for key in ("selectedCandidate", "validationTrials", "selectedValidation",
                    "trainSha256", "fitSha256"):
            self.assertEqual(variant[key], baseline[key])
        self.assertNotEqual(variant["rawContentSha256"], baseline["rawContentSha256"])

    def test_validation_only_score_and_stale_manifest_refusal(self):
        snapshot = parse_raw_snapshot(self.raw)
        rows = partition_snapshot(snapshot, self.manifest).validation
        original = boundary.score_holdout
        scored = []

        def validation_guard(trained, exposed_rows, **kwargs):
            self.assertEqual(exposed_rows, rows)
            scored.append(True)
            return original(trained, exposed_rows, **kwargs)

        with mock.patch.object(boundary, "score_holdout", side_effect=validation_guard):
            self.run_search(self.raw)
        self.assertEqual(len(scored), len(self.spec.candidates))
        changed = changed_score(self.raw, "invented-b", 101, 9)
        _, refreshed, variant = self.run_search(changed)
        _, _, baseline = self.run_search(self.raw)
        self.assertEqual(refreshed["trainIds"], self.manifest["trainIds"])
        self.assertNotEqual(variant["validationTrials"], baseline["validationTrials"])
        self.assertEqual(variant["trainSha256"], baseline["trainSha256"])
        with self.assertRaisesRegex(ValueError, "manifest|digest|snapshot"):
            self.run_search(changed, self.manifest)

    def test_incomplete_or_duplicate_trial_is_refused(self):
        incomplete = mock.Mock()
        incomplete.trials = []
        incomplete.optimize.return_value = None
        with mock.patch.object(search.optuna, "create_study", return_value=incomplete):
            with self.assertRaisesRegex(ValueError, "did not complete"):
                self.run_search(self.raw)

        class DuplicateTrial:
            def suggest_categorical(self, _name, choices):
                return choices[0]

        duplicate = mock.Mock()
        duplicate.optimize.side_effect = lambda objective, **_kwargs: (
            objective(DuplicateTrial()), objective(DuplicateTrial()))
        with mock.patch.object(search.optuna, "create_study", return_value=duplicate):
            with self.assertRaisesRegex(ValueError, "repeated declared candidate"):
                self.run_search(self.raw)

    def test_cli_writes_one_private_selection_for_existing_final_report_gate(self):
        script = ROOT / "ml" / "search_graph_mf_optuna.py"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.json"
            report = Path(temp) / "final.json"
            command = [sys.executable, str(script), "--out-selection", str(path),
                       "--out-test-report", str(report)]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertIn("test labels were not scored", first.stdout)
            self.assertFalse(report.exists())
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["format"], "split-first-selection-v1")
            self.assertEqual(record["selectedCandidate"]["id"], "graph-two-epochs")
            again = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("distinct and unused", again.stderr)
            blocked = subprocess.run([sys.executable, str(script),
                "--out-selection", str(ROOT / "web" / "public" / "data" / "selection.json"),
                "--out-test-report", str(report)], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("public or release assets", blocked.stderr)
            self.assertFalse(Path(str(path) + ".test-used").exists())
            final_command = [sys.executable, str(ROOT / "ml" / "split_first_selection.py"),
                "report-test", "--raw-ratings", str(ROOT / "fixtures" / "synthetic-split-input.json"),
                "--split-manifest", str(ROOT / "fixtures" / "synthetic-split-manifest.json"),
                "--metadata", str(ROOT / "fixtures" / "synthetic-anime-metadata.json"),
                "--selection", str(path)]
            final = subprocess.run(final_command, cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertIn("one frozen warm-user report", final.stdout)
            self.assertTrue(Path(str(path) + ".test-used").exists())
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["selectionSha256"],
                             record["selectionSha256"])
            repeat = subprocess.run(final_command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("already has a test report", repeat.stderr)


if __name__ == "__main__":
    unittest.main()
