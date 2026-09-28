"""Invented final-refit tests; no provider data or release output."""

from __future__ import annotations

import contextlib
import copy
import hashlib
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

import split_first_refit as refit  # noqa: E402
import split_first_selection as selection  # noqa: E402
from export_model_web import export_model  # noqa: E402
from model_artifact import load_numeric_model, metadata_path, save_numeric_model  # noqa: E402
from raw_interaction_split import build_split_manifest, parse_raw_snapshot  # noqa: E402
from train_only_preprocessing import load_metadata_snapshot  # noqa: E402
from verify_split_first_refit import verify_refit  # noqa: E402


RAW_PATH = ROOT / "fixtures" / "synthetic-split-input.json"
MANIFEST_PATH = ROOT / "fixtures" / "synthetic-split-manifest.json"
METADATA_PATH = ROOT / "fixtures" / "synthetic-anime-metadata.json"
CANDIDATES_PATH = ROOT / "fixtures" / "synthetic-mf-candidates.json"


def changed_score(raw: dict, user_id: str, anime_id: int, score: float) -> dict:
    value = copy.deepcopy(raw)
    matches = [row for row in value["interactions"]
               if row["userId"] == user_id and row["animeId"] == anime_id]
    if len(matches) != 1:
        raise AssertionError("Invented perturbation target is absent.")
    matches[0]["rawScore"] = score
    return value


class FinalRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
        cls.metadata = load_metadata_snapshot(METADATA_PATH)
        cls.spec = selection.parse_selection_spec(json.loads(
            CANDIDATES_PATH.read_text(encoding="utf-8")))

    def freeze(self, raw: dict, root: Path) -> tuple[object, dict, Path, Path]:
        snapshot = parse_raw_snapshot(raw)
        manifest = build_split_manifest(snapshot)
        selection_path = root / "selection.json"
        report_path = root / "test-report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            record = selection.select_on_validation(
                snapshot, manifest, self.metadata, self.spec,
                selection_path, report_path,
            )
        selection._write_new(selection_path, record)
        with contextlib.redirect_stdout(io.StringIO()):
            selection.report_frozen_test(
                snapshot, manifest, self.metadata, selection_path,
            )
        return snapshot, manifest, selection_path, report_path

    def test_file_backed_cli_writes_separate_safe_refit_without_test_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _snapshot, _manifest, selection_path, report_path = self.freeze(self.raw, root)
            out_dir = root / "refit"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "ml" / "split_first_refit.py"),
                 "--raw-ratings", str(RAW_PATH),
                 "--split-manifest", str(MANIFEST_PATH),
                 "--metadata", str(METADATA_PATH),
                 "--selection", str(selection_path),
                 "--out-dir", str(out_dir)],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
            record = json.loads((out_dir / "refit-record.json").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(completed.stdout), record)
            self.assertEqual(record["format"], "split-first-final-refit-v1")
            self.assertEqual((record["trainRows"], record["validationRows"],
                              record["testRowsExcluded"], record["refitRows"]), (7, 3, 3, 10))
            self.assertEqual(record["fitMembership"], "train-plus-validation")
            self.assertNotEqual(record["originalTrainSha256"], record["refitTrainSha256"])
            self.assertNotIn("test", record)
            self.assertNotIn("ndcgAtK", json.dumps(record))
            self.assertNotIn("invented-a", completed.stdout + completed.stderr)
            self.assertEqual(record["finalReportSha256"], refit._sha_file(report_path))
            self.assertTrue(Path(str(report_path) + ".sha256.json").is_file())
            loaded = load_numeric_model(out_dir / "model.npz")
            self.assertEqual(len(loaded.user_ids), 4)
            self.assertEqual(loaded.archive_sha256, record["numericArchiveSha256"])
            self.assertEqual(hashlib.sha256((out_dir / "model.metadata.json").read_bytes()).hexdigest(),
                             record["numericMetadataSha256"])
            self.assertEqual(refit.model_fingerprint({
                "P": loaded.p, "Q": loaded.q, "bu": loaded.bu, "bi": loaded.bi,
                "global_mean": loaded.global_mean,
            }), record["refitModelSha256"])
            self.assertTrue((out_dir / "model.metadata.json").is_file())
            web = json.loads((out_dir / "model-mf-web.compact.json").read_text(encoding="utf-8"))
            self.assertEqual(web["sourceModelSha256"], loaded.archive_sha256)
            self.assertEqual(web["animeIds"], loaded.anime_ids)
            with self.assertRaisesRegex(ValueError, "unused"):
                refit.refit_frozen_selection(
                    parse_raw_snapshot(self.raw), build_split_manifest(parse_raw_snapshot(self.raw)),
                    self.metadata, selection_path, out_dir)

    def test_test_score_change_cannot_change_refit_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_dir = root / "original"
            changed_dir = root / "changed"
            original_dir.mkdir()
            changed_dir.mkdir()
            original_snapshot, original_manifest, original_selection, _ = self.freeze(
                self.raw, original_dir)
            changed = changed_score(self.raw, "invented-c", 103, 1)
            changed_snapshot, changed_manifest, changed_selection, _ = self.freeze(
                changed, changed_dir)
            with contextlib.redirect_stdout(io.StringIO()):
                original = refit.refit_frozen_selection(
                    original_snapshot, original_manifest, self.metadata,
                    original_selection, original_dir / "refit")
                variant = refit.refit_frozen_selection(
                    changed_snapshot, changed_manifest, self.metadata,
                    changed_selection, changed_dir / "refit")
            self.assertEqual(original_manifest["trainIds"], changed_manifest["trainIds"])
            self.assertEqual(original_manifest["validationIds"], changed_manifest["validationIds"])
            self.assertEqual(original_manifest["testIds"], changed_manifest["testIds"])
            for field in ("selectedCandidateId", "refitTrainSha256", "refitFitSha256",
                          "refitModelSha256", "metadataSha256", "refitRows"):
                self.assertEqual(original[field], variant[field], field)
            self.assertNotEqual(original["rawContentSha256"], variant["rawContentSha256"])
            self.assertNotEqual(original["finalReportSha256"], variant["finalReportSha256"])
            left = load_numeric_model(original_dir / "refit" / "model.npz")
            right = load_numeric_model(changed_dir / "refit" / "model.npz")
            for field in ("p", "q", "bu", "bi"):
                np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
            self.assertEqual(left.global_mean, right.global_mean)
            self.assertEqual(left.train_user_items, right.train_user_items)

    def test_read_only_reproduction_checks_the_selected_refit_and_cli_source_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, manifest, selection_path, _ = self.freeze(self.raw, root)
            with contextlib.redirect_stdout(io.StringIO()):
                record = refit.refit_frozen_selection(
                    snapshot, manifest, self.metadata, selection_path, root / "refit")
            report = verify_refit(RAW_PATH, MANIFEST_PATH, METADATA_PATH,
                                  selection_path, root / "refit")
            self.assertEqual(report["format"], "split-first-refit-reproduction-v1")
            self.assertEqual(report["refitModelSha256"], record["refitModelSha256"])
            self.assertEqual(report["modelSeed"], self.spec.model_seed)
            self.assertEqual((report["trainRows"], report["validationRows"],
                              report["testRowsExcluded"]), (7, 3, 3))
            command = [sys.executable, str(ROOT / "ml" / "verify_split_first_refit.py"),
                       "--raw-ratings", str(RAW_PATH), "--split-manifest", str(MANIFEST_PATH),
                       "--metadata", str(METADATA_PATH), "--selection", str(selection_path),
                       "--evidence-dir", str(root / "refit")]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       text=True, check=True)
            self.assertEqual(json.loads(completed.stdout), report)
            self.assertNotIn("invented-a", completed.stdout + completed.stderr)
            copied = root / "copied-raw.json"
            copied.write_bytes(RAW_PATH.read_bytes())
            command[command.index(str(RAW_PATH))] = str(copied)
            refused = subprocess.run(command, cwd=ROOT, capture_output=True,
                                     text=True, check=False)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("source/use approval", refused.stderr)
            self.assertNotIn("invented-a", refused.stdout + refused.stderr)

    def test_rehashed_numeric_arrays_and_selection_seed_cannot_fake_refit_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, manifest, selection_path, report_path = self.freeze(self.raw, root)
            with contextlib.redirect_stdout(io.StringIO()):
                refit.refit_frozen_selection(snapshot, manifest, self.metadata,
                                             selection_path, root / "changed-archive")
            out_dir = root / "changed-archive"
            archive_path = out_dir / "model.npz"
            loaded = load_numeric_model(archive_path)
            changed_q = loaded.q.copy()
            changed_q[0, 0] += np.float32(0.125)
            save_numeric_model(
                archive_path, p=loaded.p, q=changed_q, bu=loaded.bu, bi=loaded.bi,
                global_mean=loaded.global_mean, user_ids=loaded.user_ids,
                anime_ids=loaded.anime_ids, anime_titles=loaded.anime_titles,
                train_user_items=loaded.train_user_items)
            changed = load_numeric_model(archive_path)
            export_model(archive_path, out_dir / "model-mf-web.compact.json", "compact", 8)
            record_path = out_dir / "refit-record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record.update({
                "refitModelSha256": refit.model_fingerprint({
                    "P": changed.p, "Q": changed.q, "bu": changed.bu, "bi": changed.bi,
                    "global_mean": changed.global_mean}),
                "numericArchiveSha256": changed.archive_sha256,
                "numericMetadataSha256": refit._sha_file(metadata_path(archive_path)),
                "webModelSha256": refit._sha_file(out_dir / "model-mf-web.compact.json"),
            })
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"model\.npz\.Q differs"):
                verify_refit(RAW_PATH, MANIFEST_PATH, METADATA_PATH,
                             selection_path, out_dir)

            with contextlib.redirect_stdout(io.StringIO()):
                refit.refit_frozen_selection(snapshot, manifest, self.metadata,
                                             selection_path, root / "changed-seed")
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            selected["selectionSpec"]["modelSeed"] += 1
            selected["candidateSpecSha256"] = selection._sha(selected["selectionSpec"])
            selected["selectionSha256"] = selection._sha({
                key: value for key, value in selected.items() if key != "selectionSha256"})
            selection_path.write_text(json.dumps(selected), encoding="utf-8")
            final = json.loads(report_path.read_text(encoding="utf-8"))
            final["selectionSha256"] = selected["selectionSha256"]
            report_path.write_text(json.dumps(final), encoding="utf-8")
            digest_path = Path(str(report_path) + ".sha256.json")
            digest = json.loads(digest_path.read_text(encoding="utf-8"))
            digest["selectionSha256"] = selected["selectionSha256"]
            digest["reportSha256"] = refit._sha_file(report_path)
            digest_path.write_text(json.dumps(digest), encoding="utf-8")
            marker_path = Path(str(selection_path) + ".test-used")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["selectionSha256"] = selected["selectionSha256"]
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            refit_record_path = root / "changed-seed" / "refit-record.json"
            refit_record = json.loads(refit_record_path.read_text(encoding="utf-8"))
            refit_record["selectionSha256"] = selected["selectionSha256"]
            refit_record["candidateSpecSha256"] = selected["candidateSpecSha256"]
            refit_record["finalReportSha256"] = digest["reportSha256"]
            refit_record_path.write_text(json.dumps(refit_record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection.selectedModelSha256"):
                verify_refit(RAW_PATH, MANIFEST_PATH, METADATA_PATH,
                             selection_path, root / "changed-seed")

            selected["selectionSpec"]["modelSeed"] = self.spec.model_seed
            selected_id = selected["selectedCandidate"]["id"]
            selected["selectedCandidate"]["epochs"] += 1
            for candidate in selected["selectionSpec"]["candidates"]:
                if candidate["id"] == selected_id:
                    candidate["epochs"] += 1
            selected["candidateSpecSha256"] = selection._sha(selected["selectionSpec"])
            selected["selectionSha256"] = selection._sha({
                key: value for key, value in selected.items() if key != "selectionSha256"})
            selection_path.write_text(json.dumps(selected), encoding="utf-8")
            final["selectionSha256"] = selected["selectionSha256"]
            report_path.write_text(json.dumps(final), encoding="utf-8")
            digest["selectionSha256"] = selected["selectionSha256"]
            digest["reportSha256"] = refit._sha_file(report_path)
            digest_path.write_text(json.dumps(digest), encoding="utf-8")
            marker["selectionSha256"] = selected["selectionSha256"]
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            refit_record["selectionSha256"] = selected["selectionSha256"]
            refit_record["candidateSpecSha256"] = selected["candidateSpecSha256"]
            refit_record["finalReportSha256"] = digest["reportSha256"]
            refit_record_path.write_text(json.dumps(refit_record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection.selectedModelSha256"):
                verify_refit(RAW_PATH, MANIFEST_PATH, METADATA_PATH,
                             selection_path, root / "changed-seed")

    def test_validation_score_change_changes_refit_under_same_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            base_snapshot, base_manifest, base_selection, _ = self.freeze(self.raw, first)
            altered = changed_score(self.raw, "invented-b", 101, 8.25)
            changed_snapshot, changed_manifest, changed_selection, _ = self.freeze(
                altered, second)
            with contextlib.redirect_stdout(io.StringIO()):
                base = refit.refit_frozen_selection(
                    base_snapshot, base_manifest, self.metadata,
                    base_selection, first / "refit")
                variant = refit.refit_frozen_selection(
                    changed_snapshot, changed_manifest, self.metadata,
                    changed_selection, second / "refit")
            self.assertEqual(base["selectedCandidateId"], variant["selectedCandidateId"])
            self.assertEqual(base["originalTrainSha256"], variant["originalTrainSha256"])
            self.assertNotEqual(base["refitTrainSha256"], variant["refitTrainSha256"])
            self.assertNotEqual(base["refitFitSha256"], variant["refitFitSha256"])
            self.assertNotEqual(base["refitModelSha256"], variant["refitModelSha256"])

    def test_stale_tampered_report_and_public_output_fail_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, manifest, selection_path, report_path = self.freeze(self.raw, root)
            old = json.loads(report_path.read_text(encoding="utf-8"))
            altered = changed_score(self.raw, "invented-c", 103, 1)
            stale_snapshot = parse_raw_snapshot(altered)
            stale_manifest = build_split_manifest(stale_snapshot)
            with self.assertRaisesRegex(ValueError, "raw snapshot"):
                refit.refit_frozen_selection(
                    stale_snapshot, stale_manifest, self.metadata, selection_path,
                    root / "stale")
            self.assertFalse((root / "stale").exists())
            changed = dict(old, selectedCandidateId="invented-wrong")
            report_path.write_text(json.dumps(changed), encoding="utf-8")
            digest_path = Path(str(report_path) + ".sha256.json")
            original_digest = json.loads(digest_path.read_text(encoding="utf-8"))
            digest_path.write_text(json.dumps({
                **original_digest, "reportSha256": refit._sha_file(report_path)}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Final report does not match"):
                refit.refit_frozen_selection(
                    snapshot, manifest, self.metadata, selection_path, root / "tampered")
            self.assertFalse((root / "tampered").exists())
            report_path.write_text(json.dumps(old), encoding="utf-8")
            digest_path.write_text(json.dumps(original_digest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "report bytes"):
                refit.refit_frozen_selection(
                    snapshot, manifest, self.metadata, selection_path, root / "changed-bytes")
            self.assertFalse((root / "changed-bytes").exists())
            report_path.write_text(
                json.dumps(old, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8")
            Path(str(selection_path) + ".test-used").unlink()
            with self.assertRaisesRegex(ValueError, "one-use final report"):
                refit.refit_frozen_selection(
                    snapshot, manifest, self.metadata, selection_path, root / "missing")
            self.assertFalse((root / "missing").exists())
            original_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            altered_selection = copy.deepcopy(original_selection)
            altered_selection["selectedCandidate"]["id"] = "plain-one-epoch"
            selection_path.write_text(json.dumps(altered_selection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity digest"):
                refit.refit_frozen_selection(
                    snapshot, manifest, self.metadata, selection_path, root / "bad-selection")
            self.assertFalse((root / "bad-selection").exists())
            with self.assertRaisesRegex(ValueError, "under ignored models"):
                refit.refit_frozen_selection(
                    snapshot, manifest, self.metadata, selection_path,
                    ROOT / "web" / "public" / "data" / "invented-refit")

    def test_non_fixture_cli_refuses_unapproved_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied_raw = root / "raw.json"
            copied_raw.write_text(RAW_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "ml" / "split_first_refit.py"),
                 "--raw-ratings", str(copied_raw),
                 "--split-manifest", str(MANIFEST_PATH),
                 "--metadata", str(METADATA_PATH),
                 "--selection", str(root / "absent.json"),
                 "--out-dir", str(root / "blocked")],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("recorded training source/use approval", completed.stderr)
            self.assertFalse((root / "blocked").exists())
            with_ref = subprocess.run(
                [sys.executable, str(ROOT / "ml" / "split_first_refit.py"),
                 "--raw-ratings", str(copied_raw),
                 "--split-manifest", str(MANIFEST_PATH),
                 "--metadata", str(METADATA_PATH),
                 "--selection", str(root / "absent.json"),
                 "--out-dir", str(root / "blocked"),
                 "--training-approval-ref", "https://example.test/invented-approval"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(with_ref.returncode, 0)
            self.assertIn("training has no recorded approval", with_ref.stderr)
            self.assertFalse((root / "blocked").exists())


if __name__ == "__main__":
    unittest.main()
