#!/usr/bin/env python3
"""Build an invented frozen selection and actual refit for package tests only."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
from pathlib import Path

from raw_interaction_split import load_raw_snapshot
from split_first_refit import refit_frozen_selection
from split_first_selection import (_load_json, _write_new, parse_selection_spec,
                                   report_frozen_test, select_on_validation)
from train_only_preprocessing import load_metadata_snapshot


ROOT = Path(__file__).resolve().parents[1]
REFIT_FILES = ("model.npz", "model.metadata.json", "model-mf-web.compact.json",
               "refit-record.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    args = parser.parse_args()

    # This executable must never become a shortcut for selecting or fitting
    # provider rows. Its sole admitted input is the test's declared invention.
    raw = _load_json(args.source_dir / "raw-ratings.json")
    expected = _load_json(ROOT / "fixtures" / "synthetic-split-input.json")
    expected["interactions"].append({"userId": "invented-a", "animeId": 106,
                                     "rawScore": 7})
    if raw != expected or _load_json(args.source_dir / "anime-metadata.json") != _load_json(
        ROOT / "fixtures" / "synthetic-anime-metadata.json"
    ):
        parser.exit(1, "Invented promotion refit accepts only its declared synthetic rows and metadata.\n")

    snapshot = load_raw_snapshot(args.source_dir / "raw-ratings.json")
    manifest = _load_json(args.source_dir / "split-manifest.json")
    metadata = load_metadata_snapshot(args.source_dir / "anime-metadata.json")
    spec = parse_selection_spec(_load_json(ROOT / "fixtures" / "synthetic-mf-candidates.json"))
    selection_path = args.evidence_dir / "selection.json"
    report_path = args.evidence_dir / "final-report.json"
    with contextlib.redirect_stdout(io.StringIO()):
        selection = select_on_validation(snapshot, manifest, metadata, spec,
                                         selection_path, report_path)
    _write_new(selection_path, selection)
    with contextlib.redirect_stdout(io.StringIO()):
        report_frozen_test(snapshot, manifest, metadata, selection_path)
        record = refit_frozen_selection(snapshot, manifest, metadata, selection_path,
                                        args.evidence_dir.parent / "fixture-refit",
                                        args.dataset_sha256)
    refit_dir = args.evidence_dir.parent / "fixture-refit"
    for filename in REFIT_FILES:
        shutil.move(str(refit_dir / filename), str(args.evidence_dir / filename))
    refit_dir.rmdir()
    shutil.copyfile(args.evidence_dir / "model-mf-web.compact.json",
                    args.candidate_dir / "model-mf-web.compact.json")
    print(json.dumps({"selectionSha256": selection["selectionSha256"],
                      "selectedCandidateId": selection["selectedCandidate"]["id"],
                      "refitModelSha256": record["refitModelSha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
