#!/usr/bin/env python3
"""Run the M5.9 private warm-user refit path on an invented fixture only."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

from model_artifact import load_numeric_model
from raw_interaction_split import _no_duplicate_json_keys, load_raw_snapshot, partition_snapshot
from split_first_refit import refit_frozen_selection
from split_first_selection import (_write_new, parse_selection_spec,
                                   report_frozen_test, select_on_validation)
from train_only_preprocessing import load_metadata_snapshot


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    fixture = ROOT / "fixtures"
    snapshot = load_raw_snapshot(fixture / "synthetic-split-input.json")
    manifest = json.loads(
        (fixture / "synthetic-split-manifest.json").read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_json_keys,
    )
    metadata = load_metadata_snapshot(fixture / "synthetic-anime-metadata.json")
    spec = parse_selection_spec(json.loads(
        (fixture / "synthetic-mf-candidates.json").read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_json_keys,
    ))
    partitions = partition_snapshot(snapshot, manifest)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        selection_path = root / "selection.json"
        report_path = root / "test-report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            frozen = select_on_validation(
                snapshot, manifest, metadata, spec, selection_path, report_path)
            _write_new(selection_path, frozen)
            report_frozen_test(snapshot, manifest, metadata, selection_path)
            record = refit_frozen_selection(
                snapshot, manifest, metadata, selection_path, root / "refit")
        loaded = load_numeric_model(root / "refit" / "model.npz")
        if (record["trainRows"], record["validationRows"], record["testRowsExcluded"],
            record["refitRows"]) != (
                len(partitions.train), len(partitions.validation), len(partitions.test),
                len(partitions.train) + len(partitions.validation)
            ) or record["numericArchiveSha256"] != loaded.archive_sha256:
            raise ValueError("Invented refit membership or safe artifact mismatch.")
        print(json.dumps({
            "format": "synthetic-final-refit-check-v1",
            "selectedCandidateId": record["selectedCandidateId"],
            "trainRows": record["trainRows"],
            "validationRows": record["validationRows"],
            "testRowsExcluded": record["testRowsExcluded"],
            "refitRows": record["refitRows"],
            "refitModelSha256": record["refitModelSha256"],
            "evaluation": "none in refit record",
        }, sort_keys=True))


if __name__ == "__main__":
    main()
