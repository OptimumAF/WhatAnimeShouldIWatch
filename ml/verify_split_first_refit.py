#!/usr/bin/env python3
"""Read-only reproduction of a frozen split-first MF refit from private rows."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from export_model_web import build_payload
from model_artifact import load_numeric_model, metadata_path
from raw_interaction_split import load_raw_snapshot, partition_snapshot
from split_first_graph_mf import model_fingerprint, training_arrays
from split_first_refit import (_allow_cli_source, _checked_final_report, _read_json, _sha_file)
from split_first_selection import _checked_selection, _private_output
from train_graph_mf import train_graph_mf
from train_only_preprocessing import fit_training_partition, load_metadata_snapshot


_SOURCE_LIMITS = {"raw-ratings.json": 128 * 1024 * 1024,
                  "split-manifest.json": 16 * 1024 * 1024,
                  "anime-metadata.json": 16 * 1024 * 1024}
_EVIDENCE_LIMITS = {"model.npz": 256 * 1024 * 1024,
                    "model.metadata.json": 1024 * 1024,
                    "model-mf-web.compact.json": 64 * 1024 * 1024,
                    "refit-record.json": 1024 * 1024}


def _bounded(path: Path, limit: int, field: str) -> None:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f"{field} must be a bounded regular private file.")


def _same(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} differs from the validated frozen refit.")


def verify_refit(raw: Path, split: Path, metadata_file: Path,
                 selection_path: Path, evidence_dir: Path,
                 dataset_sha256: str | None = None) -> dict[str, object]:
    """Retrain in memory and compare every fitted array; emit only aggregate digests."""
    for name, source in (("raw-ratings.json", raw), ("split-manifest.json", split),
                         ("anime-metadata.json", metadata_file)):
        _bounded(source, _SOURCE_LIMITS[name], name)
    if selection_path.is_symlink() or evidence_dir.is_symlink():
        raise ValueError("selection and refit evidence must be real private paths.")
    selection_path = _private_output(selection_path)
    _bounded(selection_path, 1024 * 1024, "selection.json")
    evidence_dir = _private_output(evidence_dir)
    if not evidence_dir.is_dir() or evidence_dir.is_symlink():
        raise ValueError("refit evidence must be a real private directory.")
    for name, limit in _EVIDENCE_LIMITS.items():
        _bounded(evidence_dir / name, limit, name)

    snapshot = load_raw_snapshot(raw)
    manifest = _read_json(split)
    metadata = load_metadata_snapshot(metadata_file)
    selection = _read_json(selection_path)
    if not isinstance(selection, dict):
        raise ValueError("selection.json must be an object.")
    report_path = _private_output(Path(selection["testReportPath"]))
    marker_path = _private_output(Path(str(selection_path) + ".test-used"))
    _bounded(report_path, 1024 * 1024, "final-report.json")
    _bounded(_private_output(Path(str(report_path) + ".sha256.json")), 1024 * 1024,
             "final-report.sha256.json")
    _bounded(marker_path, 1024 * 1024, "selection.test-used")
    spec, candidate = _checked_selection(selection, selection_path, report_path,
                                         manifest, metadata)
    partitions = partition_snapshot(snapshot, manifest)
    report_sha = _checked_final_report(selection, report_path, marker_path)

    with contextlib.redirect_stdout(io.StringIO()):
        selected_train = candidate.train(snapshot, manifest, metadata, spec.model_seed)
    _same(selected_train.fit.train_sha256, selection["trainSha256"],
          "selection.trainSha256")
    _same(selected_train.fit.fit_sha256, selection["fitSha256"],
          "selection.fitSha256")
    selected_trial = next(trial for trial in selection["validationTrials"]
                          if trial["candidateId"] == candidate.candidate_id)
    _same(model_fingerprint(selected_train.model), selected_trial["modelSha256"],
          "selection.selectedModelSha256")

    fit = fit_training_partition(partitions.train + partitions.validation, metadata)
    dataset, indexed, graph_edges = training_arrays(fit, candidate.graph_min_weight)
    with contextlib.redirect_stdout(io.StringIO()):
        expected_model = train_graph_mf(
            indexed, len(dataset.user_ids), len(dataset.anime_ids), graph_edges,
            candidate.factors, candidate.epochs, candidate.lr, candidate.reg,
            candidate.reg_bias, candidate.graph_lambda, candidate.graph_sample_rate,
            spec.model_seed)
    archive_path = evidence_dir / "model.npz"
    loaded = load_numeric_model(archive_path)
    _same(loaded.user_ids, dataset.user_ids, "model.metadata.json.userIds")
    _same(loaded.anime_ids, dataset.anime_ids, "model.npz.anime_ids")
    _same(loaded.anime_titles, dataset.anime_titles, "model.metadata.json.animeTitles")
    _same(loaded.train_user_items, indexed.train_user_items,
          "model.npz.train_item_indices")
    for name, actual in (("P", loaded.p), ("Q", loaded.q), ("bu", loaded.bu),
                         ("bi", loaded.bi)):
        if not np.array_equal(actual, np.asarray(expected_model[name], dtype=np.float32)):
            raise ValueError(f"model.npz.{name} differs from the selected split-first fit.")
    _same(loaded.global_mean, float(np.float32(expected_model["global_mean"])),
          "model.npz.global_mean")

    web_path = evidence_dir / "model-mf-web.compact.json"
    web = _read_json(web_path)
    if not isinstance(web, dict):
        raise ValueError("model-mf-web.compact.json must be an object.")
    expected_web = build_payload(archive_path, "compact", 8)
    expected_web["generatedAt"] = web.get("generatedAt")
    if dataset_sha256 is not None:
        expected_web["datasetSha256"] = dataset_sha256
    _same(web, expected_web, "model-mf-web.compact.json")

    record = _read_json(evidence_dir / "refit-record.json")
    expected_record = {
        "format": "split-first-final-refit-v1",
        "selectionSha256": selection["selectionSha256"],
        "finalReportSha256": report_sha,
        "selectedCandidateId": candidate.candidate_id,
        "candidateSpecSha256": selection["candidateSpecSha256"],
        "rawContentSha256": manifest["rawContentSha256"],
        "splitIdentitySha256": manifest["identitySha256"],
        "metadataSha256": metadata.sha256,
        "fitMembership": "train-plus-validation",
        "trainRows": len(partitions.train),
        "validationRows": len(partitions.validation),
        "testRowsExcluded": len(partitions.test),
        "refitRows": len(partitions.train) + len(partitions.validation),
        "originalTrainSha256": selection["trainSha256"],
        "refitTrainSha256": fit.train_sha256,
        "refitFitSha256": fit.fit_sha256,
        "refitModelSha256": model_fingerprint(expected_model),
        "numericArchiveSha256": loaded.archive_sha256,
        "numericMetadataSha256": _sha_file(metadata_path(archive_path)),
        "webModelSha256": _sha_file(web_path),
        "evaluation": "none; this record contains no held-out quality metric",
        "releaseStatus": "unapproved experiment artifact",
    }
    if not isinstance(record, dict) or set(record) != set(expected_record):
        raise ValueError("refit-record.json fields are unsupported or incomplete.")
    for field, expected in expected_record.items():
        _same(record[field], expected, f"refit-record.json.{field}")
    return {"format": "split-first-refit-reproduction-v1",
            "selectionSha256": selection["selectionSha256"],
            "candidateSpecSha256": selection["candidateSpecSha256"],
            "modelSeed": spec.model_seed,
            "refitModelSha256": record["refitModelSha256"],
            "numericArchiveSha256": loaded.archive_sha256,
            "trainRows": len(partitions.train),
            "validationRows": len(partitions.validation),
            "testRowsExcluded": len(partitions.test)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ratings", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--training-approval-ref")
    parser.add_argument("--dataset-sha256")
    args = parser.parse_args()
    try:
        _allow_cli_source(args.raw_ratings, args.split_manifest, args.metadata,
                          args.training_approval_ref)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Refit reproduction blocked: source/use approval missing or invalid ({type(exc).__name__}).\n")
    try:
        report = verify_refit(args.raw_ratings, args.split_manifest, args.metadata,
                              args.selection, args.evidence_dir, args.dataset_sha256)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, IndexError,
            StopIteration):
        parser.exit(1, "Refit reproduction blocked: private input or evidence mismatch.\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
