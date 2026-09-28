#!/usr/bin/env python3
"""Private final MF fit from train+validation after frozen synthetic selection."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
from pathlib import Path

from export_model_web import export_model
from model_artifact import load_numeric_model, save_numeric_model
from raw_interaction_split import RawSnapshot, _no_duplicate_json_keys, load_raw_snapshot, partition_snapshot
from split_first_graph_mf import model_fingerprint, training_arrays
from split_first_selection import _checked_selection, _private_output
from train_graph_mf import train_graph_mf
from train_only_preprocessing import MetadataSnapshot, fit_training_partition, load_metadata_snapshot


ROOT = Path(__file__).resolve().parents[1]
_TEST_REPORT_FIELDS = {
    "format", "selectionSha256", "selectedCandidateId", "modelSha256", "test", "status",
}
_TEST_METRIC_FIELDS = {"eligibleUsers", "positiveLabels", "hitsAtK", "ndcgAtK", "recallAtK"}
_TEST_STATUS = "single synthetic warm-user report; no release claim"


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_json_keys)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_unused_output(out_dir: Path) -> Path:
    output = out_dir.resolve()
    if output.exists():
        raise ValueError("Final refit output directory must be unused.")
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "models"):
        raise ValueError("Final refit output inside the repository must be under ignored models/.")
    if output.is_relative_to(ROOT / "web" / "public") or output.is_relative_to(ROOT / "release-data"):
        raise ValueError("Final refit cannot write public or release assets.")
    if not output.parent.exists():
        raise ValueError("Final refit output parent directory must already exist.")
    return output


def _checked_final_report(selection: dict, report_path: Path,
                          marker_path: Path) -> str:
    digest_path = _private_output(Path(str(report_path) + ".sha256.json"))
    if not report_path.is_file() or not marker_path.is_file() or not digest_path.is_file():
        raise ValueError("Frozen selection has no completed one-use final report, digest, and marker.")
    report = _read_json(report_path)
    marker = _read_json(marker_path)
    digest = _read_json(digest_path)
    report_sha = _sha_file(report_path)
    if digest != {"format": "split-first-final-test-digest-v1",
                  "selectionSha256": selection["selectionSha256"],
                  "reportSha256": report_sha}:
        raise ValueError("Final report bytes do not match their recorded digest.")
    if not isinstance(report, dict) or set(report) != _TEST_REPORT_FIELDS or (
        report["format"] != "split-first-final-test-v1" or
        report["status"] != _TEST_STATUS or
        report["selectionSha256"] != selection["selectionSha256"] or
        report["selectedCandidateId"] != selection["selectedCandidate"]["id"]
    ):
        raise ValueError("Final report does not match the frozen selection.")
    selected_trial = next(
        trial for trial in selection["validationTrials"]
        if trial["candidateId"] == selection["selectedCandidate"]["id"]
    )
    if report["modelSha256"] != selected_trial["modelSha256"]:
        raise ValueError("Final report selected model fingerprint changed.")
    if marker != {"format": "split-first-test-used-v1",
                  "selectionSha256": selection["selectionSha256"]}:
        raise ValueError("Final report one-use marker does not match the selection.")
    metric = report["test"]
    if not isinstance(metric, dict) or set(metric) != _TEST_METRIC_FIELDS:
        raise ValueError("Final report metric fields are malformed.")
    for field in ("eligibleUsers", "positiveLabels", "hitsAtK"):
        if type(metric[field]) is not int or metric[field] < 0:
            raise ValueError(f"Final report {field} is malformed.")
    for field in ("ndcgAtK", "recallAtK"):
        if type(metric[field]) not in (int, float) or not (
            math.isfinite(metric[field]) and 0 <= metric[field] <= 1
        ):
            raise ValueError(f"Final report {field} is malformed.")
    if metric["eligibleUsers"] < 1 or metric["positiveLabels"] < 1 or (
        metric["hitsAtK"] > metric["positiveLabels"]
    ):
        raise ValueError("Final report counts are malformed.")
    return report_sha


def refit_frozen_selection(snapshot: RawSnapshot, manifest: object,
                           metadata: MetadataSnapshot, selection_path: Path,
                           out_dir: Path) -> dict[str, object]:
    """Validate selection/report, then fit only the immutable train+validation rows."""
    selection_path = _private_output(selection_path)
    output = _private_unused_output(out_dir)
    selection = _read_json(selection_path)
    if not isinstance(selection, dict):
        raise ValueError("Frozen selection must be an object.")
    report_path = _private_output(Path(selection["testReportPath"]))
    marker_path = _private_output(Path(str(selection_path) + ".test-used"))
    spec, candidate = _checked_selection(
        selection, selection_path, report_path, manifest, metadata
    )
    partitions = partition_snapshot(snapshot, manifest)
    report_sha = _checked_final_report(selection, report_path, marker_path)

    # Recreate the selected train-only model before altering fit membership.
    with contextlib.redirect_stdout(io.StringIO()):
        selected_train = candidate.train(snapshot, manifest, metadata, spec.model_seed)
    if (selected_train.fit.train_sha256, selected_train.fit.fit_sha256) != (
        selection["trainSha256"], selection["fitSha256"]
    ):
        raise ValueError("Frozen train-only preprocessing changed before final refit.")
    selected_trial = next(
        trial for trial in selection["validationTrials"]
        if trial["candidateId"] == candidate.candidate_id
    )
    if model_fingerprint(selected_train.model) != selected_trial["modelSha256"]:
        raise ValueError("Frozen train-only model changed before final refit.")

    fit_rows = partitions.train + partitions.validation
    fit = fit_training_partition(fit_rows, metadata)
    dataset, split, graph_edges = training_arrays(fit, candidate.graph_min_weight)
    with contextlib.redirect_stdout(io.StringIO()):
        model = train_graph_mf(
            split, len(dataset.user_ids), len(dataset.anime_ids), graph_edges,
            candidate.factors, candidate.epochs, candidate.lr, candidate.reg,
            candidate.reg_bias, candidate.graph_lambda, candidate.graph_sample_rate,
            spec.model_seed,
        )
    refit_model_sha = model_fingerprint(model)
    if fit.train_sha256 == selection["trainSha256"]:
        raise ValueError("Final refit did not add validation rows to its fit.")

    output.mkdir()
    archive_path = output / "model.npz"
    sidecar_path = save_numeric_model(
        archive_path,
        p=model["P"], q=model["Q"], bu=model["bu"], bi=model["bi"],
        global_mean=float(model["global_mean"]),
        user_ids=dataset.user_ids, anime_ids=dataset.anime_ids,
        anime_titles=dataset.anime_titles,
        train_user_items=split.train_user_items,
    )
    loaded = load_numeric_model(archive_path)
    web_path = output / "model-mf-web.compact.json"
    web = export_model(archive_path, web_path, "compact", 8)
    if web["sourceModelSha256"] != loaded.archive_sha256:
        raise ValueError("Final refit web model digest disagrees with numeric model.")
    record: dict[str, object] = {
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
        "refitRows": len(fit_rows),
        "originalTrainSha256": selection["trainSha256"],
        "refitTrainSha256": fit.train_sha256,
        "refitFitSha256": fit.fit_sha256,
        "refitModelSha256": refit_model_sha,
        "numericArchiveSha256": loaded.archive_sha256,
        "numericMetadataSha256": _sha_file(sidecar_path),
        "webModelSha256": _sha_file(web_path),
        "evaluation": "none; this record contains no held-out quality metric",
        "releaseStatus": "unapproved experiment artifact",
    }
    (output / "refit-record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not sidecar_path.is_file():
        raise ValueError("Final refit metadata sidecar was not written.")
    return record


def _allow_cli_source(raw: Path, manifest: Path, metadata: Path,
                      training_approval_ref: str | None) -> None:
    # Paths alone cannot make edited or substituted data an approved fixture.
    fixture_inputs = (
        (raw, "synthetic-split-input.json",
         "cf6c53386175613610363b90ace3deb473bc93edbc348e2b099d91634bc23b18"),
        (manifest, "synthetic-split-manifest.json",
         "3b21b267a7a9eaa0b677ca7ad1a20ae208b3df2ad7b2826fefeccdfe78bd6b76"),
        (metadata, "synthetic-anime-metadata.json",
         "2a2ec17221c63a46d44d85699e72f193f31c1cf5c7f7741a6151b462776f0694"),
    )
    fixture = all(
        source.resolve() == ROOT / "fixtures" / name and
        hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected
        for source, name, expected in fixture_inputs
    )
    if fixture:
        return
    if not training_approval_ref:
        raise ValueError("Non-fixture refit requires a recorded training source/use approval.")
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_provider_data_approval import validate_approval  # noqa: PLC0415
    approval = _read_json(ROOT / "docs" / "approvals" / "provider-data.json")
    validate_approval(approval, "training", training_approval_ref, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Private final MF fit; never scores test labels.")
    parser.add_argument("--raw-ratings", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--training-approval-ref")
    args = parser.parse_args()
    try:
        _allow_cli_source(args.raw_ratings, args.split_manifest, args.metadata,
                          args.training_approval_ref)
        snapshot = load_raw_snapshot(args.raw_ratings)
        manifest = _read_json(args.split_manifest)
        metadata = load_metadata_snapshot(args.metadata)
        record = refit_frozen_selection(snapshot, manifest, metadata,
                                        args.selection, args.out_dir)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Final refit blocked: {exc}\n")
    print(json.dumps(record, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
