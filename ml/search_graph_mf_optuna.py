#!/usr/bin/env python3
"""Optuna grid over declared split-first MF candidates; validation only."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import optuna

import split_first_selection as boundary
from raw_interaction_split import RawSnapshot, load_raw_snapshot, partition_snapshot
from train_only_preprocessing import MetadataSnapshot, load_metadata_snapshot


ROOT = Path(__file__).resolve().parents[1]


def search_on_validation(snapshot: RawSnapshot, manifest: object, metadata: MetadataSnapshot,
                         spec: boundary.SelectionSpec, selection_path: Path,
                         report_path: Path) -> dict[str, Any]:
    """Visit every predeclared candidate once; test is never a trial input."""
    validation_rows = partition_snapshot(snapshot, manifest).validation
    candidates = {candidate.candidate_id: candidate for candidate in spec.candidates}
    candidate_ids = list(candidates)
    completed: dict[str, tuple[str, str, dict[str, Any]]] = {}
    sampler = optuna.samplers.GridSampler({"candidateId": candidate_ids}, seed=spec.model_seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        candidate_id = trial.suggest_categorical("candidateId", candidate_ids)
        if candidate_id in completed:
            raise ValueError(f"Optuna repeated declared candidate {candidate_id}.")
        # The underlying trainer prints aggregate epoch progress. Selection
        # emits only its frozen candidate and no held-out metric.
        with contextlib.redirect_stdout(io.StringIO()):
            result = boundary.score_validation_candidate(
                snapshot, manifest, metadata, spec, candidates[candidate_id], validation_rows)
        completed[candidate_id] = result
        return result[2]["validation"]["ndcgAtK"]

    study.optimize(objective, n_trials=len(candidate_ids), n_jobs=1, show_progress_bar=False)
    if len(study.trials) != len(candidate_ids) or set(completed) != set(candidate_ids) or any(
        trial.state != optuna.trial.TrialState.COMPLETE or
        trial.params.get("candidateId") not in completed or
        trial.value != completed[trial.params["candidateId"]][2]["validation"]["ndcgAtK"]
        for trial in study.trials
    ):
        raise ValueError("Optuna did not complete the declared validation grid exactly once.")
    ordered = [completed[candidate_id] for candidate_id in candidate_ids]
    train_sha, fit_sha = ordered[0][:2]
    if any(result[:2] != (train_sha, fit_sha) for result in ordered):
        raise ValueError("Candidate training inputs or preprocessing changed during Optuna selection.")
    return boundary.freeze_validation_trials(
        manifest, metadata, spec, selection_path, report_path,
        [result[2] for result in ordered], train_sha, fit_sha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a validation-selected synthetic Optuna MF candidate.")
    parser.add_argument("--raw-ratings", type=Path, default=ROOT / "fixtures" / "synthetic-split-input.json")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "fixtures" / "synthetic-split-manifest.json")
    parser.add_argument("--metadata", type=Path, default=ROOT / "fixtures" / "synthetic-anime-metadata.json")
    parser.add_argument("--candidates", type=Path, default=ROOT / "fixtures" / "synthetic-mf-candidates.json")
    parser.add_argument("--out-selection", type=Path, default=ROOT / "data" / "synthetic-optuna-selection-local.json")
    parser.add_argument("--out-test-report", type=Path, default=ROOT / "data" / "synthetic-optuna-final-test-local.json")
    args = parser.parse_args()
    try:
        selection_path = boundary._private_output(args.out_selection)
        report_path = boundary._private_output(args.out_test_report)
        marker_path = boundary._private_output(Path(str(selection_path) + ".test-used"))
        if (selection_path == report_path or selection_path.exists() or
                report_path.exists() or marker_path.exists()):
            raise ValueError("Selection and test paths must be distinct and unused.")
        snapshot = load_raw_snapshot(args.raw_ratings)
        manifest = boundary._load_json(args.split_manifest)
        metadata = load_metadata_snapshot(args.metadata)
        spec = boundary.parse_selection_spec(boundary._load_json(args.candidates))
        record = search_on_validation(snapshot, manifest, metadata, spec, selection_path, report_path)
        boundary._write_new(selection_path, record)
        print(f"Froze Optuna validation-selected candidate {record['selectedCandidate']['id']} "
              f"at {selection_path}; test labels were not scored.")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Split-first Optuna selection failed: {exc}\n")


if __name__ == "__main__":
    main()
