#!/usr/bin/env python3
"""Validation-only MF selection followed by one frozen synthetic test report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from raw_interaction_split import RawInteraction, RawSnapshot, load_raw_snapshot, partition_snapshot, _no_duplicate_json_keys
from split_first_graph_mf import SplitFirstTraining, fit_split_first, model_fingerprint
from train_only_preprocessing import MetadataSnapshot, load_metadata_snapshot


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_FIELDS = {"id", "factors", "epochs", "lr", "reg", "regBias", "graphLambda",
                    "graphMinWeight", "graphSampleRate", "modelScoreFloor"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_json_keys)


def _nonnegative_number(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    number = float(value)
    if (number <= 0 if positive else number < 0):
        raise ValueError(f"{field} must be {'positive' if positive else 'nonnegative'}.")
    return number


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    factors: int
    epochs: int
    lr: float
    reg: float
    reg_bias: float
    graph_lambda: float
    graph_min_weight: float
    graph_sample_rate: float
    model_score_floor: float | None

    def as_record(self) -> dict[str, Any]:
        return {"id": self.candidate_id, "factors": self.factors, "epochs": self.epochs,
                "lr": self.lr, "reg": self.reg, "regBias": self.reg_bias,
                "graphLambda": self.graph_lambda, "graphMinWeight": self.graph_min_weight,
                "graphSampleRate": self.graph_sample_rate,
                "modelScoreFloor": self.model_score_floor}

    def train(self, snapshot: RawSnapshot, manifest: object, metadata: MetadataSnapshot,
              seed: int) -> SplitFirstTraining:
        return fit_split_first(
            snapshot, manifest, metadata, factors=self.factors, epochs=self.epochs,
            lr=self.lr, reg=self.reg, reg_bias=self.reg_bias,
            graph_lambda=self.graph_lambda, graph_min_weight=self.graph_min_weight,
            graph_sample_rate=self.graph_sample_rate, seed=seed,
        )


def parse_candidate(value: object) -> Candidate:
    if not isinstance(value, dict) or set(value) != CANDIDATE_FIELDS:
        raise ValueError("Candidate requires exactly the declared MF fields.")
    candidate_id = value["id"]
    if not isinstance(candidate_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", candidate_id):
        raise ValueError("Candidate ID must be a short lowercase identifier.")
    factors, epochs = value["factors"], value["epochs"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (factors, epochs)) or not (
        1 <= factors <= 256 and 1 <= epochs <= 100
    ):
        raise ValueError("Candidate factors and epochs must be bounded positive integers.")
    lr = _nonnegative_number(value["lr"], "lr", positive=True)
    reg = _nonnegative_number(value["reg"], "reg")
    reg_bias = _nonnegative_number(value["regBias"], "regBias")
    graph_lambda = _nonnegative_number(value["graphLambda"], "graphLambda")
    graph_min_weight = _nonnegative_number(value["graphMinWeight"], "graphMinWeight")
    sample_rate = _nonnegative_number(value["graphSampleRate"], "graphSampleRate", positive=True)
    if sample_rate > 1:
        raise ValueError("graphSampleRate must be at most one.")
    score_floor = value["modelScoreFloor"]
    if score_floor is not None and (isinstance(score_floor, bool) or
                                    not isinstance(score_floor, (int, float)) or
                                    not math.isfinite(score_floor)):
        raise ValueError("modelScoreFloor must be null or finite.")
    return Candidate(candidate_id, factors, epochs, lr, reg, reg_bias, graph_lambda,
                     graph_min_weight, sample_rate, None if score_floor is None else float(score_floor))


@dataclass(frozen=True)
class SelectionSpec:
    top_k: int
    positive_raw_score_min: float
    model_seed: int
    candidates: tuple[Candidate, ...]

    def as_record(self) -> dict[str, Any]:
        return {"format": "split-first-selection-candidates-v1", "objective": "ndcg-at-k",
                "topK": self.top_k, "positiveRawScoreMin": self.positive_raw_score_min,
                "modelSeed": self.model_seed,
                "candidates": [candidate.as_record() for candidate in self.candidates]}


def parse_selection_spec(value: object) -> SelectionSpec:
    if not isinstance(value, dict) or set(value) != {
        "format", "objective", "topK", "positiveRawScoreMin", "modelSeed", "candidates"
    } or value["format"] != "split-first-selection-candidates-v1" or value["objective"] != "ndcg-at-k":
        raise ValueError("Unsupported selection candidate specification.")
    top_k, seed, raw_candidates = value["topK"], value["modelSeed"], value["candidates"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("topK must be a positive integer at most 100.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("modelSeed must be an unsigned 32-bit integer.")
    positive_min = _nonnegative_number(value["positiveRawScoreMin"], "positiveRawScoreMin")
    if not isinstance(raw_candidates, list) or not 2 <= len(raw_candidates) <= 20:
        raise ValueError("Selection requires 2–20 predeclared candidates.")
    candidates = tuple(parse_candidate(candidate) for candidate in raw_candidates)
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique.")
    return SelectionSpec(top_k, positive_min, seed, candidates)


def score_holdout(result: SplitFirstTraining, rows: tuple[RawInteraction, ...], *,
                  top_k: int, positive_raw_score_min: float,
                  model_score_floor: float | None) -> dict[str, float | int]:
    """Warm-user metric; callers choose which partition to expose."""
    user_to_index = {user_id: index for index, user_id in enumerate(result.dataset.user_ids)}
    anime_to_index = {anime_id: index for index, anime_id in enumerate(result.dataset.anime_ids)}
    positives: dict[int, set[int]] = {}
    for row in rows:
        if row.user_id not in user_to_index or row.anime_id not in anime_to_index:
            raise ValueError("A held-out interaction is outside the fixed train-user/metadata universe.")
        if row.raw_score >= positive_raw_score_min:
            positives.setdefault(user_to_index[row.user_id], set()).add(anime_to_index[row.anime_id])
    p = np.asarray(result.model["P"], dtype=np.float32)
    q = np.asarray(result.model["Q"], dtype=np.float32)
    bu = np.asarray(result.model["bu"], dtype=np.float32)
    bi = np.asarray(result.model["bi"], dtype=np.float32)
    mean = float(result.model["global_mean"])
    ndcgs: list[float] = []
    recalls: list[float] = []
    hits = 0
    for user_index, actual in sorted(positives.items()):
        scores = mean + float(bu[user_index]) + bi + q @ p[user_index]
        if not np.isfinite(scores).all():
            raise ValueError("Model produced a non-finite candidate score.")
        seen = result.split.train_user_items[user_index]
        eligible = [index for index, anime_id in enumerate(result.dataset.anime_ids)
                    if index not in seen and
                    (model_score_floor is None or float(scores[index]) >= model_score_floor)]
        eligible.sort(key=lambda index: (-float(scores[index]), result.dataset.anime_ids[index]))
        top = eligible[:top_k]
        hit_ranks = [rank for rank, index in enumerate(top, start=1) if index in actual]
        hits += len(hit_ranks)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hit_ranks)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(actual), top_k) + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
        recalls.append(len(hit_ranks) / len(actual))
    if not ndcgs:
        raise ValueError("No warm user has a positive held-out label under the fixed policy.")
    return {"eligibleUsers": len(ndcgs), "positiveLabels": sum(map(len, positives.values())),
            "hitsAtK": hits, "ndcgAtK": float(np.mean(ndcgs)),
            "recallAtK": float(np.mean(recalls))}


def score_validation_candidate(snapshot: RawSnapshot, manifest: object, metadata: MetadataSnapshot,
                               spec: SelectionSpec, candidate: Candidate,
                               validation_rows: tuple[RawInteraction, ...]
                               ) -> tuple[str, str, dict[str, Any]]:
    """Fit one declared candidate on train and score only supplied validation rows."""
    if candidate not in spec.candidates:
        raise ValueError("Validation candidate is absent from the predeclared set.")
    trained = candidate.train(snapshot, manifest, metadata, spec.model_seed)
    metric = score_holdout(trained, validation_rows, top_k=spec.top_k,
                           positive_raw_score_min=spec.positive_raw_score_min,
                           model_score_floor=candidate.model_score_floor)
    trial = {"candidateId": candidate.candidate_id, "validation": metric,
             "modelSha256": model_fingerprint(trained.model)}
    return trained.fit.train_sha256, trained.fit.fit_sha256, trial


def freeze_validation_trials(manifest: object, metadata: MetadataSnapshot, spec: SelectionSpec,
                             selection_path: Path, report_path: Path,
                             trials: Sequence[dict[str, Any]], train_sha: str,
                             fit_sha: str) -> dict[str, Any]:
    """Bind a complete predeclared validation search to the one-use report boundary."""
    def valid_trial(trial: object) -> bool:
        if not isinstance(trial, dict) or set(trial) != {
            "candidateId", "validation", "modelSha256"
        } or not isinstance(trial["validation"], dict):
            return False
        score = trial["validation"].get("ndcgAtK")
        return (isinstance(score, (int, float)) and not isinstance(score, bool) and
                math.isfinite(score) and 0 <= score <= 1 and
                isinstance(trial["modelSha256"], str) and
                re.fullmatch(r"[a-f0-9]{64}", trial["modelSha256"]) is not None)

    if not all(valid_trial(trial) for trial in trials):
        raise ValueError("Validation trials must contain finite metrics and model fingerprints.")
    if [trial.get("candidateId") for trial in trials] != [
        candidate.candidate_id for candidate in spec.candidates
    ]:
        raise ValueError("Validation trials must cover each declared candidate exactly once.")
    if not all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
               for value in (train_sha, fit_sha)):
        raise ValueError("Validation training fingerprints are missing.")
    best = sorted(trials, key=lambda trial: (-trial["validation"]["ndcgAtK"], trial["candidateId"]))[0]
    selected = next(candidate for candidate in spec.candidates if candidate.candidate_id == best["candidateId"])
    # Bind the exact snapshot only for final reporting; it is never a selection score.
    record = {"format": "split-first-selection-v1", "selectionPath": str(selection_path.resolve()),
              "testReportPath": str(report_path.resolve()),
              "rawContentSha256": manifest["rawContentSha256"],
              "identitySha256": manifest["identitySha256"],
              "metadataSha256": metadata.sha256, "trainSha256": train_sha,
              "fitSha256": fit_sha, "candidateSpecSha256": _sha(spec.as_record()),
              "selectionSpec": spec.as_record(), "selectedCandidate": selected.as_record(),
              "validationTrials": list(trials), "objective": "ndcg-at-k",
              "selectedValidation": best["validation"]}
    return {**record, "selectionSha256": _sha(record)}


def select_on_validation(snapshot: RawSnapshot, manifest: object, metadata: MetadataSnapshot,
                         spec: SelectionSpec, selection_path: Path,
                         report_path: Path) -> dict[str, Any]:
    """No access to the test partition's labels or scores while choosing a candidate."""
    partitions = partition_snapshot(snapshot, manifest)
    trials: list[dict[str, Any]] = []
    train_sha: str | None = None
    fit_sha: str | None = None
    for candidate in spec.candidates:
        candidate_train_sha, candidate_fit_sha, trial = score_validation_candidate(
            snapshot, manifest, metadata, spec, candidate, partitions.validation)
        if train_sha is None:
            train_sha, fit_sha = candidate_train_sha, candidate_fit_sha
        elif (candidate_train_sha, candidate_fit_sha) != (train_sha, fit_sha):
            raise ValueError("Candidate training inputs or preprocessing changed during validation selection.")
        trials.append(trial)
    return freeze_validation_trials(manifest, metadata, spec, selection_path, report_path,
                                    trials, train_sha, fit_sha)


def _checked_selection(record: object, selection_path: Path, report_path: Path,
                       manifest: object, metadata: MetadataSnapshot) -> tuple[SelectionSpec, Candidate]:
    fields = {"format", "selectionPath", "testReportPath", "rawContentSha256",
              "identitySha256", "metadataSha256", "trainSha256", "fitSha256",
              "candidateSpecSha256", "selectionSpec", "selectedCandidate",
              "validationTrials", "objective", "selectedValidation", "selectionSha256"}
    if not isinstance(record, dict) or set(record) != fields or record.get("format") != "split-first-selection-v1":
        raise ValueError("Unsupported frozen selection format.")
    digest = record.get("selectionSha256")
    payload = {key: value for key, value in record.items() if key != "selectionSha256"}
    if digest != _sha(payload):
        raise ValueError("Frozen selection integrity digest does not match.")
    if record.get("selectionPath") != str(selection_path.resolve()) or record.get("testReportPath") != str(report_path.resolve()):
        raise ValueError("Frozen selection path or test report path does not match.")
    if record.get("rawContentSha256") != manifest["rawContentSha256"] or record.get("identitySha256") != manifest["identitySha256"]:
        raise ValueError("Frozen selection does not match the raw snapshot and split manifest.")
    if record.get("metadataSha256") != metadata.sha256:
        raise ValueError("Frozen selection metadata snapshot changed.")
    spec = parse_selection_spec(record.get("selectionSpec"))
    if record.get("candidateSpecSha256") != _sha(spec.as_record()):
        raise ValueError("Frozen candidate specification changed.")
    selected = parse_candidate(record.get("selectedCandidate"))
    if selected not in spec.candidates:
        raise ValueError("Frozen candidate is absent from the predeclared set.")
    trials = record.get("validationTrials")
    if not isinstance(trials, list) or len(trials) != len(spec.candidates):
        raise ValueError("Frozen validation trials are missing.")
    if [trial.get("candidateId") for trial in trials] != [candidate.candidate_id for candidate in spec.candidates]:
        raise ValueError("Frozen validation trials do not match the candidate list.")
    ranked = sorted(trials, key=lambda trial: (-trial["validation"]["ndcgAtK"], trial["candidateId"]))
    if ranked[0]["candidateId"] != selected.candidate_id or record.get("selectedValidation") != ranked[0]["validation"]:
        raise ValueError("Frozen candidate disagrees with validation-only selection.")
    return spec, selected


def _private_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT / "web" / "public") or resolved.is_relative_to(ROOT / "release-data"):
        raise ValueError("Selection and test reports must stay out of public or release assets.")
    return resolved


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def report_frozen_test(snapshot: RawSnapshot, manifest: object, metadata: MetadataSnapshot,
                       selection_path: Path) -> dict[str, Any]:
    """Finalize the original frozen selection; one marker precedes test scoring."""
    selection_path = _private_output(selection_path)
    record = _load_json(selection_path)
    report_path = _private_output(Path(record["testReportPath"]))
    marker_path = _private_output(Path(str(selection_path) + ".test-used"))
    if report_path.exists() or marker_path.exists():
        raise ValueError("This frozen selection already has a test report or one-use marker.")
    spec, candidate = _checked_selection(record, selection_path, report_path, manifest, metadata)
    partitions = partition_snapshot(snapshot, manifest)
    trained = candidate.train(snapshot, manifest, metadata, spec.model_seed)
    if (trained.fit.train_sha256, trained.fit.fit_sha256) != (record["trainSha256"], record["fitSha256"]):
        raise ValueError("Frozen training data or preprocessing changed.")
    selected_trial = next(trial for trial in record["validationTrials"]
                          if trial["candidateId"] == candidate.candidate_id)
    if model_fingerprint(trained.model) != selected_trial["modelSha256"]:
        raise ValueError("Frozen model parameters no longer match validation selection.")
    # The raw snapshot is parsed and validated before this point. The marker
    # precedes scoring or inspecting the test partition for a report.
    _write_new(marker_path, {"format": "split-first-test-used-v1",
                             "selectionSha256": record["selectionSha256"]})
    metric = score_holdout(trained, partitions.test, top_k=spec.top_k,
                           positive_raw_score_min=spec.positive_raw_score_min,
                           model_score_floor=candidate.model_score_floor)
    report = {"format": "split-first-final-test-v1",
              "selectionSha256": record["selectionSha256"],
              "selectedCandidateId": candidate.candidate_id,
              "modelSha256": model_fingerprint(trained.model),
              "test": metric,
              "status": "single synthetic warm-user report; no release claim"}
    _write_new(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only selection and single frozen test report.")
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    final = commands.add_parser("report-test")
    for command in (select, final):
        command.add_argument("--raw-ratings", type=Path, required=True)
        command.add_argument("--split-manifest", type=Path, required=True)
        command.add_argument("--metadata", type=Path, required=True)
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--out-selection", type=Path, required=True)
    select.add_argument("--out-test-report", type=Path, required=True)
    final.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    try:
        snapshot = load_raw_snapshot(args.raw_ratings)
        manifest = _load_json(args.split_manifest)
        metadata = load_metadata_snapshot(args.metadata)
        if args.command == "select":
            selection_path = _private_output(args.out_selection)
            report_path = _private_output(args.out_test_report)
            marker_path = _private_output(Path(str(selection_path) + ".test-used"))
            if (selection_path == report_path or selection_path.exists() or
                    report_path.exists() or marker_path.exists()):
                raise ValueError("Selection and test paths must be distinct and unused.")
            spec = parse_selection_spec(_load_json(args.candidates))
            record = select_on_validation(snapshot, manifest, metadata, spec, selection_path, report_path)
            _write_new(selection_path, record)
            print(f"Froze validation-selected candidate {record['selectedCandidate']['id']} "
                  f"at {selection_path}; test labels were not scored.")
        else:
            report_frozen_test(snapshot, manifest, metadata, args.selection)
            record = _load_json(args.selection)
            report_path = Path(record["testReportPath"])
            print(f"Wrote one frozen synthetic test report to {report_path}.")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Split-first selection failed: {exc}\n")


if __name__ == "__main__":
    main()
