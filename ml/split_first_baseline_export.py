#!/usr/bin/env python3
"""Restricted train-only model and baseline statistics for M5.6's invented cohort."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from raw_interaction_split import _no_duplicate_json_keys, load_raw_snapshot, partition_snapshot
from split_first_graph_mf import model_fingerprint
from split_first_selection import parse_selection_spec, score_holdout
from train_graph_mf import train_graph_mf
from train_only_preprocessing import CenteredTrainRow, load_metadata_snapshot

METHODS = [
    "train-count", "metadata-score", "supported-adjusted-cosine", "genre-overlap",
    "v1-positive-pair-graph", "plain-mf", "positive-pair-mf", "unit-positive-pair-mf",
    "shrunk-positive-pair-mf", "hybrid-default-0.5",
]
HASH_FIELDS = {
    "fitSnapshotSha256", "fitManifestSha256", "metadataSha256",
    "validationSha256", "mfCandidatesSha256",
}
INPUT_FIELDS = {
    "fitSnapshotSha256": "raw",
    "fitManifestSha256": "manifest",
    "metadataSha256": "metadata",
    "validationSha256": "validation",
    "mfCandidatesSha256": "candidates",
}


def normalized_file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def parse_spec(value: object) -> dict[str, Any]:
    expected = HASH_FIELDS | {
        "format", "modelCandidateId", "suppliedCounts", "positiveRawScoreMin",
        "topK", "similarityMinSupport", "similarityShrinkage", "graphShrinkage",
        "hybridModelWeight", "missingSignalScore", "tieBreak", "objective", "methods",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Baseline specification has missing or extra fields.")
    if any(not isinstance(value[key], str) or len(value[key]) != 64 or
           any(char not in "0123456789abcdef" for char in value[key])
           for key in HASH_FIELDS):
        raise ValueError("Baseline specification has an invalid input hash.")
    for key in ("positiveRawScoreMin", "topK", "similarityMinSupport",
                "similarityShrinkage", "graphShrinkage", "missingSignalScore"):
        if type(value[key]) is not int:
            raise ValueError(f"Baseline specification {key} must be an integer.")
    if type(value["hybridModelWeight"]) not in (int, float) or not math.isfinite(
            value["hybridModelWeight"]):
        raise ValueError("Baseline specification hybridModelWeight must be finite.")
    if (value["format"] != "split-first-baseline-ablation-spec-v1" or
            value["modelCandidateId"] != "graph-two-epochs" or
            value["suppliedCounts"] != [1, 3, 5, 10] or
            value["positiveRawScoreMin"] != 7 or value["topK"] != 10 or
            value["similarityMinSupport"] != 2 or value["similarityShrinkage"] != 2 or
            value["graphShrinkage"] != 2 or value["hybridModelWeight"] != 0.5 or
            value["missingSignalScore"] != 0 or value["tieBreak"] != "anime-id-ascending" or
            value["objective"] != "mean-displayed-ndcg-at-10" or value["methods"] != METHODS):
        raise ValueError("Baseline specification protocol differs from decision 0021.")
    return value


def similarity_from_train(rows: tuple[CenteredTrainRow, ...], min_support: int,
                          shrinkage: float) -> tuple[list[dict[str, int | float]], dict[str, int]]:
    """Only train-centered co-rater observations can enter this neighborhood."""
    if min_support < 1 or not math.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("Invalid similarity support or shrinkage.")
    by_user: dict[str, list[CenteredTrainRow]] = {}
    for row in rows:
        by_user.setdefault(row.user_id, []).append(row)
    observations: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for ratings in by_user.values():
        ordered = sorted(ratings, key=lambda row: row.anime_id)
        for left, right in itertools.combinations(ordered, 2):
            observations.setdefault((left.anime_id, right.anime_id), []).append(
                (left.normalized_score, right.normalized_score)
            )
    positive: list[dict[str, int | float]] = []
    defined = 0
    negative_or_zero = 0
    low_support = 0
    for (left, right), values in sorted(observations.items()):
        if len(values) < min_support:
            low_support += 1
            continue
        dot = math.fsum(a * b for a, b in values)
        left_square = math.fsum(a * a for a, _ in values)
        right_square = math.fsum(b * b for _, b in values)
        denominator = math.sqrt(left_square * right_square)
        if denominator == 0 or not math.isfinite(denominator):
            continue
        cosine = max(-1.0, min(1.0, dot / denominator))
        defined += 1
        weight = cosine * len(values) / (len(values) + shrinkage)
        if weight <= 0:
            negative_or_zero += 1
            continue
        positive.append({
            "leftAnimeId": left, "rightAnimeId": right, "support": len(values),
            "adjustedCosine": cosine, "weight": weight,
        })
    return positive, {
        "observedPairs": len(observations), "lowSupportPairs": low_support,
        "definedSupportedPairs": defined, "nonpositiveSupportedPairs": negative_or_zero,
        "positiveSupportedPairs": len(positive),
    }


def compact_model(model: dict[str, np.ndarray | float], dataset: Any,
                  snapshot_at: str, factors: int) -> dict[str, Any]:
    return {
        "format": "model-mf-compact-v1",
        "generatedAt": snapshot_at,
        "globalMean": float(model["global_mean"]),
        "factors": factors,
        "animeIds": dataset.anime_ids,
        "titles": dataset.anime_titles,
        "biases": np.asarray(model["bi"], dtype=np.float32).tolist(),
        "embeddings": np.asarray(model["Q"], dtype=np.float32).tolist(),
    }


def build_ablation_bundle(paths: dict[str, Path], spec_value: object) -> dict[str, Any]:
    spec = parse_spec(spec_value)
    for field, input_name in INPUT_FIELDS.items():
        if normalized_file_sha(paths[input_name]) != spec[field]:
            raise ValueError(f"Baseline input {input_name} does not match {field}.")
    snapshot = load_raw_snapshot(paths["raw"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"),
                          object_pairs_hook=_no_duplicate_json_keys)
    metadata = load_metadata_snapshot(paths["metadata"])
    candidates = parse_selection_spec(json.loads(paths["candidates"].read_text(encoding="utf-8"),
                                                  object_pairs_hook=_no_duplicate_json_keys))
    selected = next((candidate for candidate in candidates.candidates
                     if candidate.candidate_id == spec["modelCandidateId"]), None)
    if selected is None or selected.model_score_floor is not None or selected.graph_lambda != 0.01 or (
            selected.graph_min_weight != 0 or selected.graph_sample_rate != 1):
        raise ValueError("Baseline MF candidate is absent or incompatible.")
    partitions = partition_snapshot(snapshot, manifest)
    # Existing fit route validates the manifest, centers train only, and calls the local pair adapter.
    with contextlib.redirect_stdout(io.StringIO()):
        fitted = selected.train(snapshot, manifest, metadata, candidates.model_seed)
    catalog = [{"animeId": anime_id, "title": title} for anime_id, title in metadata.anime
               if next(count for item_id, count, _ in fitted.fit.popularity if item_id == anime_id) > 0]
    catalog_ids = {item["animeId"] for item in catalog}
    positive_pairs = [
        {"leftAnimeId": left, "rightAnimeId": right, "weight": weight, "support": support}
        for left, right, weight, support in fitted.fit.pairs
        if weight > 0 and left in catalog_ids and right in catalog_ids
    ]
    graph_edges = fitted.graph_edges
    if graph_edges.shape[0] != len(positive_pairs) or (
            graph_edges.size and not np.all(graph_edges[:, 2] > 0)):
        raise ValueError("Negative or missing pair entered attractive regularization.")
    indexed_support = {
        (fitted.dataset.anime_ids.index(pair["leftAnimeId"]),
         fitted.dataset.anime_ids.index(pair["rightAnimeId"])): pair["support"]
        for pair in positive_pairs
    }
    unit_edges = graph_edges.copy()
    shrunk_edges = graph_edges.copy()
    for edge in unit_edges:
        edge[2] = 1.0
    for edge in shrunk_edges:
        left, right = int(edge[0]), int(edge[1])
        support = indexed_support[(left, right)]
        edge[2] *= support / (support + spec["graphShrinkage"])
    variants = {
        "plain": (np.zeros((0, 3), dtype=np.float32), 0.0),
        "unitPositive": (unit_edges, selected.graph_lambda),
        "shrunkPositive": (shrunk_edges, selected.graph_lambda),
    }
    trained: dict[str, dict[str, Any]] = {}
    for name, (edges, graph_lambda) in variants.items():
        with contextlib.redirect_stdout(io.StringIO()):
            model = train_graph_mf(
                fitted.split, len(fitted.dataset.user_ids), len(fitted.dataset.anime_ids),
                edges, selected.factors, selected.epochs, selected.lr, selected.reg,
                selected.reg_bias, graph_lambda, selected.graph_sample_rate,
                candidates.model_seed,
            )
        trained[name] = {
            "modelSha256": model_fingerprint(model),
            "model": compact_model(model, fitted.dataset, metadata.snapshot_at, selected.factors),
        }
    similarity, similarity_stats = similarity_from_train(
        fitted.fit.rows, spec["similarityMinSupport"], spec["similarityShrinkage"]
    )
    warm = score_holdout(
        fitted, partitions.validation, top_k=10,
        positive_raw_score_min=spec["positiveRawScoreMin"], model_score_floor=None,
    )
    base_bundle = {
        "format": "split-first-new-user-bundle-v1",
        "candidateId": selected.candidate_id,
        "trainSha256": fitted.fit.train_sha256,
        "fitSha256": fitted.fit.fit_sha256,
        "modelSha256": model_fingerprint(fitted.model),
        "metadataSha256": metadata.sha256,
        "fitUserCount": len(fitted.dataset.user_ids),
        "trainRowCount": len(partitions.train),
        "warmValidation": warm,
        "catalog": catalog,
        "positivePairs": positive_pairs,
        "model": compact_model(fitted.model, fitted.dataset, metadata.snapshot_at, selected.factors),
    }
    return {
        "format": "split-first-baseline-bundle-v1",
        "specSha256": hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"))
                                     .encode("utf-8")).hexdigest(),
        "baseBundle": base_bundle,
        "models": trained,
        "trainCounts": [{"animeId": anime_id, "count": count}
                        for anime_id, count, _ in fitted.fit.popularity if anime_id in catalog_ids],
        "similarityPairs": similarity,
        "audit": {
            "positivePairEdges": len(positive_pairs),
            "nonpositivePairEdgesExcluded": sum(1 for _, _, weight, _ in fitted.fit.pairs if weight <= 0),
            "similarity": similarity_stats,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export invented train-only M5.6 baseline inputs.")
    parser.add_argument("--spec", type=Path, default=Path("fixtures/synthetic-baseline-ablation-spec.json"))
    parser.add_argument("--raw", type=Path, default=Path("fixtures/synthetic-new-user-fit.json"))
    parser.add_argument("--manifest", type=Path, default=Path("fixtures/synthetic-new-user-fit-manifest.json"))
    parser.add_argument("--metadata", type=Path, default=Path("fixtures/synthetic-new-user-anime-metadata.json"))
    parser.add_argument("--validation", type=Path, default=Path("fixtures/synthetic-new-user-validation.json"))
    parser.add_argument("--candidates", type=Path, default=Path("fixtures/synthetic-mf-candidates.json"))
    args = parser.parse_args()
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"),
                          object_pairs_hook=_no_duplicate_json_keys)
        bundle = build_ablation_bundle({
            "raw": args.raw, "manifest": args.manifest, "metadata": args.metadata,
            "validation": args.validation, "candidates": args.candidates,
        }, spec)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Split-first baseline export failed: {exc}\n")
    print(json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
