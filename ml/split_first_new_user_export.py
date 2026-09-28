#!/usr/bin/env python3
"""Local train-only item-model adapter for the browser new-user fixture evaluator."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from raw_interaction_split import load_raw_snapshot, partition_snapshot, _no_duplicate_json_keys
from split_first_graph_mf import model_fingerprint
from split_first_selection import parse_selection_spec, score_holdout
from train_only_preprocessing import load_metadata_snapshot


def build_eval_bundle(raw_path: Path, manifest_path: Path, metadata_path: Path,
                      candidates_path: Path, candidate_id: str) -> dict[str, Any]:
    snapshot = load_raw_snapshot(raw_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                          object_pairs_hook=_no_duplicate_json_keys)
    metadata = load_metadata_snapshot(metadata_path)
    spec = parse_selection_spec(json.loads(candidates_path.read_text(encoding="utf-8"),
                                           object_pairs_hook=_no_duplicate_json_keys))
    selected = next((candidate for candidate in spec.candidates
                     if candidate.candidate_id == candidate_id), None)
    if selected is None:
        raise ValueError("Requested candidate is absent from the fixed specification.")
    if selected.model_score_floor is not None:
        raise ValueError("The browser model path has no candidate score floor for this evaluation.")
    partitions = partition_snapshot(snapshot, manifest)
    # The trainer prints epoch progress; keep stdout a single machine-readable JSON record.
    with contextlib.redirect_stdout(io.StringIO()):
        fitted = selected.train(snapshot, manifest, metadata, spec.model_seed)
    warm = score_holdout(fitted, partitions.validation, top_k=10,
                         positive_raw_score_min=spec.positive_raw_score_min,
                         model_score_floor=selected.model_score_floor)
    item_vectors = np.asarray(fitted.model["Q"], dtype=np.float32)
    item_biases = np.asarray(fitted.model["bi"], dtype=np.float32)
    train_counts = {anime_id: count for anime_id, count, _ in fitted.fit.popularity}
    catalog = [{"animeId": anime_id, "title": title}
               for anime_id, title in metadata.anime if train_counts[anime_id] > 0]
    catalog_ids = {entry["animeId"] for entry in catalog}
    pairs = [{"leftAnimeId": left, "rightAnimeId": right, "weight": weight, "support": support}
             for left, right, weight, support in fitted.fit.pairs
             if weight > 0 and left in catalog_ids and right in catalog_ids]
    model = {
        "format": "model-mf-compact-v1",
        "generatedAt": metadata.snapshot_at,
        "globalMean": float(fitted.model["global_mean"]),
        "factors": selected.factors,
        "animeIds": fitted.dataset.anime_ids,
        "titles": fitted.dataset.anime_titles,
        "biases": item_biases.tolist(),
        "embeddings": item_vectors.tolist(),
    }
    return {
        "format": "split-first-new-user-bundle-v1",
        "candidateId": candidate_id,
        "trainSha256": fitted.fit.train_sha256,
        "fitSha256": fitted.fit.fit_sha256,
        "modelSha256": model_fingerprint(fitted.model),
        "metadataSha256": metadata.sha256,
        "fitUserCount": len(fitted.dataset.user_ids),
        "trainRowCount": len(partitions.train),
        "warmValidation": warm,
        "catalog": catalog,
        "positivePairs": pairs,
        "model": model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a local train-only browser evaluation bundle.")
    parser.add_argument("--raw-ratings", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()
    try:
        result = build_eval_bundle(args.raw_ratings, args.split_manifest, args.metadata,
                                   args.candidates, args.candidate_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
