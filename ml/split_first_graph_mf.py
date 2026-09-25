#!/usr/bin/env python3
"""Synthetic-safe MF fit from an immutable raw split; emits no holdout metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from raw_interaction_split import load_raw_snapshot, partition_snapshot, _no_duplicate_json_keys
from train_graph_mf import Dataset, Split, train_graph_mf
from train_only_preprocessing import TrainOnlyFit, fit_training_partition, load_metadata_snapshot


def training_arrays(fit: TrainOnlyFit, min_graph_weight: float = 0.0) -> tuple[Dataset, Split, np.ndarray]:
    """Build numeric training inputs only from the fit and its fixed metadata universe."""
    if not np.isfinite(min_graph_weight) or min_graph_weight < 0:
        raise ValueError("Minimum graph weight must be finite and nonnegative.")
    user_ids = [user_id for user_id, _ in fit.user_baselines]
    anime_ids = [anime_id for anime_id, _ in fit.metadata.anime]
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    anime_to_idx = {anime_id: idx for idx, anime_id in enumerate(anime_ids)}
    indexed = [(user_to_idx[row.user_id], anime_to_idx[row.anime_id], row.normalized_score) for row in fit.rows]
    dataset = Dataset(user_ids, anime_ids, [title for _, title in fit.metadata.anime], indexed)
    seen: list[set[int]] = [set() for _ in user_ids]
    for user_idx, anime_idx, _ in indexed:
        seen[user_idx].add(anime_idx)
    split = Split(
        train_u=np.asarray([user for user, _, _ in indexed], dtype=np.int32),
        train_i=np.asarray([anime for _, anime, _ in indexed], dtype=np.int32),
        train_r=np.asarray([score for _, _, score in indexed], dtype=np.float32),
        train_user_items=seen,
        test_pos_items={},
        users_with_test=0,
    )
    edges = [(anime_to_idx[left], anime_to_idx[right], weight)
             for left, right, weight, _support in fit.pairs if weight > 0 and weight >= min_graph_weight]
    graph_edges = np.asarray(edges, dtype=np.float32).reshape(-1, 3)
    return dataset, split, graph_edges


def model_fingerprint(model: dict[str, np.ndarray | float]) -> str:
    digest = hashlib.sha256(b"wasiw-split-first-mf-v1\n")
    for name in ("P", "Q", "bu", "bi", "global_mean"):
        array = np.asarray(model[name], dtype=np.float32)
        digest.update(json.dumps([name, list(array.shape), str(array.dtype)], separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit MF from a validated raw train split without using holdout labels.")
    parser.add_argument("--raw-ratings", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--reg-bias", type=float, default=0.005)
    parser.add_argument("--graph-lambda", type=float, default=0.01)
    parser.add_argument("--graph-min-weight", type=float, default=0.0)
    parser.add_argument("--graph-sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.factors < 1 or args.epochs < 1 or args.seed < 0:
        parser.error("factors and epochs must be positive; seed must be nonnegative.")
    if not all(math.isfinite(value) for value in (
        args.lr, args.reg, args.reg_bias, args.graph_lambda,
        args.graph_min_weight, args.graph_sample_rate,
    )) or not (args.lr > 0 and args.reg >= 0 and args.reg_bias >= 0 and
              args.graph_lambda >= 0 and args.graph_min_weight >= 0 and
              0 < args.graph_sample_rate <= 1):
        parser.error("Learning rate must be positive; regularization and graph weights must be nonnegative; graph sample rate must be in (0, 1]; all must be finite.")
    try:
        snapshot = load_raw_snapshot(args.raw_ratings)
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"),
                              object_pairs_hook=_no_duplicate_json_keys)
        train = partition_snapshot(snapshot, manifest).train
        metadata = load_metadata_snapshot(args.metadata)
        fit = fit_training_partition(train, metadata)
        dataset, split, graph_edges = training_arrays(fit, args.graph_min_weight)
        model = train_graph_mf(
            split, len(dataset.user_ids), len(dataset.anime_ids), graph_edges,
            args.factors, args.epochs, args.lr, args.reg, args.reg_bias,
            args.graph_lambda, args.graph_sample_rate, args.seed,
        )
    except (OSError, UnicodeError, ValueError, KeyError) as exc:
        parser.exit(1, f"Split-first MF failed: {exc}\n")
    print(json.dumps({
        "format": "split-first-mf-fit-v1",
        "trainInteractions": len(fit.rows),
        "trainUsers": len(dataset.user_ids),
        "metadataItems": len(dataset.anime_ids),
        "trainPairCandidates": fit.pair_stats["candidatePairs"],
        "trainPairEdges": len(fit.pairs),
        "positiveRegularizationEdges": int(graph_edges.shape[0]),
        "metadataSha256": fit.metadata.sha256,
        "trainSha256": fit.train_sha256,
        "fitSha256": fit.fit_sha256,
        "modelSha256": model_fingerprint(model),
        "evaluation": "none; holdout labels were not read by preprocessing or training",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
