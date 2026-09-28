#!/usr/bin/env python3
"""Restricted M5.9 invented LightGCN/content candidates; no holdout metric or provider I/O."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from raw_interaction_split import _no_duplicate_json_keys, load_raw_snapshot, partition_snapshot
from split_first_baseline_export import INPUT_FIELDS, normalized_file_sha, parse_spec as parse_baseline_spec
from split_first_graph_mf import training_arrays
from train_only_preprocessing import fit_training_partition, load_metadata_snapshot

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_INPUTS = {
    "spec": "synthetic-experiment-spec.json",
    "baseline_spec": "synthetic-baseline-ablation-spec.json",
    "content": "synthetic-experiment-content-metadata.json",
    "raw": "synthetic-new-user-fit.json",
    "manifest": "synthetic-new-user-fit-manifest.json",
    "metadata": "synthetic-new-user-anime-metadata.json",
    "validation": "synthetic-new-user-validation.json",
    "candidates": "synthetic-mf-candidates.json",
}


def allow_fixture_cli(paths: dict[str, Path]) -> None:
    if set(paths) != set(FIXTURE_INPUTS) or any(
            paths[key].resolve() != ROOT / "fixtures" / name
            for key, name in FIXTURE_INPUTS.items()):
        raise ValueError("Non-fixture experiment CLI is held for reviewed source/use approval.")


def canonical_hash(value: object) -> str:
    def json_numbers(item: object) -> object:
        if isinstance(item, dict):
            return {key: json_numbers(child) for key, child in item.items()}
        if isinstance(item, list):
            return [json_numbers(child) for child in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    return hashlib.sha256(json.dumps(json_numbers(value), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_json_keys)


def parse_experiment_spec(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "format", "baselineSpecSha256", "contentMetadataSha256", "methods",
        "suppliedCounts", "topK", "positiveRawScoreMin", "missingSignalScore",
        "tieBreak", "objective", "lightgcn", "content",
    }:
        raise ValueError("Experiment specification has missing or extra fields.")
    for field in ("baselineSpecSha256", "contentMetadataSha256"):
        digest = value[field]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Experiment specification {field} must be a SHA-256 digest.")
    gcn = value["lightgcn"]
    content = value["content"]
    if not isinstance(gcn, dict) or not isinstance(content, dict):
        raise ValueError("Experiment specification candidate parameters are invalid.")
    expected_gcn = {
        "factors": 8, "layers": 1, "epochs": 2, "batchSize": 128,
        "lr": 0.01, "reg": 0.0001, "seed": 42,
        "centeredPositiveThreshold": 0, "pairEdges": "train-positive-only",
    }
    expected_content = {
        "featureFields": ["genres", "studios"], "dropUniversalTokens": True,
        "normalize": "item-l2", "scorer": "browser-signed-item-vector",
    }
    if (value["format"] != "split-first-experiment-spec-v1" or
            value["methods"] != ["lightgcn-bpr", "content-multihot"] or
            value["suppliedCounts"] != [1, 3, 5, 10] or
            type(value["topK"]) is not int or value["topK"] != 10 or
            type(value["positiveRawScoreMin"]) is not int or value["positiveRawScoreMin"] != 7 or
            type(value["missingSignalScore"]) is not int or value["missingSignalScore"] != 0 or
            value["tieBreak"] != "anime-id-ascending" or
            value["objective"] != "mean-displayed-ndcg-at-10" or
            set(gcn) != set(expected_gcn) or set(content) != set(expected_content) or
            any(type(gcn[key]) is not type(expected) or gcn[key] != expected
                for key, expected in expected_gcn.items()) or
            any(type(content[key]) is not type(expected) or content[key] != expected
                for key, expected in expected_content.items())):
        raise ValueError("Experiment specification differs from decision 0025.")
    return value


def parse_content_metadata(value: object, expected_ids: list[int]) -> tuple[dict[int, tuple[str, ...]], list[str]]:
    if not isinstance(value, dict) or set(value) != {"format", "source", "anime"} or (
            value["format"] != "experiment-content-metadata-v1" or
            value["source"] != "invented-fixture" or not isinstance(value["anime"], list)):
        raise ValueError("Content metadata format/source/anime is invalid.")
    features: dict[int, tuple[str, ...]] = {}
    for i, raw in enumerate(value["anime"]):
        if not isinstance(raw, dict) or set(raw) != {"animeId", "genres", "studios"}:
            raise ValueError(f"Content metadata anime[{i}] fields are invalid.")
        anime_id = raw["animeId"]
        if type(anime_id) is not int or anime_id < 1 or anime_id in features:
            raise ValueError(f"Content metadata anime[{i}].animeId is invalid or repeated.")
        tokens: list[str] = []
        for field, prefix in (("genres", "genre"), ("studios", "studio")):
            values = raw[field]
            if (not isinstance(values, list) or not values or
                    any(not isinstance(term, str) or not term or term != term.strip() or
                        len(term) > 100 for term in values) or len(values) != len(set(values))):
                raise ValueError(f"Content metadata anime[{i}].{field} is invalid.")
            tokens.extend(f"{prefix}:{term}" for term in values)
        features[anime_id] = tuple(tokens)
    if sorted(features) != expected_ids:
        raise ValueError("Content metadata anime IDs differ from fixed title metadata.")
    common = set.intersection(*(set(tokens) for tokens in features.values()))
    vocabulary = sorted(set().union(*(set(tokens) for tokens in features.values())) - common)
    if not vocabulary or any(not set(tokens).difference(common) for tokens in features.values()):
        raise ValueError("Content metadata has an empty discriminating vector.")
    return features, vocabulary


def compact_model(anime_ids: list[int], titles: list[str], snapshot_at: str,
                  vectors: np.ndarray) -> dict[str, Any]:
    if vectors.shape[0] != len(anime_ids) or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("Experiment item vectors are invalid.")
    return {
        "format": "model-mf-compact-v1", "generatedAt": snapshot_at,
        "globalMean": 0.0, "factors": int(vectors.shape[1]),
        "animeIds": anime_ids, "titles": titles,
        "biases": [0.0] * len(anime_ids),
        "embeddings": vectors.astype(np.float32).tolist(),
    }


def content_vectors(anime_ids: list[int], features: dict[int, tuple[str, ...]],
                    vocabulary: list[str]) -> np.ndarray:
    positions = {token: i for i, token in enumerate(vocabulary)}
    vectors = np.zeros((len(anime_ids), len(vocabulary)), dtype=np.float32)
    for row, anime_id in enumerate(anime_ids):
        for token in features[anime_id]:
            if token in positions:
                vectors[row, positions[token]] = 1.0
        length = float(np.linalg.norm(vectors[row]))
        if length == 0:
            raise ValueError(f"Content metadata anime {anime_id} has no retained token.")
        vectors[row] /= length
    return vectors


def build_experiment_bundle(paths: dict[str, Path], spec_value: object,
                            baseline_spec_value: object) -> dict[str, Any]:
    spec = parse_experiment_spec(spec_value)
    baseline = parse_baseline_spec(baseline_spec_value)
    if baseline_spec_value != read_json(paths["baseline_spec"]):
        raise ValueError("Experiment baseline specification value differs from its pinned file.")
    if normalized_file_sha(paths["baseline_spec"]) != spec["baselineSpecSha256"] or (
            normalized_file_sha(paths["content"]) != spec["contentMetadataSha256"]):
        raise ValueError("Experiment baseline specification or content metadata hash is stale.")
    for field, input_name in INPUT_FIELDS.items():
        if normalized_file_sha(paths[input_name]) != baseline[field]:
            raise ValueError(f"Experiment {input_name} does not match baseline {field}.")
    if (spec["topK"] != baseline["topK"] or
            spec["suppliedCounts"] != baseline["suppliedCounts"] or
            spec["positiveRawScoreMin"] != baseline["positiveRawScoreMin"] or
            spec["missingSignalScore"] != baseline["missingSignalScore"] or
            spec["tieBreak"] != baseline["tieBreak"] or spec["objective"] != baseline["objective"]):
        raise ValueError("Experiment and baseline comparison protocols differ.")
    snapshot = load_raw_snapshot(paths["raw"])
    manifest = read_json(paths["manifest"])
    metadata = load_metadata_snapshot(paths["metadata"])
    partitions = partition_snapshot(snapshot, manifest)
    fit = fit_training_partition(partitions.train, metadata)
    dataset, split, positive_pair_edges = training_arrays(fit)
    features, vocabulary = parse_content_metadata(read_json(paths["content"]), dataset.anime_ids)
    content = compact_model(dataset.anime_ids, dataset.anime_titles, metadata.snapshot_at,
                            content_vectors(dataset.anime_ids, features, vocabulary))

    # Importing the legacy module reuses its actual BPR/propagation implementation only.
    # Its full-graph loader, private local split, test evaluator, and output writer are never called.
    from eval_gnn_lightgcn import build_lightgcn_adjacency, train_lightgcn
    import torch

    positive = split.train_r > spec["lightgcn"]["centeredPositiveThreshold"]
    train_users = split.train_u[positive].astype(np.int64)
    train_items = split.train_i[positive].astype(np.int64)
    if not len(train_users):
        raise ValueError("LightGCN has no strictly positive centered training interactions.")
    adjacency = build_lightgcn_adjacency(len(dataset.user_ids), len(dataset.anime_ids),
                                          train_users, train_items, positive_pair_edges)
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        embeddings = train_lightgcn(
            len(dataset.user_ids), len(dataset.anime_ids), train_users, train_items,
            split.train_user_items, adjacency, spec["lightgcn"]["factors"],
            spec["lightgcn"]["layers"], spec["lightgcn"]["epochs"],
            spec["lightgcn"]["batchSize"], spec["lightgcn"]["lr"],
            spec["lightgcn"]["reg"], spec["lightgcn"]["seed"], torch.device("cpu"),
        )
    gcn = compact_model(dataset.anime_ids, dataset.anime_titles, metadata.snapshot_at,
                        embeddings[len(dataset.user_ids):].numpy())
    return {
        "format": "split-first-experiment-bundle-v1",
        "specSha256": canonical_hash(spec),
        "baselineSpecSha256": spec["baselineSpecSha256"],
        "trainSha256": fit.train_sha256, "fitSha256": fit.fit_sha256,
        "metadataSha256": metadata.sha256,
        "contentMetadataSha256": spec["contentMetadataSha256"],
        "trainRowCount": len(partitions.train),
        "positiveTrainInteractions": len(train_users),
        "positivePairEdges": len(positive_pair_edges),
        "contentVocabulary": vocabulary,
        "models": {
            "lightgcn-bpr": {"modelSha256": canonical_hash(gcn), "model": gcn},
            "content-multihot": {"modelSha256": canonical_hash(content), "model": content},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export restricted invented M5.9 experiment items.")
    parser.add_argument("--spec", type=Path, default=Path("fixtures/synthetic-experiment-spec.json"))
    parser.add_argument("--baseline-spec", type=Path,
                        default=Path("fixtures/synthetic-baseline-ablation-spec.json"))
    parser.add_argument("--content", type=Path,
                        default=Path("fixtures/synthetic-experiment-content-metadata.json"))
    parser.add_argument("--raw", type=Path, default=Path("fixtures/synthetic-new-user-fit.json"))
    parser.add_argument("--manifest", type=Path, default=Path("fixtures/synthetic-new-user-fit-manifest.json"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("fixtures/synthetic-new-user-anime-metadata.json"))
    parser.add_argument("--validation", type=Path,
                        default=Path("fixtures/synthetic-new-user-validation.json"))
    parser.add_argument("--candidates", type=Path,
                        default=Path("fixtures/synthetic-mf-candidates.json"))
    args = parser.parse_args()
    paths = {"spec": args.spec, "baseline_spec": args.baseline_spec, "content": args.content,
             "raw": args.raw, "manifest": args.manifest, "metadata": args.metadata,
             "validation": args.validation, "candidates": args.candidates}
    try:
        allow_fixture_cli(paths)
        bundle = build_experiment_bundle(paths, read_json(args.spec), read_json(args.baseline_spec))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Split-first experiment export failed: {exc}\n")
    print(json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
