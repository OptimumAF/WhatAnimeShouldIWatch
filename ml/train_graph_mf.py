#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


@dataclass
class Dataset:
    user_ids: List[str]
    anime_ids: List[int]
    anime_titles: List[str]
    interactions: List[Tuple[int, int, float]]


@dataclass
class Split:
    train_u: np.ndarray
    train_i: np.ndarray
    train_r: np.ndarray
    train_user_items: List[Set[int]]
    test_pos_items: Dict[int, Set[int]]
    users_with_test: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train graph-regularized matrix factorization from MAL ratings + anime graph."
    )
    parser.add_argument(
        "--ratings",
        default="data/anonymized-ratings.json",
        help="Path to anonymized-ratings.json",
    )
    parser.add_argument(
        "--graph",
        default="data/graph.json",
        help="Path to graph.json",
    )
    parser.add_argument(
        "--out-dir",
        default="models/graph_mf",
        help="Output directory for model artifacts",
    )
    parser.add_argument("--factors", type=int, default=64, help="Latent factor dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="SGD learning rate")
    parser.add_argument("--reg", type=float, default=0.01, help="L2 regularization for factors")
    parser.add_argument("--reg-bias", type=float, default=0.005, help="L2 regularization for biases")
    parser.add_argument(
        "--graph-lambda",
        type=float,
        default=0.01,
        help="Strength of anime-anime graph regularization",
    )
    parser.add_argument(
        "--graph-min-abs-weight",
        type=float,
        default=0.0,
        help="Ignore anime-anime edges where abs(weight) is below this threshold",
    )
    parser.add_argument(
        "--graph-sample-rate",
        type=float,
        default=1.0,
        help="Fraction (0,1] of graph edges sampled each epoch for regularization",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Per-user held-out ratio for evaluation",
    )
    parser.add_argument(
        "--min-ratings-for-test",
        type=int,
        default=8,
        help="Minimum ratings needed for a user to get held-out test ratings",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="Held-out rating threshold considered positive for ranking metrics",
    )
    parser.add_argument("--top-k", type=int, default=20, help="K for ranking metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _as_int_maybe(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_dataset(ratings_path: Path) -> Dataset:
    raw = json.loads(ratings_path.read_text(encoding="utf-8"))
    users_raw = raw.get("users", [])

    user_ids: List[str] = []
    anime_to_idx: Dict[int, int] = {}
    anime_ids: List[int] = []
    anime_titles: List[str] = []
    interactions: List[Tuple[int, int, float]] = []

    for user_entry in users_raw:
        user_id = str(user_entry.get("userId", "")).strip()
        if not user_id:
            continue
        user_idx = len(user_ids)
        user_ids.append(user_id)

        ratings = user_entry.get("ratings", [])
        for rating in ratings:
            anime_id = _as_int_maybe(rating.get("animeId"))
            score = rating.get("normalizedScore")
            if anime_id is None:
                continue
            if not isinstance(score, (int, float)) or math.isnan(float(score)):
                continue

            if anime_id not in anime_to_idx:
                anime_to_idx[anime_id] = len(anime_ids)
                anime_ids.append(anime_id)
                anime_titles.append(str(rating.get("title", f"Anime {anime_id}")))
            anime_idx = anime_to_idx[anime_id]
            interactions.append((user_idx, anime_idx, float(score)))

    return Dataset(
        user_ids=user_ids,
        anime_ids=anime_ids,
        anime_titles=anime_titles,
        interactions=interactions,
    )


def split_train_test(
    dataset: Dataset,
    test_ratio: float,
    min_ratings_for_test: int,
    positive_threshold: float,
    seed: int,
) -> Split:
    rng = np.random.default_rng(seed)
    per_user: List[List[Tuple[int, float]]] = [[] for _ in dataset.user_ids]
    for user_idx, anime_idx, score in dataset.interactions:
        per_user[user_idx].append((anime_idx, score))

    train_u: List[int] = []
    train_i: List[int] = []
    train_r: List[float] = []
    train_user_items: List[Set[int]] = [set() for _ in dataset.user_ids]
    test_pos_items: Dict[int, Set[int]] = {}
    users_with_test = 0

    for user_idx, ratings in enumerate(per_user):
        n = len(ratings)
        if n >= min_ratings_for_test:
            holdout_count = max(1, int(round(n * test_ratio)))
            holdout_count = min(holdout_count, n - 1)
        else:
            holdout_count = 0

        if holdout_count > 0:
            users_with_test += 1
            holdout_indices = set(rng.choice(n, size=holdout_count, replace=False).tolist())
        else:
            holdout_indices = set()

        for idx, (anime_idx, score) in enumerate(ratings):
            if idx in holdout_indices:
                if score > positive_threshold:
                    test_pos_items.setdefault(user_idx, set()).add(anime_idx)
                continue
            train_u.append(user_idx)
            train_i.append(anime_idx)
            train_r.append(score)
            train_user_items[user_idx].add(anime_idx)

    return Split(
        train_u=np.array(train_u, dtype=np.int32),
        train_i=np.array(train_i, dtype=np.int32),
        train_r=np.array(train_r, dtype=np.float32),
        train_user_items=train_user_items,
        test_pos_items=test_pos_items,
        users_with_test=users_with_test,
    )


def _parse_anime_id_from_node(node_id: str) -> int | None:
    if not isinstance(node_id, str):
        return None
    if not node_id.startswith("anime:"):
        return None
    return _as_int_maybe(node_id.split(":", 1)[1])


def load_anime_graph_edges(
    graph_path: Path,
    anime_id_to_idx: Dict[int, int],
    min_abs_weight: float,
) -> np.ndarray:
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    edges_raw = raw.get("edges", [])

    # Deduplicate undirected pairs and average by absolute edge weight.
    pair_weights: Dict[Tuple[int, int], List[float]] = {}

    for edge in edges_raw:
        if edge.get("edgeType") != "anime-anime":
            continue

        src_anime_id = _parse_anime_id_from_node(str(edge.get("source", "")))
        dst_anime_id = _parse_anime_id_from_node(str(edge.get("target", "")))
        weight = edge.get("weight")

        if src_anime_id is None or dst_anime_id is None:
            continue
        if src_anime_id == dst_anime_id:
            continue
        if not isinstance(weight, (int, float)) or math.isnan(float(weight)):
            continue

        w = abs(float(weight))
        if w < min_abs_weight:
            continue

        src_idx = anime_id_to_idx.get(src_anime_id)
        dst_idx = anime_id_to_idx.get(dst_anime_id)
        if src_idx is None or dst_idx is None:
            continue

        a, b = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
        pair_weights.setdefault((a, b), []).append(w)

    pairs: List[Tuple[int, int, float]] = []
    for (a, b), ws in pair_weights.items():
        pairs.append((a, b, float(np.mean(np.array(ws, dtype=np.float32)))))
    if not pairs:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(pairs, dtype=np.float32)


def train_graph_mf(
    split: Split,
    n_users: int,
    n_items: int,
    graph_edges: np.ndarray,
    factors: int,
    epochs: int,
    lr: float,
    reg: float,
    reg_bias: float,
    graph_lambda: float,
    graph_sample_rate: float,
    seed: int,
) -> Dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    p = (0.05 * rng.standard_normal((n_users, factors))).astype(np.float32)
    q = (0.05 * rng.standard_normal((n_items, factors))).astype(np.float32)
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)
    global_mean = float(split.train_r.mean()) if split.train_r.size else 0.0

    order = np.arange(split.train_r.shape[0], dtype=np.int32)
    graph_count = graph_edges.shape[0]
    graph_batch_size = int(round(graph_count * max(0.0, min(1.0, graph_sample_rate))))
    graph_batch_size = max(1, graph_batch_size) if graph_count > 0 else 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        rng.shuffle(order)
        se = 0.0

        for idx in order:
            u = int(split.train_u[idx])
            i = int(split.train_i[idx])
            r = float(split.train_r[idx])

            pu = p[u].copy()
            qi = q[i].copy()

            pred = global_mean + float(bu[u]) + float(bi[i]) + float(np.dot(pu, qi))
            err = r - pred
            se += err * err

            p[u] = pu + lr * ((err * qi) - (reg * pu))
            q[i] = qi + lr * ((err * pu) - (reg * qi))
            bu[u] = bu[u] + lr * (err - reg_bias * float(bu[u]))
            bi[i] = bi[i] + lr * (err - reg_bias * float(bi[i]))

        if graph_lambda > 0.0 and graph_count > 0:
            if graph_batch_size < graph_count:
                edge_indices = rng.choice(graph_count, size=graph_batch_size, replace=False)
                edge_batch = graph_edges[edge_indices]
            else:
                edge_batch = graph_edges

            for edge in edge_batch:
                a = int(edge[0])
                b = int(edge[1])
                w = float(edge[2])
                if w <= 0.0:
                    continue
                qa = q[a].copy()
                qb = q[b].copy()
                diff = qa - qb
                grad = graph_lambda * w * diff
                q[a] = qa - (lr * grad)
                q[b] = qb + (lr * grad)

        rmse = math.sqrt(se / max(1, split.train_r.shape[0]))
        dt = time.perf_counter() - t0
        print(f"Epoch {epoch:>2}/{epochs}  train_rmse={rmse:.4f}  time={dt:.2f}s")

    return {
        "P": p,
        "Q": q,
        "bu": bu,
        "bi": bi,
        "global_mean": global_mean,
    }


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, scores.shape[0]))
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def evaluate_ranking(
    model: Dict[str, np.ndarray | float],
    split: Split,
    top_k: int,
) -> Dict[str, float]:
    p = model["P"]  # type: ignore[assignment]
    q = model["Q"]  # type: ignore[assignment]
    bu = model["bu"]  # type: ignore[assignment]
    bi = model["bi"]  # type: ignore[assignment]
    global_mean = float(model["global_mean"])  # type: ignore[arg-type]

    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    bu = np.asarray(bu, dtype=np.float32)
    bi = np.asarray(bi, dtype=np.float32)

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    hit_rates: List[float] = []

    for user_idx, gt_items in split.test_pos_items.items():
        if not gt_items:
            continue

        scores = global_mean + float(bu[user_idx]) + bi + (q @ p[user_idx])
        if split.train_user_items[user_idx]:
            seen = np.fromiter(split.train_user_items[user_idx], dtype=np.int32)
            scores[seen] = -np.inf

        if not np.isfinite(scores).any():
            continue

        topk = _top_k_indices(scores, top_k)
        gt = gt_items
        hits = [1 if int(item) in gt else 0 for item in topk]
        hit_count = sum(hits)

        precision = hit_count / len(topk)
        recall = hit_count / max(1, len(gt))
        hit_rate = 1.0 if hit_count > 0 else 0.0

        dcg = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                dcg += 1.0 / math.log2(rank + 1.0)
        ideal_hits = min(len(gt), len(topk))
        idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, ideal_hits + 1))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hit_rates.append(hit_rate)

    evaluated_users = len(precisions)
    return {
        "evaluated_users": float(evaluated_users),
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "hit_rate_at_k": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def save_model(
    out_dir: Path,
    dataset: Dataset,
    split: Split,
    model: Dict[str, np.ndarray | float],
    metrics: Dict[str, float],
    args: argparse.Namespace,
    graph_edge_count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.npz"
    metrics_path = out_dir / "metrics.json"

    train_user_items_obj = np.array(
        [np.array(sorted(items), dtype=np.int32) for items in split.train_user_items],
        dtype=object,
    )

    np.savez_compressed(
        model_path,
        P=np.asarray(model["P"], dtype=np.float32),
        Q=np.asarray(model["Q"], dtype=np.float32),
        bu=np.asarray(model["bu"], dtype=np.float32),
        bi=np.asarray(model["bi"], dtype=np.float32),
        global_mean=np.array([float(model["global_mean"])], dtype=np.float32),
        user_ids=np.array(dataset.user_ids, dtype=object),
        anime_ids=np.array(dataset.anime_ids, dtype=np.int64),
        anime_titles=np.array(dataset.anime_titles, dtype=object),
        train_user_items=train_user_items_obj,
    )

    report = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modelPath": str(model_path),
        "data": {
            "ratingsPath": args.ratings,
            "graphPath": args.graph,
            "users": len(dataset.user_ids),
            "anime": len(dataset.anime_ids),
            "interactions": len(dataset.interactions),
            "trainInteractions": int(split.train_r.shape[0]),
            "usersWithHeldout": split.users_with_test,
            "usersEvaluated": int(metrics["evaluated_users"]),
            "graphRegularizationEdges": graph_edge_count,
        },
        "hyperparameters": {
            "factors": args.factors,
            "epochs": args.epochs,
            "lr": args.lr,
            "reg": args.reg,
            "regBias": args.reg_bias,
            "graphLambda": args.graph_lambda,
            "graphMinAbsWeight": args.graph_min_abs_weight,
            "graphSampleRate": args.graph_sample_rate,
            "testRatio": args.test_ratio,
            "minRatingsForTest": args.min_ratings_for_test,
            "positiveThreshold": args.positive_threshold,
            "topK": args.top_k,
            "seed": args.seed,
        },
        "metrics": {
            "precisionAtK": metrics["precision_at_k"],
            "recallAtK": metrics["recall_at_k"],
            "hitRateAtK": metrics["hit_rate_at_k"],
            "ndcgAtK": metrics["ndcg_at_k"],
        },
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model -> {model_path}")
    print(f"Saved report -> {metrics_path}")


def main() -> None:
    args = parse_args()
    ratings_path = Path(args.ratings)
    graph_path = Path(args.graph)
    out_dir = Path(args.out_dir)

    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not (0.0 < args.graph_sample_rate <= 1.0):
        raise ValueError("--graph-sample-rate must be in (0, 1].")

    print("Loading dataset...")
    dataset = load_dataset(ratings_path)
    anime_id_to_idx = {anime_id: idx for idx, anime_id in enumerate(dataset.anime_ids)}
    print(
        f"Loaded ratings: users={len(dataset.user_ids)} anime={len(dataset.anime_ids)} interactions={len(dataset.interactions)}"
    )

    print("Creating train/test split...")
    split = split_train_test(
        dataset=dataset,
        test_ratio=args.test_ratio,
        min_ratings_for_test=args.min_ratings_for_test,
        positive_threshold=args.positive_threshold,
        seed=args.seed,
    )
    print(
        f"Split complete: train_interactions={split.train_r.shape[0]} users_with_test={split.users_with_test}"
    )

    print("Loading anime-anime graph edges...")
    graph_edges = load_anime_graph_edges(
        graph_path=graph_path,
        anime_id_to_idx=anime_id_to_idx,
        min_abs_weight=args.graph_min_abs_weight,
    )
    print(f"Graph edges for regularization: {graph_edges.shape[0]}")

    print("Training model...")
    model = train_graph_mf(
        split=split,
        n_users=len(dataset.user_ids),
        n_items=len(dataset.anime_ids),
        graph_edges=graph_edges,
        factors=args.factors,
        epochs=args.epochs,
        lr=args.lr,
        reg=args.reg,
        reg_bias=args.reg_bias,
        graph_lambda=args.graph_lambda,
        graph_sample_rate=args.graph_sample_rate,
        seed=args.seed,
    )

    print("Evaluating ranking quality...")
    metrics = evaluate_ranking(model=model, split=split, top_k=args.top_k)
    print(
        "Metrics "
        f"Precision@{args.top_k}={metrics['precision_at_k']:.4f} "
        f"Recall@{args.top_k}={metrics['recall_at_k']:.4f} "
        f"HitRate@{args.top_k}={metrics['hit_rate_at_k']:.4f} "
        f"NDCG@{args.top_k}={metrics['ndcg_at_k']:.4f} "
        f"(users={int(metrics['evaluated_users'])})"
    )

    save_model(
        out_dir=out_dir,
        dataset=dataset,
        split=split,
        model=model,
        metrics=metrics,
        args=args,
        graph_edge_count=int(graph_edges.shape[0]),
    )


if __name__ == "__main__":
    main()
