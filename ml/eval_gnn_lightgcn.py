#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

import train_graph_mf as trainer

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for GNN evaluation. Install with: pip install torch"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LightGCN ranking baseline on the anime graph."
    )
    parser.add_argument(
        "--ratings",
        default="data/anonymized-ratings.compact.json",
        help="Path to ratings dataset JSON (compact or legacy)",
    )
    parser.add_argument(
        "--graph",
        default="data/graph.compact.json",
        help="Path to graph JSON (compact or legacy)",
    )
    parser.add_argument(
        "--out-dir",
        default="models/gnn_lightgcn",
        help="Output directory for GNN artifacts",
    )
    parser.add_argument("--factors", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=3, help="Graph propagation layers")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="BPR batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    parser.add_argument("--reg", type=float, default=1e-4, help="L2 regularization for BPR")
    parser.add_argument(
        "--graph-min-abs-weight",
        type=float,
        default=0.2,
        help="Minimum anime-anime weight kept for GNN graph",
    )
    parser.add_argument(
        "--anime-anime-sample-rate",
        type=float,
        default=0.3,
        help="Sample fraction of anime-anime edges used by the graph (0,1]",
    )
    parser.add_argument(
        "--disable-anime-anime",
        action="store_true",
        help="Disable anime-anime edges and use only user-anime interactions",
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
        help="Train/test rating threshold considered positive",
    )
    parser.add_argument("--top-k", type=int, default=20, help="K for ranking metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_positive_train_interactions(
    split: trainer.Split,
    positive_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if split.train_r.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    mask = split.train_r > positive_threshold
    if not np.any(mask):
        mask = np.ones_like(split.train_r, dtype=bool)
    users = split.train_u[mask].astype(np.int64, copy=False)
    items = split.train_i[mask].astype(np.int64, copy=False)
    return users, items


def maybe_sample_graph_edges(
    edges: np.ndarray,
    sample_rate: float,
    seed: int,
) -> np.ndarray:
    if edges.shape[0] == 0:
        return edges
    clamped = max(0.0, min(1.0, float(sample_rate)))
    if clamped >= 1.0:
        return edges
    rng = np.random.default_rng(seed)
    keep = int(round(edges.shape[0] * clamped))
    keep = max(1, keep)
    idx = rng.choice(edges.shape[0], size=keep, replace=False)
    return edges[idx]


def build_lightgcn_adjacency(
    n_users: int,
    n_items: int,
    train_users: np.ndarray,
    train_items: np.ndarray,
    anime_edges: np.ndarray,
) -> torch.Tensor:
    total_nodes = n_users + n_items
    src: List[int] = []
    dst: List[int] = []
    weight: List[float] = []

    # User-item bipartite edges.
    for user_idx, item_idx in zip(train_users.tolist(), train_items.tolist()):
        u = int(user_idx)
        i = n_users + int(item_idx)
        src.append(u)
        dst.append(i)
        weight.append(1.0)
        src.append(i)
        dst.append(u)
        weight.append(1.0)

    # Item-item edges from anime graph regularization edges.
    for edge in anime_edges:
        left = n_users + int(edge[0])
        right = n_users + int(edge[1])
        w = float(edge[2])
        if w <= 0:
            continue
        src.append(left)
        dst.append(right)
        weight.append(w)
        src.append(right)
        dst.append(left)
        weight.append(w)

    if not src:
        raise RuntimeError("No graph edges were built for LightGCN.")

    src_tensor = torch.tensor(src, dtype=torch.long)
    dst_tensor = torch.tensor(dst, dtype=torch.long)
    weight_tensor = torch.tensor(weight, dtype=torch.float32)

    degree = torch.zeros(total_nodes, dtype=torch.float32)
    degree.index_add_(0, src_tensor, weight_tensor)
    degree = torch.clamp(degree, min=1e-12)

    norm_weight = weight_tensor / torch.sqrt(degree[src_tensor] * degree[dst_tensor])
    indices = torch.stack([src_tensor, dst_tensor], dim=0)
    adj = torch.sparse_coo_tensor(
        indices=indices,
        values=norm_weight,
        size=(total_nodes, total_nodes),
        dtype=torch.float32,
    )
    return adj.coalesce()


def propagate_embeddings(
    base_embeddings: torch.Tensor,
    adjacency: torch.Tensor,
    layers: int,
) -> torch.Tensor:
    emb = base_embeddings
    all_embeddings = [emb]
    for _ in range(max(0, layers)):
        emb = torch.sparse.mm(adjacency, emb)
        all_embeddings.append(emb)
    stacked = torch.stack(all_embeddings, dim=0)
    return torch.mean(stacked, dim=0)


def sample_negative_items(
    user_indices: np.ndarray,
    seen_items: Sequence[Set[int]],
    n_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = np.empty_like(user_indices, dtype=np.int64)
    for idx, user_idx in enumerate(user_indices.tolist()):
        seen = seen_items[int(user_idx)]
        if len(seen) >= n_items:
            negatives[idx] = int(rng.integers(0, n_items))
            continue

        candidate = int(rng.integers(0, n_items))
        tries = 0
        while candidate in seen and tries < 60:
            candidate = int(rng.integers(0, n_items))
            tries += 1
        if candidate in seen:
            for fallback in range(n_items):
                if fallback not in seen:
                    candidate = fallback
                    break
        negatives[idx] = candidate
    return negatives


def train_lightgcn(
    n_users: int,
    n_items: int,
    train_users: np.ndarray,
    train_items: np.ndarray,
    seen_items: Sequence[Set[int]],
    adjacency: torch.Tensor,
    factors: int,
    layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    reg: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed + 1007)

    total_nodes = n_users + n_items
    embedding = torch.nn.Embedding(total_nodes, factors, device=device)
    torch.nn.init.normal_(embedding.weight, std=0.1)
    optimizer = torch.optim.Adam(embedding.parameters(), lr=lr)

    adjacency_device = adjacency.to(device)
    positives = np.arange(train_users.shape[0], dtype=np.int64)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        np_rng.shuffle(positives)
        total_loss = 0.0
        total_bpr = 0.0
        total_reg = 0.0
        steps = 0

        for offset in range(0, positives.shape[0], max(1, batch_size)):
            batch_idx = positives[offset : offset + max(1, batch_size)]
            batch_users = train_users[batch_idx]
            batch_pos_items = train_items[batch_idx]
            batch_neg_items = sample_negative_items(
                user_indices=batch_users,
                seen_items=seen_items,
                n_items=n_items,
                rng=np_rng,
            )

            users_t = torch.from_numpy(batch_users).to(device=device, dtype=torch.long)
            pos_items_t = torch.from_numpy(batch_pos_items).to(device=device, dtype=torch.long)
            neg_items_t = torch.from_numpy(batch_neg_items).to(device=device, dtype=torch.long)

            final_embeddings = propagate_embeddings(
                embedding.weight,
                adjacency_device,
                layers=layers,
            )
            user_emb = final_embeddings[users_t]
            pos_emb = final_embeddings[n_users + pos_items_t]
            neg_emb = final_embeddings[n_users + neg_items_t]

            pos_scores = torch.sum(user_emb * pos_emb, dim=1)
            neg_scores = torch.sum(user_emb * neg_emb, dim=1)
            bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            reg_loss = (
                reg
                * (
                    user_emb.pow(2).sum()
                    + pos_emb.pow(2).sum()
                    + neg_emb.pow(2).sum()
                )
                / max(1, users_t.shape[0])
            )
            loss = bpr_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_bpr += float(bpr_loss.detach().cpu())
            total_reg += float(reg_loss.detach().cpu())
            steps += 1

        dt = time.perf_counter() - t0
        avg_loss = total_loss / max(1, steps)
        avg_bpr = total_bpr / max(1, steps)
        avg_reg = total_reg / max(1, steps)
        print(
            f"Epoch {epoch:>2}/{epochs}  "
            f"loss={avg_loss:.5f} bpr={avg_bpr:.5f} reg={avg_reg:.5f} time={dt:.2f}s"
        )

    with torch.no_grad():
        final = propagate_embeddings(embedding.weight, adjacency_device, layers=layers)
    return final.detach().cpu()


def save_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    dataset: trainer.Dataset,
    split: trainer.Split,
    metrics: Dict[str, float],
    train_positive_count: int,
    anime_graph_edge_count: int,
    embeddings: torch.Tensor,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model-lightgcn.npz"
    metrics_path = out_dir / "metrics-lightgcn.json"

    n_users = len(dataset.user_ids)
    p = embeddings[:n_users].numpy().astype(np.float32)
    q = embeddings[n_users:].numpy().astype(np.float32)
    zeros_u = np.zeros((p.shape[0],), dtype=np.float32)
    zeros_i = np.zeros((q.shape[0],), dtype=np.float32)
    train_user_items_obj = np.array(
        [np.array(sorted(items), dtype=np.int32) for items in split.train_user_items],
        dtype=object,
    )

    np.savez_compressed(
        model_path,
        P=p,
        Q=q,
        bu=zeros_u,
        bi=zeros_i,
        global_mean=np.array([0.0], dtype=np.float32),
        user_ids=np.array(dataset.user_ids, dtype=object),
        anime_ids=np.array(dataset.anime_ids, dtype=np.int64),
        anime_titles=np.array(dataset.anime_titles, dtype=object),
        train_user_items=train_user_items_obj,
    )

    report = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modelType": "lightgcn",
        "modelPath": str(model_path),
        "data": {
            "ratingsPath": args.ratings,
            "graphPath": args.graph,
            "users": len(dataset.user_ids),
            "anime": len(dataset.anime_ids),
            "interactions": len(dataset.interactions),
            "trainInteractions": int(split.train_r.shape[0]),
            "positiveTrainInteractions": int(train_positive_count),
            "usersWithHeldout": split.users_with_test,
            "usersEvaluated": int(metrics["evaluated_users"]),
            "animeAnimeEdgesUsed": int(anime_graph_edge_count),
        },
        "hyperparameters": {
            "factors": args.factors,
            "layers": args.layers,
            "epochs": args.epochs,
            "batchSize": args.batch_size,
            "lr": args.lr,
            "reg": args.reg,
            "graphMinAbsWeight": args.graph_min_abs_weight,
            "animeAnimeSampleRate": args.anime_anime_sample_rate,
            "disableAnimeAnime": bool(args.disable_anime_anime),
            "testRatio": args.test_ratio,
            "minRatingsForTest": args.min_ratings_for_test,
            "positiveThreshold": args.positive_threshold,
            "topK": args.top_k,
            "seed": args.seed,
            "device": args.device,
        },
        "metrics": {
            "precisionAtK": metrics["precision_at_k"],
            "recallAtK": metrics["recall_at_k"],
            "hitRateAtK": metrics["hit_rate_at_k"],
            "ndcgAtK": metrics["ndcg_at_k"],
        },
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved LightGCN embeddings -> {model_path}")
    print(f"Saved LightGCN report -> {metrics_path}")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.anime_anime_sample_rate <= 1.0):
        raise ValueError("--anime-anime-sample-rate must be in (0, 1].")

    ratings_path = trainer.resolve_with_fallback(
        Path(args.ratings),
        [Path("data/anonymized-ratings.json")],
    )
    graph_path = trainer.resolve_with_fallback(
        Path(args.graph),
        [Path("data/graph.json")],
    )
    out_dir = Path(args.out_dir)

    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = trainer.load_dataset(ratings_path)
    anime_id_to_idx = {anime_id: idx for idx, anime_id in enumerate(dataset.anime_ids)}
    print(
        f"Loaded ratings: users={len(dataset.user_ids)} "
        f"anime={len(dataset.anime_ids)} interactions={len(dataset.interactions)}"
    )

    print("Creating train/test split...")
    split = trainer.split_train_test(
        dataset=dataset,
        test_ratio=args.test_ratio,
        min_ratings_for_test=args.min_ratings_for_test,
        positive_threshold=args.positive_threshold,
        seed=args.seed,
    )
    print(
        f"Split complete: train_interactions={split.train_r.shape[0]} "
        f"users_with_test={split.users_with_test}"
    )

    positive_users, positive_items = build_positive_train_interactions(
        split=split,
        positive_threshold=args.positive_threshold,
    )
    if positive_users.shape[0] == 0:
        raise RuntimeError("No positive training interactions available.")
    print(f"Positive train interactions: {positive_users.shape[0]}")

    anime_edges = np.zeros((0, 3), dtype=np.float32)
    if not args.disable_anime_anime:
        print("Loading anime-anime graph edges...")
        anime_edges = trainer.load_anime_graph_edges(
            graph_path=graph_path,
            anime_id_to_idx=anime_id_to_idx,
            min_abs_weight=args.graph_min_abs_weight,
        )
        anime_edges = maybe_sample_graph_edges(
            anime_edges,
            sample_rate=args.anime_anime_sample_rate,
            seed=args.seed + 3001,
        )
    print(f"Anime-anime edges used: {anime_edges.shape[0]}")

    print("Building normalized graph adjacency...")
    adjacency = build_lightgcn_adjacency(
        n_users=len(dataset.user_ids),
        n_items=len(dataset.anime_ids),
        train_users=positive_users,
        train_items=positive_items,
        anime_edges=anime_edges,
    )

    print("Training LightGCN...")
    final_embeddings = train_lightgcn(
        n_users=len(dataset.user_ids),
        n_items=len(dataset.anime_ids),
        train_users=positive_users,
        train_items=positive_items,
        seen_items=split.train_user_items,
        adjacency=adjacency,
        factors=args.factors,
        layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
        seed=args.seed,
        device=device,
    )

    print("Evaluating ranking quality...")
    model_for_eval = {
        "P": final_embeddings[: len(dataset.user_ids)].numpy().astype(np.float32),
        "Q": final_embeddings[len(dataset.user_ids) :].numpy().astype(np.float32),
        "bu": np.zeros((len(dataset.user_ids),), dtype=np.float32),
        "bi": np.zeros((len(dataset.anime_ids),), dtype=np.float32),
        "global_mean": 0.0,
    }
    metrics = trainer.evaluate_ranking(
        model=model_for_eval,
        split=split,
        top_k=args.top_k,
    )
    print(
        "Metrics "
        f"Precision@{args.top_k}={metrics['precision_at_k']:.4f} "
        f"Recall@{args.top_k}={metrics['recall_at_k']:.4f} "
        f"HitRate@{args.top_k}={metrics['hit_rate_at_k']:.4f} "
        f"NDCG@{args.top_k}={metrics['ndcg_at_k']:.4f} "
        f"(users={int(metrics['evaluated_users'])})"
    )

    save_outputs(
        out_dir=out_dir,
        args=args,
        dataset=dataset,
        split=split,
        metrics=metrics,
        train_positive_count=int(positive_users.shape[0]),
        anime_graph_edge_count=int(anime_edges.shape[0]),
        embeddings=final_embeddings,
    )


if __name__ == "__main__":
    main()
