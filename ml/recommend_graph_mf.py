#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


@dataclass
class Model:
    p: np.ndarray
    q: np.ndarray
    bu: np.ndarray
    bi: np.ndarray
    global_mean: float
    user_ids: List[str]
    anime_ids: List[int]
    anime_titles: List[str]
    train_user_items: List[Set[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend anime from a trained graph MF model."
    )
    parser.add_argument(
        "--model",
        default="models/graph_mf/model.npz",
        help="Path to model.npz from train_graph_mf.py",
    )
    parser.add_argument(
        "--user-id",
        default="",
        help="Use a known userId from training data as recommendation source",
    )
    parser.add_argument(
        "--watched",
        default="",
        help="Comma-separated anime input: '1535,9253:1.5,5114:0.7'",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Number of recommendations")
    parser.add_argument(
        "--explain-top",
        type=int,
        default=3,
        help="Number of top contributing watched anime to show per recommendation",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=float("-inf"),
        help="Filter out recommendations below this score",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of plain text",
    )
    return parser.parse_args()


def load_model(path: Path) -> Model:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    raw = np.load(path, allow_pickle=True)

    p = raw["P"].astype(np.float32)
    q = raw["Q"].astype(np.float32)
    bu = raw["bu"].astype(np.float32)
    bi = raw["bi"].astype(np.float32)
    global_mean = float(raw["global_mean"][0])
    user_ids = [str(x) for x in raw["user_ids"].tolist()]
    anime_ids = [int(x) for x in raw["anime_ids"].tolist()]
    anime_titles = [str(x) for x in raw["anime_titles"].tolist()]

    train_user_items = []
    for arr in raw["train_user_items"]:
        arr = np.asarray(arr, dtype=np.int32)
        train_user_items.append(set(int(x) for x in arr.tolist()))

    return Model(
        p=p,
        q=q,
        bu=bu,
        bi=bi,
        global_mean=global_mean,
        user_ids=user_ids,
        anime_ids=anime_ids,
        anime_titles=anime_titles,
        train_user_items=train_user_items,
    )


def parse_watched_arg(value: str) -> List[Tuple[int, float]]:
    if not value.strip():
        return []
    items: List[Tuple[int, float]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            anime_str, weight_str = token.split(":", 1)
            anime_id = int(anime_str.strip())
            weight = float(weight_str.strip())
        else:
            anime_id = int(token)
            weight = 1.0
        items.append((anime_id, weight))
    return items


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, scores.shape[0]))
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def recommend(
    model: Model,
    user_id: str,
    watched: Sequence[Tuple[int, float]],
    top_n: int,
    explain_top: int,
    min_score: float,
) -> Dict[str, object]:
    anime_id_to_idx = {anime_id: idx for idx, anime_id in enumerate(model.anime_ids)}
    idx_to_anime = model.anime_ids
    idx_to_title = model.anime_titles

    source = ""
    seen: Set[int] = set()
    watched_for_explain: List[Tuple[int, float]] = []

    if user_id:
        user_lookup = {uid: idx for idx, uid in enumerate(model.user_ids)}
        if user_id not in user_lookup:
            raise ValueError(f"user-id not found in model: {user_id}")
        user_idx = user_lookup[user_id]
        source = f"user:{user_id}"
        user_vector = model.p[user_idx]
        user_bias = float(model.bu[user_idx])
        seen = set(model.train_user_items[user_idx])
        for anime_idx in seen:
            watched_for_explain.append((anime_idx, 1.0))
    else:
        parsed = list(watched)
        if not parsed:
            raise ValueError("Provide either --user-id or --watched.")

        weighted_indices: List[Tuple[int, float]] = []
        unknown_ids: List[int] = []
        for anime_id, weight in parsed:
            idx = anime_id_to_idx.get(anime_id)
            if idx is None:
                unknown_ids.append(anime_id)
                continue
            weighted_indices.append((idx, float(weight)))
            seen.add(idx)
            watched_for_explain.append((idx, float(weight)))

        if not weighted_indices:
            raise ValueError("None of the watched anime IDs exist in the trained model.")

        weights = np.array([w for _, w in weighted_indices], dtype=np.float32)
        item_vecs = np.stack([model.q[idx] for idx, _ in weighted_indices], axis=0)
        denom = float(np.sum(np.abs(weights)))
        if denom == 0.0:
            denom = float(len(weights))
        user_vector = (weights[:, None] * item_vecs).sum(axis=0) / max(1e-6, denom)
        user_bias = 0.0
        source = f"watched:{len(weighted_indices)}"
        if unknown_ids:
            print(f"Skipped unknown anime IDs: {unknown_ids}")

    scores = model.global_mean + user_bias + model.bi + (model.q @ user_vector)
    if seen:
        seen_arr = np.fromiter(seen, dtype=np.int32)
        scores[seen_arr] = -np.inf

    if np.isfinite(scores).sum() == 0:
        raise ValueError("No candidate anime left after filtering.")

    rank_candidates = top_k_indices(scores, max(top_n * 4, top_n))
    results = []
    for idx in rank_candidates:
        score = float(scores[idx])
        if score < min_score:
            continue

        why = []
        if watched_for_explain and explain_top > 0:
            contributions = []
            for watched_idx, weight in watched_for_explain:
                contribution = float(weight * np.dot(model.q[watched_idx], model.q[idx]))
                contributions.append((watched_idx, contribution))
            contributions.sort(key=lambda x: x[1], reverse=True)
            for watched_idx, contribution in contributions[:explain_top]:
                why.append(
                    {
                        "animeId": int(idx_to_anime[watched_idx]),
                        "title": idx_to_title[watched_idx],
                        "contribution": contribution,
                    }
                )

        results.append(
            {
                "animeId": int(idx_to_anime[idx]),
                "title": idx_to_title[idx],
                "score": score,
                "why": why,
            }
        )
        if len(results) >= top_n:
            break

    return {
        "source": source,
        "count": len(results),
        "recommendations": results,
    }


def print_plain(result: Dict[str, object], explain_top: int) -> None:
    print(f"Source: {result['source']}")
    print(f"Recommendations: {result['count']}")
    print("")
    recommendations = result["recommendations"]  # type: ignore[assignment]
    for rank, item in enumerate(recommendations, start=1):
        anime_id = item["animeId"]
        title = item["title"]
        score = item["score"]
        print(f"{rank:>2}. {title} (ID {anime_id})  score={score:.4f}")
        why_items = item.get("why", [])
        if explain_top > 0 and why_items:
            why_str = ", ".join(
                f"{x['title']} ({x['contribution']:.3f})" for x in why_items
            )
            print(f"    why: {why_str}")


def main() -> None:
    args = parse_args()
    model = load_model(Path(args.model))
    watched = parse_watched_arg(args.watched)

    result = recommend(
        model=model,
        user_id=args.user_id.strip(),
        watched=watched,
        top_n=args.top_n,
        explain_top=args.explain_top,
        min_score=args.min_score,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_plain(result, explain_top=args.explain_top)


if __name__ == "__main__":
    main()
