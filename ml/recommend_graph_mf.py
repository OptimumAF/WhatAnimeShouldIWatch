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


@dataclass
class ContentEntry:
    anime_id: int
    genres: Set[str]
    studios: Set[str]
    year: int | None
    score: float | None


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
    parser.add_argument(
        "--content-features",
        default="models/graph_mf/content-features.json",
        help="Optional content feature JSON for cold-start mitigation",
    )
    parser.add_argument(
        "--content-blend",
        type=float,
        default=-1.0,
        help="Blend weight [0,1] for content score; negative enables auto blend by watched count",
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


def load_content_features(path: Path) -> Dict[int, ContentEntry]:
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    anime_raw = raw.get("anime", {})
    if not isinstance(anime_raw, dict):
        return {}

    features: Dict[int, ContentEntry] = {}
    for key, value in anime_raw.items():
        if not isinstance(value, dict):
            continue
        try:
            anime_id = int(key)
        except (TypeError, ValueError):
            continue

        genres_raw = value.get("genres", [])
        studios_raw = value.get("studios", [])
        genres = {
            str(genre).strip().lower()
            for genre in genres_raw
            if isinstance(genre, str) and genre.strip()
        }
        studios = {
            str(studio).strip().lower()
            for studio in studios_raw
            if isinstance(studio, str) and studio.strip()
        }
        year_raw = value.get("year")
        score_raw = value.get("score")
        year = int(year_raw) if isinstance(year_raw, (int, float)) else None
        score = float(score_raw) if isinstance(score_raw, (int, float)) else None

        features[anime_id] = ContentEntry(
            anime_id=anime_id,
            genres=genres,
            studios=studios,
            year=year,
            score=score,
        )

    return features


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


def content_similarity(left: ContentEntry, right: ContentEntry) -> float:
    genre_overlap = jaccard(left.genres, right.genres)
    studio_overlap = jaccard(left.studios, right.studios)

    year_score = 0.0
    if left.year is not None and right.year is not None:
        year_delta = abs(left.year - right.year)
        year_score = max(0.0, 1.0 - (year_delta / 25.0))

    mal_score_similarity = 0.0
    if left.score is not None and right.score is not None:
        mal_score_similarity = max(0.0, 1.0 - (abs(left.score - right.score) / 10.0))

    return (
        0.5 * genre_overlap
        + 0.2 * studio_overlap
        + 0.2 * year_score
        + 0.1 * mal_score_similarity
    )


def jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return len(left & right) / union_size


def scale_content_to_mf_distribution(
    content_values: np.ndarray, mf_values: np.ndarray
) -> np.ndarray:
    content_mean = float(np.mean(content_values))
    content_std = float(np.std(content_values))
    mf_mean = float(np.mean(mf_values))
    mf_std = float(np.std(mf_values))

    if content_std < 1e-8:
        return np.full_like(content_values, mf_mean, dtype=np.float32)
    if mf_std < 1e-8:
        return np.full_like(content_values, mf_mean, dtype=np.float32)

    z = (content_values - content_mean) / content_std
    return (mf_mean + (z * mf_std)).astype(np.float32)


def recommend(
    model: Model,
    user_id: str,
    watched: Sequence[Tuple[int, float]],
    top_n: int,
    explain_top: int,
    min_score: float,
    content_features: Dict[int, ContentEntry],
    content_blend: float,
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

    mf_scores = model.global_mean + user_bias + model.bi + (model.q @ user_vector)
    scores = mf_scores.copy()
    applied_content_blend = 0.0
    content_scores_output = np.full_like(scores, np.nan, dtype=np.float32)

    if not user_id and content_features and watched_for_explain:
        watched_content: List[Tuple[ContentEntry, float]] = []
        for watched_idx, weight in watched_for_explain:
            watched_anime_id = idx_to_anime[watched_idx]
            feature = content_features.get(watched_anime_id)
            if feature is not None:
                watched_content.append((feature, float(weight)))

        if watched_content:
            if content_blend >= 0.0:
                applied_content_blend = max(0.0, min(1.0, float(content_blend)))
            else:
                watched_count = len(watched_content)
                if watched_count <= 2:
                    applied_content_blend = 0.45
                elif watched_count <= 4:
                    applied_content_blend = 0.30
                else:
                    applied_content_blend = 0.15

            content_scores = np.full_like(scores, np.nan, dtype=np.float32)
            denom = float(sum(abs(weight) for _, weight in watched_content))
            denom = denom if denom > 0 else float(len(watched_content))
            for idx, anime_id in enumerate(idx_to_anime):
                candidate = content_features.get(anime_id)
                if candidate is None:
                    continue
                acc = 0.0
                for watched_entry, weight in watched_content:
                    sim = content_similarity(candidate, watched_entry)
                    acc += abs(weight) * sim
                content_scores[idx] = acc / max(1e-6, denom)
            content_scores_output = content_scores

            finite_mf = np.isfinite(scores)
            finite_content = np.isfinite(content_scores)
            blend_mask = finite_mf & finite_content
            if np.any(blend_mask) and applied_content_blend > 0.0:
                mf_masked = scores[blend_mask]
                content_masked = content_scores[blend_mask]
                content_scaled = scale_content_to_mf_distribution(content_masked, mf_masked)
                scores[blend_mask] = (
                    (1.0 - applied_content_blend) * mf_masked
                    + applied_content_blend * content_scaled
                )

    if seen:
        seen_arr = np.fromiter(seen, dtype=np.int32)
        scores[seen_arr] = -np.inf
        mf_scores[seen_arr] = -np.inf

    if np.isfinite(scores).sum() == 0:
        raise ValueError("No candidate anime left after filtering.")

    rank_candidates = top_k_indices(scores, max(top_n * 4, top_n))
    results = []
    for idx in rank_candidates:
        score = float(scores[idx])
        mf_score = float(mf_scores[idx])
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

        content_value = (
            float(content_scores_output[idx])
            if np.isfinite(content_scores_output[idx])
            else None
        )

        results.append(
            {
                "animeId": int(idx_to_anime[idx]),
                "title": idx_to_title[idx],
                "score": score,
                "mfScore": mf_score,
                "contentScore": content_value,
                "contentBlend": applied_content_blend,
                "why": why,
            }
        )
        if len(results) >= top_n:
            break

    return {
        "source": source,
        "count": len(results),
        "contentBlendApplied": applied_content_blend,
        "recommendations": results,
    }


def print_plain(result: Dict[str, object], explain_top: int) -> None:
    print(f"Source: {result['source']}")
    print(f"Recommendations: {result['count']}")
    content_blend = result.get("contentBlendApplied")
    if isinstance(content_blend, (int, float)) and float(content_blend) > 0:
        print(f"Content blend applied: {float(content_blend):.2f}")
    print("")
    recommendations = result["recommendations"]  # type: ignore[assignment]
    for rank, item in enumerate(recommendations, start=1):
        anime_id = item["animeId"]
        title = item["title"]
        score = item["score"]
        mf_score = item.get("mfScore")
        content_score = item.get("contentScore")
        extra = ""
        if isinstance(mf_score, (int, float)):
            extra += f" mf={float(mf_score):.4f}"
        if isinstance(content_score, (int, float)):
            extra += f" content={float(content_score):.4f}"
        print(f"{rank:>2}. {title} (ID {anime_id})  score={score:.4f}{extra}")
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
    content_features = load_content_features(Path(args.content_features))

    result = recommend(
        model=model,
        user_id=args.user_id.strip(),
        watched=watched,
        top_n=args.top_n,
        explain_top=args.explain_top,
        min_score=args.min_score,
        content_features=content_features,
        content_blend=args.content_blend,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_plain(result, explain_top=args.explain_top)


if __name__ == "__main__":
    main()
