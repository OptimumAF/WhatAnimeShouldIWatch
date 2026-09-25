"""Fit centering, popularity, and the existing pair graph from train rows only."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from raw_interaction_split import RawInteraction, _anime_id, _no_duplicate_json_keys


@dataclass(frozen=True)
class MetadataSnapshot:
    source: str
    snapshot_at: str
    anime: tuple[tuple[int, str], ...]
    sha256: str


@dataclass(frozen=True)
class CenteredTrainRow:
    user_id: str
    anime_id: int
    raw_score: float
    normalized_score: float


@dataclass(frozen=True)
class TrainOnlyFit:
    metadata: MetadataSnapshot
    user_baselines: tuple[tuple[str, float], ...]
    rows: tuple[CenteredTrainRow, ...]
    popularity: tuple[tuple[int, int, float], ...]
    pairs: tuple[tuple[int, int, float, int], ...]
    pair_stats: dict[str, int]
    pair_config: dict[str, int]
    train_sha256: str
    fit_sha256: str


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_metadata_snapshot(value: object) -> MetadataSnapshot:
    if not isinstance(value, dict) or set(value) != {"format", "source", "snapshotAt", "anime"}:
        raise ValueError("Metadata snapshot must contain only format, source, snapshotAt, and anime.")
    if value["format"] != "anime-metadata-snapshot-v1":
        raise ValueError("Unsupported metadata snapshot format.")
    source, snapshot_at, anime = value["source"], value["snapshotAt"], value["anime"]
    if not isinstance(source, str) or not source or source != source.strip() or len(source) > 128:
        raise ValueError("Metadata source must be a short nonempty label.")
    if not isinstance(snapshot_at, str) or not snapshot_at.endswith("Z"):
        raise ValueError("Metadata snapshotAt must be a UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Metadata snapshotAt is invalid.") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError("Metadata snapshotAt must use UTC.")
    if not isinstance(anime, list) or not anime:
        raise ValueError("Metadata anime must be a nonempty array.")
    entries: list[tuple[int, str]] = []
    seen: set[int] = set()
    for index, entry in enumerate(anime):
        if not isinstance(entry, dict) or set(entry) != {"animeId", "title"}:
            raise ValueError(f"Metadata anime[{index}] may contain only animeId and title.")
        anime_id = _anime_id(entry["animeId"], f"metadata anime[{index}]")
        title = entry["title"]
        if anime_id in seen or not isinstance(title, str) or not title or title != title.strip() or len(title) > 500:
            raise ValueError(f"Metadata anime[{index}] has a duplicate ID or invalid title.")
        seen.add(anime_id)
        entries.append((anime_id, title))
    entries.sort()
    canonical = ["anime-metadata-snapshot-v1", source, snapshot_at, entries]
    return MetadataSnapshot(source, snapshot_at, tuple(entries), hashlib.sha256(_json_bytes(canonical)).hexdigest())


def load_metadata_snapshot(path: Path) -> MetadataSnapshot:
    return parse_metadata_snapshot(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_json_keys))


def _aggregate_train_pairs(rows: tuple[CenteredTrainRow, ...]) -> tuple[
    tuple[tuple[int, int, float, int], ...], dict[str, int], dict[str, int]
]:
    users: dict[str, list[dict[str, float | int]]] = {}
    for row in rows:
        users.setdefault(row.user_id, []).append({"animeId": row.anime_id, "normalizedScore": row.normalized_score})
    payload = {"format": "train-centered-pairs-v1", "users": [
        {"userId": user_id, "ratings": ratings} for user_id, ratings in sorted(users.items())
    ]}
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["node", "--import", "tsx", "pipeline/src/aggregate-train-pairs.ts"],
        input=_json_bytes(payload), capture_output=True, cwd=repo_root, check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Train-only pair aggregation failed: {message}")
    result: Any = json.loads(completed.stdout)
    if not isinstance(result, dict) or result.get("format") != "train-pairs-v1":
        raise ValueError("Train-only pair adapter returned an invalid format.")
    pairs = tuple(tuple(pair) for pair in result["pairs"])
    return pairs, result["stats"], result["config"]


def fit_training_partition(train: tuple[RawInteraction, ...], metadata: MetadataSnapshot) -> TrainOnlyFit:
    """Fit from raw train rows; this signature cannot receive held-out labels."""
    if not train:
        raise ValueError("The training partition is empty.")
    allowed_anime = {anime_id for anime_id, _ in metadata.anime}
    by_user: dict[str, list[RawInteraction]] = {}
    seen: set[tuple[str, int]] = set()
    for row in train:
        if row.anime_id not in allowed_anime:
            raise ValueError(f"Training anime ID {row.anime_id} is absent from the fixed metadata snapshot.")
        if not math.isfinite(row.raw_score) or (row.user_id, row.anime_id) in seen:
            raise ValueError("Training rows contain a non-finite score or duplicate interaction.")
        seen.add((row.user_id, row.anime_id))
        by_user.setdefault(row.user_id, []).append(row)
    ordered = tuple(sorted(train, key=lambda row: (row.user_id, row.anime_id)))
    baselines = tuple((user_id, math.fsum(row.raw_score for row in sorted(rows, key=lambda r: r.anime_id)) / len(rows))
                      for user_id, rows in sorted(by_user.items()))
    baseline_by_user = dict(baselines)
    centered = tuple(CenteredTrainRow(row.user_id, row.anime_id, row.raw_score,
                                      row.raw_score - baseline_by_user[row.user_id]) for row in ordered)
    by_anime: dict[int, list[float]] = {anime_id: [] for anime_id, _ in metadata.anime}
    for row in ordered:
        by_anime[row.anime_id].append(row.raw_score)
    popularity = tuple((anime_id, len(scores), math.fsum(scores)) for anime_id, scores in sorted(by_anime.items()))
    pairs, pair_stats, pair_config = _aggregate_train_pairs(centered)
    train_sha = hashlib.sha256(_json_bytes([[row.user_id, row.anime_id, row.raw_score] for row in ordered])).hexdigest()
    fit_payload = {
        "metadataSha256": metadata.sha256,
        "trainSha256": train_sha,
        "baselines": baselines,
        "centered": [[row.user_id, row.anime_id, row.normalized_score] for row in centered],
        "popularity": popularity,
        "pairs": pairs,
        "pairStats": pair_stats,
        "pairConfig": pair_config,
    }
    fit_sha = hashlib.sha256(_json_bytes(fit_payload)).hexdigest()
    return TrainOnlyFit(metadata, baselines, centered, popularity, pairs, pair_stats, pair_config,
                        train_sha, fit_sha)
