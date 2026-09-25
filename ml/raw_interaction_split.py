#!/usr/bin/env python3
"""Create an immutable split manifest from raw, deduplicated interactions.

This module does not train a model or use provider services. Its manifests are
restricted local evaluation inputs and must never be copied into web assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RawInteraction:
    user_id: str
    anime_id: int
    raw_score: float
    event_at: str | None


@dataclass(frozen=True)
class RawSnapshot:
    source_format: str
    timestamp_basis: str
    interactions: tuple[RawInteraction, ...]
    duplicate_rows_dropped: int


@dataclass(frozen=True)
class RawPartitions:
    train: tuple[RawInteraction, ...]
    validation: tuple[RawInteraction, ...]
    test: tuple[RawInteraction, ...]


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object.")
    return value


def _array(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array.")
    return value


def _user_id(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 256:
        raise ValueError(f"{where}.userId must be a nonempty, unpadded string of at most 256 characters.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{where}.userId must be valid Unicode.") from exc
    return value


def _anime_id(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 2**53 - 1):
        raise ValueError(f"{where}.animeId must be a positive safe integer.")
    return value


def _raw_score(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}.rawScore must be a finite number.")
    try:
        score = float(value)
    except OverflowError as exc:
        raise ValueError(f"{where}.rawScore must be a finite number.") from exc
    if not math.isfinite(score):
        raise ValueError(f"{where}.rawScore must be a finite number.")
    return 0.0 if score == 0 else score


def _event_time(value: object, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})",
        value,
    ):
        raise ValueError(f"{where}.eventAt requires an ISO timestamp with timezone.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{where}.eventAt is invalid.") from exc
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _compact_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def interaction_id(user_id: str, anime_id: int) -> str:
    """Stable membership key; neither scores nor metadata enter the hash."""
    return hashlib.sha256(b"wasiw-interaction-v1\n" + _compact_json([user_id, anime_id])).hexdigest()


def _seeded_order(seed: int, row: RawInteraction) -> str:
    return hashlib.sha256(b"wasiw-split-seeded-v1\n" +
                          _compact_json([seed, row.user_id, row.anime_id])).hexdigest()


def parse_raw_snapshot(value: object) -> RawSnapshot:
    root = _object(value, "Input")
    source_format = root.get("format")
    timestamp_basis = "none"
    raw_rows: list[dict[str, Any]] = []

    if source_format == "raw-interactions-v1":
        timestamp_basis = root.get("timestampBasis")
        if timestamp_basis not in ("none", "verified-rating-or-viewing"):
            raise ValueError("timestampBasis must be none or verified-rating-or-viewing.")
        raw_rows = [_object(row, f"interactions[{index}]") for index, row in enumerate(
            _array(root.get("interactions"), "interactions"))]
    elif source_format == "ratings-compact-v1":
        anime = _array(root.get("anime"), "anime")
        anime_ids: list[int] = []
        for index, entry in enumerate(anime):
            entry = _array(entry, f"anime[{index}]")
            if len(entry) != 2 or not isinstance(entry[1], str):
                raise ValueError(f"anime[{index}] must contain an ID and title.")
            anime_ids.append(_anime_id(entry[0], f"anime[{index}]"))
        if len(set(anime_ids)) != len(anime_ids):
            raise ValueError("Duplicate compact anime ID.")
        for user_index, user in enumerate(_array(root.get("users"), "users")):
            user = _array(user, f"users[{user_index}]")
            if len(user) != 2:
                raise ValueError(f"users[{user_index}] must contain an ID and ratings.")
            user_id = _user_id(user[0], f"users[{user_index}]")
            for rating_index, rating in enumerate(_array(user[1], f"users[{user_index}].ratings")):
                rating = _array(rating, f"users[{user_index}].ratings[{rating_index}]")
                if len(rating) != 3:
                    raise ValueError(f"users[{user_index}].ratings[{rating_index}] needs rawScore.")
                anime_index = rating[0]
                if isinstance(anime_index, bool) or not isinstance(anime_index, int) or not (0 <= anime_index < len(anime_ids)):
                    raise ValueError(f"users[{user_index}].ratings[{rating_index}] has invalid anime index.")
                raw_rows.append({"userId": user_id, "animeId": anime_ids[anime_index],
                                 "rawScore": rating[1]})
    elif source_format is None and "users" in root:
        source_format = "ratings-legacy-v1"
        for user_index, user in enumerate(_array(root["users"], "users")):
            user = _object(user, f"users[{user_index}]")
            user_id = _user_id(user.get("userId"), f"users[{user_index}]")
            for rating_index, rating in enumerate(_array(user.get("ratings"), f"users[{user_index}].ratings")):
                rating = _object(rating, f"users[{user_index}].ratings[{rating_index}]")
                raw_rows.append({"userId": user_id, "animeId": rating.get("animeId"),
                                 "rawScore": rating.get("rawScore")})
    else:
        raise ValueError("Unsupported raw ratings format.")

    unique: dict[tuple[str, int], RawInteraction] = {}
    duplicate_rows_dropped = 0
    for index, raw in enumerate(raw_rows):
        where = f"interaction row {index}"
        user_id = _user_id(raw.get("userId"), where)
        anime_id = _anime_id(raw.get("animeId"), where)
        score = _raw_score(raw.get("rawScore"), where)
        event_at = _event_time(raw.get("eventAt"), where) if timestamp_basis == "verified-rating-or-viewing" else None
        row = RawInteraction(user_id, anime_id, score, event_at)
        key = (user_id, anime_id)
        previous = unique.get(key)
        if previous is None:
            unique[key] = row
        elif previous == row:
            duplicate_rows_dropped += 1
        else:
            raise ValueError(f"Conflicting duplicate user–anime interaction at row {index}.")
    if not unique:
        raise ValueError("No raw interactions to split.")
    return RawSnapshot(source_format, timestamp_basis, tuple(unique.values()), duplicate_rows_dropped)


def build_split_manifest(
    snapshot: RawSnapshot,
    *,
    policy: str = "auto",
    seed: int = 42,
    validation_ratio: float = 0.20,
    test_ratio: float = 0.20,
    min_user_ratings: int = 3,
) -> dict[str, Any]:
    if policy not in ("auto", "seeded", "temporal"):
        raise ValueError("Split policy must be auto, seeded, or temporal.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= 2**32 - 1):
        raise ValueError("Seed must be a nonnegative 32-bit integer.")
    if isinstance(min_user_ratings, bool) or not isinstance(min_user_ratings, int) or min_user_ratings < 3:
        raise ValueError("Minimum user ratings must be at least 3.")
    for name, ratio in (("validation", validation_ratio), ("test", test_ratio)):
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not (0 < ratio < 0.5):
            raise ValueError(f"{name} ratio must be greater than 0 and less than 0.5.")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("Holdout ratios must leave training interactions.")
    temporal = snapshot.timestamp_basis == "verified-rating-or-viewing" and policy != "seeded"
    if policy == "temporal" and not temporal:
        raise ValueError("Temporal split requires verified rating or viewing event times.")
    if temporal and any(row.event_at is None for row in snapshot.interactions):
        raise ValueError("Temporal split requires every verified eventAt.")

    by_user: dict[str, list[RawInteraction]] = {}
    for row in snapshot.interactions:
        by_user.setdefault(row.user_id, []).append(row)
    train_ids: list[str] = []
    validation_ids: list[str] = []
    test_ids: list[str] = []
    users_with_holdout = 0
    for user_id in sorted(by_user):
        ratings = by_user[user_id]
        count = len(ratings)
        if count < min_user_ratings:
            train_ids.extend(interaction_id(row.user_id, row.anime_id) for row in ratings)
            continue
        validation_count = max(1, int(Decimal(str(validation_ratio)) * count))
        test_count = max(1, int(Decimal(str(test_ratio)) * count))
        if validation_count + test_count >= count:
            raise ValueError("Holdout counts leave no training interaction.")
        users_with_holdout += 1
        if temporal:
            ordered = sorted(ratings, key=lambda row: (row.event_at, row.anime_id))
            test_start = count - test_count
            validation_start = test_start - validation_count
            if (ordered[test_start - 1].event_at == ordered[test_start].event_at or
                    ordered[validation_start - 1].event_at == ordered[validation_start].event_at):
                raise ValueError("Verified event-time tie crosses a split boundary.")
            train = ordered[:validation_start]
            validation = ordered[validation_start:test_start]
            test = ordered[test_start:]
        else:
            ordered = sorted(ratings, key=lambda row: (_seeded_order(seed, row), row.anime_id))
            test = ordered[:test_count]
            validation = ordered[test_count:test_count + validation_count]
            train = ordered[test_count + validation_count:]
        train_ids.extend(interaction_id(row.user_id, row.anime_id) for row in train)
        validation_ids.extend(interaction_id(row.user_id, row.anime_id) for row in validation)
        test_ids.extend(interaction_id(row.user_id, row.anime_id) for row in test)

    indexed = sorted(((interaction_id(row.user_id, row.anime_id), row)
                      for row in snapshot.interactions), key=lambda item: item[0])
    if len({item_id for item_id, _ in indexed}) != len(indexed):
        raise ValueError("Interaction ID collision in raw snapshot.")
    identity_bytes = b"".join((item_id + "\n").encode("ascii") for item_id, _ in indexed)
    content_bytes = b"".join(_compact_json([item_id, row.raw_score, row.event_at]) + b"\n"
                             for item_id, row in indexed)
    return {
        "format": "raw-interaction-split-v1",
        "policy": "verified-event-time-v1" if temporal else "seeded-per-user-v1",
        "sourceFormat": snapshot.source_format,
        "timestampBasis": snapshot.timestamp_basis,
        "seed": None if temporal else seed,
        "validationRatio": validation_ratio,
        "testRatio": test_ratio,
        "minUserRatings": min_user_ratings,
        "uniqueInteractions": len(snapshot.interactions),
        "duplicateRowsDropped": snapshot.duplicate_rows_dropped,
        "usersWithHoldout": users_with_holdout,
        "identitySha256": hashlib.sha256(identity_bytes).hexdigest(),
        "rawContentSha256": hashlib.sha256(content_bytes).hexdigest(),
        "trainIds": sorted(train_ids),
        "validationIds": sorted(validation_ids),
        "testIds": sorted(test_ids),
    }


def partition_snapshot(snapshot: RawSnapshot, manifest: object) -> RawPartitions:
    """Return raw rows only after checking a manifest's exact snapshot and membership."""
    if not isinstance(manifest, dict) or manifest.get("format") != "raw-interaction-split-v1":
        raise ValueError("Unsupported split manifest format.")
    policy = manifest.get("policy")
    if policy not in ("seeded-per-user-v1", "verified-event-time-v1"):
        raise ValueError("Unsupported split manifest policy.")
    try:
        expected = build_split_manifest(
            snapshot,
            policy="temporal" if policy == "verified-event-time-v1" else "seeded",
            seed=42 if policy == "verified-event-time-v1" else manifest["seed"],
            validation_ratio=manifest["validationRatio"],
            test_ratio=manifest["testRatio"],
            min_user_ratings=manifest["minUserRatings"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid split manifest configuration.") from exc
    if manifest != expected:
        raise ValueError("Split manifest does not match raw snapshot or exact partition membership.")
    by_id = {interaction_id(row.user_id, row.anime_id): row for row in snapshot.interactions}
    return RawPartitions(
        train=tuple(by_id[item_id] for item_id in manifest["trainIds"]),
        validation=tuple(by_id[item_id] for item_id in manifest["validationIds"]),
        test=tuple(by_id[item_id] for item_id in manifest["testIds"]),
    )


def _no_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key in raw ratings input.")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value} in raw ratings input.")


def load_raw_snapshot(path: Path) -> RawSnapshot:
    decoded = json.loads(path.read_text(encoding="utf-8"),
                         object_pairs_hook=_no_duplicate_json_keys,
                         parse_constant=_invalid_json_constant)
    return parse_raw_snapshot(decoded)


def manifest_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a restricted raw-interaction split manifest.")
    parser.add_argument("--input", type=Path, required=True, help="Raw ratings JSON; no provider fetch")
    parser.add_argument("--out", type=Path, required=True, help="New local manifest path")
    parser.add_argument("--check", action="store_true", help="Compare an existing manifest without writing")
    parser.add_argument("--policy", choices=("auto", "seeded", "temporal"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--min-user-ratings", type=int, default=3)
    args = parser.parse_args()
    try:
        repo_root = Path(__file__).resolve().parents[1]
        output_path = args.out.resolve()
        if not args.check and (output_path.is_relative_to(repo_root / "web" / "public") or
                               output_path.is_relative_to(repo_root / "release-data")):
            raise ValueError("Split manifests must not be written into public or release assets.")
        snapshot = load_raw_snapshot(args.input)
        manifest = build_split_manifest(
            snapshot, policy=args.policy, seed=args.seed,
            validation_ratio=args.validation_ratio, test_ratio=args.test_ratio,
            min_user_ratings=args.min_user_ratings,
        )
        expected = manifest_text(manifest)
        if args.check:
            if args.out.read_text(encoding="utf-8") != expected:
                raise ValueError("Existing split manifest is stale or does not match the raw input.")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8", newline="\n") as output:
                output.write(expected)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"Raw split failed: {exc}\n")
    print(f"{'Verified' if args.check else 'Created'} {manifest['policy']} manifest: "
          f"{manifest['uniqueInteractions']} unique interactions, "
          f"{manifest['duplicateRowsDropped']} identical duplicates removed, "
          f"{len(manifest['trainIds'])}/{len(manifest['validationIds'])}/"
          f"{len(manifest['testIds'])} train/validation/test IDs.")


if __name__ == "__main__":
    main()
