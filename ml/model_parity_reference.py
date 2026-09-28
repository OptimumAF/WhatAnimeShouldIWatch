#!/usr/bin/env python3
"""Independent Python reference for an invented browser item-model parity case."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from export_model_web import export_model
from model_artifact import load_numeric_model, save_numeric_model


ROOT = Path(__file__).resolve().parents[1]
CASES = ["signed-and-excluded", "metadata-and-allowlist", "seen-only"]


def _read_json(path: Path) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate parity field {key}.")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def parse_parity_spec(value: Any, input_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "format", "inputSha256", "numericArchiveFormat", "metadataFormat",
        "webFormat", "exportRoundDigits", "scoreAbsoluteTolerance", "tieBreak", "cases",
    } or value != {
        "format": "synthetic-model-parity-spec-v1",
        "inputSha256": _file_sha(input_path),
        "numericArchiveFormat": "model-numeric-npz-v1",
        "metadataFormat": "model-numeric-sidecar-v1",
        "webFormat": "model-mf-compact-v1",
        "exportRoundDigits": 8,
        "scoreAbsoluteTolerance": 0.00001,
        "tieBreak": "score-support-strongest-source-order",
        "cases": CASES,
    }:
        raise ValueError("Invalid or stale synthetic model parity specification.")
    return value


def reference_case(model: dict[str, Any], metadata: list[dict[str, Any]],
                   case: dict[str, Any]) -> dict[str, Any]:
    ids = model["animeIds"]
    embeddings = model["embeddings"]
    biases = model["biases"]
    factors = model["factors"]
    by_id = {anime_id: index for index, anime_id in enumerate(ids)}
    metadata_by_id = {item["animeId"]: item for item in metadata}
    preferences = case["preferences"]
    excluded_sources = {item["animeId"] for item in preferences}
    weighted = []
    for item in preferences:
        index = by_id.get(item["animeId"])
        if index is None or item["sentiment"] == "seen":
            continue
        sign = -1 if item["sentiment"] == "disliked" else 1
        weighted.append((index, sign * item["importance"] * item["confidence"]))
    raw: list[dict[str, Any]] = []
    if weighted:
        user_vector = np.zeros(factors, dtype=np.float32)
        denominator = 0.0
        for index, weight in weighted:
            denominator += abs(weight)
            for factor in range(factors):
                user_vector[factor] = np.float32(
                    float(user_vector[factor]) + embeddings[index][factor] * weight
                )
        if denominator <= 0:
            denominator = float(len(weighted))
        for factor in range(factors):
            user_vector[factor] = np.float32(float(user_vector[factor]) / denominator)
        for index, anime_id in enumerate(ids):
            if anime_id in excluded_sources:
                continue
            score = model["globalMean"] + biases[index]
            for factor in range(factors):
                score += float(user_vector[factor]) * embeddings[index][factor]
            contributions = []
            for watched_index, weight in weighted:
                similarity = sum(
                    embeddings[watched_index][factor] * embeddings[index][factor]
                    for factor in range(factors)
                )
                contributions.append(similarity * weight)
            raw.append({
                "animeId": anime_id, "score": score,
                "support": sum(value > 0 for value in contributions),
                "strongest": max(contributions),
                "sourceOrder": index,
            })
        raw.sort(key=lambda item: (-item["score"], -item["support"],
                                   -item["strongest"], item["sourceOrder"]))
    filters = case["filters"]
    blocked = excluded_sources | set(case["historySeen"]) | set(case["exclude"])
    include_only = set(case["includeOnly"])
    genre = filters["genre"].strip().lower()
    minimum_year = filters["minYear"]
    maximum_year = filters["maxYear"]
    minimum_score = filters["minScore"] or 0
    if minimum_year is not None and maximum_year is not None and minimum_year > maximum_year:
        minimum_year, maximum_year = maximum_year, minimum_year

    def allowed(item: dict[str, Any]) -> bool:
        anime_id = item["animeId"]
        if anime_id in blocked or include_only and anime_id not in include_only:
            return False
        if not (genre or minimum_year is not None or maximum_year is not None or minimum_score > 0):
            return True
        details = metadata_by_id.get(anime_id)
        return details is not None and (
            not genre or genre in {name.strip().lower() for name in details["genres"]}
        ) and (minimum_year is None or details["year"] is not None and
               details["year"] >= minimum_year) and (
            maximum_year is None or details["year"] is not None and
            details["year"] <= maximum_year
        ) and (minimum_score <= 0 or details["score"] is not None and
               details["score"] >= minimum_score)

    eligible = [item for item in raw if allowed(item)]
    eligible_ids = {item["animeId"] for item in eligible}
    return {
        "id": case["id"],
        "raw": [{"animeId": item["animeId"], "score": item["score"]} for item in raw],
        "eligible": [{"animeId": item["animeId"], "score": item["score"]} for item in eligible],
        "topKIds": [item["animeId"] for item in eligible[:case["topK"]]],
        "sourceExcludedIds": [anime_id for anime_id in ids if anime_id in excluded_sources],
        "policyExcludedIds": [item["animeId"] for item in raw
                              if item["animeId"] not in eligible_ids],
    }


def build_parity_report(input_path: Path, spec_path: Path) -> dict[str, Any]:
    spec = parse_parity_spec(_read_json(spec_path), input_path)
    fixture = _read_json(input_path)
    if fixture.get("format") != "synthetic-model-parity-input-v1" or (
        [item.get("id") for item in fixture.get("cases", [])] != CASES
    ):
        raise ValueError("Invalid synthetic model parity input.")
    source = fixture["model"]
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "model.npz"
        save_numeric_model(
            archive_path,
            p=np.asarray(source["userEmbeddings"], dtype=np.float32),
            q=np.asarray(source["embeddings"], dtype=np.float32),
            bu=np.asarray(source["userBiases"], dtype=np.float32),
            bi=np.asarray(source["biases"], dtype=np.float32),
            global_mean=source["globalMean"], user_ids=source["userIds"],
            anime_ids=source["animeIds"], anime_titles=source["titles"],
            train_user_items=source["trainUserItems"],
        )
        loaded = load_numeric_model(archive_path)
        compact_path = Path(temporary) / "model-mf-web.compact.json"
        export_model(archive_path, compact_path, "compact", spec["exportRoundDigits"])
        compact = _read_json(compact_path)
        if compact["sourceModelSha256"] != loaded.archive_sha256:
            raise ValueError("Web export sourceModelSha256 does not match its numeric archive.")
    return {
        "format": "synthetic-model-parity-reference-v1",
        "inputSha256": spec["inputSha256"],
        "archiveSha256": loaded.archive_sha256,
        "model": compact,
        "metadata": fixture["metadata"],
        "cases": [reference_case(compact, fixture["metadata"], case)
                  for case in fixture["cases"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print invented numeric model/browser parity reference.")
    parser.add_argument("--input", type=Path, default=ROOT / "fixtures" /
                        "synthetic-model-parity-input.json")
    parser.add_argument("--spec", type=Path, default=ROOT / "fixtures" /
                        "synthetic-model-parity-spec.json")
    args = parser.parse_args()
    try:
        report = build_parity_report(args.input, args.spec)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
