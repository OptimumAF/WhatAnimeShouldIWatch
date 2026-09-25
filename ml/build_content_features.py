#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from provider_scheduler import JikanRequestScheduler, RequestBudgetExceeded, RequestCancelled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build anime content features from Jikan metadata for cold-start mitigation."
    )
    parser.add_argument(
        "--model",
        default="models/graph_mf/model.npz",
        help="Path to model.npz containing anime IDs/titles",
    )
    parser.add_argument(
        "--out",
        default="models/graph_mf/content-features.json",
        help="Output JSON file for content features",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=350,
        help="Delay between Jikan requests in milliseconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per anime request",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only fetch first N anime (0 = all)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore existing output and rebuild from scratch",
    )
    return parser.parse_args()


def load_model_anime(path: Path) -> List[Tuple[int, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    raw = np.load(path, allow_pickle=True)
    anime_ids = [int(x) for x in raw["anime_ids"].tolist()]
    anime_titles = [str(x) for x in raw["anime_titles"].tolist()]
    return list(zip(anime_ids, anime_titles))


_default_scheduler = JikanRequestScheduler()


def fetch_json(
    url: str,
    max_retries: int,
    scheduler: JikanRequestScheduler | None = None,
    cancelled=lambda: False,
) -> Dict[str, object] | None:
    return (scheduler or _default_scheduler).get_json(url, max_retries, cancelled)


def parse_name_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    names: List[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        normalized = name.strip()
        if not normalized or normalized in names:
            continue
        names.append(normalized)
    return names


def parse_entry(payload: Dict[str, object], anime_id: int, fallback_title: str) -> Dict[str, object] | None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        title = fallback_title

    synopsis = data.get("synopsis")
    if not isinstance(synopsis, str):
        synopsis = ""

    year = data.get("year")
    score = data.get("score")
    season = data.get("season")

    image_url = ""
    images = data.get("images")
    if isinstance(images, dict):
        webp = images.get("webp")
        jpg = images.get("jpg")
        if isinstance(webp, dict):
            image_url = str(webp.get("large_image_url") or webp.get("image_url") or "")
        if not image_url and isinstance(jpg, dict):
            image_url = str(jpg.get("large_image_url") or jpg.get("image_url") or "")

    entry = {
        "animeId": int(anime_id),
        "title": str(title),
        "year": int(year) if isinstance(year, (int, float)) else None,
        "score": float(score) if isinstance(score, (int, float)) else None,
        "season": str(season) if isinstance(season, str) else None,
        "genres": parse_name_list(data.get("genres")),
        "studios": parse_name_list(data.get("studios")),
        "synopsis": synopsis.strip(),
        "imageUrl": image_url,
    }
    return entry


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_anime = load_model_anime(model_path)
    if args.limit > 0:
        model_anime = model_anime[: args.limit]

    features: Dict[str, Dict[str, object]] = {}
    if out_path.exists() and not args.refresh:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and isinstance(existing.get("anime"), dict):
            for key, value in existing["anime"].items():
                if isinstance(key, str) and isinstance(value, dict):
                    features[key] = value

    total = len(model_anime)
    fetched = 0
    skipped = 0
    failures = 0
    scheduler = JikanRequestScheduler(delay_ms=args.delay_ms)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda _number, _frame: stop.set())
    signal.signal(signal.SIGTERM, lambda _number, _frame: stop.set())

    for index, (anime_id, fallback_title) in enumerate(model_anime, start=1):
        key = str(anime_id)
        if key in features:
            skipped += 1
            continue

        url = f"https://api.jikan.moe/v4/anime/{anime_id}/full"
        try:
            payload = fetch_json(
                url, max_retries=max(0, args.max_retries), scheduler=scheduler,
                cancelled=stop.is_set,
            )
            if payload is None:
                failures += 1
                continue
            parsed = parse_entry(payload, anime_id=anime_id, fallback_title=fallback_title)
            if parsed is None:
                failures += 1
                continue
            features[key] = parsed
            fetched += 1
        except RequestBudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[warn] anime {anime_id} failed: {exc}")

        if index % 50 == 0 or index == total:
            report = {
                "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "jikan.moe/v4/anime/{id}/full",
                "animeCount": len(features),
                "anime": features,
            }
            out_path.write_text(json.dumps(report), encoding="utf-8")
            print(
                f"Progress {index}/{total} | cached={len(features)} fetched={fetched} skipped={skipped} failures={failures}"
            )

    report = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "jikan.moe/v4/anime/{id}/full",
        "animeCount": len(features),
        "anime": features,
    }
    out_path.write_text(json.dumps(report), encoding="utf-8")
    print(f"Saved content features -> {out_path}")
    print(f"Fetched={fetched} skipped_existing={skipped} failures={failures}")


if __name__ == "__main__":
    try:
        main()
    except RequestCancelled:
        raise SystemExit(130) from None
