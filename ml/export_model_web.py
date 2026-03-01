#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained graph MF model to web-friendly JSON."
    )
    parser.add_argument(
        "--model",
        default="models/graph_mf/model.npz",
        help="Path to model.npz from train_graph_mf.py",
    )
    parser.add_argument(
        "--out",
        default="data/model-mf-web.compact.json",
        help="Output JSON path for the web app",
    )
    parser.add_argument(
        "--format",
        choices=["compact", "legacy"],
        default="compact",
        help="Output schema format. 'compact' is significantly smaller.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=5,
        dest="round_digits",
        help="Decimal places for exported floats",
    )
    return parser.parse_args()


def quantize(value: float, digits: int) -> float:
    return round(float(value), digits)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    raw = np.load(model_path, allow_pickle=True)
    q = raw["Q"].astype(np.float32)
    bi = raw["bi"].astype(np.float32)
    anime_ids = raw["anime_ids"].astype(np.int64)
    anime_titles = raw["anime_titles"].tolist()
    global_mean = float(raw["global_mean"][0]) if "global_mean" in raw else 0.0

    if q.shape[0] != bi.shape[0] or q.shape[0] != anime_ids.shape[0]:
        raise ValueError(
            "Model arrays are inconsistent: expected Q, bi, anime_ids to have equal length."
        )

    round_digits = max(0, args.round_digits)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.format == "legacy":
        anime = []
        for idx in range(q.shape[0]):
            anime.append(
                {
                    "animeId": int(anime_ids[idx]),
                    "title": str(anime_titles[idx]),
                    "bias": quantize(float(bi[idx]), round_digits),
                    "embedding": [
                        quantize(float(v), round_digits) for v in q[idx].tolist()
                    ],
                }
            )
        payload = {
            "generatedAt": generated_at,
            "sourceModel": str(model_path),
            "globalMean": quantize(global_mean, round_digits),
            "factors": int(q.shape[1]),
            "animeCount": int(q.shape[0]),
            "anime": anime,
        }
    else:
        payload = {
            "format": "model-mf-compact-v1",
            "generatedAt": generated_at,
            "sourceModel": str(model_path),
            "globalMean": quantize(global_mean, round_digits),
            "factors": int(q.shape[1]),
            "animeCount": int(q.shape[0]),
            "animeIds": [int(x) for x in anime_ids.tolist()],
            "titles": [str(x) for x in anime_titles],
            "biases": [quantize(float(x), round_digits) for x in bi.tolist()],
            "embeddings": [
                [quantize(float(v), round_digits) for v in row.tolist()] for row in q
            ],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported web model -> {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
