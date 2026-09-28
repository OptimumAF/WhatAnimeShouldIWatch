#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from model_artifact import load_numeric_model


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


def build_payload(model_path: Path, output_format: str = "compact",
                  round_digits: int = 5) -> dict[str, object]:
    if output_format not in {"compact", "legacy"}:
        raise ValueError("Web model format must be compact or legacy.")
    loaded = load_numeric_model(model_path)
    q = loaded.q
    bi = loaded.bi
    anime_ids = loaded.anime_ids
    anime_titles = loaded.anime_titles
    global_mean = loaded.global_mean
    round_digits = max(0, round_digits)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if output_format == "legacy":
        anime = []
        for idx in range(q.shape[0]):
            anime.append(
                {
                    "animeId": anime_ids[idx],
                    "title": str(anime_titles[idx]),
                    "bias": quantize(float(bi[idx]), round_digits),
                    "embedding": [
                        quantize(float(v), round_digits) for v in q[idx].tolist()
                    ],
                }
            )
        payload = {
            "generatedAt": generated_at,
            "sourceModel": model_path.name,
            "sourceModelSha256": loaded.archive_sha256,
            "globalMean": quantize(global_mean, round_digits),
            "factors": int(q.shape[1]),
            "animeCount": int(q.shape[0]),
            "anime": anime,
        }
    else:
        payload = {
            "format": "model-mf-compact-v1",
            "generatedAt": generated_at,
            "sourceModel": model_path.name,
            "sourceModelSha256": loaded.archive_sha256,
            "globalMean": quantize(global_mean, round_digits),
            "factors": int(q.shape[1]),
            "animeCount": int(q.shape[0]),
            "animeIds": anime_ids,
            "titles": [str(x) for x in anime_titles],
            "biases": [quantize(float(x), round_digits) for x in bi.tolist()],
            "embeddings": [
                [quantize(float(v), round_digits) for v in row.tolist()] for row in q
            ],
        }
    return payload


def export_model(model_path: Path, out_path: Path, output_format: str = "compact",
                 round_digits: int = 5) -> dict[str, object]:
    payload = build_payload(model_path, output_format, round_digits)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False),
                        encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    out_path = Path(args.out)
    export_model(model_path, out_path, args.format, args.round_digits)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported web model -> {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
