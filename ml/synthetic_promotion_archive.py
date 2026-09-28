#!/usr/bin/env python3
"""Build an invented NPZ and item export for the M8.4 package tests only."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from export_model_web import build_payload
from model_artifact import load_numeric_model, metadata_path, save_numeric_model
from prepare_graph_dataset_bridge import prepare
from split_first_graph_mf import model_fingerprint
from train_only_preprocessing import load_metadata_snapshot


ROOT = Path(__file__).resolve().parents[1]
USER_IDS = ["fixture-overlap-a", "fixture-overlap-b", "fixture-opposite",
            "fixture-equal", "fixture-sparse", "fixture-empty", "fixture-duplicate-unknown"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--bridge-source-dir", type=Path)
    parser.add_argument("--dataset-sha256")
    args = parser.parse_args()
    if bool(args.bridge_source_dir) != bool(args.dataset_sha256):
        parser.error("Bridge source and dataset digest must be supplied together.")
    temp = Path(tempfile.gettempdir()).resolve()
    if not all(path.resolve().is_relative_to(temp) and path.is_dir() for path in
               (args.candidate_dir, args.evidence_dir,
                *([args.bridge_source_dir] if args.bridge_source_dir else []))):
        parser.error("Invented fixture output must use existing temporary directories.")
    template = json.loads((ROOT / "web/public/demo-data/model-mf-web.compact.json")
                          .read_text(encoding="utf-8"))
    archive = args.evidence_dir / "model.npz"
    if args.bridge_source_dir:
        prepared = prepare(args.bridge_source_dir / "raw-ratings.json",
                           args.bridge_source_dir / "split-manifest.json",
                           args.bridge_source_dir / "anime-metadata.json")
        metadata = load_metadata_snapshot(args.bridge_source_dir / "anime-metadata.json")
        user_ids = sorted({row["userId"] for row in prepared["rows"]})
        anime_ids = [anime_id for anime_id, _ in metadata.anime]
        anime_titles = [title for _, title in metadata.anime]
        existing = {anime_id: index for index, anime_id in enumerate(template["animeIds"])}
        embeddings = [template["embeddings"][existing[anime_id]] if anime_id in existing
                      else [0.0] * template["factors"] for anime_id in anime_ids]
        biases = [template["biases"][existing[anime_id]] if anime_id in existing
                  else -100.0 for anime_id in anime_ids]
        indexed = {anime_id: index for index, anime_id in enumerate(anime_ids)}
        seen = [{indexed[row["animeId"]] for row in prepared["rows"]
                 if row["userId"] == user_id} for user_id in user_ids]
    else:
        user_ids = USER_IDS
        anime_ids = template["animeIds"]
        anime_titles = template["titles"]
        embeddings = template["embeddings"]
        biases = template["biases"]
        count = len(anime_ids)
        seen = [{index % count} for index in range(len(user_ids))]
        for index in range(3):
            seen[index].add((index + len(user_ids)) % count)
    save_numeric_model(
        archive, p=np.zeros((len(user_ids), template["factors"]), dtype=np.float32),
        q=np.asarray(embeddings, dtype=np.float32),
        bu=np.zeros(len(user_ids), dtype=np.float32),
        bi=np.asarray(biases, dtype=np.float32),
        global_mean=template["globalMean"], user_ids=user_ids,
        anime_ids=anime_ids, anime_titles=anime_titles,
        train_user_items=seen,
    )
    loaded = load_numeric_model(archive)
    web = build_payload(archive, "compact", 8)
    web["generatedAt"] = template["generatedAt"]
    web["datasetSha256"] = args.dataset_sha256 or template["datasetSha256"]
    (args.candidate_dir / "model-mf-web.compact.json").write_text(
        json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "numericArchiveSha256": loaded.archive_sha256,
        "numericMetadataSha256": hashlib.sha256(metadata_path(archive).read_bytes()).hexdigest(),
        "refitModelSha256": model_fingerprint({
            "P": loaded.p, "Q": loaded.q, "bu": loaded.bu, "bi": loaded.bi,
            "global_mean": loaded.global_mean,
        }),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
