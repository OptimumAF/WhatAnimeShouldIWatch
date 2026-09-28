#!/usr/bin/env python3
"""Prepare private, split-validated fit rows for the aggregate graph bridge.

The JSON written to stdout contains user IDs and scores. It is intended only
for a local verifier pipe, never for logs, web assets, or release artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from raw_interaction_split import _no_duplicate_json_keys, load_raw_snapshot, partition_snapshot
from train_only_preprocessing import fit_training_partition, load_metadata_snapshot


def prepare(raw: Path, split: Path, metadata_path: Path) -> dict[str, object]:
    snapshot = load_raw_snapshot(raw)
    manifest = json.loads(split.read_text(encoding="utf-8"),
                          object_pairs_hook=_no_duplicate_json_keys)
    partitions = partition_snapshot(snapshot, manifest)
    metadata = load_metadata_snapshot(metadata_path)
    original_train = fit_training_partition(partitions.train, metadata)
    fit = fit_training_partition(partitions.train + partitions.validation, metadata)
    titles = dict(metadata.anime)
    return {
        "format": "private-graph-bridge-rows-v1",
        "rawContentSha256": manifest["rawContentSha256"],
        "splitIdentitySha256": manifest["identitySha256"],
        "splitManifestSha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "metadataSha256": metadata.sha256,
        "trainRows": len(partitions.train),
        "validationRows": len(partitions.validation),
        "testRowsExcluded": len(partitions.test),
        "refitTrainSha256": fit.train_sha256,
        "refitFitSha256": fit.fit_sha256,
        "originalTrainSha256": original_train.train_sha256,
        "rows": [{"userId": row.user_id, "animeId": row.anime_id,
                  "title": titles[row.anime_id], "rawScore": row.raw_score,
                  "normalizedScore": row.normalized_score} for row in fit.rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ratings", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("WASIW_PRIVATE_BRIDGE_PIPE") != "1":
        parser.exit(1, "Graph bridge input refused: use the read-only Node verifier; "
                       "the prepared row stream is private.\n")
    try:
        result = prepare(args.raw_ratings, args.split_manifest, args.metadata)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Graph bridge input refused: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
