#!/usr/bin/env python3
"""Read-only private NPZ/refit/web-export consistency gate for M8.4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from export_model_web import build_payload
from model_artifact import load_numeric_model, metadata_path
from split_first_graph_mf import model_fingerprint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Private JSON has duplicate field {key}.")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} root must be an object.")
    return value


def verify_private_archive(evidence_dir: Path, web_model_path: Path) -> None:
    """Require a safe NPZ whose fitted bytes and item export match private records."""
    archive_path = evidence_dir / "model.npz"
    loaded = load_numeric_model(archive_path)
    refit = _read(evidence_dir / "refit-record.json")
    cohort = _read(evidence_dir / "serving-cohort.json")
    web = _read(web_model_path)

    if refit.get("numericArchiveSha256") != loaded.archive_sha256 or (
        web.get("sourceModelSha256") != loaded.archive_sha256
    ):
        raise ValueError("model.npz digest differs from refit-record or web model.")
    if refit.get("numericMetadataSha256") != _sha256(metadata_path(archive_path)):
        raise ValueError("model.metadata.json digest differs from refit-record.numericMetadataSha256.")
    fingerprint = model_fingerprint({
        "P": loaded.p, "Q": loaded.q, "bu": loaded.bu, "bi": loaded.bi,
        "global_mean": loaded.global_mean,
    })
    if refit.get("refitModelSha256") != fingerprint:
        raise ValueError("model.npz fitted arrays differ from refit-record.refitModelSha256.")
    refit_rows = refit.get("refitRows")
    if type(refit_rows) is not int or sum(map(len, loaded.train_user_items)) != refit_rows:
        raise ValueError("model.npz packed fit membership differs from refit-record.refitRows.")
    fit_ids = cohort.get("trainingUsers")
    if not isinstance(fit_ids, list) or len(fit_ids) != len(loaded.user_ids) or (
        len(set(fit_ids)) != len(fit_ids) or set(fit_ids) != set(loaded.user_ids)
    ):
        raise ValueError("model.metadata.json userIds differ from serving-cohort.trainingUsers.")

    expected = build_payload(archive_path, "compact", 8)
    for field in ("format", "sourceModel", "sourceModelSha256", "globalMean",
                  "factors", "animeCount", "animeIds", "titles", "biases", "embeddings"):
        if web.get(field) != expected[field]:
            raise ValueError(f"model-mf-web.compact.json.{field} differs from the numeric export.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--web-model", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify_private_archive(args.evidence_dir, args.web_model)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Promotion archive verification failed: {exc}\n")
    print("Verified private numeric archive and item model.")


if __name__ == "__main__":
    main()
