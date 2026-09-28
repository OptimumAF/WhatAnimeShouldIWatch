"""Check an approved six-file model release without using private evaluation rows."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import urlopen

from .verify_immutable_data_release import (
    ReleaseCheckError, preflight as release_preflight,
    published_assets, validate_dispatch as release_dispatch,
)

ASSETS = (
    "release-manifest.json",
    "graph.compact.json",
    "graph-explorer.compact.json",
    "catalog.identity.json",
    "model-mf-web.compact.json",
    "model-promotion-audit.json",
)


def validate_dispatch(repo: str, tag: str, source_run_id: str,
                      artifact_name: str, base_tag: str) -> None:
    release_dispatch(repo, tag, source_run_id, artifact_name, base_tag)
    if not base_tag:
        raise ReleaseCheckError("model release requires a distinct approved data-base tag")


def preflight(repo: str, tag: str, admin_read_token: str, release_token: str,
              opener=urlopen) -> None:
    release_preflight(repo, tag, admin_read_token, release_token, opener)


def published(repo: str, tag: str, package_dir: Path, token: str,
              opener=urlopen) -> None:
    published_assets(repo, tag, package_dir, token, ASSETS, opener)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in {"inputs", "preflight", "published"}:
        print("usage: python -m scripts.verify_immutable_model_release "
              "<inputs|preflight|published> <owner/repo> <data-vtag> "
              "[mode arguments]", file=sys.stderr)
        return 2
    try:
        if argv[1] == "inputs" and len(argv) == 7:
            validate_dispatch(argv[2], argv[3], argv[4], argv[5], argv[6])
        elif argv[1] == "preflight" and len(argv) == 4:
            preflight(argv[2], argv[3],
                      os.environ.get("RELEASE_IMMUTABILITY_READ_TOKEN", ""),
                      os.environ.get("GITHUB_TOKEN", ""))
        elif argv[1] == "published" and len(argv) == 5:
            published(argv[2], argv[3], Path(argv[4]),
                      os.environ.get("GITHUB_TOKEN", ""))
        else:
            raise ReleaseCheckError("wrong argument count")
    except ReleaseCheckError as exc:
        print(f"Immutable model release blocked: {exc}", file=sys.stderr)
        return 1
    print(f"Immutable model release {argv[1]} verified for {argv[3]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
