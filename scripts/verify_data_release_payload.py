"""Keep the legacy data release route free of unpromoted model assets."""

from __future__ import annotations

import sys
from pathlib import Path


ALLOWED_GZIP = {
    "graph.compact.json.gz",
    "anonymized-ratings.compact.json.gz",
}


def verify_data_only_payload(directory: Path) -> None:
    if not directory.is_dir():
        raise ValueError(f"Missing staged release directory: {directory}")
    entries = {entry.name for entry in directory.iterdir() if entry.is_file()}
    missing = ALLOWED_GZIP - entries
    if missing:
        raise ValueError(f"Missing data release asset: {', '.join(sorted(missing))}")
    unexpected = sorted(name for name in entries if name.endswith(".gz") and name not in ALLOWED_GZIP)
    if unexpected:
        raise ValueError(
            "Data release cannot carry a model or other unreviewed gzip asset: "
            + ", ".join(unexpected)
        )
    if any(name.lower().startswith("model") for name in entries):
        raise ValueError("Data release cannot carry a model without the promotion workflow.")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_data_release_payload.py <staged-directory>", file=sys.stderr)
        return 2
    try:
        verify_data_only_payload(Path(argv[1]))
    except ValueError as exc:
        print(f"Data release blocked: {exc}", file=sys.stderr)
        return 1
    print("Data-only release payload verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
