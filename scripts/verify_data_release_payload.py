"""Refuse the retired ratings-bearing data-latest publication payload."""

from __future__ import annotations

import sys
from pathlib import Path


def verify_data_only_payload(directory: Path) -> None:
    raise ValueError(
        "Legacy data release payload is disabled by decision 0028; "
        "a verified aggregate-only bundle and publication audit are required."
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_data_release_payload.py <staged-directory>", file=sys.stderr)
        return 2
    try:
        verify_data_only_payload(Path(argv[1]))
    except ValueError as exc:
        print(f"Data release blocked: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
