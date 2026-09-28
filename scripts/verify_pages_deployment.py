"""Validate exact immutable-release inputs before a Pages build downloads assets."""

from __future__ import annotations

import re
import sys

from .verify_immutable_data_release import ReleaseCheckError
from .verify_immutable_data_release import validate_dispatch as data_dispatch
from .verify_immutable_model_release import validate_dispatch as model_dispatch

DIGEST = re.compile(r"[a-f0-9]{64}\Z")


def validate_inputs(repo: str, kind: str, tag: str, manifest_sha256: str,
                    source_run_id: str, artifact_name: str,
                    previous_tag: str) -> None:
    if kind == "data":
        data_dispatch(repo, tag, source_run_id, artifact_name, previous_tag)
    elif kind == "model":
        model_dispatch(repo, tag, source_run_id, artifact_name, previous_tag)
    else:
        raise ReleaseCheckError("Pages kind must be data or model")
    if not DIGEST.fullmatch(manifest_sha256):
        raise ReleaseCheckError("Pages manifest SHA-256 must be a lowercase digest")


def main(argv: list[str]) -> int:
    if len(argv) != 8:
        print("usage: python -m scripts.verify_pages_deployment "
              "<owner/repo> <data|model> <data-vtag> <manifest-sha256> "
              "<source-run-id> <artifact-name> <previous-tag-or-empty>",
              file=sys.stderr)
        return 2
    try:
        validate_inputs(*argv[1:])
    except ReleaseCheckError as exc:
        print(f"Pages deployment blocked: {exc}", file=sys.stderr)
        return 1
    print(f"Exact Pages {argv[2]} inputs verified for {argv[3]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
