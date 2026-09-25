"""Fail closed before a workflow consumes or publishes provider-derived data."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
APPROVALS_PATH = REPO_ROOT / "docs" / "approvals" / "provider-data.json"
SCOPES = {"training", "publication", "deployment"}


class ApprovalError(ValueError):
    pass


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApprovalError(f"missing {field}")
    return value.strip()


def validate_approval(
    manifest: object, scope: str, approval_ref: str, repo_root: Path
) -> None:
    if scope not in SCOPES:
        raise ApprovalError(f"unsupported scope: {scope}")
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise ApprovalError("unsupported approval manifest")
    approvals = manifest.get("approvals")
    if not isinstance(approvals, dict):
        raise ApprovalError("missing approvals object")
    entry = approvals.get(scope)
    if not isinstance(entry, dict) or entry.get("approved") is not True:
        raise ApprovalError(f"{scope} has no recorded approval")

    sources = entry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ApprovalError("missing sources")
    for source in sources:
        _nonempty(source, "source")
    _nonempty(entry.get("sourceBasis"), "sourceBasis")
    _nonempty(entry.get("use"), "use")
    _nonempty(entry.get("owner"), "owner")
    approved_at = _nonempty(entry.get("approvedAt"), "approvedAt")
    try:
        if date.fromisoformat(approved_at).isoformat() != approved_at:
            raise ValueError("noncanonical date")
    except ValueError as exc:
        raise ApprovalError("approvedAt must be an ISO date") from exc

    decision_ref = _nonempty(entry.get("decisionRef"), "decisionRef")
    decisions_dir = (repo_root / "docs" / "decisions").resolve()
    decision_path = (repo_root / decision_ref).resolve()
    if (
        decision_path.suffix != ".md"
        or not decision_path.is_relative_to(decisions_dir)
        or not decision_path.is_file()
    ):
        raise ApprovalError("decisionRef must name an existing docs/decisions Markdown file")

    recorded_ref = _nonempty(entry.get("approvalRef"), "approvalRef")
    supplied_ref = _nonempty(approval_ref, "PROVIDER_DATA_APPROVAL_REF")
    parsed_ref = urlparse(recorded_ref)
    if parsed_ref.scheme != "https" or not parsed_ref.netloc:
        raise ApprovalError("approvalRef must be an HTTPS URL")
    if recorded_ref != supplied_ref:
        raise ApprovalError("repository approval reference does not match the recorded scope")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_provider_data_approval.py <training|publication|deployment>", file=sys.stderr)
        return 2
    try:
        with APPROVALS_PATH.open(encoding="utf-8") as source:
            manifest = json.load(source)
        validate_approval(
            manifest, argv[1], os.environ.get("PROVIDER_DATA_APPROVAL_REF", ""), REPO_ROOT
        )
    except (OSError, json.JSONDecodeError, ApprovalError) as exc:
        print(f"Provider-data approval blocked: {exc}", file=sys.stderr)
        return 1
    print(f"Provider-data {argv[1]} approval record verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
