"""Reject data-only updates to an existing GitHub release containing a model."""

from __future__ import annotations

import json
import os
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ALLOWED_GZIP = {"graph.compact.json.gz", "anonymized-ratings.compact.json.gz"}


def verify_target(repo: str, tag: str, token: str, opener=urlopen) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("GITHUB_REPOSITORY must be an owner/repo name")
    if not tag or not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("release tag contains unsupported characters")
    if not token:
        raise ValueError("GitHub token is required to inspect the release target")
    request = Request(
        f"https://api.github.com/repos/{repo}/releases/tags/{quote(tag, safe='')}",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "WhatAnimeShouldIWatch-data-release-guard"},
    )
    try:
        with opener(request, timeout=20) as response:
            release = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return "new"
        raise ValueError(f"GitHub release lookup failed with HTTP {exc.code}") from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"GitHub release lookup failed: {exc}") from exc
    if not isinstance(release, dict) or not isinstance(release.get("assets"), list):
        raise ValueError("GitHub release lookup returned malformed assets")
    for asset in release["assets"]:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            raise ValueError("GitHub release lookup returned a malformed asset")
        if asset["name"].lower().startswith("model"):
            raise ValueError("Existing target release contains a model; data-only update blocked")
        if asset["name"].endswith(".gz") and asset["name"] not in ALLOWED_GZIP:
            raise ValueError("Existing target release contains an unreviewed gzip asset")
    return "existing-data-only"


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_data_release_target.py <owner/repo> <tag>", file=sys.stderr)
        return 2
    try:
        result = verify_target(sys.argv[1], sys.argv[2],
                               os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", ""))
    except ValueError as exc:
        print(f"Data release blocked: {exc}", file=sys.stderr)
        return 1
    print(f"Data-only release target verified: {result}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
