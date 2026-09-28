"""Fail-closed GitHub release preflight and post-publication checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ASSETS = (
    "release-manifest.json",
    "graph.compact.json",
    "graph-explorer.compact.json",
    "catalog.identity.json",
    "publication-audit.json",
)
REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
TAG = re.compile(r"data-v[A-Za-z0-9][A-Za-z0-9._-]*\Z")
ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ReleaseCheckError(ValueError):
    pass


def _inputs(repo: str, tag: str) -> None:
    if not REPO.fullmatch(repo):
        raise ReleaseCheckError("repository must be owner/repo")
    if not TAG.fullmatch(tag):
        raise ReleaseCheckError("tag must be a versioned data-v tag")


def validate_dispatch(repo: str, tag: str, source_run_id: str,
                      artifact_name: str, previous_tag: str) -> None:
    _inputs(repo, tag)
    if (not re.fullmatch(r"[0-9]{1,16}", source_run_id)
            or not (1 <= int(source_run_id) <= 2**53 - 1)):
        raise ReleaseCheckError("source run ID must be a positive safe integer")
    if not ARTIFACT_NAME.fullmatch(artifact_name):
        raise ReleaseCheckError("artifact name is invalid")
    if previous_tag and (not TAG.fullmatch(previous_tag) or previous_tag == tag):
        raise ReleaseCheckError("previous tag must be empty or a distinct versioned data-v tag")


def _get(repo: str, endpoint: str, token: str, opener=urlopen, *, allow_missing=False):
    if not token:
        raise ReleaseCheckError("required GitHub credential is missing")
    request = Request(
        f"https://api.github.com/repos/{repo}/{endpoint}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "WhatAnimeShouldIWatch-immutable-data-release-check",
        },
    )
    try:
        with opener(request, timeout=20) as response:
            return json.load(response)
    except HTTPError as exc:
        if allow_missing and exc.code == 404:
            return None
        raise ReleaseCheckError(f"GitHub {endpoint} returned HTTP {exc.code}") from exc
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise ReleaseCheckError(f"GitHub {endpoint} could not be verified") from exc


def preflight(repo: str, tag: str, admin_read_token: str, release_token: str,
              opener=urlopen) -> None:
    _inputs(repo, tag)
    setting = _get(repo, "immutable-releases", admin_read_token, opener)
    if not isinstance(setting, dict) or setting.get("enabled") is not True:
        raise ReleaseCheckError("immutable releases are not confirmed enabled")
    encoded_tag = quote(tag, safe="")
    release = _get(repo, f"releases/tags/{encoded_tag}", release_token,
                   opener, allow_missing=True)
    if release is not None:
        raise ReleaseCheckError("candidate release already exists")
    ref = _get(repo, f"git/ref/tags/{encoded_tag}", release_token,
               opener, allow_missing=True)
    if ref is not None:
        raise ReleaseCheckError("candidate tag already exists")


def published(repo: str, tag: str, package_dir: Path, token: str,
              opener=urlopen) -> None:
    _inputs(repo, tag)
    if not package_dir.is_dir() or package_dir.is_symlink():
        raise ReleaseCheckError("package directory must be real")
    entries = sorted(package_dir.iterdir())
    if [entry.name for entry in entries] != sorted(ASSETS):
        raise ReleaseCheckError("package inventory is not the exact five-file allowlist")
    for entry in entries:
        if not entry.is_file() or entry.is_symlink():
            raise ReleaseCheckError(f"package asset {entry.name} is not a regular file")
    release = _get(repo, f"releases/tags/{quote(tag, safe='')}", token, opener)
    if (not isinstance(release, dict) or release.get("tag_name") != tag
            or release.get("draft") is not False
            or release.get("immutable") is not True):
        raise ReleaseCheckError("published release is absent, draft, or not immutable")
    assets = release.get("assets")
    if not isinstance(assets, list) or len(assets) != len(ASSETS):
        raise ReleaseCheckError("published asset inventory differs from the five-file allowlist")
    by_name = {}
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            raise ReleaseCheckError("published asset metadata is malformed")
        if asset["name"] in by_name:
            raise ReleaseCheckError("published asset names repeat")
        by_name[asset["name"]] = asset
    if set(by_name) != set(ASSETS):
        raise ReleaseCheckError("published asset names differ from the five-file allowlist")
    for entry in entries:
        remote = by_name[entry.name]
        digest = hashlib.sha256()
        with entry.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if (remote.get("state") != "uploaded"
                or remote.get("size") != entry.stat().st_size
                or remote.get("digest") != f"sha256:{digest.hexdigest()}"):
            raise ReleaseCheckError(f"published asset {entry.name} differs from local bytes")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in {"inputs", "preflight", "published"}:
        print("usage: verify_immutable_data_release.py <inputs|preflight|published> "
              "<owner/repo> <data-vtag> [mode arguments]", file=sys.stderr)
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
        print(f"Immutable data release blocked: {exc}", file=sys.stderr)
        return 1
    print(f"Immutable data release {argv[1]} verified for {argv[3]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
