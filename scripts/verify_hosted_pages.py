"""Bounded post-deploy HTTP probe for one exact GitHub Pages release."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import zlib
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .verify_immutable_data_release import REPO, TAG

MANIFEST_LIMIT = 256 * 1024
ASSET_LIMIT = 256 * 1024 * 1024
TOTAL_LIMIT = 512 * 1024 * 1024
HTML_LIMIT = 2 * 1024 * 1024
DIGEST = re.compile(r"[a-f0-9]{64}\Z")


class HostedCheckError(ValueError):
    pass


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes


def _request(url: str, maximum: int, accept_gzip: bool = False) -> HttpResult:
    request = Request(url, headers={
        "Accept-Encoding": "gzip" if accept_gzip else "identity",
        "Cache-Control": "no-cache",
        "User-Agent": "WhatAnimeShouldIWatch-hosted-pages-check",
    })
    try:
        response = urlopen(request, timeout=20)
    except HTTPError as error:
        response = error
    except (URLError, OSError) as exc:
        raise HostedCheckError(f"hosted request unavailable: {url}") from exc
    try:
        with response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            compressed = headers.get("content-encoding", "").lower() == "gzip"
            limit = 64 * 1024 * 1024 if compressed else maximum
            body = response.read(limit + 1)
            if len(body) > limit:
                raise HostedCheckError(f"hosted response exceeds byte limit: {url}")
            return HttpResult(response.status, headers, body)
    except OSError as exc:
        raise HostedCheckError(f"hosted response interrupted: {url}") from exc


def _plain(result: HttpResult, maximum: int, field: str) -> bytes:
    encoding = result.headers.get("content-encoding", "").lower()
    if encoding not in ("", "identity", "gzip"):
        raise HostedCheckError(f"{field}: unsupported content encoding")
    if encoding == "gzip":
        try:
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            body = decoder.decompress(result.body, maximum + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise HostedCheckError(f"{field}: incomplete or oversized gzip")
        except (OSError, EOFError, ValueError, zlib.error) as exc:
            raise HostedCheckError(f"{field}: malformed gzip") from exc
    else:
        body = result.body
    if len(body) > maximum:
        raise HostedCheckError(f"{field}: exceeds plain-byte limit")
    return body


def _json(body: bytes, field: str) -> dict:
    try:
        value = json.loads(body.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostedCheckError(f"{field}: invalid JSON or UTF-8") from exc
    if not isinstance(value, dict):
        raise HostedCheckError(f"{field}: must be an object")
    return value


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _asset(manifest: dict, field: str, name: str) -> dict:
    item = manifest.get(field)
    if not isinstance(item, dict) or item.get("path") != name or not isinstance(
            item.get("bytes"), int) or isinstance(item.get("bytes"), bool) or not (
            0 < item["bytes"] <= ASSET_LIMIT) or not isinstance(
            item.get("sha256"), str) or not DIGEST.fullmatch(item["sha256"]):
        raise HostedCheckError(f"release-manifest.json.{field}: invalid asset")
    return item


def verify_hosted_pages(repo: str, page_url: str, kind: str, tag: str,
                        manifest_sha256: str, fetcher=_request) -> dict[str, object]:
    if not REPO.fullmatch(repo) or not TAG.fullmatch(tag) or kind not in ("data", "model") or not (
            isinstance(manifest_sha256, str) and DIGEST.fullmatch(manifest_sha256)):
        raise HostedCheckError("invalid exact hosted release inputs")
    owner, name = repo.split("/")
    base = f"https://{owner.lower()}.github.io/{name}/"
    parsed = urlparse(page_url)
    if (parsed.scheme != "https" or parsed.netloc.lower() != f"{owner.lower()}.github.io" or
            parsed.path != f"/{name}/" or parsed.query or parsed.fragment or page_url != base):
        raise HostedCheckError("Pages URL differs from the reviewed project path")
    home_response = fetcher(base, HTML_LIMIT, False)
    home = _plain(home_response, HTML_LIMIT, "index.html")
    if home_response.status != 200 or f"/{name}/assets/".encode() not in home:
        raise HostedCheckError("index.html: missing project-path app entry")
    direct_response = fetcher(base + "__direct_navigation_check__/", HTML_LIMIT, False)
    direct = _plain(direct_response, HTML_LIMIT, "404.html")
    if direct_response.status not in (200, 404) or direct != home:
        raise HostedCheckError("404.html: direct navigation differs from the app entry")
    pointer_response = fetcher(base + "data/active.json", MANIFEST_LIMIT, False)
    if pointer_response.status != 200:
        raise HostedCheckError("active.json: unavailable")
    pointer = _json(_plain(pointer_response, MANIFEST_LIMIT, "active.json"), "active.json")
    bundle_id = pointer.get("bundleId")
    if (pointer.get("format") != "active-release-bundle-v1" or pointer.get("tag") != tag or
            not isinstance(bundle_id, str) or not DIGEST.fullmatch(bundle_id) or
            pointer.get("manifestSha256") != manifest_sha256):
        raise HostedCheckError("active.json: stale or malformed release pointer")
    bundle_url = base + f"data/bundles/{bundle_id}/"
    manifest_response = fetcher(bundle_url + "release-manifest.json", MANIFEST_LIMIT, False)
    if manifest_response.status != 200:
        raise HostedCheckError("release-manifest.json: unavailable")
    manifest_bytes = _plain(manifest_response, MANIFEST_LIMIT, "release-manifest.json")
    if _sha(manifest_bytes) != manifest_sha256:
        raise HostedCheckError("release-manifest.json: byte hash differs from dispatch")
    manifest = _json(manifest_bytes, "release-manifest.json")
    if manifest.get("tag") != tag or manifest.get("bundleId") != bundle_id:
        raise HostedCheckError("release-manifest.json: identity differs from active pointer")
    entries = [
        _asset(manifest, "neighborhood", "graph.compact.json"),
        _asset(manifest, "explorer", "graph-explorer.compact.json"),
        _asset(manifest, "catalog", "catalog.identity.json"),
    ]
    model = manifest.get("model")
    if (kind == "model"):
        entries.append(_asset(manifest, "model", "model-mf-web.compact.json"))
    elif model is not None:
        raise HostedCheckError("release-manifest.json.model: data release must have no model")
    if sum(entry["bytes"] for entry in entries) > TOTAL_LIMIT:
        raise HostedCheckError("release-manifest.json: total asset bytes exceed limit")
    gzip_observed = False
    for entry in entries:
        result = fetcher(bundle_url + entry["path"], entry["bytes"],
                         entry["path"] == "graph.compact.json")
        if result.status != 200:
            raise HostedCheckError(f"{entry['path']}: unavailable")
        body = _plain(result, entry["bytes"], entry["path"])
        if len(body) != entry["bytes"] or _sha(body) != entry["sha256"]:
            raise HostedCheckError(f"{entry['path']}: hosted bytes differ from release manifest")
        gzip_observed |= result.headers.get("content-encoding", "").lower() == "gzip"
    if kind == "data":
        missing = fetcher(bundle_url + "model-mf-web.compact.json", HTML_LIMIT, False)
        if missing.status != 404:
            raise HostedCheckError("model-mf-web.compact.json: data-only release exposes a model")
    return {"tag": tag, "bundleId": bundle_id, "gzipObserved": gzip_observed,
            "securityHeaders": {key: key in home_response.headers for key in (
                "strict-transport-security", "content-security-policy",
                "x-content-type-options", "referrer-policy")}}


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print("usage: python -m scripts.verify_hosted_pages <owner/repo> "
              "<pages-url> <data|model> <data-vtag> <manifest-sha256>", file=sys.stderr)
        return 2
    for attempt in range(6):
        try:
            result = verify_hosted_pages(*argv[1:])
            print("Hosted Pages verified: " + json.dumps(result, sort_keys=True))
            return 0
        except HostedCheckError as exc:
            if attempt == 5:
                print(f"Hosted Pages blocked: {exc}", file=sys.stderr)
                return 1
            time.sleep(10)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
