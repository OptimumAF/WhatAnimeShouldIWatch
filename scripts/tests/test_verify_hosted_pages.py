import gzip
import hashlib
import json
import unittest

from scripts.verify_hosted_pages import (
    HostedCheckError, HttpResult, verify_hosted_pages,
)

REPO = "InventedOwner/InventedRepo"
BASE = "https://inventedowner.github.io/InventedRepo/"
TAG = "data-vinvented-hosted"


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture(with_model=False):
    bundle_id = "b" * 64
    graph = b'{"invented":"graph"}\n'
    explorer = b'{"invented":"explorer"}\n'
    catalog = b'{"invented":"catalog"}\n'
    model = b'{"invented":"item-only model"}\n'
    def asset(name, body):
        return {"path": name, "bytes": len(body), "sha256": sha(body)}
    manifest = {"tag": TAG, "bundleId": bundle_id,
                "neighborhood": asset("graph.compact.json", graph),
                "explorer": asset("graph-explorer.compact.json", explorer),
                "catalog": asset("catalog.identity.json", catalog),
                "model": asset("model-mf-web.compact.json", model) if with_model else None}
    manifest_bytes = encoded(manifest)
    digest = sha(manifest_bytes)
    pointer = encoded({"format": "active-release-bundle-v1", "tag": TAG,
                       "bundleId": bundle_id, "manifestSha256": digest})
    html = b'<script type="module" src="/InventedRepo/assets/app.js"></script>'
    prefix = BASE + f"data/bundles/{bundle_id}/"
    responses = {
        BASE: HttpResult(200, {"strict-transport-security": "max-age=1"}, html),
        BASE + "__direct_navigation_check__/": HttpResult(404, {}, html),
        BASE + "data/active.json": HttpResult(200, {}, pointer),
        prefix + "release-manifest.json": HttpResult(200, {}, manifest_bytes),
        prefix + "graph.compact.json": HttpResult(200, {"content-encoding": "gzip"},
                                                   gzip.compress(graph)),
        prefix + "graph-explorer.compact.json": HttpResult(200, {}, explorer),
        prefix + "catalog.identity.json": HttpResult(200, {}, catalog),
        prefix + "model-mf-web.compact.json": (
            HttpResult(200, {}, model) if with_model else HttpResult(404, {}, html)),
    }
    def fetcher(url, maximum, accept_gzip=False):
        if url not in responses:
            raise AssertionError(f"unplanned mocked URL: {url}")
        if url.endswith("graph.compact.json") and not accept_gzip:
            raise AssertionError("graph compression was not requested")
        return responses[url]
    return responses, fetcher, digest, prefix


class HostedPagesTests(unittest.TestCase):
    def test_data_and_model_paths_verify_exact_bytes_and_gzip(self):
        for kind in ("data", "model"):
            with self.subTest(kind=kind):
                _, fetcher, digest, _ = fixture(kind == "model")
                result = verify_hosted_pages(REPO, BASE, kind, TAG, digest, fetcher)
                self.assertEqual(result["tag"], TAG)
                self.assertTrue(result["gzipObserved"])
                self.assertTrue(result["securityHeaders"]["strict-transport-security"])

    def test_stale_pointer_changed_graph_direct_navigation_and_model_leak_fail(self):
        for mutation, pattern in (
            (lambda responses, prefix: responses.__setitem__(BASE + "data/active.json",
             HttpResult(200, {}, b'{}')), "active.json"),
            (lambda responses, prefix: responses.__setitem__(prefix + "graph.compact.json",
             HttpResult(200, {}, b'{}')), "graph.compact.json"),
            (lambda responses, prefix: responses.__setitem__(
                BASE + "__direct_navigation_check__/", HttpResult(404, {}, b"wrong")),
             "404.html"),
            (lambda responses, prefix: responses.__setitem__(
                prefix + "model-mf-web.compact.json", HttpResult(200, {}, b"model")),
             "exposes a model"),
            (lambda responses, prefix: responses.__setitem__(prefix + "graph.compact.json",
             HttpResult(200, {"content-encoding": "gzip"}, b"invalid")), "malformed gzip"),
        ):
            responses, fetcher, digest, prefix = fixture()
            mutation(responses, prefix)
            with self.subTest(pattern=pattern), self.assertRaisesRegex(HostedCheckError, pattern):
                verify_hosted_pages(REPO, BASE, "data", TAG, digest, fetcher)
        _, fetcher, digest, _ = fixture()
        with self.assertRaisesRegex(HostedCheckError, "project path"):
            verify_hosted_pages(REPO, BASE + "wrong/", "data", TAG, digest, fetcher)


if __name__ == "__main__":
    unittest.main()
