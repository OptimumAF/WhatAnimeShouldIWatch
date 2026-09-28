"""Mocked GitHub metadata checks; no release request is made."""

from __future__ import annotations

import io
import json
import unittest
from urllib.error import HTTPError

from scripts.verify_data_release_target import verify_target


def answer(assets):
    def open_mock(request, timeout):
        assert request.full_url.endswith("/releases/tags/data-latest")
        assert timeout == 20
        return io.BytesIO(json.dumps({"assets": assets}).encode())
    return open_mock


class DataReleaseTargetTests(unittest.TestCase):
    def test_existing_data_only_and_absent_tag_pass(self):
        self.assertEqual(verify_target("invented/repo", "data-latest", "invented-token",
                                       answer([{"name": "graph.compact.json.gz"}])),
                         "existing-data-only")

        def missing(_request, timeout):
            assert timeout == 20
            raise HTTPError("https://example.test", 404, "not found", {}, None)

        self.assertEqual(verify_target("invented/repo", "data-latest", "invented-token",
                                       missing), "new")

    def test_existing_model_and_lookup_failure_block(self):
        with self.assertRaisesRegex(ValueError, "contains a model"):
            verify_target("invented/repo", "data-latest", "invented-token",
                          answer([{"name": "model-mf-web.compact.json.gz"}]))
        with self.assertRaisesRegex(ValueError, "unreviewed gzip"):
            verify_target("invented/repo", "data-latest", "invented-token",
                          answer([{"name": "renamed-model.json.gz"}]))

        def unavailable(_request, timeout):
            assert timeout == 20
            raise HTTPError("https://example.test", 503, "unavailable", {}, None)

        with self.assertRaisesRegex(ValueError, "HTTP 503"):
            verify_target("invented/repo", "data-latest", "invented-token", unavailable)

    def test_missing_token_and_malformed_assets_block(self):
        with self.assertRaisesRegex(ValueError, "token is required"):
            verify_target("invented/repo", "data-latest", "", answer([]))
        with self.assertRaisesRegex(ValueError, "malformed asset"):
            verify_target("invented/repo", "data-latest", "invented-token",
                          answer([{"wrong": "field"}]))


if __name__ == "__main__":
    unittest.main()
