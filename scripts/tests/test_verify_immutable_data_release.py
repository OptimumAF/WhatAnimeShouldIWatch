"""Invented GitHub responses only; no release or setting mutation."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from scripts.verify_immutable_data_release import (
    ASSETS, ReleaseCheckError, preflight, published, validate_dispatch,
)


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def not_found():
    return HTTPError("https://example.test/invented", 404, "not found", {}, None)


class ImmutableDataReleaseTests(unittest.TestCase):
    def test_dispatch_input_validation(self):
        validate_dispatch("invented/repo", "data-vnew", "12345", "invented-package",
                          "data-vprior")
        validate_dispatch("invented/repo", "data-vfirst", "12345", "invented-package", "")
        with self.assertRaisesRegex(ReleaseCheckError, "source run ID"):
            validate_dispatch("invented/repo", "data-vnew", "0", "invented-package",
                              "data-vprior")
        with self.assertRaisesRegex(ReleaseCheckError, "artifact name"):
            validate_dispatch("invented/repo", "data-vnew", "123", "../package",
                              "data-vprior")
        with self.assertRaisesRegex(ReleaseCheckError, "previous tag"):
            validate_dispatch("invented/repo", "data-vnew", "123", "invented-package",
                              "data-vnew")
        with self.assertRaisesRegex(ReleaseCheckError, "previous tag"):
            validate_dispatch("invented/repo", "data-vnew", "123", "invented-package",
                              "data-latest")

    def test_preflight_requires_enabled_setting_and_absent_release_and_tag(self):
        seen = []

        def absent(request, timeout):
            self.assertEqual(timeout, 20)
            seen.append(request.full_url)
            if request.full_url.endswith("/immutable-releases"):
                self.assertEqual(request.get_header("Authorization"), "Bearer invented-admin")
                return response({"enabled": True, "enforced_by_owner": False})
            self.assertEqual(request.get_header("Authorization"), "Bearer invented-release")
            raise not_found()

        preflight("invented/repo", "data-vnew", "invented-admin", "invented-release", absent)
        self.assertEqual([url.rsplit("/", 1)[-1] for url in seen],
                         ["immutable-releases", "data-vnew", "data-vnew"])

        def disabled(request, timeout):
            return response({"enabled": False})

        with self.assertRaisesRegex(ReleaseCheckError, "not confirmed enabled"):
            preflight("invented/repo", "data-vnew", "admin", "release", disabled)
        with self.assertRaisesRegex(ReleaseCheckError, "credential is missing"):
            preflight("invented/repo", "data-vnew", "", "release", absent)
        with self.assertRaisesRegex(ReleaseCheckError, "versioned data-v"):
            preflight("invented/repo", "data-latest", "admin", "release", absent)

    def test_existing_tag_release_and_lookup_error_block_before_mutation(self):
        def existing_release(request, timeout):
            if request.full_url.endswith("/immutable-releases"):
                return response({"enabled": True})
            return response({"tag_name": "data-vnew"})

        with self.assertRaisesRegex(ReleaseCheckError, "release already exists"):
            preflight("invented/repo", "data-vnew", "admin", "release", existing_release)

        def existing_tag(request, timeout):
            if request.full_url.endswith("/immutable-releases"):
                return response({"enabled": True})
            if "/releases/" in request.full_url:
                raise not_found()
            return response({"ref": "refs/tags/data-vnew"})

        with self.assertRaisesRegex(ReleaseCheckError, "tag already exists"):
            preflight("invented/repo", "data-vnew", "admin", "release", existing_tag)

        def denied(_request, timeout):
            self.assertEqual(timeout, 20)
            raise HTTPError("https://example.test/invented", 403, "denied", {}, None)

        with self.assertRaisesRegex(ReleaseCheckError, "HTTP 403"):
            preflight("invented/repo", "data-vnew", "admin", "release", denied)

    def test_published_release_requires_immutable_exact_hashed_assets(self):
        with tempfile.TemporaryDirectory(prefix="invented-release-post-") as directory:
            package = Path(directory)
            for name in ASSETS:
                (package / name).write_text(f"invented {name}\n", encoding="utf-8")
            assets = [
                {"name": name, "state": "uploaded", "size": (package / name).stat().st_size,
                 "digest": "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest()}
                for name in ASSETS
            ]
            metadata = {"tag_name": "data-vnew", "draft": False,
                        "immutable": True, "assets": assets}

            def with_metadata(value):
                def open_mock(request, timeout):
                    self.assertEqual(timeout, 20)
                    self.assertTrue(request.full_url.endswith("/releases/tags/data-vnew"))
                    return response(value)
                return open_mock

            published("invented/repo", "data-vnew", package, "invented-token",
                      with_metadata(metadata))
            with self.assertRaisesRegex(ReleaseCheckError, "not immutable"):
                published("invented/repo", "data-vnew", package, "invented-token",
                          with_metadata({**metadata, "immutable": False}))
            with self.assertRaisesRegex(ReleaseCheckError, "inventory differs"):
                published("invented/repo", "data-vnew", package, "invented-token",
                          with_metadata({**metadata, "assets": assets[:-1]}))
            with self.assertRaisesRegex(ReleaseCheckError, "differs from local bytes"):
                published("invented/repo", "data-vnew", package, "invented-token",
                          with_metadata({**metadata, "assets": [
                              {**assets[0], "digest": "sha256:" + "0" * 64}, *assets[1:]]}))
            (package / "ratings.sqlite").write_text("invented", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseCheckError, "exact five-file"):
                published("invented/repo", "data-vnew", package, "invented-token",
                          with_metadata(metadata))


if __name__ == "__main__":
    unittest.main()
