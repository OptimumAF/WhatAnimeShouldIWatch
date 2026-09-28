"""Invented GitHub responses only; no release or repository mutation."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from scripts.verify_immutable_model_release import (
    ASSETS, ReleaseCheckError, preflight, published, validate_dispatch,
)


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def missing():
    return HTTPError("https://example.test/invented", 404, "not found", {}, None)


class ImmutableModelReleaseTests(unittest.TestCase):
    def test_dispatch_requires_exact_model_source_and_data_base(self):
        validate_dispatch("invented/repo", "data-vmodel", "12345",
                          "invented-model-package", "data-vbase")
        for inputs, reason in [
            (("invented/repo", "data-vmodel", "123", "package", ""), "data-base"),
            (("invented/repo", "data-vmodel", "123", "package", "data-vmodel"),
             "previous tag"),
            (("invented/repo", "data-vmodel", "0", "package", "data-vbase"),
             "source run ID"),
            (("invented/repo", "data-vmodel", "123", "../package", "data-vbase"),
             "artifact name"),
        ]:
            with self.subTest(inputs=inputs), self.assertRaisesRegex(ReleaseCheckError, reason):
                validate_dispatch(*inputs)

    def test_preflight_requires_admin_read_and_absent_release_and_tag(self):
        seen = []

        def available(request, timeout):
            self.assertEqual(timeout, 20)
            self.assertEqual(request.get_method(), "GET")
            seen.append(request.full_url)
            if request.full_url.endswith("/immutable-releases"):
                self.assertEqual(request.get_header("Authorization"), "Bearer invented-admin")
                return response({"enabled": True})
            self.assertEqual(request.get_header("Authorization"), "Bearer invented-release")
            raise missing()

        preflight("invented/repo", "data-vmodel", "invented-admin", "invented-release",
                  available)
        self.assertEqual([url.rsplit("/", 1)[-1] for url in seen],
                         ["immutable-releases", "data-vmodel", "data-vmodel"])
        with self.assertRaisesRegex(ReleaseCheckError, "credential is missing"):
            preflight("invented/repo", "data-vmodel", "", "token", available)

        def disabled(_request, timeout):
            self.assertEqual(timeout, 20)
            return response({"enabled": False})

        with self.assertRaisesRegex(ReleaseCheckError, "not confirmed enabled"):
            preflight("invented/repo", "data-vmodel", "admin", "token", disabled)

        def existing_release(request, timeout):
            self.assertEqual(timeout, 20)
            if request.full_url.endswith("/immutable-releases"):
                return response({"enabled": True})
            return response({"tag_name": "data-vmodel"})

        with self.assertRaisesRegex(ReleaseCheckError, "release already exists"):
            preflight("invented/repo", "data-vmodel", "admin", "token", existing_release)

        def existing_tag(request, timeout):
            self.assertEqual(timeout, 20)
            if request.full_url.endswith("/immutable-releases"):
                return response({"enabled": True})
            if "/releases/" in request.full_url:
                raise missing()
            return response({"ref": "refs/tags/data-vmodel"})

        with self.assertRaisesRegex(ReleaseCheckError, "tag already exists"):
            preflight("invented/repo", "data-vmodel", "admin", "token", existing_tag)

    def test_published_model_requires_exact_six_hashed_assets(self):
        with tempfile.TemporaryDirectory(prefix="invented-model-release-") as directory:
            package = Path(directory)
            for name in ASSETS:
                (package / name).write_text(f"invented {name}\n", encoding="utf-8")
            assets = [{"name": name, "state": "uploaded",
                       "size": (package / name).stat().st_size,
                       "digest": "sha256:" + hashlib.sha256(
                           (package / name).read_bytes()).hexdigest()}
                      for name in ASSETS]
            metadata = {"tag_name": "data-vmodel", "draft": False,
                        "immutable": True, "assets": assets}

            def remote(value):
                def open_mock(request, timeout):
                    self.assertEqual(timeout, 20)
                    self.assertTrue(request.full_url.endswith("/releases/tags/data-vmodel"))
                    return response(value)
                return open_mock

            published("invented/repo", "data-vmodel", package, "token", remote(metadata))
            for changed, reason in [
                ({**metadata, "draft": True}, "draft"),
                ({**metadata, "immutable": False}, "not immutable"),
                ({**metadata, "assets": assets[:-1]}, "inventory differs"),
                ({**metadata, "assets": [*assets, assets[0]]}, "inventory differs"),
                ({**metadata, "assets": [assets[0], assets[0], *assets[2:]]},
                 "names repeat"),
                ({**metadata, "assets": [{**assets[0], "size": 1}, *assets[1:]]},
                 "differs from local bytes"),
                ({**metadata, "assets": [{**assets[0], "state": "starter"}, *assets[1:]]},
                 "differs from local bytes"),
                ({**metadata, "assets": [{**assets[0], "digest": "sha256:" + "0" * 64},
                                         *assets[1:]]}, "differs from local bytes"),
            ]:
                with self.subTest(reason=reason), self.assertRaisesRegex(ReleaseCheckError,
                                                                         reason):
                    published("invented/repo", "data-vmodel", package, "token",
                              remote(changed))
            (package / "model.npz").write_text("invented private archive", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseCheckError, "exact six-file"):
                published("invented/repo", "data-vmodel", package, "token",
                          remote(metadata))


if __name__ == "__main__":
    unittest.main()
