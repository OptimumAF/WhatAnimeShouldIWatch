"""Invented data-only release payload checks; no publishing occurs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.verify_data_release_payload import verify_data_only_payload


class DataReleasePayloadTests(unittest.TestCase):
    def test_only_the_two_data_assets_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("graph.compact.json.gz", "anonymized-ratings.compact.json.gz"):
                (root / name).write_bytes(b"invented")
            verify_data_only_payload(root)

    def test_model_and_unexpected_gzip_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("graph.compact.json.gz", "anonymized-ratings.compact.json.gz"):
                (root / name).write_bytes(b"invented")
            for name in ("model-mf-web.compact.json.gz", "model-mf-web.compact.json",
                         "other-unreviewed.json.gz"):
                (root / name).write_bytes(b"invented")
                with self.assertRaisesRegex(ValueError, "model|unreviewed"):
                    verify_data_only_payload(root)
                (root / name).unlink()

    def test_missing_data_asset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "graph.compact.json.gz").write_bytes(b"invented")
            with self.assertRaisesRegex(ValueError, "anonymized-ratings"):
                verify_data_only_payload(root)


if __name__ == "__main__":
    unittest.main()
