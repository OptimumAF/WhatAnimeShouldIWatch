"""The retired legacy payload verifier cannot approve invented files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.verify_data_release_payload import verify_data_only_payload


class DataReleasePayloadTests(unittest.TestCase):
    def test_legacy_payload_is_blocked_before_or_after_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "disabled by decision 0028"):
                verify_data_only_payload(root)
            for name in ("graph.compact.json.gz", "anonymized-ratings.compact.json.gz"):
                (root / name).write_bytes(b"invented")
            with self.assertRaisesRegex(ValueError, "disabled by decision 0028"):
                verify_data_only_payload(root)


if __name__ == "__main__":
    unittest.main()
