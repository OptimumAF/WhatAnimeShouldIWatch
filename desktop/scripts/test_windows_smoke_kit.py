"""The M9.6 transfer kit must contain only the checked app and invented data."""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import windows_smoke_kit as kit


class WindowsSmokeKitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(kit.ARCHIVE.is_file(), "build the local smoke kit first")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def rewrite(self, change):
        destination = Path(self.directory.name) / "changed.zip"
        with zipfile.ZipFile(kit.ARCHIVE) as source, zipfile.ZipFile(destination, "w") as output:
            for member in source.infolist():
                output.writestr(member, change(member.filename, source.read(member.filename)))
        return destination

    def test_exact_invented_kit_and_package(self):
        kit.check()
        with zipfile.ZipFile(kit.ARCHIVE) as source:
            self.assertEqual(set(source.namelist()), kit.EXPECTED)
            self.assertEqual(source.read(kit.PACKAGE_NAME), kit.candidate.ARCHIVE.read_bytes())
            graph = json.loads(source.read("invented-bundle/graph.compact.json"))
            self.assertEqual(graph["format"], "graph-compact-v3")
            self.assertEqual(graph["userIds"], [])
            self.assertEqual(graph["ua"], [])
            self.assertFalse(any(name.startswith("data/") for name in source.namelist()))

    def test_changed_graph_and_extra_asset_are_rejected(self):
        changed = self.rewrite(lambda name, data: data[:-1] + b"x"
                               if name == "invented-bundle/graph.compact.json" else data)
        with self.assertRaisesRegex(kit.SmokeKitError, "differs from checked source bytes"):
            kit.check(changed)
        with zipfile.ZipFile(changed, "a") as output:
            output.writestr("invented-bundle/invented-ratings.json", b"[]")
        with self.assertRaisesRegex(kit.SmokeKitError, "inventory is not exact"):
            kit.check(changed)

    def test_source_graph_with_user_rows_is_rejected_even_when_rehashed(self):
        fixture_dir = Path(self.directory.name) / "fixtures"
        fixture_dir.mkdir()
        for name in kit.FIXTURE_NAMES:
            (fixture_dir / name).write_bytes((kit.FIXTURES / name).read_bytes())
        graph_path = fixture_dir / "graph.compact.json"
        graph = json.loads(graph_path.read_bytes())
        graph["userIds"] = ["invented-user"]
        graph_bytes = (json.dumps(graph) + "\n").encode("utf-8")
        graph_path.write_bytes(graph_bytes)
        manifest_path = fixture_dir / "release-manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["neighborhood"]["bytes"] = len(graph_bytes)
        manifest["neighborhood"]["sha256"] = hashlib.sha256(graph_bytes).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(kit, "FIXTURES", fixture_dir):
            with self.assertRaisesRegex(kit.SmokeKitError, "expected aggregate-only v3 graph"):
                kit.fixture_bytes()


if __name__ == "__main__":
    unittest.main()
