"""Negative package checks use only the locally built desktop EXE."""

import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import windows_package as package


class WindowsPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(package.ARCHIVE.is_file(), "build the local candidate package first")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def rewrite(self, change):
        destination = Path(self.directory.name) / "changed.zip"
        with zipfile.ZipFile(package.ARCHIVE) as source, zipfile.ZipFile(destination, "w") as output:
            for member in source.infolist():
                content = source.read(member.filename)
                output.writestr(member, change(member.filename, content))
        return destination

    def test_import_inventory_matches_built_exe(self):
        names = package.imports_from_pe(package.EXE.read_bytes())
        self.assertTrue(package.VC_DLLS <= set(names))
        self.assertNotIn("webview2loader.dll", names)
        package.check(package.ARCHIVE, compare_source=True)

    def test_changed_exe_is_rejected(self):
        path = self.rewrite(lambda name, content: content[:-1] + b"x" if name.endswith(".exe") else content)
        with self.assertRaisesRegex(package.PackageError, "hash/length inventory"):
            package.check(path)

    def test_extra_asset_is_rejected(self):
        path = self.rewrite(lambda _name, content: content)
        with zipfile.ZipFile(path, "a") as output:
            output.writestr(package.PREFIX + "invented-ratings.json", b"[]")
        with self.assertRaisesRegex(package.PackageError, "inventory is not exact"):
            package.check(path)

    def test_truncated_exe_is_rejected(self):
        path = self.rewrite(lambda name, content: content[:100] if name.endswith(".exe") else content)
        with self.assertRaises(package.PackageError):
            package.check(path)

    def test_hidden_zip_comment_is_rejected(self):
        path = self.rewrite(lambda _name, content: content)
        with zipfile.ZipFile(path, "a") as output:
            output.comment = b"invented hidden payload"
        with self.assertRaisesRegex(package.PackageError, "ZIP metadata"):
            package.check(path)

    def test_packaged_preflight_rejects_changed_exe(self):
        with zipfile.ZipFile(package.ARCHIVE) as candidate:
            candidate.extractall(self.directory.name)
        folder = Path(self.directory.name) / "anime-graph-desktop"
        exe = folder / "anime_graph_desktop.exe"
        data = exe.read_bytes()
        exe.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(folder / "Check-Prerequisites.ps1")],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("desktop EXE differs", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
