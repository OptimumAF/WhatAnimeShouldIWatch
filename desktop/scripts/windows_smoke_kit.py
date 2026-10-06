"""Build and verify a local, invented-data kit for the M9.6 clean-host probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import windows_package as candidate


DESKTOP = candidate.DESKTOP
ARCHIVE = DESKTOP / "target" / "package" / "anime-graph-desktop-smoke-kit-windows-x64.zip"
README = DESKTOP / "package" / "SMOKE_TEST.txt"
FIXTURES = DESKTOP / "fixtures"
PACKAGE_NAME = "anime-graph-desktop-windows-x64.zip"
FIXTURE_NAMES = (
    "catalog.identity.json",
    "graph-explorer.compact.json",
    "graph.compact.json",
    "release-manifest.json",
)
EXPECTED = {
    PACKAGE_NAME,
    "README.txt",
    *(f"invented-bundle/{name}" for name in FIXTURE_NAMES),
    "invented-missing/release-manifest.json",
}
MAX_ARCHIVE = 85 * 1024 * 1024
MAX_UNPACKED = 82 * 1024 * 1024
MAX_FIXTURE = 2 * 1024 * 1024


class SmokeKitError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeKitError(message)


def read_source(path: Path, limit: int) -> bytes:
    require(path.is_file() and 0 < path.stat().st_size <= limit,
            f"{path.name}: source file is missing or exceeds its bound")
    return path.read_bytes()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_bytes() -> dict[str, bytes]:
    files = {name: read_source(FIXTURES / name, MAX_FIXTURE) for name in FIXTURE_NAMES}
    try:
        manifest = json.loads(files["release-manifest.json"])
        graph = json.loads(files["graph.compact.json"])
        explorer = json.loads(files["graph-explorer.compact.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeKitError("invented fixture JSON cannot be parsed") from error
    require(isinstance(manifest, dict) and manifest.get("format") == "release-manifest-v1"
            and manifest.get("tag") == "data-vsynthetic-desktop-v1"
            and isinstance(manifest.get("dataset"), dict)
            and manifest["dataset"].get("source") == "synthetic-fixture"
            and manifest.get("model") is None and manifest.get("lastKnownGood") is None,
            "invented release manifest is not the expected data-only fixture")
    for role, name, parsed in (("neighborhood", "graph.compact.json", graph),
                               ("explorer", "graph-explorer.compact.json", explorer),
                               ("catalog", "catalog.identity.json", None)):
        asset = manifest.get(role)
        require(isinstance(asset, dict) and asset.get("path") == name
                and asset.get("bytes") == len(files[name])
                and asset.get("sha256") == sha256(files[name]),
                f"{name}: bytes do not match the invented manifest")
        if parsed is not None:
            require(isinstance(parsed, dict) and parsed.get("format") == "graph-compact-v3"
                    and parsed.get("userIds") == [] and parsed.get("ua") == []
                    and parsed.get("role") == ("recommendation" if role == "neighborhood" else "visualization"),
                    f"{name}: expected aggregate-only v3 graph")
    require(isinstance(graph.get("anime"), list) and len(graph["anime"]) == 8
            and isinstance(graph.get("aa"), list) and len(graph["aa"]) == 11,
            "graph.compact.json: smoke instructions no longer match the invented fixture")
    return files


def expected_bytes() -> dict[str, bytes]:
    candidate.check(candidate.ARCHIVE, compare_source=True)
    package = read_source(candidate.ARCHIVE, candidate.MAX_ARCHIVE)
    fixtures = fixture_bytes()
    try:
        template = read_source(README, 128 * 1024).decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as error:
        raise SmokeKitError("SMOKE_TEST.txt: instructions are not UTF-8") from error
    require(template.count("{PACKAGE_SHA256}") == 1 and template.count("{GRAPH_SHA256}") == 1,
            "SMOKE_TEST.txt: hash placeholders are missing or repeated")
    instructions = (template.replace("{PACKAGE_SHA256}", sha256(package))
                    .replace("{GRAPH_SHA256}", sha256(fixtures["graph.compact.json"])))
    files = {PACKAGE_NAME: package, "README.txt": instructions.encode("utf-8")}
    files.update({f"invented-bundle/{name}": data for name, data in fixtures.items()})
    files["invented-missing/release-manifest.json"] = fixtures["release-manifest.json"]
    require(set(files) == EXPECTED, "smoke kit source inventory is not exact")
    require(sum(map(len, files.values())) <= MAX_UNPACKED, "smoke kit source exceeds its bound")
    return files


def check(archive: Path = ARCHIVE, *, expected: dict[str, bytes] | None = None) -> None:
    require(archive.is_file() and 0 < archive.stat().st_size <= MAX_ARCHIVE,
            "smoke kit ZIP size is invalid")
    if expected is None:
        expected = expected_bytes()
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(EXPECTED) and set(names) == EXPECTED,
                "smoke kit file inventory is not exact")
        require(not source.comment and all(not entry.comment and not entry.extra for entry in entries),
                "smoke kit ZIP metadata is not empty")
        require(all(not entry.is_dir() and ((entry.external_attr >> 16) & 0o170000) != 0o120000
                    for entry in entries), "smoke kit contains a directory or symlink")
        require(sum(entry.file_size for entry in entries) <= MAX_UNPACKED,
                "smoke kit unpacked bytes exceed the bound")
        for entry in entries:
            with source.open(entry) as member:
                data = member.read(len(expected[entry.filename]) + 1)
                require(data == expected[entry.filename] and not member.read(1),
                        f"{entry.filename}: differs from checked source bytes")


def build(archive: Path = ARCHIVE) -> None:
    files = expected_bytes()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as output:
        for name in sorted(files):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(entry, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    check(archive, expected=files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    args = parser.parse_args()
    try:
        if args.action == "build":
            build()
        else:
            check()
    except (SmokeKitError, candidate.PackageError, OSError, zipfile.BadZipFile) as error:
        message = str(error) if isinstance(error, (SmokeKitError, candidate.PackageError)) else "file read failed"
        raise SystemExit(f"smoke kit {args.action} failed: {message}") from None
    print(f"Verified local invented smoke kit: {candidate.display_path(ARCHIVE)} "
          f"({ARCHIVE.stat().st_size} bytes; sha256 {sha256(ARCHIVE.read_bytes())})")


if __name__ == "__main__":
    main()
