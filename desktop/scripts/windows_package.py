"""Build and check a data-free Windows desktop candidate ZIP using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "desktop"
EXE = DESKTOP / "target" / "release" / "anime_graph_desktop.exe"
ARCHIVE = DESKTOP / "target" / "package" / "anime-graph-desktop-windows-x64.zip"
PREFIX = "anime-graph-desktop/"
FORMAT = "anime-desktop-portable-v1"
MAX_EXE = 64 * 1024 * 1024
MAX_ARCHIVE = 80 * 1024 * 1024
MAX_UNPACKED = 70 * 1024 * 1024
SOURCE_FILES = {
    "README.txt": DESKTOP / "package" / "README.txt",
    "Check-Prerequisites.ps1": DESKTOP / "package" / "Check-Prerequisites.ps1",
    "Launch.cmd": DESKTOP / "package" / "Launch.cmd",
}
WINDOWS_DLLS = {
    "advapi32.dll", "ntdll.dll", "kernel32.dll", "user32.dll", "ole32.dll",
    "comctl32.dll", "gdi32.dll", "api-ms-win-core-synch-l1-2-0.dll",
    "dwmapi.dll", "shlwapi.dll", "shell32.dll", "oleaut32.dll",
    "bcryptprimitives.dll", "ws2_32.dll",
    "api-ms-win-crt-math-l1-1-0.dll", "api-ms-win-crt-string-l1-1-0.dll",
    "api-ms-win-crt-convert-l1-1-0.dll", "api-ms-win-crt-runtime-l1-1-0.dll",
    "api-ms-win-crt-stdio-l1-1-0.dll", "api-ms-win-crt-locale-l1-1-0.dll",
    "api-ms-win-crt-heap-l1-1-0.dll",
}
VC_DLLS = {"vcruntime140.dll", "vcruntime140_1.dll"}


class PackageError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


def unpack(fmt: str, data: bytes, offset: int) -> tuple[int, ...]:
    size = struct.calcsize(fmt)
    require(0 <= offset <= len(data) - size, "PE header points outside the EXE")
    return struct.unpack_from(fmt, data, offset)


def imports_from_pe(data: bytes) -> list[str]:
    """Read normal and delay-load import DLL names from a bounded x64 GUI PE."""
    require(0 < len(data) <= MAX_EXE, "EXE byte length is outside the package bound")
    require(data[:2] == b"MZ", "EXE has no MZ header")
    pe = unpack("<I", data, 0x3C)[0]
    require(data[pe:pe + 4] == b"PE\0\0", "EXE has no PE signature")
    machine, section_count = unpack("<HH", data, pe + 4)
    optional_size = unpack("<H", data, pe + 20)[0]
    optional = pe + 24
    require(machine == 0x8664 and 0 < section_count <= 96, "EXE must be Windows x64")
    require(unpack("<H", data, optional)[0] == 0x20B, "EXE must be PE32+")
    require(unpack("<H", data, optional + 68)[0] == 2, "EXE must use the Windows GUI subsystem")
    require(optional_size >= 112 + 14 * 8 and optional + optional_size <= len(data), "EXE optional header is invalid")
    header_size = unpack("<I", data, optional + 60)[0]
    require(header_size <= len(data), "EXE header byte length is invalid")
    directory_count = unpack("<I", data, optional + 108)[0]
    require(directory_count >= 14, "EXE has no complete import-directory table")
    section_start = optional + optional_size
    require(section_start + section_count * 40 <= len(data), "EXE section table is truncated")
    sections = []
    for index in range(section_count):
        header = section_start + index * 40
        virtual_size, virtual_address, raw_size, raw_offset = unpack("<IIII", data, header + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset, raw_size))

    def rva_offset(rva: int) -> int:
        if rva < header_size:
            require(rva < len(data), "PE header RVA is outside the EXE")
            return rva
        for address, span, raw_offset, raw_size in sections:
            if address <= rva < address + span:
                delta = rva - address
                require(delta < raw_size and raw_offset + delta < len(data), "PE import RVA has no file bytes")
                return raw_offset + delta
        raise PackageError("PE import RVA is outside mapped sections")

    def dll_name(rva: int) -> str:
        offset = rva_offset(rva)
        end = data.find(b"\0", offset, min(offset + 256, len(data)))
        require(end > offset, "PE import DLL name is empty or too long")
        try:
            return data[offset:end].decode("ascii").lower()
        except UnicodeDecodeError as error:
            raise PackageError("PE import DLL name is not ASCII") from error

    names: set[str] = set()
    for directory_index, descriptor_size in ((1, 20), (13, 32)):
        rva, size = unpack("<II", data, optional + 112 + directory_index * 8)
        if rva == 0 and size == 0:
            continue
        require(rva != 0 and descriptor_size <= size <= 64 * 1024, "PE import directory is invalid")
        base = rva_offset(rva)
        terminated = False
        for index in range(min(size // descriptor_size, 1024)):
            offset = base + index * descriptor_size
            require(offset + descriptor_size <= len(data), "PE import descriptor is truncated")
            descriptor = data[offset:offset + descriptor_size]
            if not any(descriptor):
                terminated = True
                break
            if directory_index == 13:
                attributes, name_rva = unpack("<II", data, offset)
                require(attributes & 1 == 1, "PE delay import uses an unsupported address mode")
            else:
                name_rva = unpack("<I", data, offset + 12)[0]
            names.add(dll_name(name_rva))
        require(terminated, "PE import directory has no terminator")
    require(VC_DLLS <= names, "EXE VC runtime imports differ from the documented prerequisites")
    require(names <= WINDOWS_DLLS | VC_DLLS, f"EXE has unreviewed DLL imports: {sorted(names - WINDOWS_DLLS - VC_DLLS)}")
    return sorted(names)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(name: str, data: bytes) -> dict[str, str | int]:
    return {"path": name, "bytes": len(data), "sha256": sha256(data)}


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def source_bytes() -> dict[str, bytes]:
    require(0 < EXE.stat().st_size <= MAX_EXE, "built EXE byte length is outside the package bound")
    files = {"anime_graph_desktop.exe": EXE.read_bytes()}
    for name, path in SOURCE_FILES.items():
        require(0 < path.stat().st_size <= 128 * 1024, "package documentation or launcher exceeds the text bound")
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        files[name] = normalized.replace(b"\n", b"\r\n") if name.endswith(".cmd") else normalized
    return files


def toml_string(path: Path, section: str, key: str) -> str:
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.fullmatch(r"\s*\[([^]]+)\]\s*", line)
        if heading:
            current = heading.group(1)
        elif current == section:
            match = re.fullmatch(rf'\s*{re.escape(key)}\s*=\s*"([^"]+)"\s*', line)
            if match:
                return match.group(1)
    raise PackageError(f"{path.name}: [{section}].{key} is missing or unsupported")


def metadata() -> tuple[str, str]:
    channel = toml_string(ROOT / "rust-toolchain.toml", "toolchain", "channel")
    version = toml_string(DESKTOP / "Cargo.toml", "package", "version")
    require(re.fullmatch(r"\d+\.\d+\.\d+", channel) is not None, "Rust toolchain is not pinned")
    return channel, version


def package(archive: Path = ARCHIVE) -> None:
    files = source_bytes()
    imports = imports_from_pe(files["anime_graph_desktop.exe"])
    channel, version = metadata()
    compiler = subprocess.run(["rustc", "-vV"], cwd=ROOT, capture_output=True, text=True, check=False)
    require(compiler.returncode == 0
            and re.search(rf"^release: {re.escape(channel)}$", compiler.stdout, re.MULTILINE)
            and re.search(r"^host: x86_64-pc-windows-msvc$", compiler.stdout, re.MULTILINE),
            "active Rust compiler/host differs from the Windows x64 package contract")
    manifest = {
        "format": FORMAT,
        "target": "x86_64-pc-windows-msvc",
        "appVersion": version,
        "rustToolchain": channel,
        "imports": imports,
        "prerequisites": ["Microsoft Edge WebView2 Evergreen Runtime", "Microsoft Visual C++ x64 Redistributable"],
        "files": [file_record(name, files[name]) for name in sorted(files)],
    }
    files["package-manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as output:
        for name in sorted(files):
            entry = zipfile.ZipInfo(PREFIX + name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(entry, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    check(archive, compare_source=True)
    print(f"Verified local Windows candidate: {display_path(archive)} ({archive.stat().st_size} bytes)")


def check(archive: Path = ARCHIVE, *, compare_source: bool = False) -> None:
    require(archive.is_file() and 0 < archive.stat().st_size <= MAX_ARCHIVE, "package ZIP size is invalid")
    expected = {"anime_graph_desktop.exe", *SOURCE_FILES, "package-manifest.json"}
    with zipfile.ZipFile(archive) as package_zip:
        info = package_zip.infolist()
        names = [entry.filename for entry in info]
        require(len(names) == len(expected) and set(names) == {PREFIX + name for name in expected}, "package file inventory is not exact")
        require(not package_zip.comment and all(not entry.comment and not entry.extra for entry in info), "package ZIP metadata is not empty")
        require(all(not entry.is_dir() and ((entry.external_attr >> 16) & 0o170000) != 0o120000 for entry in info), "package contains a directory or symlink")
        require(sum(entry.file_size for entry in info) <= MAX_UNPACKED, "package unpacked bytes exceed the bound")
        files = {}
        unpacked = 0
        for name in names:
            with package_zip.open(name) as member:
                chunks = []
                while chunk := member.read(64 * 1024):
                    unpacked += len(chunk)
                    require(unpacked <= MAX_UNPACKED, "package inflated bytes exceed the bound")
                    chunks.append(chunk)
                files[name.removeprefix(PREFIX)] = b"".join(chunks)
    try:
        manifest = json.loads(files.pop("package-manifest.json"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PackageError("package manifest is not JSON") from error
    require(isinstance(manifest, dict) and set(manifest) == {
        "format", "target", "appVersion", "rustToolchain", "imports", "prerequisites", "files"
    }, "package manifest fields are invalid")
    channel, version = metadata()
    require(manifest["format"] == FORMAT and manifest["target"] == "x86_64-pc-windows-msvc"
            and manifest["rustToolchain"] == channel and manifest["appVersion"] == version,
            "package identity differs from the checked source")
    require(manifest["prerequisites"] == ["Microsoft Edge WebView2 Evergreen Runtime", "Microsoft Visual C++ x64 Redistributable"], "package prerequisites are invalid")
    require(manifest["imports"] == imports_from_pe(files["anime_graph_desktop.exe"]), "package import inventory differs from EXE")
    require(manifest["files"] == [file_record(name, files[name]) for name in sorted(files)], "package file hash/length inventory differs")
    if compare_source:
        require(files == source_bytes(), "package files differ from the built EXE or checked sources")
    print(f"Package inventory, hashes, PE imports, and prerequisites passed: {display_path(archive)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "check"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--compare-source", action="store_true", help="also compare every ZIP file to this checkout")
    args = parser.parse_args()
    try:
        if args.action == "package":
            package(args.archive)
        else:
            check(args.archive, compare_source=args.compare_source)
        return 0
    except OSError:
        print("Windows package check failed: a required local input could not be read or written", file=sys.stderr)
        return 1
    except (PackageError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Windows package check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
