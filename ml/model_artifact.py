"""Numeric-only model exchange with a hash-bound JSON metadata sidecar."""

from __future__ import annotations

import hashlib
import json
import numbers
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ARCHIVE_FORMAT = "model-numeric-npz-v1"
METADATA_FORMAT = "model-numeric-sidecar-v1"
_ARRAY_DTYPES = {
    "P": np.dtype("float32"),
    "Q": np.dtype("float32"),
    "bu": np.dtype("float32"),
    "bi": np.dtype("float32"),
    "global_mean": np.dtype("float32"),
    "anime_ids": np.dtype("int64"),
    "train_item_offsets": np.dtype("int64"),
    "train_item_indices": np.dtype("int32"),
}
_MAX_ARCHIVE_BYTES = 1_073_741_824
_MAX_ARRAY_BYTES = 2_147_483_648
_MAX_METADATA_BYTES = 134_217_728


@dataclass(frozen=True)
class NumericModel:
    p: np.ndarray
    q: np.ndarray
    bu: np.ndarray
    bi: np.ndarray
    global_mean: float
    user_ids: list[str]
    anime_ids: list[int]
    anime_titles: list[str]
    train_user_items: list[set[int]]
    archive_sha256: str


def metadata_path(model_path: Path) -> Path:
    if model_path.suffix.lower() != ".npz":
        raise ValueError("Model artifact path must end in .npz.")
    return model_path.with_name(model_path.stem + ".metadata.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Model metadata has duplicate field {key}.")
        value[key] = item
    return value


def _validate_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "format", "archiveFormat", "archiveSha256", "userCount", "animeCount",
        "factors", "userIds", "animeTitles",
    }:
        raise ValueError("Model metadata fields are unsupported or incomplete.")
    if value["format"] != METADATA_FORMAT or value["archiveFormat"] != ARCHIVE_FORMAT:
        raise ValueError("Model metadata format is unsupported.")
    digest = value["archiveSha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Model metadata archiveSha256 must be SHA-256.")
    for field, minimum in [("userCount", 0), ("animeCount", 1), ("factors", 1)]:
        number = value[field]
        if type(number) is not int or number < minimum:
            raise ValueError(f"Model metadata {field} is invalid.")
    for field, count_field in [("userIds", "userCount"), ("animeTitles", "animeCount")]:
        entries = value[field]
        if not isinstance(entries, list) or len(entries) != value[count_field] or any(
            not isinstance(entry, str) or not entry.strip() for entry in entries
        ):
            raise ValueError(f"Model metadata {field} is invalid.")
    if len(set(value["userIds"])) != value["userCount"]:
        raise ValueError("Model metadata userIds contains duplicates.")
    return value


def _validate_arrays(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    if set(arrays) != set(_ARRAY_DTYPES):
        raise ValueError("Model NPZ array membership is unsupported; legacy object arrays are refused.")
    for name, expected in _ARRAY_DTYPES.items():
        if arrays[name].dtype != expected:
            raise ValueError(f"Model NPZ {name} dtype must be {expected}; object arrays are refused.")
    p, q, bu, bi = (arrays[name] for name in ("P", "Q", "bu", "bi"))
    user_count = metadata["userCount"]
    anime_count = metadata["animeCount"]
    factors = metadata["factors"]
    if p.shape != (user_count, factors) or q.shape != (anime_count, factors) or (
        bu.shape != (user_count,) or bi.shape != (anime_count,) or
        arrays["global_mean"].shape != (1,) or
        arrays["anime_ids"].shape != (anime_count,) or
        arrays["train_item_offsets"].shape != (user_count + 1,) or
        arrays["train_item_indices"].ndim != 1
    ):
        raise ValueError("Model NPZ dimensions disagree with metadata.")
    for name in ("P", "Q", "bu", "bi", "global_mean"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Model NPZ {name} contains a nonfinite value.")
    anime_ids = arrays["anime_ids"].tolist()
    if any(anime_id < 1 for anime_id in anime_ids) or len(set(anime_ids)) != anime_count:
        raise ValueError("Model NPZ anime_ids must be positive and unique.")
    offsets = arrays["train_item_offsets"].tolist()
    indices = arrays["train_item_indices"].tolist()
    if offsets[0] != 0 or offsets[-1] != len(indices) or any(
        left > right for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError("Model NPZ train_item_offsets is invalid.")
    if any(index < 0 or index >= anime_count for index in indices):
        raise ValueError("Model NPZ train_item_indices is out of range.")
    for start, end in zip(offsets, offsets[1:]):
        group = indices[start:end]
        if group != sorted(set(group)):
            raise ValueError("Model NPZ train_item_indices must be sorted and unique per user.")


def save_numeric_model(
    model_path: Path, *, p: np.ndarray, q: np.ndarray, bu: np.ndarray,
    bi: np.ndarray, global_mean: float, user_ids: Sequence[str],
    anime_ids: Sequence[int], anime_titles: Sequence[str],
    train_user_items: Sequence[Iterable[int]],
) -> Path:
    """Write a numeric NPZ and a sidecar whose hash binds the exact archive bytes."""
    sidecar = metadata_path(model_path)
    if len(train_user_items) != len(user_ids):
        raise ValueError("Model train_user_items length must match userIds.")
    if any(isinstance(item, bool) or not isinstance(item, numbers.Integral) or item < 1
           for item in anime_ids):
        raise ValueError("Model anime_ids must be positive integers.")
    offsets = [0]
    indices: list[int] = []
    for items in train_user_items:
        values = list(items)
        if any(isinstance(item, bool) or not isinstance(item, numbers.Integral) or
               item < 0 or item >= len(anime_ids) for item in values):
            raise ValueError("Model train_user_items contains an invalid index.")
        group = sorted(set(int(item) for item in values))
        indices.extend(group)
        offsets.append(len(indices))
    arrays = {
        "P": np.asarray(p, dtype=np.float32),
        "Q": np.asarray(q, dtype=np.float32),
        "bu": np.asarray(bu, dtype=np.float32),
        "bi": np.asarray(bi, dtype=np.float32),
        "global_mean": np.asarray([global_mean], dtype=np.float32),
        "anime_ids": np.asarray(anime_ids, dtype=np.int64),
        "train_item_offsets": np.asarray(offsets, dtype=np.int64),
        "train_item_indices": np.asarray(indices, dtype=np.int32),
    }
    metadata: dict[str, object] = {
        "format": METADATA_FORMAT,
        "archiveFormat": ARCHIVE_FORMAT,
        "archiveSha256": "0" * 64,
        "userCount": len(user_ids),
        "animeCount": len(anime_ids),
        "factors": arrays["Q"].shape[1] if arrays["Q"].ndim == 2 else 0,
        "userIds": list(user_ids),
        "animeTitles": list(anime_titles),
    }
    _validate_metadata(metadata)
    _validate_arrays(arrays, metadata)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_archive: Path | None = None
    temp_metadata: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=model_path.parent, suffix=".npz",
                                         delete=False) as handle:
            temp_archive = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        if temp_archive.stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ValueError("Model NPZ exceeds the archive size limit.")
        metadata["archiveSha256"] = _sha256(temp_archive)
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent, suffix=".json", mode="w", encoding="utf-8",
            delete=False,
        ) as handle:
            temp_metadata = Path(handle.name)
            json.dump(metadata, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if temp_metadata.stat().st_size > _MAX_METADATA_BYTES:
            raise ValueError("Model metadata exceeds the size limit.")
        os.replace(temp_archive, model_path)
        temp_archive = None
        os.replace(temp_metadata, sidecar)
        temp_metadata = None
    finally:
        if temp_archive is not None:
            temp_archive.unlink(missing_ok=True)
        if temp_metadata is not None:
            temp_metadata.unlink(missing_ok=True)
    return sidecar


def load_numeric_model(model_path: Path) -> NumericModel:
    """Never executes pickle, including on old object-array archives."""
    sidecar = metadata_path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model archive not found: {model_path}")
    if not sidecar.is_file():
        raise ValueError(f"Model metadata sidecar missing: {sidecar.name}. Re-export a safe model.")
    if model_path.stat().st_size > _MAX_ARCHIVE_BYTES or sidecar.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError("Model archive or metadata exceeds the size limit.")
    metadata = _validate_metadata(json.loads(sidecar.read_text(encoding="utf-8"),
                                             object_pairs_hook=_no_duplicate_keys))
    if _sha256(model_path) != metadata["archiveSha256"]:
        raise ValueError("Model archive SHA-256 does not match metadata.")
    try:
        with zipfile.ZipFile(model_path) as archive:
            members = archive.infolist()
            expected = {name + ".npy" for name in _ARRAY_DTYPES}
            if len(members) != len(expected) or {item.filename for item in members} != expected:
                raise ValueError("Model NPZ members are unsupported; legacy object arrays are refused.")
            if sum(item.file_size for item in members) > _MAX_ARRAY_BYTES:
                raise ValueError("Model NPZ arrays exceed the size limit.")
        with np.load(model_path, allow_pickle=False) as raw:
            if set(raw.files) != set(_ARRAY_DTYPES):
                raise ValueError("Model NPZ arrays are unsupported.")
            arrays = {name: raw[name] for name in _ARRAY_DTYPES}
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Model "):
            raise
        raise ValueError("Model NPZ is invalid or uses object arrays; pickle loading is refused.") from exc
    _validate_arrays(arrays, metadata)
    offsets = arrays["train_item_offsets"].tolist()
    indices = arrays["train_item_indices"].tolist()
    groups = [set(indices[start:end]) for start, end in zip(offsets, offsets[1:])]
    return NumericModel(
        p=arrays["P"], q=arrays["Q"], bu=arrays["bu"], bi=arrays["bi"],
        global_mean=float(arrays["global_mean"][0]),
        user_ids=list(metadata["userIds"]),
        anime_ids=[int(item) for item in arrays["anime_ids"].tolist()],
        anime_titles=list(metadata["animeTitles"]),
        train_user_items=groups,
        archive_sha256=str(metadata["archiveSha256"]),
    )
