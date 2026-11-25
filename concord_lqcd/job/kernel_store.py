"""Persistent, provenance-checked storage for materialized LapH kernels."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass
from threading import Lock
import uuid
import warnings

import h5py
import numpy as np


_ANTISYM3_BLOCK_ENTRIES = 4096


def _antisym3_blocks(mode_size: int):
    block_size = min(_ANTISYM3_BLOCK_ENTRIES, math.comb(mode_size, 3))
    indices = np.empty((3, block_size), dtype=np.intp)
    filled = 0
    for first in range(mode_size - 2):
        for second in range(first + 1, mode_size - 1):
            third = second + 1
            while third < mode_size:
                count = min(mode_size - third, block_size - filled)
                stop = filled + count
                indices[0, filled:stop] = first
                indices[1, filled:stop] = second
                indices[2, filled:stop] = np.arange(third, third + count)
                third += count
                filled = stop
                if filled == block_size:
                    yield indices[0], indices[1], indices[2]
                    filled = 0
    if filled:
        yield indices[0, :filled], indices[1, :filled], indices[2, :filled]


def _antisym3_workspace_bytes(mode_size: int, dtype: np.dtype) -> int:
    element_bytes = np.dtype(dtype).itemsize
    packed_bytes = math.comb(mode_size, 3) * element_bytes
    block_size = min(_ANTISYM3_BLOCK_ENTRIES, mode_size ** 3)
    return packed_bytes + block_size * (3 * np.dtype(np.intp).itemsize + 6 * element_bytes)


def _pack_antisym3(values: np.ndarray) -> np.ndarray | None:
    flat = values.reshape(-1)
    maximum = 0.0
    for start in range(0, flat.size, _ANTISYM3_BLOCK_ENTRIES):
        block = flat[start:start + _ANTISYM3_BLOCK_ENTRIES]
        if not np.isfinite(block).all():
            return None
        maximum = max(maximum, float(np.max(np.abs(block))))
        if not math.isfinite(maximum):
            return None
    tolerance = (1e-5 if values.dtype == np.dtype(np.complex64) else 1e-11) * maximum
    mode_size = values.shape[0]
    for repeated in range(mode_size):
        for start in range(0, mode_size, _ANTISYM3_BLOCK_ENTRIES):
            stop = start + _ANTISYM3_BLOCK_ENTRIES
            for block in (values[repeated, repeated, start:stop],
                          values[repeated, start:stop, repeated],
                          values[start:stop, repeated, repeated]):
                if float(np.max(np.abs(block))) > tolerance:
                    return None
    packed = np.empty(math.comb(mode_size, 3), dtype=values.dtype)
    offset = 0
    for first, second, third in _antisym3_blocks(mode_size):
        representative = values[first, second, third]
        for indices, sign in (((second, third, first), 1),
                              ((third, first, second), 1),
                              ((first, third, second), -1),
                              ((third, second, first), -1),
                              ((second, first, third), -1)):
            difference = values[indices] - sign * representative
            if float(np.max(np.abs(difference))) > tolerance:
                return None
        packed[offset:offset + representative.size] = representative
        offset += representative.size
    return packed


def _unpack_antisym3(packed: np.ndarray, mode_size: int) -> np.ndarray:
    result = np.zeros((mode_size, mode_size, mode_size), dtype=packed.dtype)
    offset = 0
    for first, second, third in _antisym3_blocks(mode_size):
        block = packed[offset:offset + first.size]
        result[first, second, third] = block
        result[second, third, first] = block
        result[third, first, second] = block
        negative = -block
        result[first, third, second] = negative
        result[third, second, first] = negative
        result[second, first, third] = negative
        offset += first.size
    return result


class KernelStoreError(RuntimeError):
    """Raised when a kernel cache source cannot be fingerprinted safely."""


@dataclass(frozen=True)
class KernelCacheStatistics:
    """Counts of persistent-cache lookups and successful or failed writes."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    invalid_entries: int = 0
    write_failures: int = 0


class HDF5KernelStore:
    """Store one validated, uncompressed HDF5 artifact per numerical kernel."""

    format_name = "concord-kernel-cache"
    format_version = 1

    def __init__(
        self, directory: str | Path, *, configuration: int, packing: str = "none"
    ) -> None:
        if not isinstance(configuration, int) or isinstance(configuration, bool):
            raise TypeError("kernel cache configuration must be an integer")
        if packing not in ("none", "antisym3"):
            raise ValueError("kernel cache packing must be 'none' or 'antisym3'")
        self.directory = Path(directory).expanduser() / f"cfg_{configuration}"
        self.configuration = configuration
        self.packing = packing
        self._source_fingerprints: dict[
            tuple[str, int, int, int, int, int], dict[str, object]
        ] = {}
        self._fingerprint_lock = Lock()
        self._statistics_lock = Lock()
        self._statistics = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "invalid_entries": 0,
            "write_failures": 0,
        }
        self._writes_disabled = False

    @property
    def statistics(self) -> KernelCacheStatistics:
        with self._statistics_lock:
            return KernelCacheStatistics(**self._statistics)

    def record_misses(self, count: int) -> None:
        """Record lookups bypassed by a dense batch materialization."""

        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("kernel cache miss count must be a non-negative integer")
        with self._statistics_lock:
            self._statistics["misses"] += count

    def _record(self, **increments: int) -> None:
        with self._statistics_lock:
            for name, count in increments.items():
                self._statistics[name] += count

    def source_fingerprint(self, path: str | Path) -> dict[str, object]:
        """Return a SHA-256 identity for the complete immutable source file."""

        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise KernelStoreError(f"kernel cache source is not a file: {source}")
        with self._fingerprint_lock:
            before = source.stat()
            signature = (
                str(source),
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            cached = self._source_fingerprints.get(signature)
            if cached is not None:
                return dict(cached)

            digest = hashlib.sha256()
            try:
                with source.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        digest.update(chunk)
            except OSError as exc:
                raise KernelStoreError(
                    f"cannot checksum kernel cache source {source}: {exc}"
                ) from exc

            after = source.stat()
            after_signature = (
                str(source),
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if after_signature != signature:
                raise KernelStoreError(
                    f"kernel cache source changed while being checksummed: {source}"
                )
            fingerprint: dict[str, object] = {
                "path": str(source),
                "size_bytes": before.st_size,
                "sha256": digest.hexdigest(),
            }
            self._source_fingerprints[signature] = fingerprint
            return dict(fingerprint)

    @staticmethod
    def canonical_manifest(manifest: dict[str, object]) -> str:
        return json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def entry_key(self, manifest_json: str) -> str:
        return hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()

    def entry_path(self, key: str) -> Path:
        if len(key) != 64 or any(value not in "0123456789abcdef" for value in key):
            raise ValueError("kernel cache key must be a lowercase SHA-256 digest")
        return self.directory / key[:2] / f"{key}.h5"

    def contains(self, key: str) -> bool:
        return self.entry_path(key).is_file()

    @staticmethod
    def _decode_attribute(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _array_checksum(array: np.ndarray) -> str:
        contiguous = np.asarray(array)
        if not contiguous.flags.c_contiguous:
            contiguous = np.ascontiguousarray(contiguous)
        return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()

    def load_array(
        self,
        key: str,
        manifest_json: str,
        *,
        expected_shape: tuple[int, ...],
        expected_dtype: np.dtype,
        expected_axes: tuple[str, ...],
        expected_scalar_type: str,
    ) -> np.ndarray | None:
        path = self.entry_path(key)
        if not path.is_file():
            self._record(misses=1)
            return None
        try:
            with h5py.File(path, "r") as handle:
                if self._decode_attribute(handle.attrs["format"]) != self.format_name:
                    return self._invalid_entry(path, "format identifier mismatch")
                if int(handle.attrs["format_version"]) != self.format_version:
                    return self._invalid_entry(path, "format version mismatch")
                if self._decode_attribute(handle.attrs["key"]) != key:
                    return self._invalid_entry(path, "entry key mismatch")
                if self._decode_attribute(handle.attrs["manifest_json"]) != manifest_json:
                    return self._invalid_entry(path, "provenance manifest mismatch")
                dataset = handle["values"]
                storage_packing = self._decode_attribute(dataset.attrs.get("storage_packing", "none"))
                if storage_packing == "none":
                    if tuple(dataset.shape) != expected_shape:
                        return self._invalid_entry(path, "tensor shape mismatch")
                elif storage_packing == "antisym3":
                    if self.packing != "antisym3":
                        return self._invalid_entry(path, "packed storage is not enabled")
                    if (len(expected_shape) != 3 or len(set(expected_shape)) != 1
                            or expected_shape[0] < 3
                            or tuple(dataset.shape) != (math.comb(expected_shape[0], 3),)
                            or np.dtype(expected_dtype) not in (np.dtype(np.complex64), np.dtype(np.complex128))):
                        return self._invalid_entry(path, "packed tensor shape or dtype mismatch")
                else:
                    return self._invalid_entry(path, "unsupported storage packing")
                if np.dtype(dataset.dtype) != np.dtype(expected_dtype):
                    return self._invalid_entry(path, "tensor dtype mismatch")
                if json.loads(
                    self._decode_attribute(dataset.attrs["shape_json"])
                ) != list(expected_shape):
                    return self._invalid_entry(path, "recorded tensor shape mismatch")
                if json.loads(
                    self._decode_attribute(dataset.attrs["axes_json"])
                ) != list(expected_axes):
                    return self._invalid_entry(path, "tensor axis metadata mismatch")
                if (
                    self._decode_attribute(dataset.attrs["scalar_type"])
                    != expected_scalar_type
                ):
                    return self._invalid_entry(path, "recorded scalar type mismatch")
                array = np.asarray(dataset[...])
                expected_checksum = self._decode_attribute(
                    dataset.attrs["sha256"]
                )
                if self._array_checksum(array) != expected_checksum:
                    return self._invalid_entry(path, "tensor checksum mismatch")
                if storage_packing == "antisym3":
                    for start in range(0, array.size, _ANTISYM3_BLOCK_ENTRIES):
                        if not np.isfinite(array[start:start + _ANTISYM3_BLOCK_ENTRIES]).all():
                            return self._invalid_entry(path, "nonfinite packed tensor")
                    array = _unpack_antisym3(array, expected_shape[0])
                self._record(hits=1)
                return array if array.flags.c_contiguous else np.ascontiguousarray(array)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self._record(misses=1, invalid_entries=1)
            warnings.warn(
                f"ignoring unreadable ConCorD kernel cache entry {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _invalid_entry(self, path: Path, reason: str) -> None:
        self._record(misses=1, invalid_entries=1)
        warnings.warn(
            f"ignoring invalid ConCorD kernel cache entry {path}: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def store_array(
        self,
        key: str,
        manifest_json: str,
        array: np.ndarray,
        *,
        axes: tuple[str, ...],
        allow_antisym3: bool = False,
    ) -> bool:
        values = np.asarray(array)
        if not values.flags.c_contiguous:
            values = np.ascontiguousarray(values)
        if self._writes_disabled:
            self._record(write_failures=1)
            return False
        storage_packing = "none"
        stored_values = values
        if (self.packing == "antisym3" and allow_antisym3
                and values.ndim == 3 and len(set(values.shape)) == 1
                and values.shape[0] >= 3
                and values.dtype in (np.dtype(np.complex64), np.dtype(np.complex128))):
            packed = _pack_antisym3(values)
            if packed is not None:
                stored_values = packed
                storage_packing = "antisym3"
        path = self.entry_path(key)
        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{key}.",
                suffix=f".{uuid.uuid4().hex}.partial.h5",
                dir=path.parent,
            )
            os.close(descriptor)
            temporary = Path(name)
            with h5py.File(temporary, "w") as handle:
                handle.attrs["format"] = self.format_name
                handle.attrs["format_version"] = self.format_version
                handle.attrs["key"] = key
                handle.attrs["manifest_json"] = manifest_json
                dataset = handle.create_dataset(
                    "values",
                    data=stored_values,
                    compression=None,
                )
                dataset.attrs["axes_json"] = json.dumps(axes)
                dataset.attrs["shape_json"] = json.dumps(values.shape)
                dataset.attrs["storage_packing"] = storage_packing
                dataset.attrs["scalar_type"] = (
                    "complex64"
                    if values.dtype == np.dtype(np.complex64)
                    else "complex128"
                    if values.dtype == np.dtype(np.complex128)
                    else str(values.dtype)
                )
                dataset.attrs["sha256"] = self._array_checksum(stored_values)
                handle.flush()
            file_descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
            os.replace(temporary, path)
            temporary = None
            self._record(writes=1)
            return True
        except OSError as exc:
            self._writes_disabled = True
            self._record(write_failures=1)
            warnings.warn(
                f"could not persist ConCorD kernel cache entry {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = ["HDF5KernelStore", "KernelCacheStatistics", "KernelStoreError"]
