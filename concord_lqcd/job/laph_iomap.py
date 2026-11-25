"""Read native Chroma-LapH and QUDA-LapH smeared-gauge IOMap files.

The LapH applications serialize gauge links in lexicographic ``x, y, z, t``
site order inside their native IOMap container.  Chroma-LapH may store either
three spatial links in one four-dimensional record or three links in one
record per timeslice.  QUDA-LapH stores one four-dimensional record containing
all four directions.  This reader understands those layouts without requiring
QDP++ or QUDA at ConCorD run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Literal

import numpy as np


_IO_HEADER_BYTES = 33
_FILE_ID_4D = "Laph--SmearedGaugeField4D"
_FILE_ID_TIMESLICES = "Laph--SmearedGaugeField"
_DIRECTION_NAMES = ("x", "y", "z", "t")
_COLOR_MATRIX_COMPLEX_VALUES = 9


class LaphGaugeIOMapError(ValueError):
    """Raised when a native LapH smeared-gauge file is malformed or unsupported."""


@dataclass(frozen=True)
class LaphGaugeIOMapMetadata:
    """Validated storage metadata for one native smeared-gauge file."""

    path: Path
    file_id: str
    storage: Literal["four_dimensional", "time_sliced"]
    byte_order: Literal["little", "big"]
    checksums_present: bool
    header_xml: str
    lattice_extent: tuple[int, int, int, int]
    directions: tuple[str, ...]
    scalar_type: Literal["complex64", "complex128"]


@dataclass(frozen=True)
class _GaugeStorage:
    metadata: LaphGaugeIOMapMetadata
    dtype: np.dtype
    slice_offsets: dict[tuple[int, int], int]


def _read_exact(handle, size: int, *, role: str) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise LaphGaugeIOMapError(f"truncated LapH IOMap while reading {role}")
    return value


def _unpack(handle, format_spec: str, *, role: str):
    size = struct.calcsize(format_spec)
    return struct.unpack(format_spec, _read_exact(handle, size, role=role))


def _validate_record_bounds(
    *,
    position: int,
    record_size: int,
    map_start: int,
    checksums_present: bool,
) -> None:
    checksum_bytes = 4 if checksums_present else 0
    record_end = position + 8 + record_size + checksum_bytes
    if position < 0 or record_size < 0 or record_end > map_start:
        raise LaphGaugeIOMapError("LapH IOMap record lies outside the data region")


def _complex_dtype(byte_order: str, bytes_per_site: int) -> tuple[np.dtype, str]:
    if bytes_per_site == _COLOR_MATRIX_COMPLEX_VALUES * 8:
        return np.dtype(f"{byte_order}c8"), "complex64"
    if bytes_per_site == _COLOR_MATRIX_COMPLEX_VALUES * 16:
        return np.dtype(f"{byte_order}c16"), "complex128"
    raise LaphGaugeIOMapError(
        "smeared gauge field must store a 3x3 complex color matrix per site; "
        f"got {bytes_per_site} bytes per site"
    )


def _read_iomap_header(path: Path):
    file_size = path.stat().st_size
    if file_size < _IO_HEADER_BYTES + 8 + 1 + 4:
        raise LaphGaugeIOMapError("file is too small to be a LapH IOMap")

    handle = path.open("rb")
    try:
        endian_marker = _read_exact(handle, 1, role="endianness marker")
        if endian_marker == b"L":
            byte_order = "<"
            byte_order_name = "little"
        elif endian_marker == b"B":
            byte_order = ">"
            byte_order_name = "big"
        else:
            raise LaphGaugeIOMapError(
                f"invalid LapH IOMap endianness marker {endian_marker!r}"
            )

        raw_file_id = _read_exact(handle, 32, role="file identifier")
        try:
            file_id = raw_file_id.decode("ascii").rstrip(" ")
        except UnicodeDecodeError as exc:
            raise LaphGaugeIOMapError("LapH IOMap file identifier is not ASCII") from exc
        if file_id not in {_FILE_ID_4D, _FILE_ID_TIMESLICES}:
            raise LaphGaugeIOMapError(
                f"unsupported LapH IOMap file identifier {file_id!r}"
            )

        (relative_map_start,) = _unpack(
            handle,
            f"{byte_order}Q",
            role="file-map location",
        )
        map_start = _IO_HEADER_BYTES + relative_map_start
        if map_start <= handle.tell() or map_start >= file_size:
            raise LaphGaugeIOMapError("invalid LapH IOMap file-map location")

        checksum_marker = _read_exact(handle, 1, role="checksum marker")
        if checksum_marker not in {b"N", b"Y"}:
            raise LaphGaugeIOMapError(
                f"invalid LapH IOMap checksum marker {checksum_marker!r}"
            )
        checksums_present = checksum_marker == b"Y"

        (header_size,) = _unpack(
            handle,
            f"{byte_order}i",
            role="XML header size",
        )
        if header_size < 0 or handle.tell() + header_size > map_start:
            raise LaphGaugeIOMapError("invalid LapH IOMap XML header size")
        raw_header = _read_exact(handle, header_size, role="XML header")
        try:
            header_xml = raw_header.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LaphGaugeIOMapError("LapH IOMap XML header is not UTF-8") from exc

        handle.seek(map_start)
        nrecords, key_size = _unpack(
            handle,
            f"{byte_order}QQ",
            role="file-map header",
        )
        if nrecords <= 0:
            raise LaphGaugeIOMapError("LapH smeared-gauge IOMap contains no records")
        if nrecords > 1_000_000:
            raise LaphGaugeIOMapError("LapH IOMap contains an unreasonable record count")
        if key_size != 4:
            raise LaphGaugeIOMapError(
                f"LapH smeared-gauge IOMap requires four-byte integer keys, got {key_size}"
            )
        keys = _unpack(
            handle,
            f"{byte_order}{nrecords}I",
            role="file-map keys",
        )
        positions = _unpack(
            handle,
            f"{byte_order}{nrecords}Q",
            role="file-map record positions",
        )
        if checksums_present:
            _read_exact(handle, 4, role="file-map checksum")
        if _read_exact(handle, 1, role="file-map ending marker") != b"E":
            raise LaphGaugeIOMapError("invalid LapH IOMap file-map ending marker")
        absolute_positions = tuple(_IO_HEADER_BYTES + value for value in positions)
        if len(set(keys)) != len(keys):
            raise LaphGaugeIOMapError("LapH IOMap contains duplicate record keys")
        if len(set(absolute_positions)) != len(absolute_positions):
            raise LaphGaugeIOMapError("LapH IOMap contains duplicate record positions")
        return (
            handle,
            file_id,
            byte_order,
            byte_order_name,
            checksums_present,
            header_xml,
            map_start,
            dict(zip(keys, absolute_positions, strict=True)),
        )
    except Exception:
        handle.close()
        raise


def _parse_four_dimensional(path: Path) -> _GaugeStorage:
    (
        handle,
        file_id,
        byte_order,
        byte_order_name,
        checksums_present,
        header_xml,
        map_start,
        records,
    ) = _read_iomap_header(path)
    try:
        if set(records) != {0}:
            raise LaphGaugeIOMapError(
                "four-dimensional LapH smeared-gauge IOMap must contain key zero"
            )
        position = records[0]
        handle.seek(position)
        (record_size,) = _unpack(handle, f"{byte_order}Q", role="record size")
        _validate_record_bounds(
            position=position,
            record_size=record_size,
            map_start=map_start,
            checksums_present=checksums_present,
        )
        value_start = handle.tell()
        (direction_count,) = _unpack(
            handle,
            f"{byte_order}i",
            role="gauge direction count",
        )
        if direction_count not in {3, 4}:
            raise LaphGaugeIOMapError(
                f"expected three or four gauge directions, got {direction_count}"
            )

        reference_extent: tuple[int, int, int, int] | None = None
        reference_bytes_per_site: int | None = None
        slice_offsets: dict[tuple[int, int], int] = {}
        for direction in range(direction_count):
            ndim, lx, ly, lz, lt, bytes_per_site = _unpack(
                handle,
                f"{byte_order}6I",
                role=f"direction {direction} lattice header",
            )
            if ndim != 4 or min(lx, ly, lz, lt) <= 0:
                raise LaphGaugeIOMapError(
                    f"invalid four-dimensional lattice header for direction {direction}"
                )
            extent = (lx, ly, lz, lt)
            if reference_extent is None:
                reference_extent = extent
                reference_bytes_per_site = bytes_per_site
            elif (
                extent != reference_extent
                or bytes_per_site != reference_bytes_per_site
            ):
                raise LaphGaugeIOMapError(
                    "gauge directions use inconsistent lattice extents or precision"
                )
            data_start = handle.tell()
            spatial_sites = lx * ly * lz
            for time in range(lt):
                slice_offsets[(time, direction)] = (
                    data_start + time * spatial_sites * bytes_per_site
                )
            handle.seek(spatial_sites * lt * bytes_per_site, 1)

        if handle.tell() - value_start != record_size:
            raise LaphGaugeIOMapError(
                "four-dimensional smeared-gauge record size does not match its payload"
            )
        assert reference_extent is not None
        assert reference_bytes_per_site is not None
        dtype, scalar_type = _complex_dtype(
            byte_order,
            reference_bytes_per_site,
        )
        metadata = LaphGaugeIOMapMetadata(
            path=path,
            file_id=file_id,
            storage="four_dimensional",
            byte_order=byte_order_name,
            checksums_present=checksums_present,
            header_xml=header_xml,
            lattice_extent=reference_extent,
            directions=_DIRECTION_NAMES[:direction_count],
            scalar_type=scalar_type,
        )
        return _GaugeStorage(metadata, dtype, slice_offsets)
    finally:
        handle.close()


def _parse_time_sliced(path: Path) -> _GaugeStorage:
    (
        handle,
        file_id,
        byte_order,
        byte_order_name,
        checksums_present,
        header_xml,
        map_start,
        records,
    ) = _read_iomap_header(path)
    try:
        times = tuple(sorted(records))
        if times != tuple(range(len(times))):
            raise LaphGaugeIOMapError(
                "time-sliced LapH smeared-gauge records must be contiguous from zero"
            )

        reference_spatial_extent: tuple[int, int, int] | None = None
        reference_bytes_per_site: int | None = None
        direction_count: int | None = None
        slice_offsets: dict[tuple[int, int], int] = {}
        for time in times:
            position = records[time]
            handle.seek(position)
            (record_size,) = _unpack(handle, f"{byte_order}Q", role="record size")
            _validate_record_bounds(
                position=position,
                record_size=record_size,
                map_start=map_start,
                checksums_present=checksums_present,
            )
            value_start = handle.tell()
            (current_direction_count,) = _unpack(
                handle,
                f"{byte_order}i",
                role="gauge direction count",
            )
            if current_direction_count != 3:
                raise LaphGaugeIOMapError(
                    "time-sliced LapH smeared-gauge files must contain three "
                    "spatial directions"
                )
            if direction_count is None:
                direction_count = current_direction_count
            elif current_direction_count != direction_count:
                raise LaphGaugeIOMapError(
                    "time-sliced records use inconsistent direction counts"
                )

            for direction in range(current_direction_count):
                ndim, lx, ly, lz, bytes_per_site = _unpack(
                    handle,
                    f"{byte_order}5I",
                    role=f"timeslice {time} direction {direction} lattice header",
                )
                if ndim != 3 or min(lx, ly, lz) <= 0:
                    raise LaphGaugeIOMapError(
                        f"invalid spatial lattice header at time {time}, "
                        f"direction {direction}"
                    )
                spatial_extent = (lx, ly, lz)
                if reference_spatial_extent is None:
                    reference_spatial_extent = spatial_extent
                    reference_bytes_per_site = bytes_per_site
                elif (
                    spatial_extent != reference_spatial_extent
                    or bytes_per_site != reference_bytes_per_site
                ):
                    raise LaphGaugeIOMapError(
                        "time-sliced gauge records use inconsistent extents or precision"
                    )
                slice_offsets[(time, direction)] = handle.tell()
                handle.seek(lx * ly * lz * bytes_per_site, 1)

            if handle.tell() - value_start != record_size:
                raise LaphGaugeIOMapError(
                    f"smeared-gauge record {time} size does not match its payload"
                )

        assert reference_spatial_extent is not None
        assert reference_bytes_per_site is not None
        assert direction_count is not None
        dtype, scalar_type = _complex_dtype(
            byte_order,
            reference_bytes_per_site,
        )
        metadata = LaphGaugeIOMapMetadata(
            path=path,
            file_id=file_id,
            storage="time_sliced",
            byte_order=byte_order_name,
            checksums_present=checksums_present,
            header_xml=header_xml,
            lattice_extent=(*reference_spatial_extent, len(times)),
            directions=_DIRECTION_NAMES[:direction_count],
            scalar_type=scalar_type,
        )
        return _GaugeStorage(metadata, dtype, slice_offsets)
    finally:
        handle.close()


class LaphGaugeIOMapReader:
    """Validated random-access reader for native LapH smeared gauge links."""

    def __init__(self, path: str | Path):
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise LaphGaugeIOMapError(f"LapH smeared-gauge file does not exist: {resolved}")
        with resolved.open("rb") as handle:
            marker_and_id = _read_exact(handle, 33, role="file identifier")
        try:
            file_id = marker_and_id[1:].decode("ascii").rstrip(" ")
        except UnicodeDecodeError as exc:
            raise LaphGaugeIOMapError("LapH IOMap file identifier is not ASCII") from exc
        if file_id == _FILE_ID_4D:
            storage = _parse_four_dimensional(resolved)
        elif file_id == _FILE_ID_TIMESLICES:
            storage = _parse_time_sliced(resolved)
        else:
            raise LaphGaugeIOMapError(
                f"unsupported LapH IOMap file identifier {file_id!r}"
            )
        self._storage = storage

    @property
    def metadata(self) -> LaphGaugeIOMapMetadata:
        return self._storage.metadata

    def _direction_index(self, direction: str | int) -> int:
        if isinstance(direction, bool):
            raise TypeError("gauge direction must be a name or integer index")
        if isinstance(direction, int):
            index = direction
        elif isinstance(direction, str):
            normalized = direction.strip().lower()
            try:
                index = self.metadata.directions.index(normalized)
            except ValueError as exc:
                raise LaphGaugeIOMapError(
                    f"gauge direction {direction!r} is unavailable; file contains "
                    f"{self.metadata.directions}"
                ) from exc
        else:
            raise TypeError("gauge direction must be a name or integer index")
        if index < 0 or index >= len(self.metadata.directions):
            raise LaphGaugeIOMapError(
                f"gauge direction index {index} is outside the stored range"
            )
        return index

    def read_timeslice(
        self,
        direction: str | int,
        time: int,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        """Read one link direction at one Euclidean time.

        The returned shape is ``[z, y, x, color_row, color_column]``.  With
        ``copy=False`` the result is a read-only memory map backed by the native
        file; callers must not mutate it.
        """
        if not isinstance(time, int) or isinstance(time, bool):
            raise TypeError("gauge-link time must be an integer")
        lx, ly, lz, lt = self.metadata.lattice_extent
        if time < 0 or time >= lt:
            raise LaphGaugeIOMapError(
                f"gauge-link time {time} is outside the lattice range [0, {lt})"
            )
        direction_index = self._direction_index(direction)
        offset = self._storage.slice_offsets[(time, direction_index)]
        view = np.memmap(
            self.metadata.path,
            mode="r",
            dtype=self._storage.dtype,
            offset=offset,
            shape=(lz, ly, lx, 3, 3),
            order="C",
        )
        if copy:
            return np.array(view, copy=True)
        return view


__all__ = [
    "LaphGaugeIOMapError",
    "LaphGaugeIOMapMetadata",
    "LaphGaugeIOMapReader",
]
