"""Native Chroma-LapH and QUDA-LapH smeared-gauge reader tests."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from concord_lqcd.job.laph_iomap import LaphGaugeIOMapError, LaphGaugeIOMapReader


_BASE = 33


def _gauge_values(
    *,
    directions: int,
    extent: tuple[int, int, int, int],
    dtype: np.dtype,
) -> np.ndarray:
    lx, ly, lz, lt = extent
    count = directions * lt * lz * ly * lx * 9
    real = np.arange(count, dtype=np.float64).reshape(
        directions,
        lt,
        lz,
        ly,
        lx,
        3,
        3,
    )
    values = real + 1j * (real + 0.25)
    return values.astype(dtype)


def _write_iomap(
    path: Path,
    *,
    file_id: str,
    records: dict[int, bytes],
    byte_order: str,
    checksums: bool = False,
) -> None:
    marker = b"L" if byte_order == "<" else b"B"
    header = b"<SmearedGaugeField/>"
    prefix = marker + file_id.encode("ascii").ljust(32, b" ")
    data = bytearray(struct.pack(f"{byte_order}Q", 0))
    data.extend(b"Y" if checksums else b"N")
    data.extend(struct.pack(f"{byte_order}i", len(header)))
    data.extend(header)

    positions: list[int] = []
    for _, value in sorted(records.items()):
        positions.append(len(data))
        data.extend(struct.pack(f"{byte_order}Q", len(value)))
        data.extend(value)
        if checksums:
            data.extend(struct.pack(f"{byte_order}I", 0))

    map_start = len(data)
    ordered_keys = tuple(sorted(records))
    data.extend(struct.pack(f"{byte_order}QQ", len(records), 4))
    data.extend(struct.pack(f"{byte_order}{len(records)}I", *ordered_keys))
    data.extend(struct.pack(f"{byte_order}{len(records)}Q", *positions))
    if checksums:
        data.extend(struct.pack(f"{byte_order}I", 0))
    data.extend(b"E")
    struct.pack_into(f"{byte_order}Q", data, 0, map_start)
    path.write_bytes(prefix + data)


def _four_dimensional_record(
    values: np.ndarray,
    *,
    byte_order: str,
) -> bytes:
    directions, lt, lz, ly, lx, _, _ = values.shape
    bytes_per_site = values.dtype.itemsize * 9
    data = bytearray(struct.pack(f"{byte_order}i", directions))
    for direction in range(directions):
        data.extend(
            struct.pack(
                f"{byte_order}6I",
                4,
                lx,
                ly,
                lz,
                lt,
                bytes_per_site,
            )
        )
        data.extend(values[direction].tobytes(order="C"))
    return bytes(data)


def _time_sliced_records(
    values: np.ndarray,
    *,
    byte_order: str,
) -> dict[int, bytes]:
    directions, lt, lz, ly, lx, _, _ = values.shape
    bytes_per_site = values.dtype.itemsize * 9
    records: dict[int, bytes] = {}
    for time in range(lt):
        data = bytearray(struct.pack(f"{byte_order}i", directions))
        for direction in range(directions):
            data.extend(
                struct.pack(
                    f"{byte_order}5I",
                    3,
                    lx,
                    ly,
                    lz,
                    bytes_per_site,
                )
            )
            data.extend(values[direction, time].tobytes(order="C"))
        records[time] = bytes(data)
    return records


@pytest.mark.parametrize(
    ("byte_order", "scalar_type"),
    (("<", "complex128"), (">", "complex64")),
)
def test_reads_four_dimensional_chroma_and_quda_layouts(
    tmp_path,
    byte_order,
    scalar_type,
):
    dtype = np.dtype(f"{byte_order}{'c16' if scalar_type == 'complex128' else 'c8'}")
    extent = (3, 2, 2, 4)
    values = _gauge_values(directions=4, extent=extent, dtype=dtype)
    path = tmp_path / "smeared_gauge"
    _write_iomap(
        path,
        file_id="Laph--SmearedGaugeField4D",
        records={0: _four_dimensional_record(values, byte_order=byte_order)},
        byte_order=byte_order,
    )

    reader = LaphGaugeIOMapReader(path)

    assert reader.metadata.lattice_extent == extent
    assert reader.metadata.directions == ("x", "y", "z", "t")
    assert reader.metadata.scalar_type == scalar_type
    assert reader.metadata.storage == "four_dimensional"
    np.testing.assert_array_equal(reader.read_timeslice("y", 2), values[1, 2])
    mapped = reader.read_timeslice(3, 1, copy=False)
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, values[3, 1])


def test_reads_time_sliced_chroma_layout(tmp_path):
    byte_order = "<"
    dtype = np.dtype("<c16")
    extent = (3, 2, 2, 4)
    values = _gauge_values(directions=3, extent=extent, dtype=dtype)
    path = tmp_path / "smeared_gauge_timeslices"
    _write_iomap(
        path,
        file_id="Laph--SmearedGaugeField",
        records=_time_sliced_records(values, byte_order=byte_order),
        byte_order=byte_order,
    )

    reader = LaphGaugeIOMapReader(path)

    assert reader.metadata.lattice_extent == extent
    assert reader.metadata.directions == ("x", "y", "z")
    assert reader.metadata.storage == "time_sliced"
    np.testing.assert_array_equal(reader.read_timeslice("z", 3), values[2, 3])


def test_reads_checksum_bearing_iomap_layout(tmp_path):
    values = _gauge_values(
        directions=3,
        extent=(2, 2, 2, 2),
        dtype=np.dtype("<c16"),
    )
    path = tmp_path / "checksummed_gauge"
    _write_iomap(
        path,
        file_id="Laph--SmearedGaugeField4D",
        records={0: _four_dimensional_record(values, byte_order="<")},
        byte_order="<",
        checksums=True,
    )

    reader = LaphGaugeIOMapReader(path)

    assert reader.metadata.checksums_present is True
    np.testing.assert_array_equal(reader.read_timeslice("x", 1), values[0, 1])


def test_rejects_missing_temporal_link_and_invalid_time(tmp_path):
    values = _gauge_values(
        directions=3,
        extent=(2, 2, 2, 2),
        dtype=np.dtype("<c16"),
    )
    path = tmp_path / "spatial_only"
    _write_iomap(
        path,
        file_id="Laph--SmearedGaugeField4D",
        records={0: _four_dimensional_record(values, byte_order="<")},
        byte_order="<",
    )
    reader = LaphGaugeIOMapReader(path)

    with pytest.raises(LaphGaugeIOMapError, match="unavailable"):
        reader.read_timeslice("t", 0)
    with pytest.raises(LaphGaugeIOMapError, match="outside"):
        reader.read_timeslice("x", 2)


def test_rejects_unrelated_native_file(tmp_path):
    path = tmp_path / "wrong_file"
    path.write_bytes(b"L" + b"not-a-laph-gauge-file".ljust(32, b" ") + b"x" * 32)

    with pytest.raises(LaphGaugeIOMapError, match="unsupported"):
        LaphGaugeIOMapReader(path)


def test_rejects_invalid_file_map_ending_marker(tmp_path):
    values = _gauge_values(
        directions=3,
        extent=(2, 2, 2, 2),
        dtype=np.dtype("<c16"),
    )
    path = tmp_path / "bad_footer"
    _write_iomap(
        path,
        file_id="Laph--SmearedGaugeField4D",
        records={0: _four_dimensional_record(values, byte_order="<")},
        byte_order="<",
    )
    payload = bytearray(path.read_bytes())
    payload[-1] = ord("X")
    path.write_bytes(payload)

    with pytest.raises(LaphGaugeIOMapError, match="ending marker"):
        LaphGaugeIOMapReader(path)
