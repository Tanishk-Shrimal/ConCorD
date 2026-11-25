"""Validation-tool plumbing for Chroma spatial LapH reference fields."""

from __future__ import annotations

import struct

import h5py
import numpy as np
import pytest
import torch

from concord_lqcd.compiler.physics_ir import OperationKind, OperationRecord
from concord_lqcd.job.data import _apply_laph_operations
from tools.validate_spatial_laph_reference import _metrics, compare


def _write_gauge(path, values: np.ndarray) -> None:
    byte_order = "<"
    directions, lt, lz, ly, lx, _, _ = values.shape
    record = bytearray(struct.pack(f"{byte_order}i", directions))
    for direction in range(directions):
        record.extend(
            struct.pack(
                f"{byte_order}6I",
                4,
                lx,
                ly,
                lz,
                lt,
                values.dtype.itemsize * 9,
            )
        )
        record.extend(values[direction].tobytes(order="C"))

    file_id = b"Laph--SmearedGaugeField4D".ljust(32, b" ")
    header = b"<SmearedGaugeField/>"
    body = bytearray(struct.pack(f"{byte_order}Q", 0))
    body.extend(b"N")
    body.extend(struct.pack(f"{byte_order}i", len(header)))
    body.extend(header)
    record_position = len(body)
    body.extend(struct.pack(f"{byte_order}Q", len(record)))
    body.extend(record)
    map_start = len(body)
    body.extend(struct.pack(f"{byte_order}QQ", 1, 4))
    body.extend(struct.pack(f"{byte_order}I", 0))
    body.extend(struct.pack(f"{byte_order}Q", record_position))
    body.extend(b"E")
    struct.pack_into(f"{byte_order}Q", body, 0, map_start)
    path.write_bytes(b"L" + file_id + body)


def _reference_values(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    level_first = np.moveaxis(array, -1, 0)
    return np.stack((level_first.real, level_first.imag), axis=-1)


def _elemental_values(source: torch.Tensor, transformed: torch.Tensor) -> np.ndarray:
    source_modes = source.detach().cpu().numpy().reshape(-1, source.shape[-1])
    transformed_modes = transformed.detach().cpu().numpy().reshape(
        -1, transformed.shape[-1]
    )
    elemental = source_modes.conj().T @ transformed_modes
    return np.stack((elemental.real, elemental.imag), axis=-1)


@pytest.mark.parametrize(
    "reference_format",
    (
        "chroma_laph_spatial_reference_v1",
        "chroma_laph_spatial_reference_v2",
    ),
)
def test_reference_comparator_accepts_matching_transport_and_derivative(
    tmp_path,
    reference_format,
):
    generator = np.random.default_rng(20260914)
    lt, lz, ly, lx, colors, modes = 2, 2, 2, 3, 3, 3
    eigenvectors = generator.normal(size=(lt, lz, ly, lx, colors, modes))
    eigenvectors = eigenvectors + 1j * generator.normal(size=eigenvectors.shape)
    eigenvectors = eigenvectors.astype(np.complex128)
    eigenvector_path = tmp_path / "eigenvectors.h5"
    with h5py.File(eigenvector_path, "w") as handle:
        dataset = handle.create_dataset("eigenvectors", data=eigenvectors)
        dataset.attrs["axes"] = '["time", "z", "y", "x", "color", "mode"]'

    gauge_values = generator.normal(size=(3, lt, lz, ly, lx, 3, 3))
    gauge_values = gauge_values + 1j * generator.normal(size=gauge_values.shape)
    gauge_values = gauge_values.astype(np.dtype("<c16"))
    gauge_path = tmp_path / "smeared_gauge"
    _write_gauge(gauge_path, gauge_values)

    levels = np.asarray([0, 2], dtype=np.int32)
    source = torch.as_tensor(
        np.take(eigenvectors[0], levels, axis=-1),
        dtype=torch.complex128,
    )
    links = {
        axis: torch.as_tensor(gauge_values[index, 0], dtype=torch.complex128).unsqueeze(0)
        for index, axis in enumerate(("x", "y", "z"))
    }
    batched_source = source.unsqueeze(0)
    transport = _apply_laph_operations(
        batched_source,
        (OperationRecord(OperationKind.TRANSPORT, {"path": ("+x", "+y")}),),
        links.__getitem__,
    )[0]
    derivative = _apply_laph_operations(
        batched_source,
        (OperationRecord(OperationKind.DERIVATIVE, {"direction": "+z"}),),
        links.__getitem__,
    )[0]

    reference_path = tmp_path / "reference.h5"
    with h5py.File(reference_path, "w") as handle:
        handle.attrs["format"] = reference_format
        handle.attrs["time_slice"] = 0
        compact = reference_format.endswith("_v2")
        if compact:
            handle.attrs["representation"] = "elementals"
        handle.create_dataset("levels", data=levels)
        source_dataset = handle.create_dataset(
            "source",
            data=(
                _elemental_values(source, source)
                if compact
                else _reference_values(source)
            ),
        )
        source_dataset.attrs["dimension_order"] = (
            "bra_level,ket_level,complex_pair"
            if compact
            else "level,z,y,x,color,complex_pair"
        )
        transport_group = handle.create_group("transport")
        transport_dataset = transport_group.create_dataset(
            "plus_x_plus_y",
            data=(
                _elemental_values(source, transport)
                if compact
                else _reference_values(transport)
            ),
        )
        transport_dataset.attrs["dimension_order"] = (
            "bra_level,ket_level,complex_pair"
            if compact
            else "level,z,y,x,color,complex_pair"
        )
        transport_dataset.attrs["path"] = "+x,+y"
        derivative_group = handle.create_group("derivative")
        derivative_dataset = derivative_group.create_dataset(
            "z",
            data=(
                _elemental_values(source, derivative)
                if compact
                else _reference_values(derivative)
            ),
        )
        derivative_dataset.attrs["dimension_order"] = (
            "bra_level,ket_level,complex_pair"
            if compact
            else "level,z,y,x,color,complex_pair"
        )
        derivative_dataset.attrs["direction"] = "+z"

    report = compare(
        reference_path=reference_path,
        eigenvector_path=eigenvector_path,
        eigenvector_dataset="/eigenvectors",
        gauge_path=gauge_path,
        device="cpu",
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    assert report["passed"] is True
    assert set(report["comparisons"]) == {
        "source",
        "transport/plus_x_plus_y",
        "derivative/z",
    }
    assert report["representation"] == ("elementals" if compact else "fields")
    assert all(
        comparison["max_tolerance_ratio"] <= 1.0
        for comparison in report["comparisons"].values()
    )


def test_reference_metrics_report_finite_tolerance_ratio_near_zero():
    result = _metrics(
        np.asarray([1.0e-15 + 0.0j]),
        np.asarray([0.0 + 0.0j]),
        rtol=5.0e-13,
        atol=5.0e-14,
    )

    assert result["passed"] is True
    assert result["max_tolerance_ratio"] == pytest.approx(0.02)
    assert result["max_relative"] > 1.0e290
