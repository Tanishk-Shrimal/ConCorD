#!/usr/bin/env python3
"""Compare ConCorD spatial LapH operations with a Chroma reference file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from concord_lqcd.compiler.physics_ir import OperationKind, OperationRecord
from concord_lqcd.job.data import _apply_laph_operations
from concord_lqcd.job.laph_iomap import LaphGaugeIOMapReader


EXPECTED_EIGENVECTOR_AXES = ("time", "z", "y", "x", "color", "mode")
FIELD_REFERENCE_FORMAT = "chroma_laph_spatial_reference_v1"
ELEMENTAL_REFERENCE_FORMAT = "chroma_laph_spatial_reference_v2"
FIELD_REFERENCE_ORDER = "level,z,y,x,color,complex_pair"
ELEMENTAL_REFERENCE_ORDER = "bra_level,ket_level,complex_pair"


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _axes(dataset: h5py.Dataset) -> tuple[str, ...]:
    try:
        value = dataset.attrs["axes"]
    except KeyError as exc:
        raise ValueError(f"{dataset.name} has no required axes attribute") from exc
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        axes = tuple(_text(item) for item in value)
    else:
        raw = _text(value)
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = [item.strip() for item in raw.split(",")]
        axes = tuple(_text(item) for item in decoded)
    if axes != EXPECTED_EIGENVECTOR_AXES:
        raise ValueError(
            f"{dataset.name} axes are {axes}, expected {EXPECTED_EIGENVECTOR_AXES}"
        )
    return axes


def _complex_field(dataset: h5py.Dataset) -> np.ndarray:
    order = _text(dataset.attrs.get("dimension_order", ""))
    if order != FIELD_REFERENCE_ORDER:
        raise ValueError(
            f"{dataset.name} dimension_order is {order!r}, "
            f"expected {FIELD_REFERENCE_ORDER!r}"
        )
    values = np.asarray(dataset)
    if values.ndim != 6 or values.shape[-1] != 2:
        raise ValueError(
            f"{dataset.name} must have shape [level,z,y,x,color,2], got {values.shape}"
        )
    return values[..., 0] + 1j * values[..., 1]


def _complex_elemental(dataset: h5py.Dataset) -> np.ndarray:
    order = _text(dataset.attrs.get("dimension_order", ""))
    if order != ELEMENTAL_REFERENCE_ORDER:
        raise ValueError(
            f"{dataset.name} dimension_order is {order!r}, "
            f"expected {ELEMENTAL_REFERENCE_ORDER!r}"
        )
    values = np.asarray(dataset)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(
            f"{dataset.name} must have shape [bra_level,ket_level,2], "
            f"got {values.shape}"
        )
    return values[..., 0] + 1j * values[..., 1]


def _load_reference(
    path: Path,
) -> tuple[
    int,
    np.ndarray,
    str,
    dict[str, tuple[OperationRecord | None, np.ndarray]],
]:
    with h5py.File(path, "r") as handle:
        file_format = _text(handle.attrs.get("format", ""))
        if file_format == FIELD_REFERENCE_FORMAT:
            representation = "fields"
            decoder = _complex_field
        elif file_format == ELEMENTAL_REFERENCE_FORMAT:
            representation = _text(handle.attrs.get("representation", ""))
            if representation != "elementals":
                raise ValueError(
                    "v2 spatial reference representation must be 'elementals'"
                )
            decoder = _complex_elemental
        else:
            raise ValueError(f"unsupported spatial reference format {file_format!r}")
        time_slice = int(handle.attrs["time_slice"])
        levels = np.asarray(handle["levels"], dtype=np.int64)
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("reference levels must be a non-empty one-dimensional dataset")
        source = decoder(handle["source"])
        if source.shape[0] != levels.size:
            raise ValueError("reference source level axis disagrees with /levels")
        if representation == "elementals" and source.shape != (
            levels.size,
            levels.size,
        ):
            raise ValueError("reference source elemental axes disagree with /levels")

        operations: dict[str, tuple[OperationRecord | None, np.ndarray]] = {}
        for name, dataset in handle["transport"].items():
            path_value = tuple(
                item.strip()
                for item in _text(dataset.attrs["path"]).split(",")
                if item.strip()
            )
            operations[f"transport/{name}"] = (
                OperationRecord(OperationKind.TRANSPORT, {"path": path_value}),
                decoder(dataset),
            )
        for name, dataset in handle["derivative"].items():
            direction = _text(dataset.attrs["direction"])
            operations[f"derivative/{name}"] = (
                OperationRecord(OperationKind.DERIVATIVE, {"direction": direction}),
                decoder(dataset),
            )
    return time_slice, levels, representation, {"source": (None, source), **operations}


def _load_eigenvectors(
    path: Path,
    dataset_name: str,
    time_slice: int,
    levels: np.ndarray,
) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise ValueError(f"eigenvector dataset {dataset_name!r} is absent from {path}")
        dataset = handle[dataset_name]
        _axes(dataset)
        if dataset.ndim != 6:
            raise ValueError(f"{dataset.name} must have rank six, got {dataset.ndim}")
        if time_slice < 0 or time_slice >= dataset.shape[0]:
            raise ValueError(f"time slice {time_slice} is outside {dataset.name}")
        if int(levels[-1]) >= dataset.shape[-1]:
            raise ValueError(
                f"LapH level {int(levels[-1])} is outside mode axis of {dataset.name}"
            )
        selected = [np.asarray(dataset[time_slice, ..., int(level)]) for level in levels]
    return np.stack(selected, axis=-1)


def _reference_to_concord(values: np.ndarray) -> np.ndarray:
    return np.transpose(values, (1, 2, 3, 4, 0))


def _elemental(source: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    source_modes = source.reshape(-1, source.shape[-1])
    transformed_modes = transformed.reshape(-1, transformed.shape[-1])
    return source_modes.conj().T @ transformed_modes


def _metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "max_absolute": None,
            "max_relative": None,
            "max_tolerance_ratio": None,
        }
    difference = np.abs(actual - expected)
    max_absolute = float(np.max(difference, initial=0.0))
    denominator = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    max_relative = float(np.max(difference / denominator, initial=0.0))
    tolerance = np.maximum(
        atol + rtol * np.abs(expected),
        np.finfo(np.float64).tiny,
    )
    max_tolerance_ratio = float(np.max(difference / tolerance, initial=0.0))
    return {
        "passed": bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_absolute": max_absolute,
        "max_relative": max_relative,
        "max_tolerance_ratio": max_tolerance_ratio,
    }


def compare(
    *,
    reference_path: Path,
    eigenvector_path: Path,
    eigenvector_dataset: str,
    gauge_path: Path,
    device: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    time_slice, levels, representation, reference = _load_reference(reference_path)
    eigenvectors = _load_eigenvectors(
        eigenvector_path,
        eigenvector_dataset,
        time_slice,
        levels,
    )
    torch_device = torch.device(device)
    source = torch.as_tensor(
        np.ascontiguousarray(eigenvectors),
        dtype=torch.complex128,
        device=torch_device,
    ).unsqueeze(0)

    gauge_reader = LaphGaugeIOMapReader(gauge_path)
    if gauge_reader.metadata.lattice_extent[:3] != (
        eigenvectors.shape[2],
        eigenvectors.shape[1],
        eigenvectors.shape[0],
    ):
        raise ValueError(
            "smeared-gauge spatial extent does not match the normalized eigenvectors"
        )
    gauge_links = {
        axis: torch.as_tensor(
            np.ascontiguousarray(gauge_reader.read_timeslice(axis, time_slice)),
            dtype=torch.complex128,
            device=torch_device,
        ).unsqueeze(0)
        for axis in ("x", "y", "z")
    }

    results: dict[str, dict[str, Any]] = {}
    for name, (operation, expected_raw) in reference.items():
        if operation is None:
            actual = source[0].detach().cpu().numpy()
        else:
            actual = _apply_laph_operations(
                source,
                (operation,),
                gauge_links.__getitem__,
            )[0].detach().cpu().numpy()
        if representation == "fields":
            expected = _reference_to_concord(expected_raw)
        else:
            expected = expected_raw
            actual = _elemental(eigenvectors, actual)
        results[name] = _metrics(actual, expected, rtol=rtol, atol=atol)

    return {
        "passed": all(result["passed"] for result in results.values()),
        "reference": str(reference_path),
        "eigenvectors": str(eigenvector_path),
        "eigenvector_dataset": eigenvector_dataset,
        "gauge": str(gauge_path),
        "time_slice": time_slice,
        "levels": [int(level) for level in levels],
        "representation": representation,
        "device": str(torch_device),
        "rtol": rtol,
        "atol": atol,
        "comparisons": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eigenvectors", type=Path, required=True)
    parser.add_argument("--eigenvector-dataset", default="/eigenvectors")
    parser.add_argument("--gauge", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rtol", type=float, default=5.0e-13)
    parser.add_argument("--atol", type=float, default=5.0e-14)
    parser.add_argument("--json-report", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = compare(
        reference_path=args.reference,
        eigenvector_path=args.eigenvectors,
        eigenvector_dataset=args.eigenvector_dataset,
        gauge_path=args.gauge,
        device=args.device,
        rtol=args.rtol,
        atol=args.atol,
    )
    for name, result in report["comparisons"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status:4s} {name:32s} "
            f"max_abs={result['max_absolute']!s:>12s} "
            f"max_tol={result['max_tolerance_ratio']!s:>12s}"
        )
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.json_report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
