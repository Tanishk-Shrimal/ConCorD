#!/usr/bin/env python3
"""Compare a spectator-displaced proton with an independent field oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import opt_einsum as oe
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from concord_lqcd.core.dirac_basis import get_spin_matrices
from tools.validate_spatial_laph_correlators import (
    DEFAULT_PROPAGATION_AXES,
    SpatialReference,
    load_fixed_source_perambulator,
    load_spatial_reference,
    load_concord_tensors,
)


def _torch_complex_dtype(values: np.ndarray) -> torch.dtype:
    return torch.complex64 if values.dtype == np.complex64 else torch.complex128


def _color_epsilon(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    epsilon = torch.zeros((3, 3, 3), dtype=dtype, device=device)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1
    return epsilon


def direct_baryon_elemental(
    reference: SpatialReference,
    *,
    operation_path: str,
    transformed_leg: int,
    is_anti: bool,
    device: str,
) -> torch.Tensor:
    """Build ``epsilon_abc V0_a V1_b V2_c`` from independent Chroma fields."""
    if reference.representation != "fields":
        raise ValueError("baryon validation requires a v1 field reference")
    if transformed_leg not in (0, 1, 2):
        raise ValueError("transformed_leg must be 0, 1, or 2")
    normalized = operation_path.strip("/")
    try:
        transformed = reference.operations[normalized]
    except KeyError as exc:
        raise ValueError(
            f"{reference.path}: operation /{normalized} is absent"
        ) from exc

    fields = [reference.source, reference.source, reference.source]
    fields[transformed_leg] = transformed
    torch_device = torch.device(device)
    dtype = _torch_complex_dtype(reference.source)
    tensors: list[torch.Tensor] = []
    for field in fields:
        values = np.ascontiguousarray(field.reshape(len(reference.levels), -1, 3))
        tensor = torch.as_tensor(values, dtype=dtype, device=torch_device)
        tensors.append(tensor.conj() if is_anti else tensor)

    return oe.contract(
        "isa,jsb,ksc,abc->ijk",
        *tensors,
        _color_epsilon(dtype=dtype, device=torch_device),
        optimize="auto-hq",
        backend="torch",
    )


def direct_proton_correlator(
    *,
    sink_elemental: torch.Tensor,
    source_elemental: torch.Tensor,
    up_perambulator: np.ndarray,
    down_perambulator: np.ndarray,
    spin_basis: str,
    device: str,
) -> torch.Tensor:
    """Evaluate the independent direct-minus-exchange proton contraction."""
    torch_device = torch.device(device)
    dtype = sink_elemental.dtype
    up = torch.as_tensor(up_perambulator, dtype=dtype, device=torch_device)
    down = torch.as_tensor(down_perambulator, dtype=dtype, device=torch_device)
    matrices = get_spin_matrices(spin_basis)
    gamma4 = torch.as_tensor(matrices["gammaT"].copy(), dtype=dtype, device=torch_device)
    cg5 = torch.as_tensor(matrices["Cg5"].copy(), dtype=dtype, device=torch_device)
    cg5_dagger = cg5.conj().T

    direct = oe.contract(
        "ijk,lmn,ab,ialr,jbms,kdnq,rA,sB,qD,BA->dD",
        sink_elemental,
        source_elemental,
        cg5,
        up,
        down,
        up,
        gamma4,
        gamma4,
        gamma4,
        cg5_dagger,
        optimize="auto-hq",
        backend="torch",
    )
    exchange = oe.contract(
        "ijk,lmn,ab,ianq,jbms,kdlr,qD,sB,rA,BA->dD",
        sink_elemental,
        source_elemental,
        cg5,
        up,
        down,
        up,
        gamma4,
        gamma4,
        gamma4,
        cg5_dagger,
        optimize="auto-hq",
        backend="torch",
    )
    return direct - exchange


def _complex_payload(values: np.ndarray | complex) -> dict[str, Any]:
    array = np.asarray(values)
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def _matrix_metrics(
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
            "max_scaled": None,
        }
    difference = np.abs(actual - expected)
    max_absolute = float(np.max(difference, initial=0.0))
    channel_scale = max(float(np.max(np.abs(expected), initial=0.0)), np.finfo(float).tiny)
    reference_frobenius_norm = float(np.linalg.norm(expected))
    difference_frobenius_norm = float(np.linalg.norm(actual - expected))
    frobenius_scale = max(reference_frobenius_norm, np.finfo(float).tiny)
    tolerance = np.maximum(
        atol + rtol * np.abs(expected),
        np.finfo(float).tiny,
    )
    return {
        "passed": bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "reference_maximum_magnitude": channel_scale,
        "reference_frobenius_norm": reference_frobenius_norm,
        "max_absolute": max_absolute,
        "max_scaled": float(max_absolute / channel_scale),
        "difference_frobenius_norm": difference_frobenius_norm,
        "frobenius_relative": float(difference_frobenius_norm / frobenius_scale),
        "max_tolerance_ratio": float(np.max(difference / tolerance, initial=0.0)),
        "concord": _complex_payload(actual),
        "reference": _complex_payload(expected),
    }


def compare(
    *,
    source_reference_path: Path,
    sink_reference_path: Path,
    up_perambulator_path: Path,
    down_perambulator_path: Path,
    perambulator_dataset: str,
    perambulator_axes: Sequence[str],
    source_field: str,
    concord_output_path: Path,
    correlator_id: str,
    configuration: int,
    source_time: int,
    sink_time: int,
    source_operation: str,
    sink_operation: str,
    transformed_leg: int,
    spin_basis: str,
    device: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    source_reference = load_spatial_reference(
        source_reference_path,
        (source_operation,),
    )
    sink_reference = load_spatial_reference(
        sink_reference_path,
        (sink_operation,),
    )
    if source_reference.time_slice != source_time:
        raise ValueError(
            f"source reference time is {source_reference.time_slice}, expected {source_time}"
        )
    if sink_reference.time_slice != sink_time:
        raise ValueError(
            f"sink reference time is {sink_reference.time_slice}, expected {sink_time}"
        )
    if source_reference.levels != sink_reference.levels:
        raise ValueError("source and sink Chroma references use different LapH levels")

    levels = source_reference.levels
    up = load_fixed_source_perambulator(
        path=up_perambulator_path,
        dataset_template=perambulator_dataset,
        source_time=source_time,
        sink_time=sink_time,
        axes=perambulator_axes,
        sink_levels=levels,
        source_levels=levels,
        source_field=source_field,
        spin_basis=spin_basis,
    )
    down = load_fixed_source_perambulator(
        path=down_perambulator_path,
        dataset_template=perambulator_dataset,
        source_time=source_time,
        sink_time=sink_time,
        axes=perambulator_axes,
        sink_levels=levels,
        source_levels=levels,
        source_field=source_field,
        spin_basis=spin_basis,
    )
    actual = load_concord_tensors(
        path=concord_output_path,
        correlator_ids=(correlator_id,),
        configuration=configuration,
        source_time=source_time,
        sink_time=sink_time,
    )[correlator_id]
    if actual.shape != (4, 4):
        raise ValueError(
            f"correlator {correlator_id!r} must be a 4x4 spin matrix, got {actual.shape}"
        )

    sink_elemental = direct_baryon_elemental(
        sink_reference,
        operation_path=sink_operation,
        transformed_leg=transformed_leg,
        is_anti=False,
        device=device,
    )
    source_elemental = direct_baryon_elemental(
        source_reference,
        operation_path=source_operation,
        transformed_leg=transformed_leg,
        is_anti=True,
        device=device,
    )
    expected_tensor = direct_proton_correlator(
        sink_elemental=sink_elemental,
        source_elemental=source_elemental,
        up_perambulator=up,
        down_perambulator=down,
        spin_basis=spin_basis,
        device=device,
    )
    expected = expected_tensor.detach().cpu().numpy()
    metrics = _matrix_metrics(actual, expected, rtol=rtol, atol=atol)

    pplus = np.asarray(get_spin_matrices(spin_basis)["Pplus"], dtype=np.complex128)
    actual_positive = complex(np.trace(pplus @ actual))
    expected_positive = complex(np.trace(pplus @ expected))
    metrics["positive_parity"] = {
        "concord": _complex_payload(actual_positive),
        "reference": _complex_payload(expected_positive),
        "absolute_difference": float(abs(actual_positive - expected_positive)),
    }
    return {
        "passed": metrics["passed"],
        "configuration": configuration,
        "source_time": source_time,
        "sink_time": sink_time,
        "levels": list(levels),
        "spin_basis": spin_basis,
        "device": str(torch.device(device)),
        "correlator_id": correlator_id,
        "transformed_leg": transformed_leg,
        "transformed_field": "spectator_u" if transformed_leg == 2 else "non_spectator",
        "source_operation": source_operation.strip("/"),
        "sink_operation": sink_operation.strip("/"),
        "source_reference": str(source_reference_path),
        "sink_reference": str(sink_reference_path),
        "up_perambulator": str(up_perambulator_path),
        "down_perambulator": str(down_perambulator_path),
        "concord_output": str(concord_output_path),
        "rtol": rtol,
        "atol": atol,
        "comparison": metrics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", type=Path, required=True)
    parser.add_argument("--sink-reference", type=Path, required=True)
    parser.add_argument("--up-perambulator", type=Path, required=True)
    parser.add_argument("--down-perambulator", type=Path)
    parser.add_argument(
        "--perambulator-dataset",
        default="/perambulators/src_t_{fixed_source_time}/data",
    )
    parser.add_argument(
        "--perambulator-axes",
        default=",".join(DEFAULT_PROPAGATION_AXES),
    )
    parser.add_argument("--source-field", choices=("qbar", "qdagger"), default="qdagger")
    parser.add_argument("--concord-output", type=Path, required=True)
    parser.add_argument(
        "--correlator-id",
        default="proton_spectator_transport_plus_x",
    )
    parser.add_argument("--configuration", type=int, required=True)
    parser.add_argument("--source-time", type=int, required=True)
    parser.add_argument("--sink-time", type=int, required=True)
    parser.add_argument("--source-operation", default="transport/plus_x")
    parser.add_argument("--sink-operation", default="transport/plus_x")
    parser.add_argument("--transformed-leg", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--spin-basis", choices=("DR", "DP"), default="DP")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rtol", type=float, default=5.0e-13)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--json-report", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    axes = tuple(axis.strip() for axis in args.perambulator_axes.split(",") if axis.strip())
    report = compare(
        source_reference_path=args.source_reference,
        sink_reference_path=args.sink_reference,
        up_perambulator_path=args.up_perambulator,
        down_perambulator_path=args.down_perambulator or args.up_perambulator,
        perambulator_dataset=args.perambulator_dataset,
        perambulator_axes=axes,
        source_field=args.source_field,
        concord_output_path=args.concord_output,
        correlator_id=args.correlator_id,
        configuration=args.configuration,
        source_time=args.source_time,
        sink_time=args.sink_time,
        source_operation=args.source_operation,
        sink_operation=args.sink_operation,
        transformed_leg=args.transformed_leg,
        spin_basis=args.spin_basis,
        device=args.device,
        rtol=args.rtol,
        atol=args.atol,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
