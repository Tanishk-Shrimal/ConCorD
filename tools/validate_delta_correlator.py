#!/usr/bin/env python3
"""Independent rest-frame Delta++ Wick and spin-3/2 validation."""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import opt_einsum as oe


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools._independent_laph import (
    EIGENVECTOR_AXES,
    baryon_elemental,
    hdf5_text,
    read_eigenvector_times,
    read_propagators,
)


def dp_gamma_matrices() -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.diag([1, -1]).astype(np.complex128)
    zero = np.zeros((2, 2), dtype=np.complex128)
    # Chroma Dirac-Pauli basis: gamma_k = [[0, -i sigma_k], [i sigma_k, 0]] for every k.
    spatial = tuple(
        np.block([[zero, -1j * sigma], [1j * sigma, zero]])
        for sigma in (sigma_x, sigma_y, sigma_z)
    )
    temporal = np.diag([1, 1, -1, -1]).astype(np.complex128)
    return spatial, temporal


def rest_spin_projector(spatial_gammas: tuple[np.ndarray, ...]) -> np.ndarray:
    identity = np.eye(4, dtype=np.complex128)
    return np.stack([
        np.stack([
            (identity if row == column else 0) - spatial_gammas[row] @ spatial_gammas[column] / 3
            for column in range(3)
        ])
        for row in range(3)
    ])


def project_delta(
    values: np.ndarray,
    spatial_gammas: tuple[np.ndarray, ...],
    temporal_gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unprojected = np.asarray(values, dtype=np.complex128)
    if unprojected.ndim != 5 or unprojected.shape[1:] != (3, 3, 4, 4):
        raise ValueError("Delta matrix must have shape (Nt,3,3,4,4)")
    projector = rest_spin_projector(spatial_gammas)
    flattened_projector = projector.transpose(0, 2, 1, 3).reshape(12, 12)
    identity = np.eye(4, dtype=np.complex128)
    parity_plus = (identity + temporal_gamma) / 2
    parity_minus = (identity - temporal_gamma) / 2
    projected = np.empty_like(unprojected)
    positive = np.empty(unprojected.shape[0], dtype=np.complex128)
    negative = np.empty_like(positive)
    for time, matrix in enumerate(unprojected):
        flattened = matrix.transpose(0, 2, 1, 3).reshape(12, 12)
        projected[time] = (
            flattened_projector @ flattened @ flattened_projector
        ).reshape(3, 4, 3, 4).transpose(0, 2, 1, 3)
        trace_matrix = sum(projected[time, direction, direction] for direction in range(3))
        positive[time] = np.trace(parity_plus @ trace_matrix)
        negative[time] = np.trace(parity_minus @ trace_matrix)
    return projected, positive, negative


def _permutation_sign(permutation: tuple[int, int, int]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def evaluate_reference(
    sink_elemental: np.ndarray,
    source_elemental: np.ndarray,
    propagator: np.ndarray,
) -> np.ndarray:
    """Sum six signed contractions for epsilon (u^T C gamma_i u) u."""
    sink_elemental = np.asarray(sink_elemental, dtype=np.complex128)
    source_elemental = np.asarray(source_elemental, dtype=np.complex128)
    propagator = np.asarray(propagator, dtype=np.complex128)
    if sink_elemental.ndim != 3 or len(set(sink_elemental.shape)) != 1:
        raise ValueError("sink elemental must have shape (Nv,Nv,Nv)")
    if source_elemental.shape != sink_elemental.shape:
        raise ValueError("source and sink elementals must have the same shape")
    mode_size = sink_elemental.shape[0]
    if propagator.shape != (mode_size, 4, mode_size, 4):
        raise ValueError("propagator must have shape (Nv,4,Nv,4)")

    spatial_gammas, gamma4 = dp_gamma_matrices()
    charge_conjugation = spatial_gammas[1] @ gamma4
    diquarks = tuple(charge_conjugation @ gamma for gamma in spatial_gammas)
    source_modes = "lmn"
    source_spins = "rsq"
    total = np.zeros((3, 3, 4, 4), dtype=np.complex128)
    for sink_direction, sink_diquark in enumerate(diquarks):
        for source_direction, source_diquark in enumerate(diquarks):
            for permutation in permutations((0, 1, 2)):
                paired = tuple(
                    source_modes[index] + source_spins[index]
                    for index in permutation
                )
                equation = (
                    "ijk,lmn,ab,BA,"
                    f"ia{paired[0]},jb{paired[1]},kd{paired[2]},"
                    "rA,sB,qD->dD"
                )
                total[sink_direction, source_direction] += _permutation_sign(permutation) * oe.contract(
                    equation,
                    sink_elemental,
                    source_elemental,
                    sink_diquark,
                    source_diquark.conj().T,
                    propagator,
                    propagator,
                    propagator,
                    gamma4,
                    gamma4,
                    gamma4,
                    optimize="greedy",
                )
    return total


def _write_operator_source(path: Path) -> None:
    path.write_text(
        "\n".join((
            "from concord_lqcd.core import Correlator, build_baryon_operator, reset_registry",
            "",
            "def build():",
            "    reset_registry()",
            "    delta = [build_baryon_operator(('u', 'u', 'u'), f'Cg{direction}', pos_label='x', spin_labels=('alpha', 'beta', 'rho')) for direction in range(1, 4)]",
            "    return {f'delta_{sink + 1}_{source + 1}': Correlator(delta[sink], delta[source].dagger()) for sink in range(3) for source in range(3)}",
            "",
        )),
        encoding="utf-8",
    )


def _job_spec(
    *,
    perambulator: Path,
    eigenvectors: Path,
    eigenvector_dataset: str,
    output_dir: Path,
    configuration: int,
    source_time: int,
    sink_times: tuple[int, ...],
    mode_size: int,
    lattice: tuple[int, int, int, int],
):
    from concord_lqcd.job import JobSpec

    _write_operator_source(output_dir / "operators.py")
    temporal_extent, x_extent, y_extent, z_extent = lattice
    return JobSpec.model_validate({
        "schema_version": 1,
        "job": {"name": f"delta_pp_cfg{configuration}"},
        "operators": {"file": str(output_dir / "operators.py"), "factory": "build"},
        "plan": {"file": str(output_dir / "delta.concord")},
        "lattice": {
            "spatial_extent": [x_extent, y_extent, z_extent],
            "temporal_extent": temporal_extent,
        },
        "compile": {
            "mode_size": mode_size,
            "spin_basis": "DP",
            "scalar_type": "complex128",
            "contraction": {"strategy": "auto-hq"},
        },
        "run": {
            "device": "cpu",
            "time_batch_size": 1,
            "configurations": {"values": [configuration]},
            "times": {"contexts": [
                {
                    "id": f"sink_t{sink_time}",
                    "operator_times": {2: [sink_time, source_time]},
                }
                for sink_time in sink_times
            ]},
        },
        "data": {
            "eigenvectors": {
                "provider": "hdf5_distillation",
                "path": str(eigenvectors),
                "dataset": eigenvector_dataset,
                "axes": list(EIGENVECTOR_AXES),
            },
            "propagation": {
                "provider": "hdf5_perambulator",
                "ordinary": {
                    "files": {"u": str(perambulator)},
                    "dataset": "/perambulators/src_t_{source_time}/data",
                    "axes": ["sink_time", "source_spin", "sink_spin", "source_mode", "sink_mode"],
                    "source_field": "qdagger",
                },
            },
        },
        "output": {"file": str(output_dir / "concord_result.h5"), "on_existing": "error"},
    })


def _stored_values(path: Path, count: int) -> np.ndarray:
    values = np.empty((count, 3, 3, 4, 4), dtype=np.complex128)
    found: set[tuple[int, int]] = set()
    with h5py.File(path, "r") as handle:
        for group in handle["correlators"].values():
            identifier = hdf5_text(group.attrs.get("correlator_id", ""))
            if not identifier.startswith("delta_"):
                raise ValueError(f"unexpected correlator {identifier}")
            _, sink, source = identifier.split("_")
            row, column = int(sink) - 1, int(source) - 1
            if row not in range(3) or column not in range(3) or (row, column) in found:
                raise ValueError(f"duplicate or invalid Delta channel {identifier}")
            data = np.asarray(group["values"], dtype=np.complex128)
            if data.shape != (1, count, 4, 4):
                raise ValueError(f"invalid Delta output shape {data.shape}")
            values[:, row, column] = data[0]
            found.add((row, column))
    if len(found) != 9:
        raise ValueError(f"expected nine Delta channels, found {len(found)}")
    return values


def validate(
    *,
    perambulator: Path,
    eigenvectors: Path,
    output_dir: Path,
    mode_size: int = 8,
    source_time: int = 0,
    sink_times: tuple[int, ...] = (1, 7, 23),
    configuration: int = 1161,
    eigenvector_dataset: str = "/eigenvectors",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict[str, Any]:
    if mode_size < 3 or not sink_times or len(set(sink_times)) != len(sink_times):
        raise ValueError("choose Nv >= 3 and distinct sink times")
    if any(time == source_time for time in sink_times):
        raise ValueError("sink times must differ from the source time")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (
        output_dir / "independent_reference.h5",
        output_dir / "delta.concord",
        output_dir / "concord_result.h5",
        output_dir / "report.json",
    )
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite validation artifacts: " + ", ".join(existing))

    times = (source_time,) + sink_times
    slices, lattice = read_eigenvector_times(
        eigenvectors, eigenvector_dataset, times, mode_size
    )
    blocks = read_propagators({"u": perambulator}, times, mode_size, lattice[0])
    elementals = {time: baryon_elemental(slices[time], mode_size) for time in times}
    expected = np.stack([
        evaluate_reference(
            elementals[sink_time],
            elementals[source_time].conj(),
            blocks[("u", sink_time, source_time)],
        )
        for sink_time in sink_times
    ])
    spatial_gammas, gamma4 = dp_gamma_matrices()
    reference_projected, reference_positive, reference_negative = project_delta(
        expected, spatial_gammas, gamma4
    )
    with h5py.File(output_dir / "independent_reference.h5", "w") as handle:
        handle.attrs["construction"] = "independent_six_permutation_delta_pp"
        handle.attrs["source_time"] = source_time
        handle.attrs["sink_times"] = sink_times
        handle.attrs["mode_size"] = mode_size
        handle.create_dataset("unprojected", data=expected)
        handle.create_dataset("spin_projected", data=reference_projected)
        handle.create_dataset("parity_plus", data=reference_positive)
        handle.create_dataset("parity_minus", data=reference_negative)

    from concord_lqcd.job import run_job, write_job_plan

    spec = _job_spec(
        perambulator=perambulator,
        eigenvectors=eigenvectors,
        eigenvector_dataset=eigenvector_dataset,
        output_dir=output_dir,
        configuration=configuration,
        source_time=source_time,
        sink_times=sink_times,
        mode_size=mode_size,
        lattice=lattice,
    )
    plan = write_job_plan(spec)
    summary = run_job(spec, plan)
    if not summary.all_complete or summary.executed_work_units != len(sink_times):
        raise RuntimeError("Delta ConCorD job did not complete all requested channels and times")
    actual = _stored_values(spec.output.file, len(sink_times))
    actual_projected, actual_positive, actual_negative = project_delta(
        actual, spatial_gammas, gamma4
    )
    difference = np.abs(actual - expected)
    time_scales = np.max(np.abs(expected), axis=(1, 2, 3, 4))
    time_errors = np.max(difference, axis=(1, 2, 3, 4))
    time_scaled = np.divide(
        time_errors,
        time_scales,
        out=np.full_like(time_errors, np.inf),
        where=time_scales > 0,
    )
    projected_scale = max(float(np.max(np.abs(reference_projected))), np.finfo(float).tiny)
    positive_scale = float(np.max(np.abs(reference_positive)))
    negative_scale = float(np.max(np.abs(reference_negative)))
    projected_scaled = float(
        np.max(np.abs(actual_projected - reference_projected)) / projected_scale
    )
    parity_scaled = float(max(
        np.max(np.abs(actual_positive - reference_positive))
        / max(positive_scale, np.finfo(float).tiny),
        np.max(np.abs(actual_negative - reference_negative))
        / max(negative_scale, np.finfo(float).tiny),
    ))
    report = {
        "passed": bool(
            np.allclose(actual, expected, rtol=rtol, atol=atol)
            and np.all(time_scales > atol)
            and np.all(time_scaled <= rtol)
            and positive_scale > atol
            and negative_scale > atol
            and projected_scaled <= rtol
            and parity_scaled <= rtol
        ),
        "max_absolute": float(np.max(difference)),
        "max_per_time_scaled": float(np.max(time_scaled)),
        "max_spin_projected_scaled": projected_scaled,
        "max_parity_projected_scaled": parity_scaled,
        "max_reference_magnitude": float(np.max(np.abs(expected))),
        "minimum_reference_magnitude": float(np.min(time_scales)),
        "max_positive_parity_magnitude": positive_scale,
        "max_negative_parity_magnitude": negative_scale,
        "configuration": configuration,
        "source_time": source_time,
        "sink_times": sink_times,
        "mode_size": mode_size,
        "wick_terms_per_channel": 6,
        "correlator_channels": 9,
        "source_perambulator": str(perambulator),
        "eigenvectors": str(eigenvectors),
        "concord_output": str(spec.output.file),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perambulator", type=Path, required=True)
    parser.add_argument("--eigenvectors", type=Path, required=True)
    parser.add_argument("--eigenvector-dataset", default="/eigenvectors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", type=int, default=1161)
    parser.add_argument("--mode-size", type=int, default=8)
    parser.add_argument("--source-time", type=int, default=0)
    parser.add_argument("--sink-times", type=int, nargs="+", default=(1, 7, 23))
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    report = validate(
        perambulator=args.perambulator,
        eigenvectors=args.eigenvectors,
        eigenvector_dataset=args.eigenvector_dataset,
        output_dir=args.output_dir,
        configuration=args.configuration,
        mode_size=args.mode_size,
        source_time=args.source_time,
        sink_times=tuple(args.sink_times),
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
