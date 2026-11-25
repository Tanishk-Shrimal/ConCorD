#!/usr/bin/env python3
"""Compare a nonzero-momentum LapH loop with an independent NumPy oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import h5py
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

EIGENVECTOR_AXES = ("time", "z", "y", "x", "color", "mode")
PERAMBULATOR_ORDER = "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
MOMENTUM = (1, 0, 0)


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _eigenvector_metadata(path: Path, dataset_name: str) -> tuple[int, int, int, int]:
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise ValueError(f"eigenvector dataset {dataset_name!r} is absent from {path}")
        dataset = handle[dataset_name]
        axes_raw = dataset.attrs.get("axes")
        if axes_raw is None:
            raise ValueError(f"{dataset.name} has no required axes attribute")
        axes = tuple(json.loads(_text(axes_raw)))
        if axes != EIGENVECTOR_AXES:
            raise ValueError(f"{dataset.name} axes {axes} != {EIGENVECTOR_AXES}")
        if dataset.ndim != 6 or dataset.shape[4] != 3:
            raise ValueError(
                f"{dataset.name} must have shape [Nt,Nz,Ny,Nx,3,Nv], got {dataset.shape}"
            )
        if not np.issubdtype(dataset.dtype, np.complexfloating):
            raise ValueError(f"{dataset.name} must contain complex eigenvectors")
        temporal_extent, _, _, spatial_x, _, mode_size = dataset.shape
        return int(temporal_extent), int(spatial_x), int(mode_size), int(dataset.shape[2])


def _perambulator_metadata(path: Path, temporal_extent: int, mode_size: int) -> None:
    expected_shape = (temporal_extent, 4, 4, mode_size, mode_size)
    with h5py.File(path, "r") as handle:
        if "perambulators" not in handle:
            raise ValueError(f"{path} has no /perambulators group")
        groups = handle["perambulators"]
        source_times = sorted(
            (int(match.group(1)), name)
            for name in groups
            if (match := re.fullmatch(r"src_t_(\d+)", name))
        )
        if [time for time, _ in source_times] != list(range(temporal_extent)):
            raise ValueError(
                f"{path} must contain dense source groups src_t_0 through "
                f"src_t_{temporal_extent - 1}"
            )
        for _, name in source_times:
            dataset = groups[name]["data"]
            if dataset.shape != expected_shape:
                raise ValueError(
                    f"{dataset.name} shape {dataset.shape} != {expected_shape}"
                )
            if _text(dataset.attrs.get("dimension_order", "")) != PERAMBULATOR_ORDER:
                raise ValueError(f"{dataset.name} has an unexpected dimension_order")
            if not np.issubdtype(dataset.dtype, np.complexfloating):
                raise ValueError(f"{dataset.name} must contain complex perambulators")


def _reference_values(
    perambulator_path: Path,
    eigenvector_path: Path,
    *,
    eigenvector_dataset: str,
    temporal_extent: int,
    spatial_extents: tuple[int, int, int],
    mode_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_extent, y_extent, z_extent = spatial_extents
    if MOMENTUM[1] != 0 or MOMENTUM[2] != 0:
        raise ValueError("the current oracle is for p=(1,0,0)")
    x_phase = np.exp(
        2j * np.pi * np.arange(x_extent, dtype=np.float64) / x_extent
    ).reshape(1, 1, x_extent)

    values = np.empty(temporal_extent, dtype=np.complex128)
    opposite_momentum_values = np.empty(temporal_extent, dtype=np.complex128)
    with h5py.File(perambulator_path, "r") as perambulator_file, h5py.File(
        eigenvector_path, "r"
    ) as eigenvector_file:
        eigenvectors = eigenvector_file[eigenvector_dataset]
        perambulators = perambulator_file["perambulators"]
        for time in range(temporal_extent):
            modes = np.asarray(eigenvectors[time], dtype=np.complex128)
            if modes.shape != (z_extent, y_extent, x_extent, 3, mode_size):
                raise ValueError(
                    f"eigenvector slice {time} has shape {modes.shape}, expected "
                    f"{(z_extent, y_extent, x_extent, 3, mode_size)}"
                )
            elemental = np.einsum(
                "zyxci,zyxcj,zyx->ij",
                modes.conj(),
                modes,
                x_phase,
                optimize=True,
            )
            opposite_elemental = np.einsum(
                "zyxci,zyxcj,zyx->ij",
                modes.conj(),
                modes,
                x_phase.conj(),
                optimize=True,
            )
            equal_time_perambulator = np.asarray(
                perambulators[f"src_t_{time}"]["data"][time],
                dtype=np.complex128,
            )
            values[time] = -np.einsum(
                "aaij,ij->",
                equal_time_perambulator,
                elemental,
                optimize=True,
            )
            opposite_momentum_values[time] = -np.einsum(
                "aaij,ij->",
                equal_time_perambulator,
                opposite_elemental,
                optimize=True,
            )
    return values, opposite_momentum_values


def _write_operator_source(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from concord_lqcd import Correlator, Operator, gamma_identity, u, ubar",
                "",
                "def build():",
                "    loop = Operator(ubar('x','a','s1') * gamma_identity('s1','s2') * u('x','a','s2'), momentum=(1,0,0))",
                "    return {'momentum_scalar_loop': Correlator(loop)}",
            ]
        ),
        encoding="utf-8",
    )


def _job_spec(
    *,
    perambulator_path: Path,
    eigenvector_path: Path,
    output_dir: Path,
    temporal_extent: int,
    spatial_extents: tuple[int, int, int],
    mode_size: int,
    configuration: int,
) -> Any:
    from concord_lqcd.job import JobSpec

    operator_path = output_dir / "operators.py"
    _write_operator_source(operator_path)
    return JobSpec.model_validate(
        {
            "schema_version": 1,
            "job": {"name": f"disconnected_momentum_loop_cfg{configuration}"},
            "operators": {"file": str(operator_path), "factory": "build"},
            "plan": {"file": str(output_dir / "momentum_loop.concord")},
            "lattice": {
                "spatial_extent": list(spatial_extents),
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
                "times": {
                    "contexts": [
                        {
                            "id": f"loop_t{time}",
                            "operator_times": {1: [time]},
                        }
                        for time in range(temporal_extent)
                    ]
                },
            },
            "data": {
                "eigenvectors": {
                    "provider": "hdf5_distillation",
                    "path": str(eigenvector_path),
                    "dataset": "/eigenvectors",
                    "axes": list(EIGENVECTOR_AXES),
                },
                "propagation": {
                    "provider": "hdf5_perambulator",
                    "ordinary": {
                        "files": {"u": str(perambulator_path)},
                        "dataset": "/perambulators/src_t_{source_time}/data",
                        "axes": [
                            "sink_time",
                            "source_spin",
                            "sink_spin",
                            "source_mode",
                            "sink_mode",
                        ],
                    },
                },
            },
            "output": {
                "file": str(output_dir / "concord_result.h5"),
                "on_existing": "error",
            },
        }
    )


def _concord_values(path: Path, temporal_extent: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        matches = [
            group
            for group in handle["correlators"].values()
            if _text(group.attrs.get("correlator_id", "")) == "momentum_scalar_loop"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one momentum_scalar_loop result in {path}")
        values = np.asarray(matches[0]["values"], dtype=np.complex128)
    if values.shape != (1, temporal_extent):
        raise ValueError(
            f"ConCorD result has shape {values.shape}, expected {(1, temporal_extent)}"
        )
    return values[0]


def _metrics(actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any]:
    difference = np.abs(actual - expected)
    max_absolute = float(np.max(difference, initial=0.0))
    denominator = float(np.max(np.abs(expected), initial=0.0))
    max_scaled = max_absolute / max(denominator, np.finfo(np.float64).tiny)
    return {
        "passed": bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        "max_absolute": max_absolute,
        "max_scaled": max_scaled,
        "max_reference_magnitude": denominator,
        "max_actual_magnitude": float(np.max(np.abs(actual), initial=0.0)),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (
        output_dir / "independent_reference.h5",
        output_dir / "momentum_loop.concord",
        output_dir / "concord_result.h5",
        output_dir / "report.json",
    )
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite validation artifacts: " + ", ".join(existing)
        )

    temporal_extent, spatial_x, mode_size, spatial_y = _eigenvector_metadata(
        args.eigenvectors,
        args.eigenvector_dataset,
    )
    with h5py.File(args.eigenvectors, "r") as handle:
        eigenvector_shape = handle[args.eigenvector_dataset].shape
    spatial_z = int(eigenvector_shape[1])
    spatial_extents = (spatial_x, spatial_y, spatial_z)
    _perambulator_metadata(args.perambulator, temporal_extent, mode_size)

    print("Building direct NumPy reference before ConCorD execution", flush=True)
    expected, opposite_momentum = _reference_values(
        args.perambulator,
        args.eigenvectors,
        eigenvector_dataset=args.eigenvector_dataset,
        temporal_extent=temporal_extent,
        spatial_extents=spatial_extents,
        mode_size=mode_size,
    )
    with h5py.File(output_dir / "independent_reference.h5", "w") as handle:
        handle.attrs["construction"] = "independent_numpy_disconnected_momentum_loop"
        handle.attrs["formula"] = (
            "-sum_{a,i,j} tau(t,t)[a,a,i,j] "
            "sum_{z,y,x,c} V(t,z,y,x,c,i)^* V(t,z,y,x,c,j) "
            "exp(+2*pi*i*x/Lx)"
        )
        handle.attrs["momentum"] = MOMENTUM
        handle.attrs["source_perambulator"] = str(args.perambulator)
        handle.attrs["eigenvectors"] = str(args.eigenvectors)
        handle.create_dataset("values", data=expected)
        handle.create_dataset("opposite_momentum_values", data=opposite_momentum)

    from concord_lqcd.job import run_job, write_job_plan

    spec = _job_spec(
        perambulator_path=args.perambulator.resolve(),
        eigenvector_path=args.eigenvectors.resolve(),
        output_dir=output_dir,
        temporal_extent=temporal_extent,
        spatial_extents=spatial_extents,
        mode_size=mode_size,
        configuration=args.configuration,
    )
    if not os.environ.get("XDG_CACHE_HOME"):
        os.environ["XDG_CACHE_HOME"] = str(output_dir / "cache")
    plan = write_job_plan(spec)
    print("Running the ConCorD CPU job", flush=True)
    summary = run_job(spec, plan)
    if not summary.all_complete:
        raise RuntimeError("ConCorD did not complete all source-time loop work units")
    actual = _concord_values(spec.output.file, temporal_extent)
    result = _metrics(actual, expected, rtol=args.rtol, atol=args.atol)
    result["max_opposite_momentum_separation"] = float(
        np.max(np.abs(expected - opposite_momentum), initial=0.0)
    )
    result["max_difference_if_phase_sign_is_reversed"] = float(
        np.max(np.abs(actual - opposite_momentum), initial=0.0)
    )
    result.update(
        {
            "configuration": args.configuration,
            "momentum": list(MOMENTUM),
            "temporal_extent": temporal_extent,
            "spatial_extents": list(spatial_extents),
            "mode_size": mode_size,
            "executed_work_units": summary.executed_work_units,
            "source_perambulator": str(args.perambulator.resolve()),
            "eigenvectors": str(args.eigenvectors.resolve()),
            "concord_output": str(spec.output.file),
        }
    )
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perambulator", type=Path, required=True)
    parser.add_argument("--eigenvectors", type=Path, required=True)
    parser.add_argument("--eigenvector-dataset", default="/eigenvectors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", type=int, default=1161)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    result = validate(args)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
