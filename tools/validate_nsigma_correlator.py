#!/usr/bin/env python3
"""Independent raw proton-isoscalar-scalar two-point validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import h5py
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools._independent_baryon_meson import ReferenceResult, evaluate_reference
from tools._independent_laph import (
    EIGENVECTOR_AXES,
    baryon_elemental,
    hdf5_text,
    read_eigenvector_times,
    read_propagators,
)


SIGMA_BASIS = (
    ("sigma_u", ("u", "u"), 1 / np.sqrt(2)),
    ("sigma_d", ("d", "d"), 1 / np.sqrt(2)),
)


def evaluate_sigma_reference(
    sink_elemental: np.ndarray,
    source_elemental: np.ndarray,
    propagators: Mapping[tuple[str, int, int], np.ndarray],
    *,
    sink_time: int,
    source_time: int,
) -> tuple[np.ndarray, dict[tuple[str, str], ReferenceResult]]:
    components: dict[tuple[str, str], ReferenceResult] = {}
    total = np.zeros((4, 4), dtype=np.complex128)
    for sink_name, sink_meson, sink_weight in SIGMA_BASIS:
        for source_name, source_meson, source_weight in SIGMA_BASIS:
            result = evaluate_reference(
                sink_elemental,
                source_elemental,
                propagators,
                sink_time=sink_time,
                source_time=source_time,
                sink_meson_flavors=sink_meson,
                source_meson_flavors=source_meson,
                meson_spin="scalar",
            )
            components[sink_name, source_name] = result
            total += sink_weight * source_weight * result.total
    return total, components


def _write_operator_source(path: Path) -> None:
    path.write_text(
        "\n".join((
            "from math import sqrt",
            "from concord_lqcd.core import Correlator, Operator, build_baryon_operator, d, dbar, gamma_identity, reset_registry, u, ubar",
            "",
            "def build():",
            "    reset_registry()",
            "    proton = build_baryon_operator(('u', 'd', 'u'), 'Cg5', pos_label='x', color_labels=('a', 'b', 'c'), spin_labels=('alpha', 'beta', 'rho'))",
            "    sigma_u = Operator(ubar('y', 'k', 'mu') * gamma_identity('mu', 'nu') * u('y', 'k', 'nu'), momentum=(0, 0, 0))",
            "    sigma_d = Operator(dbar('y', 'k', 'mu') * gamma_identity('mu', 'nu') * d('y', 'k', 'nu'), momentum=(0, 0, 0))",
            "    operator = proton * ((sigma_u + sigma_d) / sqrt(2))",
            "    return {'nsigma_raw': Correlator(operator, operator.dagger())}",
            "",
        )),
        encoding="utf-8",
    )


def _job_spec(
    *,
    paths: Mapping[str, Path],
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
        "job": {"name": f"nsigma_raw_cfg{configuration}"},
        "operators": {"file": str(output_dir / "operators.py"), "factory": "build"},
        "plan": {"file": str(output_dir / "nsigma.concord")},
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
                    "files": {flavor: str(path) for flavor, path in paths.items()},
                    "dataset": "/perambulators/src_t_{source_time}/data",
                    "axes": ["sink_time", "source_spin", "sink_spin", "source_mode", "sink_mode"],
                    "source_field": "qdagger",
                },
            },
        },
        "output": {"file": str(output_dir / "concord_result.h5"), "on_existing": "error"},
    })


def _stored_values(path: Path, count: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        groups = [
            group for group in handle["correlators"].values()
            if hdf5_text(group.attrs.get("correlator_id", "")) == "nsigma_raw"
        ]
        if len(groups) != 1:
            raise ValueError("expected one N-sigma correlator in the output")
        values = np.asarray(groups[0]["values"], dtype=np.complex128)
    if values.shape != (1, count, 4, 4):
        raise ValueError(f"unexpected N-sigma output shape {values.shape}")
    return values[0]


def validate(
    *,
    perambulator: Path,
    eigenvectors: Path,
    output_dir: Path,
    mode_size: int = 8,
    source_time: int = 0,
    sink_times: tuple[int, ...] = (1, 7, 23),
    configuration: int = 1161,
    down_perambulator: Path | None = None,
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
        output_dir / "nsigma.concord",
        output_dir / "concord_result.h5",
        output_dir / "report.json",
        output_dir / "operators.py",
    )
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite validation artifacts: " + ", ".join(existing))

    times = (source_time,) + sink_times
    slices, lattice = read_eigenvector_times(
        eigenvectors, eigenvector_dataset, times, mode_size
    )
    paths = {"u": perambulator, "d": down_perambulator or perambulator}
    blocks = read_propagators(paths, times, mode_size, lattice[0])
    elementals = {time: baryon_elemental(slices[time], mode_size) for time in times}
    references = tuple(
        evaluate_sigma_reference(
            elementals[sink_time],
            elementals[source_time].conj(),
            blocks,
            sink_time=sink_time,
            source_time=source_time,
        )
        for sink_time in sink_times
    )
    expected = np.stack([result[0] for result in references])
    component_order = tuple(references[0][1])
    weights = {name: weight for name, _, weight in SIGMA_BASIS}

    def weighted_sector(components: Mapping[tuple[str, str], ReferenceResult], sector: str) -> np.ndarray:
        return sum(
            (
                weights[key[0]] * weights[key[1]]
                * getattr(components[key], sector)
                for key in component_order
            ),
            np.zeros((4, 4), dtype=np.complex128),
        )

    cross_time = np.stack([weighted_sector(components, "cross_time") for _, components in references])
    same_timeslice = np.stack([
        weighted_sector(components, "same_timeslice") for _, components in references
    ])
    vacuum_vacuum = np.stack([
        sum(
            (
                weights[key[0]] * weights[key[1]] * term.value
                for key in component_order
                for term in components[key].terms
                if (4, 3) in term.pairing and (6, 5) in term.pairing
            ),
            np.zeros((4, 4), dtype=np.complex128),
        )
        for _, components in references
    ])
    term_count = sum(len(result.terms) for result in references[0][1].values())
    local_count = sum(
        term.same_timeslice
        for result in references[0][1].values()
        for term in result.terms
    )
    with h5py.File(output_dir / "independent_reference.h5", "w") as handle:
        handle.attrs["construction"] = "independent_ordered_grassmann_pairings"
        handle.attrs["source_time"] = source_time
        handle.attrs["sink_times"] = sink_times
        handle.attrs["mode_size"] = mode_size
        handle.attrs["vacuum_subtracted"] = False
        handle.attrs["component_order"] = json.dumps(component_order)
        handle.create_dataset("values", data=expected)
        handle.create_dataset("cross_time", data=cross_time)
        handle.create_dataset("same_timeslice", data=same_timeslice)
        handle.create_dataset("vacuum_vacuum", data=vacuum_vacuum)
        handle.create_dataset(
            "component_values",
            data=np.stack([
                [components[key].total for key in component_order]
                for _, components in references
            ]),
        )

    from concord_lqcd.job import run_job, write_job_plan

    spec = _job_spec(
        paths=paths,
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
        raise RuntimeError("N-sigma ConCorD job did not complete all requested times")
    actual = _stored_values(spec.output.file, len(sink_times))
    difference = np.abs(actual - expected)
    max_absolute = float(np.max(difference))
    scale = max(float(np.max(np.abs(expected))), np.finfo(np.float64).tiny)
    time_scales = np.max(np.abs(expected), axis=(1, 2))
    time_errors = np.max(difference, axis=(1, 2))
    time_scaled = np.divide(
        time_errors,
        time_scales,
        out=np.full_like(time_errors, np.inf),
        where=time_scales > 0,
    )
    report = {
        "passed": bool(
            np.allclose(actual, expected, rtol=rtol, atol=atol)
            and np.all(time_scales > atol)
            and np.all(time_scaled <= rtol)
        ),
        "max_absolute": max_absolute,
        "max_scaled": max_absolute / scale,
        "max_per_time_scaled": float(np.max(time_scaled)),
        "minimum_reference_magnitude": float(np.min(time_scales)),
        "max_reference_magnitude": float(np.max(np.abs(expected))),
        "max_cross_time_magnitude": float(np.max(np.abs(cross_time))),
        "max_same_timeslice_magnitude": float(np.max(np.abs(same_timeslice))),
        "max_vacuum_vacuum_magnitude": float(np.max(np.abs(vacuum_vacuum))),
        "vacuum_subtracted": False,
        "configuration": configuration,
        "source_time": source_time,
        "sink_times": sink_times,
        "mode_size": mode_size,
        "terms_per_time": term_count,
        "same_timeslice_terms_per_time": local_count,
        "source_perambulator": str(perambulator),
        "eigenvectors": str(eigenvectors),
        "concord_output": str(spec.output.file),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perambulator", type=Path, required=True)
    parser.add_argument("--down-perambulator", type=Path)
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
        down_perambulator=args.down_perambulator,
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
