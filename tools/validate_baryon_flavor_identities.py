#!/usr/bin/env python3
"""Independent NumPy check of Delta+, Lambda, proton and Delta++ at rest, and their flavor identities.

With D_G(q1, q2) q3 = eps_abc (q1_a^T G q2_b) q3_c, the operators are

  Delta++_i = D_{C gamma_i}(u, u) u,
  Delta+_i  = [2 D_{C gamma_i}(u, d) u + D_{C gamma_i}(u, u) d] / sqrt(3),
  p         = D_{C gamma_5}(u, d) u,
  Lambda    = [2 D_{C gamma_5}(u, d) s + D_{C gamma_5}(u, s) d - D_{C gamma_5}(d, s) u] / sqrt(6),

plus two negative controls, Delta+ with the first coefficient 1/sqrt(3) and Lambda with the last sign +.
All 30 two-point functions (the 3x3 Delta matrices of the three Delta families, p, Lambda and the
Lambda control) are compared with ConCorD.

The reference enumerates the quark pairings of every pair of sink and source terms itself, with the
permutation sign, and contracts LapH elementals, perambulators and spin matrices with opt_einsum. It
imports nothing from ConCorD. Its conventions (DP gammas, C = gamma_2 gamma_4, gamma_4 on the source
spins, conjugated source elemental) are those of validate_delta_correlator.py, whose Delta++
reference it reproduces as a self-check.

When u, d and s are bound to the same perambulator file, Delta+_i = Delta++_i and Lambda = p hold
exactly (D_{C gamma_i} is flavor symmetric, D_{C gamma_5} antisymmetric), and the tool also checks
these identities on the ConCorD output; the two controls must then differ at order one.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from math import sqrt
from pathlib import Path
import sys
from typing import Any, Mapping

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
from tools.validate_delta_correlator import evaluate_reference as delta_pp_reference


# A negative control passes when it differs from its partner by more than this (relative).
CONTROL_MINIMUM = 1e-3


def dp_spin_matrices() -> dict[str, np.ndarray]:
    """C gamma_i, C gamma_5 and gamma_4 in the DP basis, C = gamma_2 gamma_4."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.diag([1, -1]).astype(np.complex128)
    zero = np.zeros((2, 2), dtype=np.complex128)
    spatial = [np.block([[zero, -1j * s], [1j * s, zero]]) for s in (sigma_x, sigma_y, sigma_z)]
    gamma4 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    charge_conjugation = spatial[1] @ gamma4
    # An overall sign of gamma_5 cancels in <O Obar>.
    gamma5 = spatial[0] @ spatial[1] @ spatial[2] @ gamma4
    matrices = {f"Cg{index + 1}": charge_conjugation @ spatial[index] for index in range(3)}
    matrices["Cg5"] = charge_conjugation @ gamma5
    matrices["g4"] = gamma4
    return matrices


SPIN = dp_spin_matrices()

# One operator term: (coefficient, diquark matrix, flavors of q1 q2 q3).
Term = tuple[float, str, str]


def operator_terms() -> dict[str, list[Term]]:
    operators: dict[str, list[Term]] = {}
    for direction in (1, 2, 3):
        diquark = f"Cg{direction}"
        operators[f"delta_pp_{direction}"] = [(1.0, diquark, "uuu")]
        operators[f"delta_p_{direction}"] = [(2 / sqrt(3), diquark, "udu"), (1 / sqrt(3), diquark, "uud")]
        operators[f"delta_p_control_{direction}"] = [
            (1 / sqrt(3), diquark, "udu"), (1 / sqrt(3), diquark, "uud")
        ]
    operators["proton"] = [(1.0, "Cg5", "udu")]
    operators["lambda"] = [
        (2 / sqrt(6), "Cg5", "uds"), (1 / sqrt(6), "Cg5", "usd"), (-1 / sqrt(6), "Cg5", "dsu")
    ]
    operators["lambda_control"] = [
        (2 / sqrt(6), "Cg5", "uds"), (1 / sqrt(6), "Cg5", "usd"), (1 / sqrt(6), "Cg5", "dsu")
    ]
    return operators


DELTA_FAMILIES = ("delta_pp", "delta_p", "delta_p_control")
SCALAR_FAMILIES = ("proton", "lambda", "lambda_control")


def correlator_pairs() -> dict[str, tuple[str, str, str]]:
    """Correlator name -> (family, sink operator, source operator)."""
    pairs = {}
    for family in DELTA_FAMILIES:
        for sink in (1, 2, 3):
            for source in (1, 2, 3):
                pairs[f"{family}_{sink}_{source}"] = (family, f"{family}_{sink}", f"{family}_{source}")
    for name in SCALAR_FAMILIES:
        pairs[name] = (name, name, name)
    return pairs


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def two_point(
    sink_terms: list[Term],
    source_terms: list[Term],
    sink_elemental: np.ndarray,
    source_elemental_conj: np.ndarray,
    propagators: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Sum over sink terms, source terms and flavor-allowed pairings of eps (q1 G q2) q3.

    ``propagators[f]`` has shape (Nv, 4, Nv, 4) as returned by ``read_propagators``.
    Returns the (sink spin, source spin) matrix of the open third quark.
    """
    modes, spins = "lmn", "rsq"
    total = np.zeros((4, 4), dtype=np.complex128)
    for sink_coefficient, sink_diquark, sink_flavors in sink_terms:
        for source_coefficient, source_diquark, source_flavors in source_terms:
            for permutation in permutations(range(3)):
                if any(sink_flavors[slot] != source_flavors[permutation[slot]] for slot in range(3)):
                    continue
                paired = [modes[permutation[slot]] + spins[permutation[slot]] for slot in range(3)]
                equation = (
                    f"ijk,lmn,ab,BA,ia{paired[0]},jb{paired[1]},kd{paired[2]},rA,sB,qD->dD"
                )
                weight = sink_coefficient * np.conj(source_coefficient) * _permutation_sign(permutation)
                total += weight * oe.contract(
                    equation,
                    sink_elemental,
                    source_elemental_conj,
                    SPIN[sink_diquark],
                    SPIN[source_diquark].conj().T,
                    propagators[sink_flavors[0]],
                    propagators[sink_flavors[1]],
                    propagators[sink_flavors[2]],
                    SPIN["g4"],
                    SPIN["g4"],
                    SPIN["g4"],
                    optimize="greedy",
                )
    return total


OPERATOR_SOURCE = '''\
from math import sqrt

from concord_lqcd.core import Correlator, Operator, d, epsilon, s, spin_matrix, u


def _baryon(diquark, q1, q2, q3):
    return Operator(
        epsilon("a", "b", "c")
        * spin_matrix(diquark, "al", "be")
        * q1("x", "a", "al")
        * q2("x", "b", "be")
        * q3("x", "c", "de"),
        momentum=(0, 0, 0),
    )


def build():
    ops = {}
    for i in (1, 2, 3):
        g = f"Cg{i}"
        ops[f"delta_pp_{i}"] = _baryon(g, u, u, u)
        ops[f"delta_p_{i}"] = (2 / sqrt(3)) * _baryon(g, u, d, u) + (1 / sqrt(3)) * _baryon(g, u, u, d)
        ops[f"delta_p_control_{i}"] = (1 / sqrt(3)) * _baryon(g, u, d, u) + (1 / sqrt(3)) * _baryon(g, u, u, d)
    ops["proton"] = _baryon("Cg5", u, d, u)
    ops["lambda"] = (2 * _baryon("Cg5", u, d, s) + _baryon("Cg5", u, s, d) - _baryon("Cg5", d, s, u)) / sqrt(6)
    ops["lambda_control"] = (2 * _baryon("Cg5", u, d, s) + _baryon("Cg5", u, s, d) + _baryon("Cg5", d, s, u)) / sqrt(6)

    correlators = {}
    for family in ("delta_pp", "delta_p", "delta_p_control"):
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                correlators[f"{family}_{i}_{j}"] = Correlator(ops[f"{family}_{i}"], ops[f"{family}_{j}"].dagger())
    for name in ("proton", "lambda", "lambda_control"):
        correlators[name] = Correlator(ops[name], ops[name].dagger())
    return correlators
'''


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
    contraction_strategy: str,
):
    from concord_lqcd.job import JobSpec

    (output_dir / "operators.py").write_text(OPERATOR_SOURCE, encoding="utf-8")
    temporal_extent, x_extent, y_extent, z_extent = lattice
    return JobSpec.model_validate({
        "schema_version": 1,
        "job": {"name": f"baryon_flavor_identities_cfg{configuration}"},
        "operators": {"file": str(output_dir / "operators.py"), "factory": "build"},
        "plan": {"file": str(output_dir / "baryon_flavor.concord")},
        "lattice": {
            "spatial_extent": [x_extent, y_extent, z_extent],
            "temporal_extent": temporal_extent,
        },
        "compile": {
            "mode_size": mode_size,
            "spin_basis": "DP",
            "scalar_type": "complex128",
            "contraction": {"strategy": contraction_strategy},
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


def _stored_values(path: Path, sink_times: tuple[int, ...], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """ConCorD correlators as (sink time, sink spin, source spin) arrays, in ``sink_times`` order."""
    values: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        stored_sinks = [
            json.loads(hdf5_text(entry))["2"][0] for entry in handle["work/operator_times_json"][()]
        ]
        if sorted(stored_sinks) != sorted(sink_times):
            raise ValueError(f"ConCorD output has sink times {stored_sinks}, expected {list(sink_times)}")
        for group in handle["correlators"].values():
            identifier = hdf5_text(group.attrs.get("correlator_id", ""))
            if identifier not in names or identifier in values:
                raise ValueError(f"unexpected or duplicate correlator {identifier!r}")
            data = np.asarray(group["values"], dtype=np.complex128)
            if data.shape != (1, len(sink_times), 4, 4):
                raise ValueError(f"invalid output shape {data.shape} for {identifier}")
            values[identifier] = np.stack([data[0, stored_sinks.index(time)] for time in sink_times])
    missing = sorted(set(names) - set(values))
    if missing:
        raise ValueError(f"ConCorD output lacks correlators {missing}")
    return values


def _global_relative(actual: np.ndarray, target: np.ndarray) -> float:
    """max |actual - target| / max |target| over all sink times and spins."""
    return float(np.max(np.abs(actual - target)) / max(float(np.max(np.abs(target))), np.finfo(float).tiny))


def _identities(actual: Mapping[str, np.ndarray]) -> dict[str, float]:
    pairs = [(sink, source) for sink in (1, 2, 3) for source in (1, 2, 3)]
    return {
        "delta_p_vs_delta_pp": max(
            _global_relative(actual[f"delta_p_{i}_{j}"], actual[f"delta_pp_{i}_{j}"]) for i, j in pairs
        ),
        "lambda_vs_proton": _global_relative(actual["lambda"], actual["proton"]),
        "delta_p_control_vs_delta_pp": min(
            _global_relative(actual[f"delta_p_control_{i}_{j}"], actual[f"delta_pp_{i}_{j}"]) for i, j in pairs
        ),
        "lambda_control_vs_proton": _global_relative(actual["lambda_control"], actual["proton"]),
    }


def validate(
    *,
    perambulator: Path,
    eigenvectors: Path,
    output_dir: Path,
    mode_size: int = 8,
    source_time: int = 0,
    sink_times: tuple[int, ...] = (1, 2, 7, 23),
    configuration: int = 1161,
    down_perambulator: Path | None = None,
    strange_perambulator: Path | None = None,
    eigenvector_dataset: str = "/eigenvectors",
    contraction_strategy: str = "greedy",
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
        output_dir / "operators.py",
        output_dir / "baryon_flavor.concord",
        output_dir / "concord_result.h5",
        output_dir / "report.json",
    )
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite validation artifacts: " + ", ".join(existing))

    paths = {
        "u": perambulator,
        "d": down_perambulator or perambulator,
        "s": strange_perambulator or perambulator,
    }
    one_perambulator = len({path.resolve() for path in paths.values()}) == 1
    times = (source_time,) + sink_times
    slices, lattice = read_eigenvector_times(eigenvectors, eigenvector_dataset, times, mode_size)
    blocks = read_propagators(paths, times, mode_size, lattice[0])
    elementals = {time: baryon_elemental(slices[time], mode_size) for time in times}
    source_conj = elementals[source_time].conj()

    operators = operator_terms()
    pairs = correlator_pairs()
    names = tuple(pairs)
    expected: dict[str, np.ndarray] = {}
    for name, (_, sink, source) in pairs.items():
        expected[name] = np.stack([
            two_point(
                operators[sink],
                operators[source],
                elementals[sink_time],
                source_conj,
                {flavor: blocks[(flavor, sink_time, source_time)] for flavor in paths},
            )
            for sink_time in sink_times
        ])

    # Self-check: the generic enumeration against the Delta++ tool's six-permutation reference.
    selfcheck = 0.0
    for index, sink_time in enumerate(sink_times):
        delta_pp = delta_pp_reference(
            elementals[sink_time], source_conj, blocks[("u", sink_time, source_time)]
        )
        scale = max(float(np.max(np.abs(delta_pp))), np.finfo(float).tiny)
        for sink in (1, 2, 3):
            for source in (1, 2, 3):
                selfcheck = max(selfcheck, float(np.max(np.abs(
                    expected[f"delta_pp_{sink}_{source}"][index] - delta_pp[sink - 1, source - 1]
                ))) / scale)

    with h5py.File(output_dir / "independent_reference.h5", "w") as handle:
        handle.attrs["construction"] = "independent_signed_quark_pairings_per_operator_term"
        handle.attrs["source_time"] = source_time
        handle.attrs["sink_times"] = sink_times
        handle.attrs["mode_size"] = mode_size
        handle.attrs["axes"] = json.dumps(["sink_time", "sink_spin", "source_spin"])
        for name, value in expected.items():
            handle.create_dataset(name, data=value)

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
        contraction_strategy=contraction_strategy,
    )
    plan = write_job_plan(spec)
    summary = run_job(spec, plan)
    if not summary.all_complete or summary.executed_work_units != len(sink_times):
        raise RuntimeError("ConCorD job did not complete all requested correlators and times")
    actual = _stored_values(spec.output.file, sink_times, names)

    per_correlator: dict[str, dict[str, float]] = {}
    by_family: dict[str, float] = {}
    all_close = True
    max_absolute = 0.0
    minimum_scale = np.inf
    maximum_scale = 0.0
    for name, (family, _, _) in pairs.items():
        reference = expected[name]
        difference = np.abs(actual[name] - reference)
        time_scales = np.max(np.abs(reference), axis=(1, 2))
        time_errors = np.max(difference, axis=(1, 2))
        time_scaled = np.divide(
            time_errors, time_scales, out=np.full_like(time_errors, np.inf), where=time_scales > 0
        )
        all_close = all_close and bool(np.allclose(actual[name], reference, rtol=rtol, atol=atol))
        max_absolute = max(max_absolute, float(np.max(difference)))
        minimum_scale = min(minimum_scale, float(np.min(time_scales)))
        maximum_scale = max(maximum_scale, float(np.max(time_scales)))
        per_correlator[name] = {
            "max_per_time_scaled": float(np.max(time_scaled)),
            "minimum_reference_magnitude": float(np.min(time_scales)),
        }
        by_family[family] = max(by_family.get(family, 0.0), float(np.max(time_scaled)))
    max_scaled = max(row["max_per_time_scaled"] for row in per_correlator.values())

    identities = _identities(actual) if one_perambulator else None
    identities_pass = identities is None or (
        identities["delta_p_vs_delta_pp"] <= rtol
        and identities["lambda_vs_proton"] <= rtol
        and identities["delta_p_control_vs_delta_pp"] > CONTROL_MINIMUM
        and identities["lambda_control_vs_proton"] > CONTROL_MINIMUM
    )
    report = {
        "passed": bool(
            all_close
            and minimum_scale > atol
            and max_scaled <= rtol
            and selfcheck <= rtol
            and identities_pass
        ),
        "max_absolute": max_absolute,
        "max_per_time_scaled": max_scaled,
        "max_per_time_scaled_by_family": by_family,
        "max_reference_magnitude": maximum_scale,
        "minimum_reference_magnitude": minimum_scale,
        "reference_selfcheck_vs_delta_pp_tool": selfcheck,
        "identities_checked": one_perambulator,
        "identities_concord": identities,
        "control_minimum": CONTROL_MINIMUM,
        "configuration": configuration,
        "source_time": source_time,
        "sink_times": sink_times,
        "mode_size": mode_size,
        "contraction_strategy": contraction_strategy,
        "correlators": len(names),
        "per_correlator": per_correlator,
        "perambulators": {flavor: str(path) for flavor, path in paths.items()},
        "eigenvectors": str(eigenvectors),
        "concord_output": str(spec.output.file),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perambulator", type=Path, required=True,
                        help="u perambulator; also used for d and s unless they are given")
    parser.add_argument("--down-perambulator", type=Path)
    parser.add_argument("--strange-perambulator", type=Path)
    parser.add_argument("--eigenvectors", type=Path, required=True)
    parser.add_argument("--eigenvector-dataset", default="/eigenvectors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", type=int, default=1161)
    parser.add_argument("--mode-size", type=int, default=8)
    parser.add_argument("--source-time", type=int, default=0)
    parser.add_argument("--sink-times", type=int, nargs="+", default=(1, 2, 7, 23))
    parser.add_argument("--contraction-strategy", default="greedy",
                        help="ConCorD compile.contraction.strategy (greedy is deterministic and compiles "
                             "multi-quark systems fastest)")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    report = validate(
        perambulator=args.perambulator,
        down_perambulator=args.down_perambulator,
        strange_perambulator=args.strange_perambulator,
        eigenvectors=args.eigenvectors,
        eigenvector_dataset=args.eigenvector_dataset,
        output_dir=args.output_dir,
        configuration=args.configuration,
        mode_size=args.mode_size,
        source_time=args.source_time,
        sink_times=tuple(args.sink_times),
        contraction_strategy=args.contraction_strategy,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_correlator"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
