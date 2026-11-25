#!/usr/bin/env python3
"""Independent NumPy check of multi-baryon two-point functions: NN 1S0, NN 3S1 and the triton.

Systems (P = eps (u^T C gamma_5 d) u, N = eps (u^T C gamma_5 d) d, each projected to its own momentum
at its own position; k is --momentum, default (0, 0, 1) in units of 2 pi / L):

  nn_1s0  2x2 matrix of [P(p) P(-p)] coupled with C gamma_5, p = 0 and p = k  (I = 1, 1S0);
  nn_3s1  2x2 matrix of {[P(p) N(-p)] - [N(p) P(-p)]} / sqrt(2) coupled with C gamma_3  (I = 0, 3S1);
  triton  P [N C gamma_5 N] at rest, open proton spin (2880 quark pairings);
  proton  one local proton at rest.

The reference imports nothing from ConCorD. An operator term is a coefficient, a list of local
baryons eps_abc (q1_a^T G q2_b) q3_c (each with its own momentum) and spin matrices that couple
the open q3 spins of two baryons. For every pair of sink and source terms the reference enumerates
all flavor-allowed pairings of the 3K quark slots, with the parity of the permutation as sign (the
direct pairing has sign +1), and contracts with opt_einsum:
  sink:   E_b(p)[m m m] G_b[a a] M[a a],   propagator P_f[m a; M r] for each sink slot,
  source: conj(E_b(p))[M M M] conj(G_b)[A A] conj(M)[A A],  gamma_4[r A] on each source spin.
These are the conventions of validate_delta_correlator.py (DP gammas, C = gamma_2 gamma_4); every
run also reproduces that tool's Delta++ reference for one baryon as a self-check. Elementals use
the phase exp(+i 2 pi p.x / L).
"""

from __future__ import annotations

import argparse
from itertools import permutations, product
import json
from math import sqrt
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np
import opt_einsum as oe


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools._independent_laph import (
    EIGENVECTOR_AXES,
    hdf5_text,
    read_eigenvector_times,
    read_propagators,
)
from tools.validate_delta_correlator import evaluate_reference as delta_pp_reference


SYSTEMS = ("proton", "nn_1s0", "nn_3s1", "triton")
REST = (0, 0, 0)


def dp_spin_matrices() -> dict[str, np.ndarray]:
    """C gamma_i, C gamma_5 and gamma_4 in the DP basis, C = gamma_2 gamma_4."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.diag([1, -1]).astype(np.complex128)
    zero = np.zeros((2, 2), dtype=np.complex128)
    spatial = [np.block([[zero, -1j * s], [1j * s, zero]]) for s in (sigma_x, sigma_y, sigma_z)]
    gamma4 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    charge_conjugation = spatial[1] @ gamma4
    matrices = {f"Cg{index + 1}": charge_conjugation @ spatial[index] for index in range(3)}
    matrices["Cg5"] = charge_conjugation @ (spatial[0] @ spatial[1] @ spatial[2] @ gamma4)
    matrices["g4"] = gamma4
    return matrices


SPIN = dp_spin_matrices()
EPSILON = np.zeros((3, 3, 3), dtype=np.complex128)
EPSILON[0, 1, 2] = EPSILON[1, 2, 0] = EPSILON[2, 0, 1] = 1
EPSILON[2, 1, 0] = EPSILON[0, 2, 1] = EPSILON[1, 0, 2] = -1


def momentum_elemental(vectors: np.ndarray, momentum: Sequence[int]) -> np.ndarray:
    """sum_x exp(+i 2 pi p.x / L) eps_abc V_a,i V_b,j V_c,k for one time slice, axes (z,y,x,color,mode)."""
    vectors = np.asarray(vectors)
    if vectors.ndim != 5 or vectors.shape[3] != 3:
        raise ValueError("one LapH eigenvector timeslice must have shape (z,y,x,3,Nv)")
    nz, ny, nx = vectors.shape[:3]
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    px, py, pz = momentum
    phase = np.exp(2j * np.pi * (px * x / nx + py * y / ny + pz * z / nz)).reshape(-1)
    modes = vectors.reshape(-1, 3, vectors.shape[-1]).astype(np.complex128)
    out = np.zeros((modes.shape[-1],) * 3, dtype=np.complex128)
    for start in range(0, len(modes), 256):
        block, block_phase = modes[start:start + 256], phase[start:start + 256]
        out += oe.contract("s,sai,sbj,sck,abc->ijk", block_phase, block, block, block, EPSILON, optimize="greedy")
    return out


# A baryon is (diquark matrix, flavors of q1 q2 q3, momentum); a coupling is (matrix, baryon i, baryon j)
# acting on the open q3 spins; a term is (coefficient, baryons, couplings).
Baryon = tuple[str, str, tuple[int, int, int]]
Coupling = tuple[str, int, int]
Term = tuple[float, list[Baryon], list[Coupling]]


def baryon(gamma: str, flavors: str, momentum: Sequence[int] = REST) -> Baryon:
    return (gamma, flavors, tuple(momentum))


def term(coefficient: float, baryons: list[Baryon], couplings: list[Coupling]) -> Term:
    return (coefficient, baryons, couplings)


def pairings(sink_flavors: str, source_flavors: str) -> Iterator[tuple[list[int], int]]:
    """All bijections sink slot -> source slot with equal flavors, with their permutation parity."""
    slots = {}
    for flavor in set(sink_flavors):
        sink_slots = [index for index, value in enumerate(sink_flavors) if value == flavor]
        source_slots = [index for index, value in enumerate(source_flavors) if value == flavor]
        if len(sink_slots) != len(source_slots):
            return
        slots[flavor] = (sink_slots, source_slots)
    flavors = sorted(slots)
    for choice in product(*(permutations(slots[flavor][1]) for flavor in flavors)):
        permutation = [0] * len(sink_flavors)
        for flavor, targets in zip(flavors, choice):
            for sink_slot, target in zip(slots[flavor][0], targets):
                permutation[sink_slot] = target
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(len(permutation))
            for right in range(left + 1, len(permutation))
        )
        yield permutation, (-1 if inversions % 2 else 1)


def term_pair(
    sink: Term,
    source: Term,
    sink_elementals: Mapping[tuple[int, int, int], np.ndarray],
    source_elementals: Mapping[tuple[int, int, int], np.ndarray],
    propagators: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Signed sum over all flavor-allowed pairings of one sink term with one source term.

    ``source_elementals`` are unconjugated; ``propagators[f]`` has shape (Nv, 4, Nv, 4) as returned
    by ``read_propagators``. The result has the open sink spins, then the open source spins, as axes.
    """
    sink_coefficient, sink_baryons, sink_couplings = sink
    source_coefficient, source_baryons, source_couplings = source
    count = 3 * len(sink_baryons)
    symbols = iter(oe.get_symbol(index) for index in range(10_000))
    m = [next(symbols) for _ in range(count)]  # sink modes
    a = [next(symbols) for _ in range(count)]  # sink spins
    M = [next(symbols) for _ in range(count)]  # source modes
    r = [next(symbols) for _ in range(count)]  # source spins of the propagators
    A = [next(symbols) for _ in range(count)]  # source spins of the operator
    fixed_terms, fixed_operands = [], []
    for index, (gamma, _, momentum) in enumerate(sink_baryons):
        fixed_terms += [m[3 * index] + m[3 * index + 1] + m[3 * index + 2], a[3 * index] + a[3 * index + 1]]
        fixed_operands += [sink_elementals[momentum], SPIN[gamma]]
    for gamma, first, second in sink_couplings:
        fixed_terms.append(a[3 * first + 2] + a[3 * second + 2])
        fixed_operands.append(SPIN[gamma])
    for index, (gamma, _, momentum) in enumerate(source_baryons):
        fixed_terms += [M[3 * index] + M[3 * index + 1] + M[3 * index + 2], A[3 * index] + A[3 * index + 1]]
        fixed_operands += [source_elementals[momentum].conj(), SPIN[gamma].conj()]
    for gamma, first, second in source_couplings:
        fixed_terms.append(A[3 * first + 2] + A[3 * second + 2])
        fixed_operands.append(SPIN[gamma].conj())
    for slot in range(count):
        fixed_terms.append(r[slot] + A[slot])
        fixed_operands.append(SPIN["g4"])
    open_sink = [
        a[3 * index + 2] for index in range(len(sink_baryons))
        if all(index not in (first, second) for _, first, second in sink_couplings)
    ]
    open_source = [
        A[3 * index + 2] for index in range(len(source_baryons))
        if all(index not in (first, second) for _, first, second in source_couplings)
    ]
    output = "".join(open_sink + open_source)
    sink_flavors = "".join(flavor for _, flavors, _ in sink_baryons for flavor in flavors)
    source_flavors = "".join(flavor for _, flavors, _ in source_baryons for flavor in flavors)
    total = 0
    for permutation, sign in pairings(sink_flavors, source_flavors):
        terms = fixed_terms + [
            m[slot] + a[slot] + M[permutation[slot]] + r[permutation[slot]] for slot in range(count)
        ]
        operands = fixed_operands + [propagators[sink_flavors[slot]] for slot in range(count)]
        total = total + sign * oe.contract(",".join(terms) + "->" + output, *operands, optimize="greedy")
    return sink_coefficient * np.conj(source_coefficient) * total


def correlator(
    sink_operator: list[Term],
    source_operator: list[Term],
    sink_elementals: Mapping[tuple[int, int, int], np.ndarray],
    source_elementals: Mapping[tuple[int, int, int], np.ndarray],
    propagators: Mapping[str, np.ndarray],
) -> np.ndarray:
    return sum(
        term_pair(sink, source, sink_elementals, source_elementals, propagators)
        for sink in sink_operator
        for source in source_operator
    )


def pairing_count(sink_operator: list[Term], source_operator: list[Term]) -> int:
    """Number of signed quark pairings summed by ``correlator``."""
    return sum(
        sum(1 for _ in pairings(
            "".join(flavor for _, flavors, _ in sink[1] for flavor in flavors),
            "".join(flavor for _, flavors, _ in source[1] for flavor in flavors),
        ))
        for sink in sink_operator
        for source in source_operator
    )


def operator_terms(system: str, momentum: Sequence[int] = (0, 0, 1)) -> dict[str, list[Term]]:
    """Operator basis of ``system``: index -> list of terms."""
    p1 = tuple(momentum)

    def neg(p):
        return tuple(-x for x in p)

    def P(p=REST):
        return baryon("Cg5", "udu", p)

    def N(p=REST):
        return baryon("Cg5", "udd", p)

    if system == "proton":
        return {"0": [term(1.0, [P()], [])]}
    if system == "nn_1s0":
        return {str(i): [term(1.0, [P(p), P(neg(p))], [("Cg5", 0, 1)])] for i, p in enumerate((REST, p1))}
    if system == "nn_3s1":
        return {
            str(i): [
                term(1 / sqrt(2), [P(p), N(neg(p))], [("Cg3", 0, 1)]),
                term(-1 / sqrt(2), [N(p), P(neg(p))], [("Cg3", 0, 1)]),
            ]
            for i, p in enumerate((REST, p1))
        }
    if system == "triton":
        return {"0": [term(1.0, [P(), N(), N()], [("Cg5", 1, 2)])]}
    raise ValueError(f"unknown system {system!r}; choose one of {SYSTEMS}")


OPERATOR_SOURCE = '''\
from itertools import count
from math import sqrt

from concord_lqcd.core import Correlator, Operator, d, epsilon, spin_matrix, u

_uid = count()


def _labels(n, prefix):
    k = next(_uid)
    return [f"{prefix}{k}_{i}" for i in range(n)]


def nucleon(third, pos, p=(0, 0, 0), open_spin=None):
    """eps_abc (u_a^T C g5 d_b) q3_c with q3 = u (proton) or d (neutron)."""
    a, b, c = _labels(3, "c")
    al, be = _labels(2, "s")
    spin = open_spin or _labels(1, "o")[0]
    return Operator(
        epsilon(a, b, c) * spin_matrix("Cg5", al, be) * u(pos, a, al) * d(pos, b, be) * third(pos, c, spin),
        momentum=p,
    )


def proton(pos, p=(0, 0, 0), open_spin=None):
    return nucleon(u, pos, p, open_spin)


def neutron(pos, p=(0, 0, 0), open_spin=None):
    return nucleon(d, pos, p, open_spin)


def build(system, momentum):
    p0, p1 = (0, 0, 0), tuple(momentum)

    def neg(p):
        return tuple(-x for x in p)

    if system == "proton":
        ops = {"0": proton("x", open_spin="sN")}
    elif system == "nn_1s0":
        ops = {str(i): proton("x", p, open_spin="s1") * proton("y", neg(p), open_spin="s2")
               * spin_matrix("Cg5", "s1", "s2") for i, p in enumerate((p0, p1))}
    elif system == "nn_3s1":
        ops = {str(i): (proton("x", p, open_spin="s1") * neutron("y", neg(p), open_spin="s2")
                        * spin_matrix("Cg3", "s1", "s2")
                        - neutron("x", p, open_spin="s1") * proton("y", neg(p), open_spin="s2")
                        * spin_matrix("Cg3", "s1", "s2")) / sqrt(2) for i, p in enumerate((p0, p1))}
    elif system == "triton":
        ops = {"0": proton("x", open_spin="sN") * neutron("y", open_spin="s1")
               * neutron("z", open_spin="s2") * spin_matrix("Cg5", "s1", "s2")}
    else:
        raise ValueError(system)
    return {f"{system}_{i}_{j}": Correlator(ops[i], ops[j].dagger()) for i in ops for j in ops}
'''


def _job_spec(
    *,
    system: str,
    momentum: tuple[int, int, int],
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
        "job": {"name": f"multibaryon_{system}_cfg{configuration}"},
        "operators": {
            "file": str(output_dir / "operators.py"),
            "factory": "build",
            "parameters": {"system": system, "momentum": list(momentum)},
        },
        "plan": {"file": str(output_dir / "multibaryon.concord")},
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


def _stored_values(
    path: Path, sink_times: tuple[int, ...], shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """ConCorD correlators reshaped to the reference shapes, in ``sink_times`` order."""
    values: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        stored_sinks = [
            json.loads(hdf5_text(entry))["2"][0] for entry in handle["work/operator_times_json"][()]
        ]
        if sorted(stored_sinks) != sorted(sink_times):
            raise ValueError(f"ConCorD output has sink times {stored_sinks}, expected {list(sink_times)}")
        for group in handle["correlators"].values():
            identifier = hdf5_text(group.attrs.get("correlator_id", ""))
            if identifier not in shapes or identifier in values:
                raise ValueError(f"unexpected or duplicate correlator {identifier!r}")
            data = np.asarray(group["values"], dtype=np.complex128)
            if data.shape[:2] != (1, len(sink_times)) or data[0, 0].size != int(np.prod(shapes[identifier])):
                raise ValueError(f"invalid output shape {data.shape} for {identifier}")
            values[identifier] = np.stack([
                data[0, stored_sinks.index(time)].reshape(shapes[identifier]) for time in sink_times
            ])
    missing = sorted(set(shapes) - set(values))
    if missing:
        raise ValueError(f"ConCorD output lacks correlators {missing}")
    return values


def validate(
    *,
    perambulator: Path,
    eigenvectors: Path,
    output_dir: Path,
    system: str,
    mode_size: int = 8,
    source_time: int = 0,
    sink_times: tuple[int, ...] = (1, 5),
    configuration: int = 1161,
    down_perambulator: Path | None = None,
    eigenvector_dataset: str = "/eigenvectors",
    momentum: tuple[int, int, int] = (0, 0, 1),
    contraction_strategy: str = "greedy",
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> dict[str, Any]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose one of {SYSTEMS}")
    if mode_size < 3 or not sink_times or len(set(sink_times)) != len(sink_times):
        raise ValueError("choose Nv >= 3 and distinct sink times")
    if any(time == source_time for time in sink_times):
        raise ValueError("sink times must differ from the source time")
    momentum = tuple(int(component) for component in momentum)
    if len(momentum) != 3 or (system.startswith("nn_") and momentum == REST):
        raise ValueError("the NN systems need a nonzero three-momentum")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (
        output_dir / "independent_reference.h5",
        output_dir / "operators.py",
        output_dir / "multibaryon.concord",
        output_dir / "concord_result.h5",
        output_dir / "report.json",
    )
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite validation artifacts: " + ", ".join(existing))

    paths = {"u": perambulator, "d": down_perambulator or perambulator}
    times = (source_time,) + sink_times
    slices, lattice = read_eigenvector_times(eigenvectors, eigenvector_dataset, times, mode_size)
    blocks = read_propagators(paths, times, mode_size, lattice[0])
    operators = operator_terms(system, momentum)
    momenta = {REST} | {
        p for operator in operators.values() for _, baryons, _ in operator for _, _, p in baryons
    }
    elementals = {(t, p): momentum_elemental(slices[t], p) for t in times for p in momenta}

    def sink_elementals(sink_time: int) -> dict[tuple[int, int, int], np.ndarray]:
        return {p: elementals[(sink_time, p)] for p in momenta}

    source_elementals = sink_elementals(source_time)
    started = perf_counter()
    expected: dict[str, np.ndarray] = {}
    pairings_per_correlator: dict[str, int] = {}
    for sink in operators:
        for source in operators:
            name = f"{system}_{sink}_{source}"
            expected[name] = np.stack([
                np.asarray(correlator(
                    operators[sink],
                    operators[source],
                    sink_elementals(sink_time),
                    source_elementals,
                    {flavor: blocks[(flavor, sink_time, source_time)] for flavor in paths},
                ))
                for sink_time in sink_times
            ])
            pairings_per_correlator[name] = pairing_count(operators[sink], operators[source])
    reference_seconds = perf_counter() - started

    # Self-check: the generic enumeration on one Delta++ against the Delta++ tool's reference.
    delta_pp = [term(1.0, [baryon("Cg1", "uuu")], [])]
    selfcheck = 0.0
    for sink_time in sink_times:
        tool = delta_pp_reference(
            elementals[(sink_time, REST)], elementals[(source_time, REST)].conj(),
            blocks[("u", sink_time, source_time)],
        )[0, 0]
        generic = correlator(
            delta_pp, delta_pp, {REST: elementals[(sink_time, REST)]}, {REST: elementals[(source_time, REST)]},
            {"u": blocks[("u", sink_time, source_time)]},
        )
        scale = max(float(np.max(np.abs(tool))), np.finfo(float).tiny)
        selfcheck = max(selfcheck, float(np.max(np.abs(generic - tool))) / scale)

    with h5py.File(output_dir / "independent_reference.h5", "w") as handle:
        handle.attrs["construction"] = "independent_signed_quark_pairings_per_operator_term"
        handle.attrs["system"] = system
        handle.attrs["momentum"] = momentum
        handle.attrs["source_time"] = source_time
        handle.attrs["sink_times"] = sink_times
        handle.attrs["mode_size"] = mode_size
        for name, value in expected.items():
            dataset = handle.create_dataset(name, data=value)
            dataset.attrs["pairings"] = pairings_per_correlator[name]

    from concord_lqcd.job import run_job, write_job_plan

    spec = _job_spec(
        system=system,
        momentum=momentum,
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
    actual = _stored_values(
        spec.output.file, sink_times, {name: value.shape[1:] for name, value in expected.items()}
    )

    per_correlator: dict[str, dict[str, float]] = {}
    per_sink = np.zeros(len(sink_times))
    max_absolute = 0.0
    minimum_scale = np.inf
    maximum_scale = 0.0
    for name, reference in expected.items():
        flat_reference = reference.reshape(len(sink_times), -1)
        difference = np.abs(actual[name].reshape(len(sink_times), -1) - flat_reference)
        time_scales = np.max(np.abs(flat_reference), axis=1)
        time_errors = np.max(difference, axis=1)
        time_scaled = np.divide(
            time_errors, time_scales, out=np.full_like(time_errors, np.inf), where=time_scales > 0
        )
        per_sink = np.maximum(per_sink, time_scaled)
        max_absolute = max(max_absolute, float(np.max(difference)))
        minimum_scale = min(minimum_scale, float(np.min(time_scales)))
        maximum_scale = max(maximum_scale, float(np.max(time_scales)))
        per_correlator[name] = {
            "max_per_time_scaled": float(np.max(time_scaled)),
            "minimum_reference_magnitude": float(np.min(time_scales)),
            "pairings": pairings_per_correlator[name],
        }
    max_scaled = float(np.max(per_sink))
    report = {
        "passed": bool(max_scaled <= rtol and minimum_scale > atol and selfcheck <= rtol),
        "system": system,
        "max_absolute": max_absolute,
        "max_per_time_scaled": max_scaled,
        "max_per_time_scaled_by_sink": {str(t): float(value) for t, value in zip(sink_times, per_sink)},
        "max_reference_magnitude": maximum_scale,
        "minimum_reference_magnitude": minimum_scale,
        "reference_selfcheck_vs_delta_pp_tool": selfcheck,
        "reference_seconds": round(reference_seconds, 1),
        "configuration": configuration,
        "source_time": source_time,
        "sink_times": sink_times,
        "mode_size": mode_size,
        "contraction_strategy": contraction_strategy,
        "momentum": momentum if system.startswith("nn_") else None,
        "correlators": len(expected),
        "per_correlator": per_correlator,
        "perambulators": {flavor: str(path) for flavor, path in paths.items()},
        "eigenvectors": str(eigenvectors),
        "concord_output": str(spec.output.file),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--perambulator", type=Path, required=True,
                        help="u perambulator; also used for d unless --down-perambulator is given")
    parser.add_argument("--down-perambulator", type=Path)
    parser.add_argument("--eigenvectors", type=Path, required=True)
    parser.add_argument("--eigenvector-dataset", default="/eigenvectors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", type=int, default=1161)
    parser.add_argument("--mode-size", type=int, default=8)
    parser.add_argument("--source-time", type=int, default=0)
    parser.add_argument("--sink-times", type=int, nargs="+", default=(1, 5))
    parser.add_argument("--momentum", type=int, nargs=3, default=(0, 0, 1), metavar=("PX", "PY", "PZ"),
                        help="back-to-back momentum of the second NN operator, in units of 2 pi / L")
    parser.add_argument("--contraction-strategy", default="greedy",
                        help="ConCorD compile.contraction.strategy (greedy is deterministic and compiles "
                             "multi-quark systems fastest)")
    parser.add_argument("--rtol", type=float, default=1e-10,
                        help="bound on max|C - C_ref| / max|C_ref| per correlator and sink time")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="every per-time reference magnitude must exceed this (multi-baryon "
                             "correlators can be far below 1e-12 in lattice units)")
    args = parser.parse_args()
    report = validate(
        system=args.system,
        perambulator=args.perambulator,
        down_perambulator=args.down_perambulator,
        eigenvectors=args.eigenvectors,
        eigenvector_dataset=args.eigenvector_dataset,
        output_dir=args.output_dir,
        configuration=args.configuration,
        mode_size=args.mode_size,
        source_time=args.source_time,
        sink_times=tuple(args.sink_times),
        momentum=tuple(args.momentum),
        contraction_strategy=args.contraction_strategy,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_correlator"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
