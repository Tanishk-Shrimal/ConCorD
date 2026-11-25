"""Test helper: deterministic analytical plan costs for representative correlators.

The fixtures declare the shared orthonormal LapH basis used by built-in jobs.
This is a compiler benchmark, not a wall-clock or device benchmark. It
materializes no lattice tensors and executes no numerical contractions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.core import (
    Correlator,
    CorrMatrix,
    Operator,
    build_baryon_operator,
    c,
    cbar,
    d,
    dbar,
    gamma5,
    reset_registry,
    u,
    ubar,
)


PLAN_COST_BASELINE_VERSION = 1


def _meson_correlator() -> Correlator:
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * u("y", "a", "s2")
    )
    source = Operator(
        ubar("x", "b", "s3")
        * gamma5("s3", "s4")
        * c("x", "b", "s4")
    )
    return Correlator(sink, source)


def _baryon_correlator() -> Correlator:
    reset_registry()
    baryon = build_baryon_operator(
        ["l", "l", "s"],
        "Cg5",
        momentum=(0, 0, 0),
    )
    return CorrMatrix([baryon])[0, 0]


def _tetraquark_correlator() -> Correlator:
    reset_registry()
    sink = Operator(
        dbar("x", "a", "s1")
        * u("x", "b", "s2")
        * ubar("x", "c", "s3")
        * d("x", "d", "s4"),
        momenta={"x": (0, 0, 0)},
    )
    source = Operator(
        ubar("y", "a", "s5")
        * d("y", "b", "s6")
        * dbar("y", "c", "s7")
        * u("y", "d", "s8"),
        momenta={"y": (0, 0, 0)},
    )
    return Correlator(sink, source)


def collect_plan_costs(
    *,
    mode_size: int = 4,
    contraction_strategy: str = "greedy",
    physical_memory_limit_bytes: int | None = None,
) -> dict[str, object]:
    """Compile the fixed fixture set and return a JSON-compatible report."""

    config = RuntimeCompilationConfig(
        mode_size=mode_size,
        contraction_strategy=contraction_strategy,
        physical_memory_limit_bytes=physical_memory_limit_bytes,
        shared_orthonormal_laph_basis=True,
    )
    cases: dict[str, object] = {}
    for name, builder in (
        ("meson", _meson_correlator),
        ("baryon", _baryon_correlator),
        ("tetraquark", _tetraquark_correlator),
    ):
        compilation = compile_correlators({name: builder()}, config)
        cases[name] = asdict(compilation.cost_report)
    return {
        "baseline_schema_version": PLAN_COST_BASELINE_VERSION,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report deterministic ConCorD compiler-plan costs."
    )
    parser.add_argument(
        "--mode-size",
        type=int,
        default=4,
        help="LapH mode dimension used by the analytical size model (default: 4).",
    )
    parser.add_argument(
        "--strategy",
        default="greedy",
        help="Contraction strategy, including reuse-aware (default: greedy).",
    )
    parser.add_argument(
        "--physical-memory-limit-bytes",
        type=int,
        default=None,
        help="Optional hard abstract buffer-pool capacity in bytes.",
    )
    args = parser.parse_args()
    report = collect_plan_costs(
        mode_size=args.mode_size,
        contraction_strategy=args.strategy,
        physical_memory_limit_bytes=args.physical_memory_limit_bytes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
