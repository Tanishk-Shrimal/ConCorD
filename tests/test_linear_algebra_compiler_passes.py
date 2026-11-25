"""Unit tests for Wick caching and operator-combination compilation."""

from concord_lqcd.core import (
    Operator,
    CorrMatrix,
    Correlator,
    build_baryon_operator,
    reset_registry,
    cbar,
    sbar,
    u,
    gamma5,
)
from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.compiler.wick_cache import WickCache


def setup_function():
    reset_registry()


def test_wick_cache_hit_rate():
    op1 = Operator(cbar('x_1', 'c1', 'a') * gamma5('a', 'b') * u('x_1', 'c1', 'b'), momentum=(0,0,0))
    op2 = Operator(sbar('x_2', 'c2', 'c_sp') * gamma5('c_sp', 'd_sp') * u('x_2', 'c2', 'd_sp'), momentum=(0,0,0))

    # Same basis operators used across multiple linear combinations
    comb1 = op1 + op2
    comb2 = 2.0 * op1 + 3.0 * op2

    mat = CorrMatrix([comb1, comb2], [comb1, comb2])

    cache = WickCache()
    compilation = compile_correlators(
        mat,
        RuntimeCompilationConfig(mode_size=2),
        wick_cache=cache,
    )

    assert compilation.correlator_ids == ("C[0,0]", "C[0,1]", "C[1,0]", "C[1,1]")
    # Basis pairs (op1, op1^\dagger), (op1, op2^\dagger), (op2, op1^\dagger), (op2, op2^\dagger)
    # Total unique pairs = 4. All subsequent queries must hit the cache!
    assert cache.misses == 4
    assert cache.hits > 0


def test_open_spin_linear_combination_has_one_output_interface():
    proton = build_baryon_operator(("u", "d", "u"), spin_labels=("alpha", "beta", "rho"))
    neutron = build_baryon_operator(("d", "u", "d"), spin_labels=("alpha", "beta", "rho"))
    combined = proton + neutron

    compilation = compile_correlators(
        {"nucleon_sum": Correlator(combined, combined.dagger())},
        RuntimeCompilationConfig(mode_size=3),
    )

    output = compilation.logical_dag.correlators[0]
    assert len(output.output_indices) == 2
    assert output.output_indices == ("spin:op0:rho", "spin:op1:rho")
