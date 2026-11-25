"""Unit tests for N-point correlator compilation in ConCorD."""

import pytest
from concord_lqcd.core import (
    Operator,
    Correlator,
    u,
    ubar,
    d,
    dbar,
    c,
    cbar,
    gamma5,
    gammaX,
    point,
)
from concord_lqcd.core.wick import wick_contract
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.compiler import lower_correlator_to_kernel_recipes_and_propagations


def test_three_point_correlator_compilation():
    # 1. Define 3-point meson -> local POINT vector current -> meson operators
    op_snk = Operator(
        cbar("x1", "c1", "s1")
        * gamma5("s1", "s2")
        * u("x1", "c1", "s2"),
        momentum=(0, 0, 1),
    )
    op_ins = Operator(
        point(ubar("x2", "c2", "s3"))
        * gammaX("s3", "s4")
        * point(u("x2", "c2", "s4")),
        momentum=(0, 0, 0),
    )
    op_src = Operator(
        ubar("x3", "c3", "s5")
        * gamma5("s5", "s6")
        * c("x3", "c3", "s6"),
        momentum=(0, 0, -1),
    )

    # 2. Build 3-point Correlator
    corr_3pt = Correlator(op_snk, op_ins, op_src)
    assert len(corr_3pt.operators) == 3
    assert len(corr_3pt.symbolic_operators) == 3

    # 3. Contract Wick terms
    wick_terms = wick_contract(corr_3pt)
    assert len(wick_terms) > 0

    # 4. Lower through Stage 5 propagation-resource identification
    phys_ir = lower_to_physics_ir(corr_3pt, wick_terms)
    assert len(phys_ir.correlators) >= 1

    kernel_recipes, prop_recipes = lower_correlator_to_kernel_recipes_and_propagations(
        phys_ir.correlators[0]
    )
    assert len(kernel_recipes) == 3
    assert len(prop_recipes) >= 1
    assert any(p.is_generalized for p in prop_recipes)
    generalized = next(p for p in prop_recipes if p.is_generalized)
    assert generalized.resource_kind == "generalized_perambulator"
    assert generalized.resource_id.startswith("generalized_perambulator_")
