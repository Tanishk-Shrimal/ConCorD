"""Regression tests for the symbolic-frontend/compiler boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from concord_lqcd.compiler.expansion import ExpansionPass
from concord_lqcd.compiler.normalization import CompositeOperator
from concord_lqcd.compiler.prewick import lower_correlator_to_physics_ir
from concord_lqcd.compiler.wick_cache import WickCache
from concord_lqcd.core import (
    Apply,
    Correlator,
    Operator,
    Path as LatticePath,
    derivative,
    transport,
    u,
)


ROOT = Path(__file__).resolve().parents[1]


def test_operator_keeps_user_expression_until_compiler_boundary():
    q = u("x", "a", "s")
    expr = derivative(transport(q, "+x", "-x", "+y"), "+z")
    op = Operator(expr, momenta={"x": (0, 0, 0)})

    assert op.expr == expr
    assert isinstance(op.expr, Apply)
    assert isinstance(op.expr.operand, Apply)
    assert op.expr.operand.operator.path.hops == ("+x", "-x", "+y")


def test_expansion_canonicalizes_before_merging_linear_combination_terms():
    q1 = u("x", "a", "s")
    op_unsimplified = Operator(
        transport(q1, "+x", "-x", "+y"),
        momenta={"x": (0, 0, 0)},
    )

    q2 = u("x", "a", "s")
    op_simplified = Operator(
        transport(q2, "+y"),
        momenta={"x": (0, 0, 0)},
    )

    # These are intentionally distinct frontend expressions. At the compiler
    # boundary they normalize to the same basis operator and cancel exactly.
    assert op_unsimplified != op_simplified
    corr = Correlator(op_unsimplified - op_simplified)
    assert ExpansionPass.run_correlator(corr) == ()


def test_wick_cache_applies_compiler_normalization_before_contraction():
    q = u("x", "a", "s")
    op = Operator(
        derivative(transport(q, "+x", "-x", "+y"), "+z"),
        momenta={"x": (0, 0, 0)},
    )

    cache = WickCache()
    compiled_corr, terms = cache.get_or_contract_operators((op, op.dagger()))

    assert terms
    normalized = compiled_corr.operators[0]
    assert isinstance(normalized, Operator)
    assert isinstance(normalized.expr, Apply)
    assert isinstance(normalized.expr.operator, CompositeOperator)

    outer, inner = normalized.expr.operator.operators
    assert outer.direction == "+z"
    assert inner.path == LatticePath(("+y",))

    # The user's original frontend object remains untouched.
    assert isinstance(op.expr.operand, Apply)
    assert op.expr.operand.operator.path.hops == ("+x", "-x", "+y")


def test_plain_frontend_construction_does_not_import_compiler_modules():
    source = textwrap.dedent(
        """
        import sys
        import concord_lqcd

        q = concord_lqcd.u("x", "a", "s")
        expr = concord_lqcd.derivative(q, "x")
        op = concord_lqcd.Operator(expr, momenta={"x": (0, 0, 0)})

        assert op.expr is expr
        loaded = [
            name for name in sys.modules
            if name == "concord_lqcd.compiler" or name.startswith("concord_lqcd.compiler.")
        ]
        assert not loaded, loaded
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_frontend_rejects_invalid_path_hops_early():
    with pytest.raises(ValueError, match="Invalid lattice hop"):
        LatticePath(("north",))


def test_frontend_rejects_non_expression_apply_operand():
    from concord_lqcd.core import CovariantDerivative

    with pytest.raises(TypeError, match="Apply operand must be an Expr instance"):
        Apply(CovariantDerivative("x"), object())  # type: ignore[arg-type]


def test_projection_helpers_are_public_top_level_exports():
    import concord_lqcd

    assert concord_lqcd.Projection is not None
    assert concord_lqcd.projected is not None
    assert concord_lqcd.laph is not None
    assert concord_lqcd.point is not None


def test_small_correlator_weight_survives_compiler_expansion():
    operator = Operator(u("x", "color", "spin"), momentum=(0, 0, 0))
    correlator = Correlator(1e-8 * operator, 1e-8 * operator.dagger())
    expected_weight = complex(1e-8) * complex(1e-8)
    frontend_terms = tuple(correlator.iter_basis_terms())
    compiler_terms = ExpansionPass.run_correlator(correlator)

    assert len(frontend_terms) == len(compiler_terms) == 1
    assert frontend_terms[0][0] == compiler_terms[0][0] == expected_weight


def test_compiler_normalization_preserves_near_cancellation():
    unsimplified = Operator(transport(u("x", "color", "spin"), "+x", "-x", "+y"))
    simplified = Operator(transport(u("x", "color", "spin"), "+y"))
    coefficient = 1 - 2 ** -52
    terms = ExpansionPass.run_correlator(Correlator(unsimplified - coefficient * simplified))

    assert len(terms) == 1
    assert terms[0][0] == complex(1 - coefficient)


@pytest.mark.parametrize("coefficient", [1e308, 1e308j])
def test_normalized_duplicate_overflow_fails_before_wick_contraction(coefficient):
    unsimplified = Operator(transport(u("x", "color", "spin"), "+x", "-x", "+y"))
    simplified = Operator(transport(u("x", "color", "spin"), "+y"))
    correlator = Correlator(coefficient * unsimplified + coefficient * simplified)
    cache = WickCache()

    with pytest.raises(ValueError, match="Combined correlator coefficients must be finite"):
        ExpansionPass.run_correlator(correlator)
    with pytest.raises(ValueError, match="Combined correlator coefficients must be finite"):
        lower_correlator_to_physics_ir(correlator, cache)
    assert cache.misses == cache.hits == 0


def test_correlator_product_overflow_fails_before_wick_contraction():
    operator = Operator(u("x", "color", "spin"))
    correlator = Correlator(1e200 * operator, 1e200 * operator.dagger())
    cache = WickCache()

    with pytest.raises(ValueError, match="coefficient product must be finite"):
        lower_correlator_to_physics_ir(correlator, cache)
    assert cache.misses == cache.hits == 0
