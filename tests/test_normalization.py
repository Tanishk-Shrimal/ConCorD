"""
Unit tests for AST Normalization Pass and CompositeOperator.
"""

import pytest
from concord_lqcd.core import (
    u, derivative, transport, shift, Axis, Path, Apply,
    CovariantDerivative, ParallelTransport, Shift, reset_registry
)
from concord_lqcd.compiler.normalization import normalize_expr, CompositeOperator


def test_normalization_pass():
    reset_registry()
    q = u("x", "a", "alpha")

    # Path simplification during normalization
    t_unsimplified = transport(q, "+x", "-x", "+y")
    norm_t = normalize_expr(t_unsimplified)
    assert isinstance(norm_t, Apply)
    assert norm_t.operator == ParallelTransport(Path(("+y",)))

    # Nested Apply fusion into CompositeOperator
    d_xy = derivative(derivative(q, "y"), "x")
    norm_d_xy = normalize_expr(d_xy)
    assert isinstance(norm_d_xy, Apply)
    assert isinstance(norm_d_xy.operator, CompositeOperator)
    assert norm_d_xy.operator.operators == (CovariantDerivative("+x"), CovariantDerivative("+y"))
    assert norm_d_xy.operand == q
