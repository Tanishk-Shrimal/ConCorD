"""
Unit tests for Physics IR dataclasses and extract_operations.
"""

import pytest
from concord_lqcd.core import (
    Operator,
    u, ubar, d, dbar, gamma5, delta,
    derivative, transport, shift, projected, laph, point,
    CovariantDerivative, ParallelTransport, Shift, Projection,
    get_registry, reset_registry
)
from concord_lqcd.compiler.physics_ir import (
    OperationKind,
    Representation,
    OperationRecord,
    QuarkRecord,
    SpinTensorRecord,
    ColorTensorRecord,
    LogicalQuarkLine,
    LogicalCorrelator,
    PhysicsIR,
)
from concord_lqcd.compiler.physics_lowering import extract_operations


def test_extract_operations():
    reset_registry()
    q = u("x", "a", "alpha")

    # Simple field (defaults to LapH representation for distillation)
    leaf, recs, rep = extract_operations(q)
    assert leaf == q
    assert len(recs) == 0
    assert rep == Representation.LAPH

    # Projected derivative field: D_x P_laph q
    d_p_q = derivative(laph(q), "x")
    leaf, recs, rep = extract_operations(d_p_q)
    assert leaf == q
    assert len(recs) == 1
    assert recs[0].kind == OperationKind.DERIVATIVE
    assert recs[0].parameters == {"direction": "+x"}
    assert rep == Representation.LAPH

    # Point representation field: D_x point(q)
    d_pt_q = derivative(point(q), "x")
    leaf, recs, rep = extract_operations(d_pt_q)
    assert leaf == q
    assert len(recs) == 1
    assert recs[0].kind == OperationKind.DERIVATIVE
    assert rep == Representation.POINT


def test_physics_ir_preserves_field_operations_and_explicit_tensors():
    from concord_lqcd.core import Correlator
    from concord_lqcd.core.wick import wick_contract
    from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir

    reset_registry()
    sink = Operator(
        dbar("x", "a", "alpha") * gamma5("alpha", "beta")
        * transport(u("x", "b", "beta"), "+x", "-y")
        * delta("a", "b"),
        momentum=(0, 0, 0),
    )
    corr = Correlator(sink, sink.dagger())
    logical = lower_to_physics_ir(corr, wick_contract(corr)).correlators[0]

    assert any(
        line.quark_operations
        and line.quark_operations[0].kind == OperationKind.TRANSPORT
        for line in logical.quark_lines
    )
    assert any(t.kind == "gamma5" for t in logical.spin_tensors)
    assert any(t.kind == "delta" for t in logical.color_tensors)
