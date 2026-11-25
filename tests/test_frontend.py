"""
test_frontend.py

Comprehensive test suite for the ConCorD symbolic frontend.
Validates position tracking, derivatives, gauge links, Hermitian
conjugates, and index validation.
"""

from concord_lqcd.core.expr import Product
from concord_lqcd.core.indices import get_registry, reset_registry
from concord_lqcd.core.fields import u, d, ubar, dbar, QuarkField
from concord_lqcd.core.gamma import gamma5, gammaT, GammaMatrix
from concord_lqcd.core.color import delta, epsilon
from concord_lqcd.core.operator import Operator, derivative, _dagger_expr
from concord_lqcd.core.operations import Apply, CovariantDerivative
from concord_lqcd.core.gauge import link, GaugeLink
from concord_lqcd.core.validate import validate


# ── Index Registry & Position Tracking ────────────────────────────────

def test_field_indices_and_positions_are_independent_of_registry():
    reset_registry()
    reg = get_registry()

    q1 = d('x', 'a', 'alpha')
    q2 = d('y', 'b', 'beta')

    assert q1.pos.space == 'position'
    assert q1.color.space == 'color'
    assert reg.all_indices() == []
    assert len(q1.indices) == 3
    assert q1.pos.label == 'x'


# ── Gauge Links ───────────────────────────────────────────────────────

def test_gauge_link_creation():
    reset_registry()

    U = link('x', 'y', 'a', 'b')
    assert isinstance(U, GaugeLink)
    assert U.start_pos.label == 'x' and U.end_pos.label == 'y'


def test_gauge_link_dagger():
    reset_registry()

    U = link('x', 'y', 'a', 'b')
    U_dag = _dagger_expr(U)
    assert U_dag.start_pos.label == 'y' and U_dag.end_pos.label == 'x'
    assert U_dag.start_color.label == 'b' and U_dag.end_color.label == 'a'


# ── Derivatives ───────────────────────────────────────────────────────

def test_derivative_attributes():
    reset_registry()

    q = u('x', 'a', 'alpha')
    dq = derivative(q, 'z')

    assert isinstance(dq, Apply)
    assert dq.operator == CovariantDerivative('+z')


def test_derivative_dagger():
    """Conjugating D_z q retains D_z for conjugated-basis realization."""
    reset_registry()

    q = u('x', 'a', 'alpha')
    dq = derivative(q, 'z')
    dq_dag = _dagger_expr(dq)

    assert isinstance(dq_dag, Product)

    app_dag = None
    for factor in dq_dag.factors:
        if isinstance(factor, Apply):
            app_dag = factor
            break
    assert app_dag is not None
    assert app_dag.operator == CovariantDerivative('+z')
    assert app_dag.operand.is_anti is True


# ── Dirac Adjoint ─────────────────────────────────────────────────────

def test_dirac_adjoint():
    """q†_α = q̄_β (γ_t)_{β,α}"""
    reset_registry()

    q = u('x', 'a', 'alpha')
    q_dag = _dagger_expr(q)

    assert isinstance(q_dag, Product)
    assert len(q_dag.factors) == 2

    qbar_node, gamma_node = q_dag.factors
    assert isinstance(qbar_node, QuarkField) and qbar_node.is_anti
    assert isinstance(gamma_node, GammaMatrix) and gamma_node.kind == 'gammaT'

    # Check dummy index injection
    dummy_index = qbar_node.spin
    assert dummy_index.label.startswith("__dagger")
    assert gamma_node.left == dummy_index
    assert gamma_node.right.label == 'alpha'


# ── Operator Validation ──────────────────────────────────────────────

def test_valid_nonlocal_pion():
    """d̄(x) γ₅ U(x,y) u(y) — gauge-invariant non-local pion."""
    reset_registry()

    valid_op = Operator(
        dbar('x', 'a', 'alpha') * gamma5('alpha', 'beta')
        * link('x', 'y', 'a', 'b') * u('y', 'b', 'beta')
    )
    ok, errors = validate(valid_op)
    assert ok, f"Valid operator failed validation: {errors}"


def test_dangling_color_index_produces_warning():
    """Uncontracted color index 'c' and 'b' should produce warnings."""
    reset_registry()

    op = Operator(
        dbar('x', 'a', 'alpha') * gamma5('alpha', 'beta')
        * link('x', 'y', 'a', 'c') * u('y', 'b', 'beta')
    )
    ok, messages = validate(op)
    # Free color indices are warnings, not errors
    assert ok, "Free color indices should not cause validation failure"
    assert any("WARNING" in m and "uncontracted" in m for m in messages)
