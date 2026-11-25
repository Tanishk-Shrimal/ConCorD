"""Tests for concord_lqcd.core.operator — Operator, derivative, dagger."""

import numpy as np
import pytest

from concord_lqcd.core import (
    Operator, Product, QuarkField, GammaMatrix,
    derivative, u, ubar, d, dbar, gamma5, gammaX,
    reset_registry, spin_matrix,
)
from concord_lqcd.core.dirac_basis import get_spin_matrices
from concord_lqcd.core.operator import _dagger_expr


def test_operator_with_momentum():
    reset_registry()
    expr = dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β')
    op = Operator(expr, momentum=(0, 0, 1))
    assert op.expr == expr


def test_operator_without_momentum():
    reset_registry()
    expr = dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β')
    op = Operator(expr)
    assert op.momentum is None


def test_operator_with_momenta_derives_total():
    reset_registry()
    expr = (dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β') *
            dbar('y', 'b', 'γ') * gamma5('γ', 'δ') * u('y', 'b', 'δ'))
    op = Operator(expr, momenta={'x': (1, 0, 0), 'y': (-1, 0, 0)})
    assert op.total_momentum == (0, 0, 0)
    assert op.momenta == {'x': (1, 0, 0), 'y': (-1, 0, 0)}


def test_operator_momentum_vs_momenta_mutually_exclusive():
    reset_registry()
    expr = dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β')
    with pytest.raises(ValueError, match='Pass either `momentum` or `momenta`, not both'):
        Operator(expr, momentum=(1, 0, 0), momenta={'x': (1, 0, 0)})


def test_operator_metadata():
    reset_registry()
    expr = dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β')
    op = Operator(expr, momentum=(0, 0, 0), name='pion', irrep='A1')
    meta = op.metadata
    assert meta['name'] == 'pion'
    assert meta['irrep'] == 'A1'


def test_operator_indices_delegates_to_expr():
    reset_registry()
    expr = dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β')
    op = Operator(expr)
    assert op.indices == expr.indices


def test_operator_reindexed_namespaces_all_indices():
    reset_registry()
    expr = (dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β') *
            dbar('y', 'b', 'γ') * gamma5('γ', 'δ') * u('y', 'b', 'δ'))
    op = Operator(expr, momenta={'x': (1, 0, 0), 'y': (-1, 0, 0)})
    renamed = op.reindexed('snk')
    labels = [idx.label for idx in renamed.indices]
    assert all(label.startswith('snk:') for label in labels)
    assert len(set(labels)) == 8
    
    # Momenta dict should be updated with new position labels
    assert renamed.momenta == {'snk:x': (1, 0, 0), 'snk:y': (-1, 0, 0)}


def test_operator_composition_mul():
    reset_registry()
    b1 = Operator(u('x', 'a', 'α1') * d('x', 'b', 'β1'), momenta={'x': (1, 0, 0)})
    b2 = Operator(u('y', 'd', 'α2') * d('y', 'e', 'β2'), momenta={'y': (-1, 0, 0)})
    coupling = gammaX('α1', 'α2')
    
    op = b1 * b2 * coupling
    assert isinstance(op, Operator)
    assert op.momenta == {'x': (1, 0, 0), 'y': (-1, 0, 0)}


def test_derivative_creates_quark_with_derivative():
    from concord_lqcd.core import Apply, CovariantDerivative
    reset_registry()
    field = u('x', 'a', 'α')
    d_field = derivative(field, 'x')
    assert isinstance(d_field, Apply)
    assert d_field.operator == CovariantDerivative('+x')
    assert d_field.operand.flavor == 'u'
    assert d_field.operand.is_anti is False


def test_derivative_invalid_direction_raises():
    reset_registry()
    field = u('x', 'a', 'α')
    with pytest.raises(ValueError):
        derivative(field, 'w')


def test_derivative_preserves_indices():
    reset_registry()
    field = u('x', 'a', 'α')
    d_field = derivative(field, 'y')
    assert d_field.indices == field.indices


def test_derivative_on_non_quark_raises():
    reset_registry()
    g = gamma5('α', 'β')
    with pytest.raises(TypeError):
        derivative(g, 'x')


def test_dagger_quark_produces_adjoint():
    reset_registry()
    q = u('x', 'a', 'alpha')
    q_dag = _dagger_expr(q)

    assert isinstance(q_dag, Product)
    assert len(q_dag) == 2

    qbar_node, gamma_node = q_dag[0], q_dag[1]
    assert isinstance(qbar_node, QuarkField) and qbar_node.is_anti
    assert isinstance(gamma_node, GammaMatrix) and gamma_node.kind == 'gammaT'

    assert qbar_node.spin.label.startswith('__dagger')
    assert gamma_node.left == qbar_node.spin
    assert gamma_node.right.label == 'alpha'


def test_dagger_reverses_product_order():
    reset_registry()
    f1 = dbar('x', 'a', 'α')
    g = gamma5('α', 'β')
    f2 = u('x', 'a', 'β')
    op = Operator(f1 * g * f2, momentum=(1, 2, 3))
    dag = op.dagger()

    dag_expr = dag.expr
    assert isinstance(dag_expr, Product)
    assert len(dag_expr) == 5


def test_dagger_negates_momentum():
    reset_registry()
    expr = (dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β') *
            dbar('y', 'b', 'γ') * gamma5('γ', 'δ') * u('y', 'b', 'δ'))
    op = Operator(expr, momenta={'x': (1, 0, 0), 'y': (-1, 2, 0)})
    dag = op.dagger()
    assert dag.momenta == {'x': (-1, 0, 0), 'y': (1, -2, 0)}
    assert dag.total_momentum == (0, -2, 0)


def test_dagger_preserves_metadata():
    reset_registry()
    expr = u('x', 'a', 'α')
    op = Operator(expr, momentum=(0, 0, 1), name='test')
    dag = op.dagger()
    assert dag.metadata['name'] == 'test'


def test_dagger_none_momentum():
    reset_registry()
    expr = u('x', 'a', 'α')
    op = Operator(expr)
    dag = op.dagger()
    assert dag.momentum is None


@pytest.mark.parametrize("component", [1.9, -0.5, 1.0, "1", True, float("nan"), float("inf")])
@pytest.mark.parametrize("use_position_map", [False, True])
def test_momentum_rejects_noninteger_components(component, use_position_map):
    reset_registry()
    expression = u("x", "color", "spin")
    momentum = (component, 0, 0)
    arguments = {"momenta": {"x": momentum}} if use_position_map else {"momentum": momentum}

    with pytest.raises(TypeError, match="momentum components must be integers"):
        Operator(expression, **arguments)


@pytest.mark.parametrize("momentum", [(), (1, 2), (1, 2, 3, 4)])
def test_momentum_rejects_wrong_triplet_length(momentum):
    reset_registry()

    with pytest.raises(ValueError, match="3-tuple of integers"):
        Operator(u("x", "color", "spin"), momentum=momentum)


@pytest.mark.parametrize("momentum", [(1, -2, 0), [1, -2, 0], np.array([1, -2, 0], dtype=np.int64)])
def test_integer_momentum_sequences_are_normalized(momentum):
    reset_registry()
    operator = Operator(u("x", "color", "spin"), momentum=momentum)

    assert operator.momenta == {"x": (1, -2, 0)}
    assert all(type(component) is int for component in operator.momenta["x"])


def test_momentum_rejects_unknown_position_label():
    reset_registry()

    with pytest.raises(ValueError, match=r"\['y'\].*Available position labels: \['x'\]"):
        Operator(u("x", "color", "spin"), momenta={"y": (1, 0, 0)})


def test_partial_momentum_map_uses_existing_positions():
    reset_registry()
    expression = u("x", "color", "spin") * d("y", "other_color", "other_spin")
    operator = Operator(expression, momenta={"x": (1, 0, 0)})

    assert operator.momenta == {"x": (1, 0, 0)}
    assert operator.total_momentum == (1, 0, 0)


def test_explicit_position_momentum_cannot_be_none():
    reset_registry()

    with pytest.raises(TypeError, match="triplet of integers"):
        Operator(u("x", "color", "spin"), momenta={"x": None})


def test_operator_rejects_non_expression_operand():
    with pytest.raises(TypeError, match="Operator expression must be an Expr instance"):
        Operator(object())


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_named_spin_matrix_double_dagger_restores_original(basis):
    matrices = get_spin_matrices(basis)
    for kind in matrices:
        reset_registry()
        original = spin_matrix(kind, "left_spin", "right_spin")
        adjoint = _dagger_expr(original)

        assert adjoint.kind in matrices
        np.testing.assert_array_equal(matrices[adjoint.kind], matrices[kind].conj().T)
        assert _dagger_expr(adjoint) == original


def _bilinear_spin_value(operator, matrices):
    factors = operator.expr.factors
    fields = [factor for factor in factors if isinstance(factor, QuarkField)]
    antifield = next(field for field in fields if field.is_anti)
    quark = next(field for field in fields if not field.is_anti)
    spin_factors = [factor for factor in factors if isinstance(factor, GammaMatrix)]
    indices = tuple(dict.fromkeys(index for factor in spin_factors for index in factor.indices))
    positions = {index: position for position, index in enumerate(indices)}
    operands = []
    for factor in spin_factors:
        operands.extend((matrices[factor.kind], [positions[factor.left], positions[factor.right]]))
    operands.append([positions[antifield.spin], positions[quark.spin]])
    return np.einsum(*operands, optimize=True)


@pytest.mark.parametrize("basis", ["DR", "DP"])
@pytest.mark.parametrize("kind", ["gamma5", "gammaX", "gammaY", "A2", "Pplus"])
def test_repeated_bilinear_dagger_preserves_numerical_meaning(basis, kind):
    reset_registry()
    matrices = get_spin_matrices(basis)
    operator = Operator(
        ubar("x", "color", "left_spin")
        * spin_matrix(kind, "left_spin", "right_spin")
        * u("x", "color", "right_spin"),
        momentum=(1, -2, 0),
    )
    adjoint = operator.dagger()
    restored = adjoint.dagger()
    repeated = restored.dagger()
    temporal = matrices["gammaT"]
    expected_adjoint = temporal @ matrices[kind].conj().T @ temporal

    np.testing.assert_allclose(_bilinear_spin_value(adjoint, matrices), expected_adjoint, rtol=0, atol=1e-14)
    np.testing.assert_allclose(_bilinear_spin_value(restored, matrices), matrices[kind], rtol=0, atol=1e-14)
    np.testing.assert_allclose(_bilinear_spin_value(repeated, matrices), expected_adjoint, rtol=0, atol=1e-14)
    assert restored.momenta == operator.momenta
    assert repeated.momenta == adjoint.momenta
