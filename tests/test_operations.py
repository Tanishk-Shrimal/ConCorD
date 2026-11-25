"""
Unit tests for transformation algebra, LinearOperators, Path, and Apply AST nodes.
"""

import pytest
from concord_lqcd.core import (
    Space, Locality, Axis, FieldType, Path, Product,
    LinearOperator, Shift, ParallelTransport, CovariantDerivative, Projection, Apply,
    QuarkField, u, ubar, d, dbar, GammaMatrix, gamma5, gammaX,
    derivative, transport, shift, projected, laph, point,
    Operator, get_registry, reset_registry
)
from concord_lqcd.core.operator import _dagger_expr, _reindex_expr


def test_axis_and_path():
    p1 = Path(("+x", "+x", "-y"))
    assert p1.hops == ("+x", "+x", "-y")
    assert p1.locality == Locality.NONLOCAL

    p2 = Path(("+y",))
    assert p2.locality == Locality.NEAREST_NEIGHBOR

    p_rev = p1.reverse()
    assert p_rev.hops == ("+y", "-x", "-x")

    p_add = p1 + p2
    assert p_add.hops == ("+x", "+x", "-y", "+y")
    assert p_add.simplify().hops == ("+x", "+x")


def test_linear_operators():
    dx = CovariantDerivative(Axis.X)
    assert dx.direction == "+x"
    assert dx.acts_on == frozenset({Space.COORDINATE, Space.COLOR})
    assert dx.locality == Locality.NEAREST_NEIGHBOR
    assert dx.dagger() == CovariantDerivative("-x")

    pt = ParallelTransport(Path(("+x", "+y")))
    assert pt.acts_on == frozenset({Space.COORDINATE, Space.COLOR})
    assert pt.locality == Locality.NONLOCAL
    assert pt.dagger() == ParallelTransport(Path(("-y", "-x")))

    sh = Shift(Path(("+z",)))
    assert sh.acts_on == frozenset({Space.COORDINATE})
    assert sh.locality == Locality.NEAREST_NEIGHBOR
    assert sh.dagger() == Shift(Path(("-z",)))


def test_apply_node_construction_and_validation():
    reset_registry()
    q = u("x", "a", "alpha")

    # Valid Apply on QuarkField
    app1 = Apply(CovariantDerivative("x"), q)
    assert app1.indices == q.indices
    assert app1.operator == CovariantDerivative("+x")
    assert app1.operand == q

    # Valid nested Apply
    app2 = Apply(CovariantDerivative("y"), app1)
    assert app2.indices == q.indices
    assert app2.operator == CovariantDerivative("+y")
    assert app2.operand == app1

    # Invalid operand: Product
    g5 = gamma5("alpha", "beta")
    with pytest.raises(TypeError, match="Apply operand cannot be a Product expression"):
        Apply(CovariantDerivative("x"), q * g5)

    # Invalid operator: not a LinearOperator
    with pytest.raises(TypeError, match="Apply operator must be a LinearOperator instance"):
        Apply("invalid_operator", q)  # type: ignore


def test_convenience_helpers():
    reset_registry()
    q = u("x", "a", "alpha")

    d_x = derivative(q, "x")
    assert isinstance(d_x, Apply)
    assert d_x.operator == CovariantDerivative("+x")

    d_x_bwd = derivative(q, "x", is_backward=True)
    assert isinstance(d_x_bwd, Apply)
    assert d_x_bwd.operator == CovariantDerivative("-x")

    t_xy = transport(q, "+x", "+y")
    assert isinstance(t_xy, Apply)
    assert t_xy.operator == ParallelTransport(Path(("+x", "+y")))

    s_z = shift(q, "+z")
    assert isinstance(s_z, Apply)
    assert s_z.operator == Shift(Path(("+z",)))

    prj = projected(q, "laph")
    assert isinstance(prj, Apply)
    assert prj.operator == Projection("laph")

    assert laph(q).operator == Projection("laph")
    assert point(q).operator == Projection("point")


def test_dagger_and_reindex_on_apply():
    """Antiquark realization conjugates the original field-operation chain."""
    reset_registry()
    q = u("x", "a", "alpha")

    d_x = derivative(q, "x")
    dag_d_x = _dagger_expr(d_x)
    assert isinstance(dag_d_x, Product)
    app_factor = dag_d_x.factors[0]
    assert isinstance(app_factor, Apply)
    assert app_factor.operator == CovariantDerivative("+x")
    assert app_factor.operand.is_anti == True
    assert app_factor.operand.flavor == "u"

    d_xy = derivative(derivative(q, "y"), "x")
    dag_d_xy = _dagger_expr(d_xy)
    assert isinstance(dag_d_xy, Product)
    app_xy = dag_d_xy.factors[0]
    assert isinstance(app_xy, Apply)
    assert app_xy.operator == CovariantDerivative("+x")
    assert isinstance(app_xy.operand, Apply)
    assert app_xy.operand.operator == CovariantDerivative("+y")

    # Reindex Apply
    mapping = {}
    reindexed_d_x = _reindex_expr(d_x, "test_ns", mapping)
    assert isinstance(reindexed_d_x, Apply)
    assert reindexed_d_x.operator == CovariantDerivative("+x")
    assert reindexed_d_x.operand.pos.label == "test_ns:x"
