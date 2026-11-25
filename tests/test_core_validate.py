"""Tests for concord_lqcd.core.validate — index validation."""

from concord_lqcd.core import (
    Operator, validate,
    u, d, dbar, s, epsilon, delta, gamma5, derivative,
    reset_registry, get_registry,
)
from concord_lqcd.core.propagator import Propagator


def test_valid_pion():
    reset_registry()
    expr = (
        dbar('x', 'a', 'α') * gamma5('α', 'β')
        * u('x', 'b', 'β') * delta('a', 'b')
    )
    op = Operator(expr)
    ok, errors = validate(op)
    assert ok, f"Expected valid, got errors: {errors}"
    assert errors == []


def test_valid_derivative_operator_recurses_through_apply():
    reset_registry()
    expr = (
        dbar('x', 'a', 'α') * gamma5('α', 'β')
        * derivative(u('x', 'b', 'β'), '+x') * delta('a', 'b')
    )
    op = Operator(expr)
    ok, errors = validate(op)
    assert ok, f"Expected valid derivative operator, got errors: {errors}"


def test_baryon_with_free_spin_index_is_valid():
    """Baryons carry an open Dirac index — free spin indices are allowed."""
    reset_registry()
    expr = (
        epsilon('a', 'b', 'c')
        * u('x', 'a', 'α')
        * d('x', 'b', 'β')
        * u('x', 'c', 'γ')
        * gamma5('α', 'δ')
        * gamma5('β', 'δ')
        * gamma5('γ', 'η')
    )
    op = Operator(expr)
    ok, errors = validate(op)
    # η is a free (open) spinor index — this is valid
    assert ok, f"Expected valid baryon, got errors: {errors}"
    # Position index 'x' appears 3 times but position indices are exempt
    assert not any("position" in err for err in errors)


def test_fully_contracted_baryon():
    reset_registry()
    expr = epsilon('a', 'b', 'c') * epsilon('a', 'b', 'c')
    op = Operator(expr)
    ok, errors = validate(op)
    assert ok, f"Expected valid, got errors: {errors}"


def test_triple_index_is_invalid():
    reset_registry()
    expr = u('x', 'a', 'α') * d('x', 'a', 'β') * s('x', 'a', 'γ')
    op = Operator(expr)
    ok, errors = validate(op)
    assert not ok
    assert any("'a'" in e and "3 times" in e for e in errors)


def test_propagator_leaf_validates():
    """Validation accepts the internal propagator leaf produced by Wick contraction."""
    reset_registry()
    reg = get_registry()
    prop = Propagator(
        'u',
        reg.position('x'), reg.color('a'), reg.spin('α'),
        reg.position('y'), reg.color('b'), reg.spin('β'),
    )
    op = Operator(prop * gamma5('α', 'β') * delta('a', 'b'))
    ok, errors = validate(op)
    assert ok, errors
