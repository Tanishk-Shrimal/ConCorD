"""Unit tests for OperatorLike protocol and OperatorCombination linear algebra."""

import pytest
from concord_lqcd.core import (
    Operator, cbar, sbar, u, gamma5,
    reset_registry, OperatorLike, OperatorCombination
)


def setup_function():
    reset_registry()


def test_operator_like_protocol():
    op = Operator(cbar('x_1', 'c1', 'a') * gamma5('a', 'b') * u('x_1', 'c1', 'b'), momentum=(0,0,0))
    assert isinstance(op, OperatorLike)

    comb = 0.5 * op
    assert isinstance(comb, OperatorLike)
    assert isinstance(comb, OperatorCombination)


def test_operator_combination_self_canonicalization():
    op1 = Operator(cbar('x_1', 'c1', 'a') * gamma5('a', 'b') * u('x_1', 'c1', 'b'), momentum=(0,0,0))
    op2 = Operator(sbar('x_2', 'c2', 'c_sp') * gamma5('c_sp', 'd_sp') * u('x_2', 'c2', 'd_sp'), momentum=(0,0,0))

    # O1 + 2*O2 - O1 + 3*O2 should auto-canonicalize to 5*O2
    comb = op1 + 2.0 * op2 - op1 + 3.0 * op2
    assert isinstance(comb, OperatorCombination)
    assert len(comb.terms) == 1

    coeff, op = comb.terms[0]
    assert coeff == pytest.approx(5.0 + 0.0j)
    assert op == op2


def test_operator_combination_nested_algebra():
    op1 = Operator(cbar('x_1', 'c1', 'a') * gamma5('a', 'b') * u('x_1', 'c1', 'b'), momentum=(0,0,0))
    op2 = Operator(sbar('x_2', 'c2', 'c_sp') * gamma5('c_sp', 'd_sp') * u('x_2', 'c2', 'd_sp'), momentum=(0,0,0))

    A = op1 + op2
    B = op1 - op2
    C = 2.0 * A + 3.0 * B  # 2*(O1+O2) + 3*(O1-O2) = 5*O1 - 1*O2

    assert len(C.terms) == 2
    term_dict = {op: c for c, op in C.terms}
    assert term_dict[op1] == pytest.approx(5.0 + 0.0j)
    assert term_dict[op2] == pytest.approx(-1.0 + 0.0j)


def test_operator_combination_dagger():
    op1 = Operator(cbar('x_1', 'c1', 'a') * gamma5('a', 'b') * u('x_1', 'c1', 'b'), momentum=(0,0,1))
    comb = (1.0 + 2.0j) * op1
    dag_comb = comb.dagger()

    assert len(dag_comb.terms) == 1
    coeff, dag_op = dag_comb.terms[0]
    assert coeff == pytest.approx(1.0 - 2.0j)
    assert dag_op.momenta == {"x_1": (0, 0, -1)}


@pytest.mark.parametrize("coefficient", [1e-16, 1e-50, 1e-300, 1e-50j])
def test_small_nonzero_coefficients_are_preserved(coefficient):
    operator = Operator(u("x", "color", "spin"))
    combination = coefficient * operator

    assert combination.terms == ((complex(coefficient), operator),)
    assert (combination + combination).terms == ((2 * coefficient, operator),)
    assert (combination - combination).terms == ()
    assert combination.dagger().terms[0][0] == complex(coefficient).conjugate()


def test_small_combination_can_be_rescaled():
    operator = Operator(u("x", "color", "spin"))

    assert ((1e-16 * operator) * 1e16).terms == ((1 + 0j, operator),)


def test_near_cancellation_does_not_drop_residual():
    operator = Operator(u("x", "color", "spin"))
    coefficient = 1 - 2 ** -52
    combination = operator - coefficient * operator

    assert combination.terms == ((complex(1 - coefficient), operator),)


@pytest.mark.parametrize("scalar", [1e-16, -1e-50, 1e-50j])
def test_nonzero_scalar_constants_cannot_be_added_or_subtracted(scalar):
    operator = Operator(u("x", "color", "spin"))
    combination = 2 * operator

    for expression in (operator, combination):
        with pytest.raises(TypeError, match="non-zero constant"):
            expression + scalar
        with pytest.raises(TypeError, match="non-zero constant"):
            expression - scalar
        with pytest.raises(TypeError, match="non-zero constant"):
            scalar + expression
        with pytest.raises(TypeError, match="non-zero constant"):
            scalar - expression


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), complex(0, float("inf"))])
def test_nonfinite_coefficients_raise(coefficient):
    operator = Operator(u("x", "color", "spin"))

    with pytest.raises(ValueError, match="coefficients must be finite"):
        coefficient * operator


def test_duplicate_coefficient_overflow_raises():
    operator = Operator(u("x", "color", "spin"))

    with pytest.raises(ValueError, match="coefficients must be finite"):
        OperatorCombination([(1e308, operator), (1e308, operator)])
