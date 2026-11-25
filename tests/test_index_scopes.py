"""Expression-local index validation and capture-free operator algebra."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from concord_lqcd.core import (
    COLOR,
    SPIN,
    Correlator,
    Index,
    IndexConflictError,
    IndexRegistry,
    Operator,
    QuarkField,
    build_baryon_operator,
    delta,
    epsilon,
    gamma5,
    gamma_identity,
    get_registry,
    link,
    reset_registry,
    spin_matrix,
    u,
    ubar,
    validate,
)
from concord_lqcd.core.dirac_basis import get_spin_matrices
from concord_lqcd.core.operator import _dagger_expr


@pytest.fixture
def explicit_registry():
    reset_registry()
    registry = get_registry()
    yield registry
    reset_registry()


def _pion(position, color_label, left_spin, right_spin):
    return Operator(
        ubar(position, color_label, left_spin)
        * gamma5(left_spin, right_spin)
        * u(position, color_label, right_spin),
        momentum=(0, 0, 0),
    )


def _spin_value(operator, basis, output_labels=()):
    matrices = get_spin_matrices(basis)
    factors = getattr(operator.expr, "factors", (operator.expr,))
    labels = tuple(dict.fromkeys(index.label for factor in factors for index in factor.indices))
    positions = {label: position for position, label in enumerate(labels)}
    operands = []
    for factor in factors:
        operands.extend((matrices[factor.kind], [positions[index.label] for index in factor.indices]))
    operands.append([positions[label] for label in output_labels])
    return np.einsum(*operands, optimize=True)


def test_operator_rejects_same_label_in_different_spaces():
    with pytest.raises(IndexConflictError, match="'shared'.*'color'.*'spin'"):
        Operator(u("x", "shared", "shared"))


def test_partial_expressions_cannot_bypass_operator_label_checks():
    field = u("x", "shared", "left_spin")
    Operator(field)
    matrix = gamma5("shared", "right_spin")

    with pytest.raises(IndexConflictError, match="'shared'.*'color'.*'spin'"):
        Operator(field * matrix)


def test_manually_constructed_indices_obey_the_same_scope_rule():
    field = QuarkField("u", False, Index("x", "position"), Index("shared", COLOR), Index("shared", SPIN))

    with pytest.raises(IndexConflictError, match="'shared'"):
        Operator(field)


def test_independent_operators_can_reuse_labels_across_spaces():
    first = _pion("x", "a", "alpha", "beta")
    second = _pion("y", "color", "a", "b")

    assert validate(first) == (True, [])
    assert validate(second) == (True, [])
    assert first == second
    assert hash(first) == hash(second)
    correlator = Correlator(first, second.dagger())
    first_indices, second_indices = (set(operator.indices) for operator in correlator.symbolic_operators)
    assert first_indices.isdisjoint(second_indices)


def test_factories_and_operator_algebra_do_not_touch_shared_registry(explicit_registry):
    explicit_registry.spin("color")
    explicit_registry.color("left_spin")
    explicit_registry.spin("sink:color")
    before = explicit_registry.all_indices()
    operator = _pion("x", "color", "left_spin", "right_spin")

    delta("color", "other_color")
    epsilon("color", "other_color", "third_color")
    link("x", "y", "color", "other_color")
    operator.dagger()
    operator.reindexed("sink")
    Correlator(operator, operator.dagger())

    assert explicit_registry.all_indices() == before


def test_failed_operator_does_not_change_explicit_registry(explicit_registry):
    explicit_registry.color("sentinel")
    before = explicit_registry.all_indices()

    with pytest.raises(IndexConflictError):
        Operator(u("x", "shared", "shared"))
    with pytest.raises(TypeError, match="momentum components must be integers"):
        Operator(u("x", "color", "spin"), momentum=(1.5, 0, 0))

    assert explicit_registry.all_indices() == before
    assert validate(_pion("x", "color", "alpha", "beta")) == (True, [])


def test_explicit_registry_still_checks_its_own_scope():
    registry = IndexRegistry((Index("color", COLOR), Index("spin", SPIN)))

    assert registry.color("color") is registry.color("color")
    with pytest.raises(IndexConflictError, match="'color'"):
        registry.spin("color")
    assert registry.fresh(SPIN).label not in {"color", "spin"}


@pytest.mark.parametrize("slot", [0, 1, 2])
def test_dagger_dummies_avoid_existing_labels_in_every_space(slot):
    labels = ["x", "color", "spin"]
    labels[slot] = "__dagger_1"
    operator = Operator(u(*labels))
    adjoint = operator.dagger()
    antifield = next(factor for factor in adjoint.expr.factors if isinstance(factor, QuarkField))

    assert antifield.spin.label not in set(labels)
    assert validate(adjoint)[0]


def test_dagger_reserves_all_factor_labels_before_recursion():
    expression = u("x", "color", "__dagger_1") * gamma5("other_spin", "__dagger_2")
    adjoint = _dagger_expr(expression)
    antifield = next(factor for factor in adjoint.factors if isinstance(factor, QuarkField))

    assert antifield.spin.label not in {index.label for index in expression.indices}


def test_dagger_is_independent_of_unrelated_construction():
    operator = _pion("x", "color", "left_spin", "right_spin")
    before = operator.dagger()
    _pion("other_x", "other_color", "other_left", "other_right").dagger()
    after = operator.dagger()

    assert before.expr == after.expr
    assert before == after


def test_completed_meson_products_keep_independent_summed_indices():
    first = _pion("x", "color", "alpha", "beta")
    second = _pion("y", "color", "alpha", "beta")
    original_first, original_second = first.expr, second.expr
    combined = first * second

    assert validate(combined) == (True, [])
    assert all(count == 2 for index, count in Counter(combined.indices).items() if index.space != "position")
    assert combined.momenta == {"x": (0, 0, 0), "y": (0, 0, 0)}
    assert first.expr == original_first
    assert second.expr == original_second


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_closed_spin_products_are_products_of_traces(basis):
    first = Operator(gamma_identity("shared", "shared"))
    second = Operator(spin_matrix("Pplus", "shared", "shared"))
    combined = first * second
    matrices = get_spin_matrices(basis)
    expected = np.trace(matrices["gamma_id"]) * np.trace(matrices["Pplus"])

    np.testing.assert_allclose(_spin_value(combined, basis), expected, rtol=0, atol=1e-14)
    assert abs(expected) > 7


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_closed_spin_adjoint_products_preserve_independent_traces(basis):
    first = Operator(gamma_identity("shared", "shared")).dagger()
    second = Operator(spin_matrix("Pplus", "shared", "shared")).dagger()
    combined = first * second
    matrices = get_spin_matrices(basis)
    expected = np.trace(matrices["gamma_id"].conj().T) * np.trace(matrices["Pplus"].conj().T)

    assert validate(combined) == (True, [])
    np.testing.assert_allclose(_spin_value(combined, basis), expected, rtol=0, atol=1e-14)
    assert abs(expected) > 7


def test_composed_adjoint_fields_have_independent_generated_indices():
    first = Operator(u("x", "left_color", "left_spin")).dagger()
    second = Operator(u("y", "right_color", "right_spin")).dagger()
    combined = first * second

    assert validate(combined)[0]
    antifields = [factor for factor in combined.expr.factors if isinstance(factor, QuarkField)]
    assert antifields[0].spin != antifields[1].spin


def test_composition_respects_dagger_product_order():
    first = _pion("x", "color", "alpha", "beta")
    second = _pion("y", "color", "alpha", "beta")

    assert (first * second).dagger() == second.dagger() * first.dagger()


def test_alpha_equivalent_scopes_reuse_wick_contractions():
    from concord_lqcd.compiler.wick_cache import WickCache

    first = _pion("x", "a", "alpha", "beta")
    second = _pion("y", "color", "a", "b")
    cache = WickCache()
    _, first_terms = cache.get_or_contract_operators((first, first.dagger()))
    _, second_terms = cache.get_or_contract_operators((second, second.dagger()))

    assert first_terms
    assert second_terms
    assert cache.misses == 1
    assert cache.hits == 1


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_shared_open_spin_indices_keep_matrix_couplings(basis):
    first = Operator(gamma5("left_spin", "shared"))
    second = Operator(gamma5("shared", "right_spin"))
    combined = first * second
    matrices = get_spin_matrices(basis)

    np.testing.assert_allclose(
        _spin_value(combined, basis, ("left_spin", "right_spin")),
        matrices["gamma5"] @ matrices["gamma5"],
        rtol=0, atol=1e-14,
    )


@pytest.mark.parametrize("bound_first", [False, True])
@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_summed_spin_indices_cannot_capture_another_operand_free_index(basis, bound_first):
    closed = Operator(spin_matrix("Pplus", "shared", "shared"))
    opened = Operator(gamma_identity("shared", "free_spin"))
    combined = closed * opened if bound_first else opened * closed
    matrices = get_spin_matrices(basis)

    np.testing.assert_allclose(
        _spin_value(combined, basis, ("shared", "free_spin")),
        np.trace(matrices["Pplus"]) * matrices["gamma_id"],
        rtol=0, atol=1e-14,
    )


def test_baryon_products_preserve_external_spin_couplings():
    first = build_baryon_operator(("u", "d", "u"), pos_label="x", spin_labels=("alpha", "beta", "rho_left"))
    second = build_baryon_operator(("u", "d", "u"), pos_label="y", spin_labels=("alpha", "beta", "rho_right"))
    combined = first * second * gamma_identity("rho_left", "rho_right")

    assert validate(combined) == (True, [])
    counts = Counter(combined.indices)
    assert counts[Index("rho_left", SPIN)] == 2
    assert counts[Index("rho_right", SPIN)] == 2


def test_bare_field_products_preserve_written_contractions():
    expression = ubar("x", "color", "alpha") * gamma5("alpha", "beta") * u("x", "color", "beta")
    operator = Operator(expression)

    assert operator.expr is expression
    assert {index.label for index in operator.indices} == {"x", "color", "alpha", "beta"}


def test_parallel_operator_construction_has_no_shared_index_history():
    def construct(number):
        if number % 2:
            return _pion("x", "a", "alpha", "beta").dagger()
        return _pion("y", "color", "a", "b").dagger()

    with ThreadPoolExecutor(max_workers=4) as executor:
        operators = list(executor.map(construct, range(16)))

    assert all(validate(operator) == (True, []) for operator in operators)
    assert all(operator == operators[0] for operator in operators)
