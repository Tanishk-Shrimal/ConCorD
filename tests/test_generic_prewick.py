"""Tests for topology-independent pre-Wick N-point handling."""

from concord_lqcd.core import Operator, Correlator, CorrMatrix, u, ubar, gamma5, gammaX
from concord_lqcd.compiler import expansion, wick_cache
from concord_lqcd.compiler.expansion import ExpansionPass
from concord_lqcd.compiler.prewick import lower_correlator_to_physics_ir
from concord_lqcd.compiler.wick_cache import WickCache


def _bilinear(pos: str, gamma, s1: str, s2: str) -> Operator:
    return Operator(
        ubar(pos, "a", s1) * gamma(s1, s2) * u(pos, "a", s2),
        momentum=(0, 0, 0),
    )


def test_generic_three_point_expansion_uses_complete_operator_tuple():
    sink = _bilinear("x", gamma5, "s1", "s2")
    current_a = _bilinear("y", gammaX, "s3", "s4")
    current_b = _bilinear("y", gamma5, "s5", "s6")
    source = _bilinear("z", gamma5, "s7", "s8")

    corr = Correlator(sink, current_a + 2.0 * current_b, source)
    terms = ExpansionPass.run_correlator(corr)

    assert len(terms) == 2
    assert {weight for weight, _ in terms} == {1.0 + 0.0j, 2.0 + 0.0j}
    assert all(len(operators) == 3 for _, operators in terms)
    assert all(operators[0] == sink for _, operators in terms)
    assert all(operators[2] == source for _, operators in terms)


def test_wick_cache_keys_on_ordered_npoint_tuple():
    op_a = _bilinear("x", gamma5, "s1", "s2")
    op_b = _bilinear("y", gammaX, "s3", "s4")
    op_c = _bilinear("z", gamma5, "s5", "s6")

    cache = WickCache()
    corr1, terms1 = cache.get_or_contract_operators((op_a, op_b, op_c))
    corr2, terms2 = cache.get_or_contract_operators((op_a, op_b, op_c))

    assert corr1 is corr2
    assert terms1 is terms2
    assert cache.misses == 1
    assert cache.hits == 1

    cache.get_or_contract_operators((op_a, op_c, op_b))
    assert cache.misses == 2


def test_linear_combination_weight_reaches_physics_ir():
    op_a = _bilinear("x", gamma5, "s1", "s2")
    op_b = _bilinear("y", gammaX, "s3", "s4")

    physics_ir = lower_correlator_to_physics_ir(Correlator(op_a + 2.0 * op_b))

    # Each one-point bilinear has one Wick contraction.  The fermionic sign is
    # common to both terms, so the magnitudes expose the frontend weights.
    assert len(physics_ir.correlators) == 2
    assert sorted(abs(term.coefficient) for term in physics_ir.correlators) == [1.0, 2.0]


def test_matrix_expansion_matches_complex_change_of_basis():
    basis = (
        _bilinear("x", gamma5, "s1", "s2"),
        _bilinear("x", gammaX, "s1", "s2"),
    )
    source_basis = tuple(operator.dagger() for operator in basis)
    rows = ((2 + 1j, -1 + 3j), (3 - 2j, 4 + 1j))
    combinations = [
        coefficients[0] * basis[0] + coefficients[1] * basis[1]
        for coefficients in rows
    ]
    matrix = CorrMatrix(combinations)

    for row_index, sink_weights in enumerate(rows):
        for column_index, source_weights in enumerate(rows):
            expected = {
                (basis[sink_index], source_basis[source_index]):
                sink_weight * source_weight.conjugate()
                for sink_index, sink_weight in enumerate(sink_weights)
                for source_index, source_weight in enumerate(source_weights)
            }
            actual = {
                operators: weight
                for weight, operators in ExpansionPass.run_correlator(matrix[row_index, column_index])
            }
            assert actual == expected


def test_prewick_does_not_repeat_normalization_or_unused_namespace_copies(monkeypatch):
    first = _bilinear("x", gamma5, "s1", "s2")
    second = _bilinear("x", gammaX, "s1", "s2")
    matrix = CorrMatrix([first + second, 2 * first + 3 * second])
    cache = WickCache()
    normalization_calls = []
    namespaces = []
    original_normalize = expansion.normalize_operator
    original_reindexed = Operator.reindexed

    def record_normalization(operator):
        normalization_calls.append(operator)
        return original_normalize(operator)

    def reject_repeated_normalization(operator):
        raise AssertionError("Pre-Wick tuples have already been normalized")

    def record_reindexed(operator, namespace):
        namespaces.append(namespace)
        return original_reindexed(operator, namespace)

    monkeypatch.setattr(expansion, "normalize_operator", record_normalization)
    monkeypatch.setattr(wick_cache, "normalize_operator", reject_repeated_normalization)
    monkeypatch.setattr(Operator, "reindexed", record_reindexed)

    for _, _, correlator in matrix:
        physics_ir = lower_correlator_to_physics_ir(correlator, cache)
        assert physics_ir.correlators

    assert cache.misses == 4
    assert cache.hits == 12
    assert len(normalization_calls) == 32
    assert len(namespaces) == 8
