"""Tests for concord_lqcd.core.correlator — CorrMatrix and Correlator."""

import pytest

from concord_lqcd.core import (
    Operator, Correlator, CorrMatrix,
    dbar, u, gamma5, gammaX, delta,
    reset_registry,
)


def _make_pion():
    reset_registry()
    return Operator(
        dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'b', 'β') * delta('a', 'b'),
        momentum=(0, 0, 1),
    )


def _make_rho():
    reset_registry()
    return Operator(
        dbar('x', 'a', 'α') * gammaX('α', 'β') * u('x', 'b', 'β') * delta('a', 'b'),
    )


def test_corrmatrix_shape_2x2():
    pion = _make_pion()
    rho = _make_rho()
    mat = CorrMatrix([pion, rho])
    assert mat.shape == (2, 2)


def test_corrmatrix_shape_1x1():
    pion = _make_pion()
    mat = CorrMatrix([pion])
    assert mat.shape == (1, 1)


def test_corrmatrix_getitem_returns_correlator():
    pion = _make_pion()
    rho = _make_rho()
    rho_dag = rho.dagger()
    mat = CorrMatrix([pion, rho])
    c01 = mat[0, 1]
    assert isinstance(c01, Correlator)
    assert c01.operators[0] is pion


def test_corrmatrix_diagonal_element():
    pion = _make_pion()
    rho = _make_rho()
    pion_dag = pion.dagger()
    mat = CorrMatrix([pion, rho])
    c00 = mat[0, 0]
    assert c00.operators[0] is pion


def test_corrmatrix_default_source_daggers_complex_combination_only_once():
    pion = _make_pion()
    sink = (1 + 2j) * pion

    default = CorrMatrix([sink])
    source_coefficient, source_operator = default.sources[0].terms[0]
    assert source_coefficient == 1 - 2j
    assert source_operator == pion.dagger()
    assert [weight for weight, _ in default[0, 0].iter_basis_terms()] == [5 + 0j]

    explicit = CorrMatrix([sink], [sink])
    source_coefficient, source_operator = explicit.sources[0].terms[0]
    assert source_coefficient == 1 + 2j
    assert source_operator == pion
    assert [weight for weight, _ in explicit[0, 0].iter_basis_terms()] == [-3 + 4j]


def test_correlator_multilinear_expansion_multiplies_complex_coefficients():
    pion = _make_pion()
    rho = _make_rho()
    sink = (1 + 2j) * pion + rho
    source = (2 - 1j) * pion

    terms = list(Correlator(sink, source).iter_basis_terms())

    assert len(terms) == 2
    assert all(basis.is_basis and basis.operators[1] == pion for _, basis in terms)
    weights = {basis.operators[0]: weight for weight, basis in terms}
    assert weights == {pion: 4 + 3j, rho: 2 - 1j}


@pytest.mark.parametrize(
    "coefficients",
    [(1e200, 1e200), (1e200j, 1e200j), (1e100, 1e100, 1e200)],
)
def test_correlator_rejects_coefficient_product_overflow(coefficients):
    pion = _make_pion()
    correlator = Correlator(*(coefficient * pion for coefficient in coefficients))

    with pytest.raises(ValueError, match="coefficient product must be finite"):
        tuple(correlator.iter_basis_terms())


def test_correlator_preserves_large_finite_coefficient_product():
    pion = _make_pion()
    correlator = Correlator(1e150 * pion, 1e150 * pion.dagger())
    terms = tuple(correlator.iter_basis_terms())

    assert len(terms) == 1
    assert terms[0][0] == complex(1e150) * complex(1e150)


def test_correlator_creates_symbolic_namespaces_only_when_needed(monkeypatch):
    pion = _make_pion()
    source = pion.dagger()
    namespaces = []
    original_reindexed = Operator.reindexed

    def record_reindexed(operator, namespace):
        namespaces.append(namespace)
        return original_reindexed(operator, namespace)

    monkeypatch.setattr(Operator, "reindexed", record_reindexed)
    correlator = Correlator(pion, source)

    assert correlator.operators == (pion, source)
    assert namespaces == []
    symbolic_operators = correlator.symbolic_operators
    assert len(namespaces) == 2
    assert correlator.symbolic_operators is symbolic_operators
    assert correlator.operator_fields
    assert len(namespaces) == 2


def test_basis_expansion_does_not_create_unused_symbolic_namespaces(monkeypatch):
    pion = _make_pion()
    rho = _make_rho()
    source = pion.dagger()
    correlator = Correlator(pion + 2 * rho, source)

    def reject_unused_reindexing(operator, namespace):
        raise AssertionError("Expansion does not need symbolic namespace copies")

    monkeypatch.setattr(Operator, "reindexed", reject_unused_reindexing)
    terms = tuple(correlator.iter_basis_terms())

    assert len(terms) == 2
    assert all(basis.is_basis and basis.operators[1] == source for _, basis in terms)


def test_nonbasis_correlator_still_rejects_symbolic_operator_access():
    pion = _make_pion()
    correlator = Correlator(2 * pion)

    with pytest.raises(ValueError, match="contains linear combinations"):
        _ = correlator.symbolic_operators


def test_correlator_symbolic_operator_namespaces():
    pion = _make_pion()
    rho = _make_rho()
    corr = Correlator(pion, rho)
    op0, op1 = corr.symbolic_operators
    op0_labels = {idx.label for idx in op0.indices}
    op1_labels = {idx.label for idx in op1.indices}
    assert all(':op0' in label for label in op0_labels)
    assert all(':op1' in label for label in op1_labels)
    assert op0_labels.isdisjoint(op1_labels)


def test_correlators_get_unique_namespaces():
    pion = _make_pion()
    rho = _make_rho()
    corr1 = Correlator(pion, rho)
    corr2 = Correlator(pion, rho)
    labels1 = {idx.label for op in corr1.symbolic_operators for idx in op.indices}
    labels2 = {idx.label for op in corr2.symbolic_operators for idx in op.indices}
    assert labels1.isdisjoint(labels2)

def test_correlator_single_operator_allowed():
    pion = _make_pion()

    corr = Correlator(pion)

    assert len(corr.operators) == 1
    assert corr.operators[0] == pion

def test_correlator_requires_at_least_one_operator():
    with pytest.raises(
        ValueError,
        match="at least one operator",
    ):
        Correlator()

def test_corrmatrix_iteration_yields_all_triples():
    pion = _make_pion()
    rho = _make_rho()
    mat = CorrMatrix([pion, rho])
    triples = list(mat)
    assert len(triples) == 4
    for i, j, corr in triples:
        assert isinstance(i, int)
        assert isinstance(j, int)
        assert isinstance(corr, Correlator)
    indices = [(i, j) for i, j, _ in triples]
    assert indices == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_corrmatrix_diagonal():
    pion = _make_pion()
    rho = _make_rho()
    pion_dag = pion.dagger()
    rho_dag = rho.dagger()
    mat = CorrMatrix([pion, rho])
    diag = list(mat.diagonal())
    assert len(diag) == 2
    i0, c0 = diag[0]
    i1, c1 = diag[1]
    assert i0 == 0
    assert c0.operators[0] is pion
    assert i1 == 1
    assert c1.operators[0] is rho


def test_corrmatrix_empty_raises():
    with pytest.raises(ValueError, match="at least one sink operator"):
        CorrMatrix([])


def test_corrmatrix_out_of_range_raises():
    pion = _make_pion()
    mat = CorrMatrix([pion])
    with pytest.raises(IndexError):
        _ = mat[1, 0]

    with pytest.raises(IndexError):
        _ = mat[0, 1]
