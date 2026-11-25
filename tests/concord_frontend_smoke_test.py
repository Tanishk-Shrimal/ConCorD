from math import sqrt

from concord_lqcd import *
from concord_lqcd.core.wick import wick_contract
from concord_lqcd.compiler.expansion import ExpansionPass


def diag_u(pos, gamma_factory=gamma5, p=(0, 0, 0)):
    return Operator(
        ubar(pos, "a", "s1")
        * gamma_factory("s1", "s2")
        * u(pos, "a", "s2"),
        momentum=p,
    )


def test_structural_identity():
    a = Operator(
        ubar("x", "a", "s1") * gamma5("s1", "s2") * u("x", "a", "s2"),
        momentum=(1, 0, 0),
        label="first",
    )
    b = Operator(
        ubar("y", "c", "alpha") * gamma5("alpha", "beta") * u("y", "c", "beta"),
        momentum=(1, 0, 0),
        label="second",
    )
    assert a == b
    assert hash(a) == hash(b)
    assert len(a - b) == 0
    assert a.dagger() == b.dagger()


def test_scalar_division():
    a = diag_u("x")
    combo = (a + a) / 2
    assert len(combo) == 1
    coeff, op = combo.terms[0]
    assert coeff == 1
    assert op == a
    half = a / 2
    assert half.terms[0][0] == 0.5


def test_npoint_linear_expansion():
    sink = diag_u("x", gamma5)
    jx = diag_u("y", gammaX)
    jy = diag_u("z", gammaY)
    src = diag_u("w", gamma5).dagger()

    corr = Correlator(sink, (jx + 1j * jy) / sqrt(2), src)
    assert corr.n_operators == 3
    assert not corr.is_basis

    terms = list(corr.iter_basis_terms())
    assert len(terms) == 2
    weights = {complex(round(w.real, 15), round(w.imag, 15)) for w, _ in terms}
    expected = {
        complex(round(1 / sqrt(2), 15), 0),
        complex(0, round(1 / sqrt(2), 15)),
    }
    assert weights == expected
    assert all(basis.is_basis for _, basis in terms)
    assert all(len(wick_contract(basis)) > 0 for _, basis in terms)


def test_basis_arbitrary_npoint_and_onepoint():
    one = Correlator(diag_u("x"))
    assert len(wick_contract(one)) == 1

    three = Correlator(diag_u("x"), diag_u("y", gammaX), diag_u("z", gammaY))
    assert three.is_basis
    assert len(wick_contract(three)) == 6


def test_corrmatrix_index_preserves_combination():
    a = diag_u("x", gamma5)
    b = diag_u("y", gammaX)
    op = (a + 1j * b) / sqrt(2)
    matrix = CorrMatrix([op])
    entry = matrix[0, 0]
    assert not entry.is_basis
    assert len(list(entry.iter_basis_terms())) == 4

    expanded = ExpansionPass.run_correlator(entry)
    assert len(expanded) == 4


def main():
    test_structural_identity()
    test_scalar_division()
    test_npoint_linear_expansion()
    test_basis_arbitrary_npoint_and_onepoint()
    test_corrmatrix_index_preserves_combination()
    print("All frontend smoke tests passed.")


if __name__ == "__main__":
    main()
