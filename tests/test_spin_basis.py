"""Tests for the selectable Chroma spin-basis conventions."""

import numpy as np
import pytest

from concord_lqcd.core.dirac_basis import SPIN_MATRICES, get_spin_matrices


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_spin_basis_satisfies_euclidean_clifford_algebra(basis):
    matrices = get_spin_matrices(basis)
    gammas = [matrices[f"g{index}"] for index in range(1, 5)]
    identity = np.eye(4, dtype=complex)

    for left, gamma_left in enumerate(gammas):
        for right, gamma_right in enumerate(gammas):
            expected = 2.0 * identity if left == right else np.zeros((4, 4))
            np.testing.assert_allclose(
                gamma_left @ gamma_right + gamma_right @ gamma_left,
                expected,
                atol=1.0e-14,
            )


# QDP++ GammaConst<4,n>, written as the action on a spinor r: (gamma r)_0 .. (gamma r)_3.
_QDP_ACTION = {
    1: ((3, 1j), (2, 1j), (1, -1j), (0, -1j)),   # gamma_1
    2: ((3, -1), (2, 1), (1, 1), (0, -1)),       # gamma_2
    4: ((2, 1j), (3, -1j), (0, -1j), (1, 1j)),   # gamma_3
    8: ((2, 1), (3, 1), (0, 1), (1, 1)),         # gamma_4
}


def _qdp_gamma(n):
    matrix = np.zeros((4, 4), dtype=complex)
    for row, (column, value) in enumerate(_QDP_ACTION[n]):
        matrix[row, column] = value
    return matrix


def test_dr_gammas_are_the_chroma_qdp_degrand_rossi_matrices():
    matrices = get_spin_matrices("DR")
    for kinds, n in ((("g1", "gammaX", "V1"), 1), (("g2", "gammaY", "V2"), 2),
                     (("g3", "gammaZ", "V3"), 4), (("g4", "gammaT", "V4"), 8)):
        for kind in kinds:
            np.testing.assert_array_equal(matrices[kind], _qdp_gamma(n), err_msg=kind)


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_chroma_gamma_index_is_the_ordered_product(basis):
    matrices = get_spin_matrices(basis)
    gammas = [matrices[f"g{index}"] for index in range(1, 5)]
    for index in range(16):
        expected = np.eye(4, dtype=complex)
        for bit, gamma in enumerate(gammas):
            if (index >> bit) & 1:
                expected = expected @ gamma
        np.testing.assert_allclose(matrices[f"chroma_{index}"], expected, atol=1.0e-14)
    np.testing.assert_allclose(matrices["chroma_15"], matrices["gamma5"], atol=1.0e-14)


def test_dp_spatial_gammas_have_the_same_dirac_pauli_form_for_every_direction():
    matrices = get_spin_matrices("DP")
    zero = np.zeros((2, 2), dtype=complex)
    paulis = (
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    )
    for index, sigma in enumerate(paulis, start=1):
        expected = np.block([[zero, -1j * sigma], [1j * sigma, zero]])
        np.testing.assert_allclose(matrices[f"g{index}"], expected, atol=1.0e-14)

def test_dp_temporal_gamma_matches_dirac_pauli_convention():
    temporal = get_spin_matrices("DP")["gammaT"]
    np.testing.assert_allclose(
        temporal,
        np.diag([1.0, 1.0, -1.0, -1.0]),
        atol=1.0e-14,
    )


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_diquark_cgamma5_is_consistent_with_c_and_gamma5(basis):
    matrices = get_spin_matrices(basis)
    np.testing.assert_allclose(
        matrices["Cg5"],
        matrices["C"] @ matrices["gamma5"],
        atol=1.0e-14,
    )


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_spin_matrix_cache_reuses_the_read_only_table(basis):
    assert get_spin_matrices(basis) is get_spin_matrices(basis)


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_spin_matrix_table_rejects_replacement_addition_and_deletion(basis):
    matrices = get_spin_matrices(basis)
    expected = matrices["gamma5"].copy()

    with pytest.raises(TypeError):
        matrices["gamma5"] = np.zeros((4, 4), dtype=complex)
    with pytest.raises(TypeError):
        matrices["custom"] = np.eye(4, dtype=complex)
    with pytest.raises(TypeError):
        del matrices["gamma5"]

    np.testing.assert_array_equal(matrices["gamma5"], expected)
    assert "custom" not in matrices


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_all_spin_matrices_and_views_reject_writes(basis):
    for kind, matrix in get_spin_matrices(basis).items():
        assert matrix.shape == (4, 4), kind
        assert matrix.dtype == np.dtype("complex128"), kind
        assert not matrix.flags.writeable, kind
        expected = matrix.copy()

        with pytest.raises(ValueError):
            matrix[0, 0] = 7.0
        with pytest.raises(ValueError):
            matrix.setflags(write=True)
        with pytest.raises(ValueError):
            matrix.T[0, 0] = 7.0
        with pytest.raises(ValueError):
            np.asarray(matrix).setflags(write=True)
        with pytest.raises(ValueError):
            matrix.base.setflags(write=True)

        np.testing.assert_array_equal(matrix, expected)


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_spin_matrix_copy_can_be_edited_without_changing_either_basis(basis):
    references = {
        name: get_spin_matrices(name)["gamma5"].copy()
        for name in ("DR", "DP")
    }
    editable = get_spin_matrices(basis)["gamma5"].copy()
    assert editable.flags.writeable
    editable[0, 0] += 7.0
    assert not np.array_equal(editable, references[basis])

    for name, expected in references.items():
        for alias in ("g5", "gamma5", "P"):
            np.testing.assert_array_equal(get_spin_matrices(name)[alias], expected)


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_spin_matrix_cache_rebuild_preserves_values_and_old_references(basis):
    matrices = get_spin_matrices(basis)
    references = {kind: matrix.copy() for kind, matrix in matrices.items()}

    get_spin_matrices.cache_clear()
    rebuilt = get_spin_matrices(basis)

    assert rebuilt is not matrices
    assert rebuilt.keys() == matrices.keys()
    for kind, expected in references.items():
        np.testing.assert_array_equal(rebuilt[kind], expected)
        np.testing.assert_array_equal(matrices[kind], expected)
        assert not rebuilt[kind].flags.writeable
        assert not matrices[kind].flags.writeable


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_named_spin_adjoints_remain_consistent_with_their_matrices(basis):
    matrices = get_spin_matrices(basis)
    for kind, matrix in matrices.items():
        if not kind.endswith("_dag"):
            np.testing.assert_array_equal(matrices[f"{kind}_dag"], matrix.conj().T)


def test_dr_compatibility_alias_stays_read_only_after_cache_rebuild():
    get_spin_matrices.cache_clear()
    matrices = get_spin_matrices("DR")

    assert SPIN_MATRICES.keys() == matrices.keys()
    for kind, matrix in SPIN_MATRICES.items():
        np.testing.assert_array_equal(matrix, matrices[kind])
        assert not matrix.flags.writeable

    with pytest.raises(TypeError):
        SPIN_MATRICES["gamma5"] = np.eye(4, dtype=complex)
    with pytest.raises(ValueError):
        SPIN_MATRICES["gamma5"][0, 0] = 7.0


@pytest.mark.parametrize("basis", ["DR", "DP"])
def test_charge_conjugation_symmetries_hold_exactly_in_stored_matrices(basis):
    matrices = get_spin_matrices(basis)
    for mu in range(1, 5):
        np.testing.assert_array_equal(matrices[f"Cg{mu}"].T, matrices[f"Cg{mu}"])
    for kind in ("C", "Cg5", "Cg5Upper"):
        np.testing.assert_array_equal(matrices[kind].T, -matrices[kind])
    for matrix in matrices.values():
        np.testing.assert_array_equal(matrix * 8, np.round(matrix * 8))


def test_dp_matrices_are_the_exact_dirac_pauli_transform_of_dr():
    root = np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, -1, 0]], dtype=complex) / np.sqrt(2.0)
    dr, dp = get_spin_matrices("DR"), get_spin_matrices("DP")
    for mu in range(1, 5):
        np.testing.assert_allclose(dp[f"g{mu}"], root.conj().T @ dr[f"g{mu}"] @ root, rtol=0, atol=1e-15)
        np.testing.assert_allclose(dp[f"Cg{mu}"], root.T @ dr[f"Cg{mu}"] @ root, rtol=0, atol=1e-15)
