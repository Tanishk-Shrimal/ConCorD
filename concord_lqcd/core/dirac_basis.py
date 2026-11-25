"""Numerical Dirac-basis matrices shared by symbolic and runtime layers."""

from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Literal

import numpy as np

SpinBasis = Literal["DR", "DP"]


def _dirac_to_dr_integer_matrix() -> np.ndarray:
    return np.array(
        [[0, 1, 0, -1], [-1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, -1, 0]],
        dtype=complex,
    )


def _dirac_to_dr_matrix() -> np.ndarray:
    return _dirac_to_dr_integer_matrix() / np.sqrt(2.0)


# U = R / sqrt(2) with an integer matrix R. Every change of basis is computed as R^dag M R / 2,
# so matrices with entries in {0, +-1, +-i, +-1/2, ...} stay exact. Multiplying by U twice would
# leave 1e-16 rounding (0.9999999999999998) and hide exact symmetries such as (C g_mu)^T = C g_mu.
_R_DP_TO_DR = _dirac_to_dr_integer_matrix()
_U_DP_TO_DR = _dirac_to_dr_matrix()

# Chroma/QDP++ DeGrand-Rossi matrices (GammaConst<4,1|2|4|8> in qdp_primspinvec.h).  Up to
# version 0.1.0, gamma_1 and gamma_3 had the opposite sign (they were the transposes of these).
_DR_GAMMAS = (
    np.array(
        [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
        dtype=complex,
    ),
    np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex),
    np.array(
        [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
        dtype=complex,
    ),
    np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=complex),
)
_DR_GAMMA5 = np.diag([1, 1, -1, -1]).astype(complex)


def _ordinary_change_of_basis(matrix: np.ndarray, basis: SpinBasis) -> np.ndarray:
    if basis == "DR":
        return matrix
    return _R_DP_TO_DR.conj().T @ matrix @ _R_DP_TO_DR / 2.0


def _diquark_change_of_basis(matrix: np.ndarray, basis: SpinBasis) -> np.ndarray:
    if basis == "DR":
        return matrix
    return _R_DP_TO_DR.T @ matrix @ _R_DP_TO_DR / 2.0


def _build_spin_matrices(basis: SpinBasis) -> dict[str, np.ndarray]:
    if basis not in ("DR", "DP"):
        raise ValueError(f"spin basis must be 'DR' or 'DP', got {basis!r}")

    gammas = tuple(_ordinary_change_of_basis(matrix, basis) for matrix in _DR_GAMMAS)
    gamma5 = _ordinary_change_of_basis(_DR_GAMMA5, basis)
    identity = np.eye(4, dtype=complex)
    gamma1, gamma2, gamma3, gamma4 = gammas
    c_dr = _DR_GAMMAS[1] @ _DR_GAMMAS[3]
    c = _diquark_change_of_basis(c_dr, basis)
    cg = tuple(
        _diquark_change_of_basis(c_dr @ matrix, basis) for matrix in _DR_GAMMAS
    )
    temporal = gamma4
    pplus = (identity + temporal) / 2.0
    pminus = (identity - temporal) / 2.0
    sigma12 = 0.5 * (gamma1 @ gamma2 - gamma2 @ gamma1)
    sigma13 = 0.5 * (gamma1 @ gamma3 - gamma3 @ gamma1)
    sigma14 = 0.5 * (gamma1 @ gamma4 - gamma4 @ gamma1)
    sigma23 = 0.5 * (gamma2 @ gamma3 - gamma3 @ gamma2)
    sigma24 = 0.5 * (gamma2 @ gamma4 - gamma4 @ gamma2)
    sigma34 = 0.5 * (gamma3 @ gamma4 - gamma4 @ gamma3)

    upper_cg5_dp = np.zeros((4, 4), dtype=complex)
    upper_cg5_dp[0, 1] = -1.0
    upper_cg5_dp[1, 0] = 1.0
    upper_pplus_dp = np.zeros((4, 4), dtype=complex)
    upper_pplus_dp[0, 0] = 0.5
    upper_pplus_dp[1, 1] = 0.5
    if basis == "DP":
        upper_cg5 = upper_cg5_dp
        upper_pplus = upper_pplus_dp
    else:
        upper_cg5 = _R_DP_TO_DR.conj() @ upper_cg5_dp @ _R_DP_TO_DR.conj().T / 2.0
        upper_pplus = _R_DP_TO_DR @ upper_pplus_dp @ _R_DP_TO_DR.conj().T / 2.0

    matrices = {
        "Identity": identity,
        "C": c,
        "Cg1": cg[0],
        "Cg2": cg[1],
        "Cg3": cg[2],
        "Cg4": cg[3],
        "Cg5": _diquark_change_of_basis(c_dr @ _DR_GAMMA5, basis),
        "g5": gamma5,
        "gamma5": gamma5,
        "g4": gamma4,
        "gammaT": temporal,
        "g1": gamma1,
        "gammaX": gamma1,
        "g2": gamma2,
        "gammaY": gamma2,
        "g3": gamma3,
        "gammaZ": gamma3,
        "gamma_id": identity,
        "Pplus": pplus,
        "Pminus": pminus,
        "Cg5Upper": upper_cg5,
        "PplusUpper": upper_pplus,
        "S": identity,
        "P": gamma5,
        "V1": gamma1,
        "V2": gamma2,
        "V3": gamma3,
        "V4": gamma4,
        "A1": gamma5 @ gamma1,
        "A2": gamma5 @ gamma2,
        "A3": gamma5 @ gamma3,
        "A4": gamma5 @ gamma4,
        "T12": sigma12,
        "T13": sigma13,
        "T14": sigma14,
        "T23": sigma23,
        "T24": sigma24,
        "T34": sigma34,
    }

    def get_chroma_gamma(index: int, gamma_set: tuple[np.ndarray, ...]) -> np.ndarray:
        matrix = identity
        for bit, gamma in enumerate(gamma_set):
            if (index >> bit) & 1:
                matrix = matrix @ gamma
        return matrix

    dr_chroma_gammas = tuple(get_chroma_gamma(index, _DR_GAMMAS) for index in range(16))
    for index in range(16):
        gamma_matrix = get_chroma_gamma(index, gammas)
        matrices[f"chroma_{index}"] = gamma_matrix
        matrices[f"chroma_src_{index}"] = temporal @ gamma_matrix.conj().T @ temporal
        matrices[f"C_{index}"] = _diquark_change_of_basis(
            c_dr @ dr_chroma_gammas[index], basis
        )

    for key, matrix in tuple(matrices.items()):
        matrices[f"{key}_dag"] = matrix.conj().T
    return matrices


def _readonly_matrix(matrix: np.ndarray) -> np.ndarray:
    """Detach a matrix into an immutable byte-backed NumPy array."""
    return np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)


@lru_cache(maxsize=2)
def get_spin_matrices(basis: SpinBasis = "DR") -> Mapping[str, np.ndarray]:
    """Return shared, read-only spin matrices in the requested Chroma basis.

    Use ``get_spin_matrices(basis)[kind].copy()`` for an editable matrix.
    """
    return MappingProxyType({
        kind: _readonly_matrix(matrix)
        for kind, matrix in _build_spin_matrices(basis).items()
    })


SPIN_MATRICES = get_spin_matrices("DR")
