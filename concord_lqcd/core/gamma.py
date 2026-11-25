"""
Gamma matrix constructors.

Usage::

    from concord_lqcd import *

    g = gamma5('α', 'β')    # γ₅ with spin indices α, β

All gamma factory functions create typed spin-index values without shared state.
"""

from __future__ import annotations

from typing import Any

from .expr import Expr
from .indices import Index, SPIN


class GammaMatrix(Expr):
    """An indexed gamma matrix Γ(left_spin, right_spin).

    Parameters
    ----------
    kind : str
        Identifier for the gamma type ('gamma5', 'gammaX', etc.).
    left : Index
        Left spin index.
    right : Index
        Right spin index.
    """

    __slots__ = ("_kind", "_left", "_right")

    def __init__(self, kind: str, left: Index, right: Index) -> None:
        self._kind = kind
        self._left = left
        self._right = right

    @property
    def output_type(self) -> Any:
        from .operations import FieldType
        return FieldType.TENSOR

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def left(self) -> Index:
        return self._left

    @property
    def right(self) -> Index:
        return self._right

    @property
    def indices(self) -> tuple[Index, ...]:
        return (self._left, self._right)

    def __repr__(self) -> str:
        return f"{self._kind}({self._left.label},{self._right.label})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GammaMatrix):
            return NotImplemented
        return (
            self._kind == other._kind
            and self._left == other._left
            and self._right == other._right
        )

    def __hash__(self) -> int:
        return hash((self._kind, self._left, self._right))


# ── Factory helpers ────────────────────────────────────────────────────

def _make_gamma(kind: str, left_label: str, right_label: str) -> GammaMatrix:
    return GammaMatrix(kind, Index(left_label, SPIN), Index(right_label, SPIN))


def gamma5(left: str, right: str) -> GammaMatrix:
    """γ₅ matrix."""
    return _make_gamma("gamma5", left, right)


def gammaX(left: str, right: str) -> GammaMatrix:
    """γ_x (spatial-x) matrix."""
    return _make_gamma("gammaX", left, right)


def gammaY(left: str, right: str) -> GammaMatrix:
    """γ_y (spatial-y) matrix."""
    return _make_gamma("gammaY", left, right)


def gammaZ(left: str, right: str) -> GammaMatrix:
    """γ_z (spatial-z) matrix."""
    return _make_gamma("gammaZ", left, right)


def gammaT(left: str, right: str) -> GammaMatrix:
    """γ_t (temporal) matrix."""
    return _make_gamma("gammaT", left, right)


def gamma_identity(left: str, right: str) -> GammaMatrix:
    """Identity matrix in spin space (δ_αβ)."""
    return _make_gamma("gamma_id", left, right)


def spin_matrix(kind: str, left: str, right: str) -> GammaMatrix:
    """Generic named spin matrix.

    The numeric backend resolves ``kind`` through ``get_spin_matrices`` in the
    selected spin basis. This is useful for charge-conjugation and projected
    matrices such as ``C``, ``Cg5``, or ``Pplus`` without adding a one-off
    constructor for every key.
    """
    return _make_gamma(kind, left, right)


def charge_conjugation(left: str, right: str) -> GammaMatrix:
    """Charge-conjugation matrix C."""
    return spin_matrix("C", left, right)


def cgamma5(left: str, right: str) -> GammaMatrix:
    """Diquark matrix Cγ₅."""
    return spin_matrix("Cg5", left, right)
