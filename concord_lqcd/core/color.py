"""
Color tensor constructors.

Usage::

    from concord_lqcd import *

    d = delta('a', 'b')            # Kronecker δ_ab in color space
    e = epsilon('a', 'b', 'c')     # Levi-Civita ε_abc in color space

All factory functions create typed color-index values without shared state.
"""

from __future__ import annotations

from typing import Any

from .expr import Expr
from .indices import Index, COLOR


class ColorDelta(Expr):
    """Kronecker delta δ(a, b) in color space.

    Parameters
    ----------
    left : Index
        First color index.
    right : Index
        Second color index.
    """

    __slots__ = ("_left", "_right")

    def __init__(self, left: Index, right: Index) -> None:
        self._left = left
        self._right = right

    @property
    def output_type(self) -> Any:
        from .operations import FieldType
        return FieldType.TENSOR

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
        return f"δ({self._left.label},{self._right.label})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColorDelta):
            return NotImplemented
        return self._left == other._left and self._right == other._right

    def __hash__(self) -> int:
        return hash(("delta", self._left, self._right))


class ColorEpsilon(Expr):
    """Levi-Civita tensor ε(a, b, c) in color space.

    Parameters
    ----------
    i, j, k : Index
        Three color indices.
    """

    __slots__ = ("_i", "_j", "_k")

    def __init__(self, i: Index, j: Index, k: Index) -> None:
        self._i = i
        self._j = j
        self._k = k

    @property
    def output_type(self) -> Any:
        from .operations import FieldType
        return FieldType.TENSOR

    @property
    def i(self) -> Index:
        return self._i

    @property
    def j(self) -> Index:
        return self._j

    @property
    def k(self) -> Index:
        return self._k

    @property
    def indices(self) -> tuple[Index, ...]:
        return (self._i, self._j, self._k)

    def __repr__(self) -> str:
        return f"ε({self._i.label},{self._j.label},{self._k.label})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColorEpsilon):
            return NotImplemented
        return (
            self._i == other._i
            and self._j == other._j
            and self._k == other._k
        )

    def __hash__(self) -> int:
        return hash(("epsilon", self._i, self._j, self._k))


# ── Factory functions ──────────────────────────────────────────────────

def delta(a: str, b: str) -> ColorDelta:
    """Kronecker delta δ(a, b) in color space."""
    return ColorDelta(Index(a, COLOR), Index(b, COLOR))


def epsilon(a: str, b: str, c: str) -> ColorEpsilon:
    """Levi-Civita tensor ε(a, b, c) in color space."""
    return ColorEpsilon(Index(a, COLOR), Index(b, COLOR), Index(c, COLOR))
