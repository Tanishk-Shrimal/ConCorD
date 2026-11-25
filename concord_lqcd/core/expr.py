"""
Expression tree for the symbolic frontend.

Every user-visible symbolic object is an Expr.  The composite node is
Product (ordered tensor product, built with ``*``).

Field-level expressions remain multiplicative tensor expressions.
Linear combinations live one layer above this tree in ``OperatorCombination``;
this keeps field algebra separate from operator-space linear algebra while
allowing arbitrary projected/rotated operator combinations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .indices import Index


# ── Base ───────────────────────────────────────────────────────────────

class Expr(ABC):
    """Base class for every symbolic expression node."""

    @property
    @abstractmethod
    def indices(self) -> tuple[Index, ...]:
        """All indices carried by this node (not recursively)."""
        ...

    def __mul__(self, other: Expr) -> Product:
        if not isinstance(other, Expr):
            return NotImplemented
        return Product._make(self, other)

    def __rmul__(self, other: Expr) -> Product:
        if not isinstance(other, Expr):
            return NotImplemented
        return Product._make(other, self)


# ── Product ────────────────────────────────────────────────────────────

class Product(Expr):
    """Ordered product of symbolic expressions.

    Products are *flat*: ``(A * B) * C`` stores ``(A, B, C)`` rather
    than a nested tree.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors: tuple[Expr, ...]) -> None:
        self._factors = factors

    @classmethod
    def _make(cls, left: Expr, right: Expr) -> Product:
        parts: list[Expr] = []
        if isinstance(left, Product):
            parts.extend(left._factors)
        else:
            parts.append(left)
        if isinstance(right, Product):
            parts.extend(right._factors)
        else:
            parts.append(right)
        return cls(tuple(parts))

    @property
    def factors(self) -> tuple[Expr, ...]:
        return self._factors

    @property
    def indices(self) -> tuple[Index, ...]:
        out: list[Index] = []
        for f in self._factors:
            out.extend(f.indices)
        return tuple(out)

    def __len__(self) -> int:
        return len(self._factors)

    def __getitem__(self, i: int) -> Expr:
        return self._factors[i]

    def __iter__(self):
        return iter(self._factors)

    def __repr__(self) -> str:
        return " * ".join(repr(f) for f in self._factors)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Product):
            return NotImplemented
        return self._factors == other._factors

    def __hash__(self) -> int:
        return hash(self._factors)