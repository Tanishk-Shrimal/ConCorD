"""OperatorLike Protocol and OperatorCombination Linear Algebra Abstraction.

Enables formal symbolic linear combinations sum_i (c_i * O_i) of basis Operator instances,
self-canonicalizing on construction while decoupling linear algebra from physicsExpr.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable, Iterable
from math import isfinite

from .operator import Operator


@runtime_checkable
class OperatorLike(Protocol):
    """Protocol for any object that behaves like a symbolic operator or linear combination."""

    def iter_terms(self) -> Iterator[tuple[complex, Operator]]:
        """Yield (coefficient, Operator) tuples for all terms in the combination."""
        ...

    def dagger(self) -> OperatorLike:
        """Return the Hermitian conjugate operator or combination."""
        ...


class OperatorCombination:
    """Represents a formal linear combination sum_i (c_i * O_i) of basis Operator instances.

    Properties:
    - Immutable and self-canonicalizing upon construction.
    - Combines duplicate basis operators and removes exactly zero coefficients.
    - Preserves exposed spin/color interfaces when identifying duplicates.
    - Fully supports scalar multiplication, addition, subtraction, and daggering.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: Iterable[tuple[complex | float | int, Operator]]):
        term_map: dict[Operator, complex] = {}
        for coeff, op in terms:
            if not isinstance(op, Operator):
                raise TypeError(f"Expected Operator instance in term, got {type(op).__name__}")
            c = complex(coeff)
            if not isfinite(c.real) or not isfinite(c.imag):
                raise ValueError("Operator coefficients must be finite.")
            combined = term_map.get(op, 0.0 + 0.0j) + c
            if not isfinite(combined.real) or not isfinite(combined.imag):
                raise ValueError("Combined operator coefficients must be finite.")
            term_map[op] = combined

        filtered_terms = tuple(
            (c, op) for op, c in term_map.items() if c != 0
        )

        # Sort by mathematical operator identity rather than user-chosen dummy labels.
        self._terms: tuple[tuple[complex, Operator], ...] = tuple(
            sorted(filtered_terms, key=lambda pair: repr(pair[1].structural_key))
        )

    @classmethod
    def from_element(cls, item: OperatorLike) -> OperatorCombination:
        """Wrap an OperatorLike object into an OperatorCombination."""
        if isinstance(item, OperatorCombination):
            return item
        if isinstance(item, Operator):
            return cls([(1.0, item)])
        return cls(item.iter_terms())

    @property
    def terms(self) -> tuple[tuple[complex, Operator], ...]:
        return self._terms

    def iter_terms(self) -> Iterator[tuple[complex, Operator]]:
        yield from self._terms

    def canonicalize(self) -> OperatorCombination:
        """Return self (OperatorCombination is self-canonicalizing on construction)."""
        return self

    def dagger(self) -> OperatorCombination:
        """Hermitian conjugate of the linear combination: sum_i (c_i * O_i)^dag = sum_i (c_i* * O_i^dag)."""
        return OperatorCombination((c.conjugate(), op.dagger()) for c, op in self._terms)

    def __add__(self, other: OperatorLike | complex | float | int) -> OperatorCombination:
        if isinstance(other, (int, float, complex)):
            if other == 0:
                return self
            raise TypeError("Cannot add scalar non-zero constant to an OperatorCombination.")
        other_comb = OperatorCombination.from_element(other)
        return OperatorCombination(self._terms + other_comb._terms)

    def __radd__(self, other: OperatorLike | complex | float | int) -> OperatorCombination:
        return self.__add__(other)

    def __sub__(self, other: OperatorLike | complex | float | int) -> OperatorCombination:
        if isinstance(other, (int, float, complex)):
            if other == 0:
                return self
            raise TypeError("Cannot subtract scalar non-zero constant from an OperatorCombination.")
        other_comb = OperatorCombination.from_element(other)
        neg_terms = ((-c, op) for c, op in other_comb._terms)
        return OperatorCombination(self._terms + tuple(neg_terms))

    def __rsub__(self, other: OperatorLike | complex | float | int) -> OperatorCombination:
        return (-self).__add__(other)

    def __neg__(self) -> OperatorCombination:
        return OperatorCombination(((-c, op) for c, op in self._terms))

    def __truediv__(self, scalar: complex | float | int) -> OperatorCombination:
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide an OperatorCombination by zero.")
        inv = 1.0 / complex(scalar)
        return OperatorCombination(((inv * c, op) for c, op in self._terms))

    def __mul__(self, scalar: complex | float | int | OperatorLike) -> OperatorCombination:
        if isinstance(scalar, (int, float, complex)):
            s = complex(scalar)
            return OperatorCombination(((s * c, op) for c, op in self._terms))
        if isinstance(scalar, OperatorLike):
            # Tensor product multiplication across all terms: (sum_i a_i A_i) * (sum_j b_j B_j)
            other_comb = OperatorCombination.from_element(scalar)
            prod_terms = []
            for c1, op1 in self._terms:
                for c2, op2 in other_comb._terms:
                    prod_terms.append((c1 * c2, op1 * op2))
            return OperatorCombination(prod_terms)
        return NotImplemented

    def __rmul__(self, scalar: complex | float | int | OperatorLike) -> OperatorCombination:
        if isinstance(scalar, (int, float, complex)):
            s = complex(scalar)
            return OperatorCombination(((s * c, op) for c, op in self._terms))
        if isinstance(scalar, OperatorLike):
            other_comb = OperatorCombination.from_element(scalar)
            return other_comb.__mul__(self)
        return NotImplemented

    def __len__(self) -> int:
        return len(self._terms)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OperatorCombination):
            return self._terms == other._terms
        if isinstance(other, Operator):
            return len(self._terms) == 1 and self._terms[0] == (1.0 + 0.0j, other)
        return False

    def __repr__(self) -> str:
        if not self._terms:
            return "OperatorCombination(0)"
        parts = []
        for coeff, op in self._terms:
            if coeff.imag == 0:
                c_str = f"{coeff.real:.4g}" if coeff.real != 1.0 else ""
            else:
                c_str = f"({coeff:.4g})"
            if c_str:
                parts.append(f"{c_str}*{op}")
            else:
                parts.append(f"{op}")
        return " + ".join(parts)
