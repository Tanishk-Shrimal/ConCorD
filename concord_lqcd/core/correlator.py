"""
Correlation functions and two-point correlation matrices.

``Correlator`` is the generic ordered N-point frontend object.  Every
insertion may be either a basis ``Operator`` or an ``OperatorCombination``.
Linear combinations are expanded only when requested, keeping the symbolic
frontend independent of Wick contraction and backend lowering.
"""

from __future__ import annotations

from itertools import count, product
from math import isfinite
from typing import Iterator, Sequence

from .fields import QuarkField, quark_fields
from .operator import Operator
from .operator_expression import OperatorLike, OperatorCombination


class Correlator:
    """Ordered symbolic N-point correlation function.

    Examples
    --------
    A basis three-point function::

        C = Correlator(O_sink, J, O_source)

    An insertion may itself be a linear combination::

        C = Correlator(O_sink, Jx + 1j * Jy, O_source)

    The latter is not Wick-contracted directly.  ``iter_basis_terms()``
    expands it into weighted basis correlators first.
    """

    __slots__ = ("_operator_combinations", "_symbolic_operators", "_symbolic_namespace")

    _counter = count()

    def __init__(self, *operators: OperatorLike) -> None:
        if len(operators) < 1:
            raise ValueError("A correlation function requires at least one operator.")

        combinations: list[OperatorCombination] = []
        for i, op in enumerate(operators):
            try:
                combinations.append(OperatorCombination.from_element(op))
            except (AttributeError, TypeError) as exc:
                raise TypeError(
                    f"Correlator insertion {i} must be an Operator or OperatorLike "
                    f"linear combination, got {type(op).__name__}."
                ) from exc

        self._operator_combinations = tuple(combinations)
        self._symbolic_operators: tuple[Operator, ...] | None = None
        self._symbolic_namespace: str | None = None

    @classmethod
    def _with_symbolic_namespace(
        cls, operators: Sequence[Operator], namespace: str,
    ) -> Correlator:
        """Create a compiler-owned basis context without changing user objects."""
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("symbolic namespace must be a non-empty string")
        correlator = cls(*operators)
        if not correlator.is_basis:
            raise ValueError("a symbolic namespace requires unit-weight basis operators")
        correlator._symbolic_namespace = namespace
        return correlator

    @property
    def operator_combinations(self) -> tuple[OperatorCombination, ...]:
        """All insertions in a uniform linear-combination representation."""
        return self._operator_combinations

    @property
    def operators(self) -> tuple[Operator | OperatorCombination, ...]:
        """User-facing insertions.

        A plain unit-weight basis insertion is returned as ``Operator`` for
        backward compatibility.  Nontrivial linear combinations remain
        ``OperatorCombination`` objects.
        """
        out: list[Operator | OperatorCombination] = []
        for comb in self._operator_combinations:
            if len(comb.terms) == 1 and comb.terms[0][0] == 1.0 + 0.0j:
                out.append(comb.terms[0][1])
            else:
                out.append(comb)
        return tuple(out)

    @property
    def is_basis(self) -> bool:
        """True iff every insertion is one unit-weight basis Operator."""
        return all(
            len(comb.terms) == 1 and comb.terms[0][0] == 1.0 + 0.0j
            for comb in self._operator_combinations
        )

    @property
    def is_zero(self) -> bool:
        """True if at least one insertion is the zero linear combination."""
        return any(len(comb.terms) == 0 for comb in self._operator_combinations)

    @property
    def symbolic_operators(self) -> tuple[Operator, ...]:
        """Collision-free basis operators used by Wick contraction.

        A correlator containing a nontrivial linear combination does not have
        one unique symbolic operator sequence.  Expand it first with
        ``iter_basis_terms()``. Basis namespace copies are built on first
        access and reused by Wick contraction and lowering.
        """
        if not self.is_basis:
            raise ValueError(
                "This Correlator contains linear combinations. Expand it with "
                "`iter_basis_terms()` before Wick contraction or lowering."
            )
        if self._symbolic_operators is None:
            basis_operators = tuple(
                combination.terms[0][1] for combination in self._operator_combinations
            )
            namespace = self._symbolic_namespace
            if namespace is None:
                namespace = f"c{next(self._counter)}"
            self._symbolic_operators = tuple(
                operator.reindexed(f"{namespace}:op{insertion_index}")
                for insertion_index, operator in enumerate(basis_operators)
            )
        return self._symbolic_operators

    @property
    def operator_fields(self) -> tuple[list[QuarkField], ...]:
        return tuple(quark_fields(op.expr) for op in self.symbolic_operators)

    @property
    def n_operators(self) -> int:
        return len(self._operator_combinations)

    @property
    def operator_ids(self) -> tuple[int, ...]:
        return tuple(range(self.n_operators))

    def iter_basis_terms(self) -> Iterator[tuple[complex, Correlator]]:
        """Expand arbitrary insertion-wise linear combinations.

        For a product of insertion-wise sums, the emitted weight is the
        product of the selected insertion coefficients, yielding
        ``(a_i * b_j * ..., Correlator(O_i, J_j, ...))``.
        Dagger conjugation is already handled by ``OperatorCombination.dagger``;
        no additional coefficient conjugation is performed here.
        Products that overflow the finite complex representation raise
        ``ValueError`` rather than passing nonfinite coefficients to compilation.
        """
        if self.is_zero:
            return

        term_lists = [comb.terms for comb in self._operator_combinations]
        for selected in product(*term_lists):
            weight = 1.0 + 0.0j
            basis_ops: list[Operator] = []
            for insertion_index, (coefficient, operator) in enumerate(selected):
                weight *= coefficient
                if not isfinite(weight.real) or not isfinite(weight.imag):
                    raise ValueError(
                        "Correlator coefficient product must be finite; "
                        f"overflow at insertion {insertion_index}."
                    )
                basis_ops.append(operator)
            if weight != 0:
                yield weight, Correlator(*basis_ops)

    def __repr__(self) -> str:
        ops_str = " | ".join(repr(op) for op in self.operators)
        return f"⟨{ops_str}⟩"


class CorrMatrix:
    """M×N matrix of two-point correlator expressions.

    Both rows and columns may contain basis ``Operator`` objects or arbitrary
    ``OperatorCombination`` expressions.  Matrix indexing preserves the full
    linear combination; it never selects only the first basis term.
    """

    def __init__(
        self,
        sinks: Sequence[OperatorLike],
        sources: Sequence[OperatorLike] | None = None,
    ) -> None:
        if not sinks:
            raise ValueError("CorrMatrix requires at least one sink operator.")
        self._sinks = [OperatorCombination.from_element(op) for op in sinks]
        if sources is None:
            self._sources = [op.dagger() for op in self._sinks]
        else:
            self._sources = [OperatorCombination.from_element(op) for op in sources]
        self._n_rows = len(self._sinks)
        self._n_cols = len(self._sources)

    @property
    def sinks(self) -> list[OperatorCombination]:
        return list(self._sinks)

    @property
    def sources(self) -> list[OperatorCombination]:
        return list(self._sources)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n_rows, self._n_cols)

    def __getitem__(self, ij: tuple[int, int]) -> Correlator:
        i, j = ij
        if not (0 <= i < self._n_rows and 0 <= j < self._n_cols):
            raise IndexError(
                f"Index ({i}, {j}) out of range for {self._n_rows}×{self._n_cols} CorrMatrix."
            )
        return Correlator(self._sinks[i], self._sources[j])

    def __iter__(self) -> Iterator[tuple[int, int, Correlator]]:
        for i in range(self._n_rows):
            for j in range(self._n_cols):
                yield i, j, self[i, j]

    def diagonal(self) -> Iterator[tuple[int, Correlator]]:
        if self._n_rows != self._n_cols:
            raise ValueError("Cannot get diagonal of non-square matrix.")
        for i in range(self._n_rows):
            yield i, self[i, i]

    def __repr__(self) -> str:
        header = f"CorrMatrix({self._n_rows}×{self._n_cols})\n"
        rows: list[str] = []
        for i in range(self._n_rows):
            row_entries = []
            for j in range(self._n_cols):
                row_entries.append(f"C[{i},{j}]")
            rows.append("  " + "  ".join(row_entries))
        return header + "\n".join(rows)
