"""Normalize and expand symbolic correlators into basis operator tuples."""

from __future__ import annotations

from math import isfinite

from concord_lqcd.core.operator import Operator
from concord_lqcd.core.correlator import Correlator
from concord_lqcd.compiler.normalization import normalize_operator


class ExpansionPass:
    """Compiler-side normalization and linear-combination expansion.

    Frontend ``Operator`` objects remain exactly as constructed by the user.
    Basis operators are normalized here before duplicate terms are merged, so
    canonicalization retains the same algebraic simplification semantics that
    previously happened during frontend construction.
    """

    @staticmethod
    def run_correlator(
        correlator: Correlator,
    ) -> tuple[tuple[complex, tuple[Operator, ...]], ...]:
        """Expand an arbitrary N-point correlator into normalized basis tuples."""
        term_map: dict[tuple[Operator, ...], complex] = {}

        for weight, basis_correlator in correlator.iter_basis_terms():
            operators = basis_correlator.operators
            if not all(isinstance(op, Operator) for op in operators):
                raise TypeError(
                    "Correlator.iter_basis_terms() must emit basis Operator insertions."
                )
            basis_tuple = tuple(normalize_operator(op) for op in operators)
            combined_weight = term_map.get(basis_tuple, 0.0 + 0.0j) + weight
            if not isfinite(combined_weight.real) or not isfinite(combined_weight.imag):
                raise ValueError(
                    "Combined correlator coefficients must be finite after normalization."
                )
            term_map[basis_tuple] = combined_weight

        return tuple(
            (weight, operators)
            for operators, weight in term_map.items()
            if weight != 0
        )
