"""Compiler-layer Wick contraction cache.

Caches Wick contractions by the complete ordered tuple of compiler-normalized
basis ``Operator`` insertions. Two-point callers remain supported through
``get_or_contract``.
"""

from __future__ import annotations

import hashlib
import json

from concord_lqcd.core.operator import Operator
from concord_lqcd.core.correlator import Correlator
from concord_lqcd.core.wick import wick_contract, WickTerm
from concord_lqcd.compiler.normalization import normalize_operator


def _basis_namespace(operators: tuple[Operator, ...]) -> str:
    """Stable naming for the supported normalized basis, independent of hash seeds."""
    payload = tuple(operator.structural_key for operator in operators)
    encoded = json.dumps(payload, default=repr, ensure_ascii=False, separators=(",", ":"))
    return "basis_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


class WickCache:
    """Cache symbolic Wick contractions for arbitrary basis N-point correlators."""

    def __init__(self) -> None:
        self._cache: dict[tuple[Operator, ...], tuple[Correlator, list[WickTerm]]] = {}
        self._namespace_keys: dict[str, tuple[Operator, ...]] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get_or_contract_operators(
        self,
        operators: tuple[Operator, ...],
    ) -> tuple[Correlator, list[WickTerm]]:
        """Normalize, Wick-contract, and cache one ordered basis-operator tuple."""
        if not operators:
            raise ValueError("WickCache requires at least one basis operator.")
        if not all(isinstance(op, Operator) for op in operators):
            raise TypeError("WickCache accepts only basis Operator insertions.")

        key = tuple(normalize_operator(op) for op in operators)
        return self._get_or_contract_normalized(key)

    def _get_or_contract_normalized(
        self,
        key: tuple[Operator, ...],
    ) -> tuple[Correlator, list[WickTerm]]:
        """Reuse tuples already normalized at the compiler expansion boundary."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        namespace = _basis_namespace(key)
        existing_key = self._namespace_keys.get(namespace)
        if existing_key is not None and existing_key != key:
            raise ValueError("compiler basis namespace collision between distinct operator tuples")
        correlator = Correlator._with_symbolic_namespace(key, namespace)
        terms = wick_contract(correlator)
        result = (correlator, terms)
        self._cache[key] = result
        self._namespace_keys[namespace] = key
        return result

    def get_or_contract(
        self,
        sink_op: Operator,
        source_op: Operator,
    ) -> tuple[Correlator, list[WickTerm]]:
        """Backward-compatible two-point cache entry point."""
        return self.get_or_contract_operators((sink_op, source_op))

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def clear(self) -> None:
        self._cache.clear()
        self._namespace_keys.clear()
        self._hits = 0
        self._misses = 0
