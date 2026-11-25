"""
AST normalization for the ConCorD compiler.

The symbolic frontend records exactly what the user constructs. This compiler
pass canonicalizes field-operation chains before Wick contraction and lowering:
paths are simplified and nested ``Apply`` nodes are fused into an internal
``CompositeOperator``.
"""

from __future__ import annotations

from concord_lqcd.core.expr import Expr, Product
from concord_lqcd.core.fields import QuarkField
from concord_lqcd.core.operator import Operator
from concord_lqcd.core.operations import (
    Apply,
    LinearOperator,
    Shift,
    ParallelTransport,
    Space,
    Locality,
)


class CompositeOperator(LinearOperator):
    """Internal compiler-only sequence of field operations.

    This node is produced only by compiler normalization and is never inserted
    into expressions by frontend construction.
    """

    def __init__(self, operators: tuple[LinearOperator, ...]) -> None:
        self._operators = operators

    @property
    def operators(self) -> tuple[LinearOperator, ...]:
        return self._operators

    @property
    def acts_on(self) -> frozenset[Space]:
        spaces: set[Space] = set()
        for op in self._operators:
            spaces.update(op.acts_on)
        return frozenset(spaces)

    @property
    def locality(self) -> Locality:
        if any(op.locality == Locality.NONLOCAL for op in self._operators):
            return Locality.NONLOCAL
        if any(op.locality == Locality.NEAREST_NEIGHBOR for op in self._operators):
            return Locality.NEAREST_NEIGHBOR
        return Locality.LOCAL

    def dagger(self) -> CompositeOperator:
        return CompositeOperator(tuple(op.dagger() for op in reversed(self._operators)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeOperator):
            return NotImplemented
        return self._operators == other._operators

    def __hash__(self) -> int:
        return hash(("CompositeOperator", self._operators))

    def __repr__(self) -> str:
        return f"Composite({self._operators!r})"


def _collect_ops_and_leaf(node: Expr) -> tuple[list[LinearOperator], QuarkField]:
    ops: list[LinearOperator] = []
    curr = node
    while isinstance(curr, Apply):
        ops.append(curr.operator)
        curr = curr.operand
    if not isinstance(curr, QuarkField):
        raise TypeError(
            f"Apply chain must terminate on a QuarkField, got {type(curr).__name__}."
        )
    return ops, curr


def normalize_op(op: LinearOperator) -> LinearOperator:
    """Canonicalize one primitive field operation without changing its meaning."""
    if isinstance(op, Shift):
        return Shift(op.path.simplify())
    if isinstance(op, ParallelTransport):
        return ParallelTransport(op.path.simplify())
    return op


def normalize_expr(expr: Expr) -> Expr:
    """Return the compiler-normalized form of one frontend expression."""
    if isinstance(expr, Apply):
        ops, leaf = _collect_ops_and_leaf(expr)
        normalized_ops = tuple(normalize_op(op) for op in ops)

        if len(normalized_ops) == 1:
            return Apply(normalized_ops[0], leaf)
        return Apply(CompositeOperator(normalized_ops), leaf)

    if isinstance(expr, Product):
        return Product(tuple(normalize_expr(factor) for factor in expr.factors))

    return expr


def normalize_operator(operator: Operator) -> Operator:
    """Normalize one basis operator at the frontend/compiler boundary.

    Momentum assignments and user metadata are preserved exactly. The returned
    operator is a compiler-side canonical copy; the user's original symbolic
    object is not mutated.
    """
    if not isinstance(operator, Operator):
        raise TypeError(
            f"normalize_operator expects Operator, got {type(operator).__name__}."
        )
    return Operator(
        normalize_expr(operator.expr),
        momenta=operator.momenta or None,
        **operator.metadata,
    )
