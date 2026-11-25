"""
Operator wrapper, derivative expression, and dagger operation.

Usage::

    from concord_lqcd import *

    pion = Operator(
        dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β'),
        momentum=(0, 0, 1),
    )

    d_k = Operator(
        dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β'),
        constituent_momenta={"D": (1, 0, 0), "K": (-1, 0, 0)},
    )

    # Derivative wraps a specific field
    du = derivative(u('x', 'a', 'α'), direction='x')
"""

from __future__ import annotations

from collections import Counter
from numbers import Integral

from .expr import Expr, Product
from .indices import Index, IndexRegistry, _fresh_index, SPIN, COLOR, POSITION
from .fields import QuarkField
from .gamma import GammaMatrix
from .color import ColorDelta, ColorEpsilon
from .gauge import GaugeLink
from .propagator import Propagator
from .operations import (
    Apply,
    Axis,
    CovariantDerivative,
    LinearOperator,
    ParallelTransport,
    Path,
    Shift,
)

Momentum = tuple[int, int, int]


def _normalize_momentum(momentum: Momentum) -> Momentum:
    try:
        components = tuple(momentum)
    except TypeError as exc:
        raise TypeError("momentum must be a triplet of integers.") from exc
    if len(components) != 3:
        raise ValueError("momentum must be a 3-tuple of integers.")
    if any(
        isinstance(component, bool) or not isinstance(component, Integral)
        for component in components
    ):
        raise TypeError("momentum components must be integers; no truncation or rounding is performed.")
    return tuple(int(component) for component in components)


# ── Derivative & Transport Helpers ─────────────────────────────────────

def derivative(field: Expr, direction: str | Axis, is_backward: bool = False) -> Expr:
    """Applies a covariant derivative to a QuarkField or nested Apply node."""
    if isinstance(direction, str):
        dir_clean = direction.lstrip("+-")
        valid_dirs = {"x", "y", "z", "t"}
        if dir_clean not in valid_dirs:
            raise ValueError(f"Direction must be one of {valid_dirs}.")
        axis_dir = Axis(dir_clean)
        if direction.startswith("-"):
            is_backward = not is_backward
    elif isinstance(direction, Axis):
        axis_dir = direction
    else:
        raise TypeError(f"Direction must be str or Axis, got {type(direction).__name__}.")

    final_dir = -axis_dir if is_backward else axis_dir
    return Apply(CovariantDerivative(final_dir), field)


def transport(field: Expr, *hops: str | Axis) -> Expr:
    """Applies a gauge-covariant parallel transport along a Path of hops."""
    if not hops:
        raise ValueError("transport requires at least one hop direction.")
    return Apply(ParallelTransport(Path(hops)), field)


def shift(field: Expr, *hops: str | Axis) -> Expr:
    """Applies a pure coordinate shift along a Path of hops without gauge link transport."""
    if not hops:
        raise ValueError("shift requires at least one hop direction.")
    return Apply(Shift(Path(hops)), field)


def projected(field: Expr, projector: str = "laph") -> Expr:
    """Applies a physical projection operator to a field-level operand."""
    from .operations import Projection
    return Apply(Projection(projector), field)


def laph(field: Expr) -> Expr:
    """Applies a LapH subspace projection operator to a field-level operand."""
    from .operations import Projection
    return Apply(Projection("laph"), field)


def point(field: Expr) -> Expr:
    """Applies a point spatial representation to a field-level operand."""
    from .operations import Projection
    return Apply(Projection("point"), field)


# ── Dagger ─────────────────────────────────────────────────────────────

def _dagger_expr(expr: Expr, used_labels: set[str] | None = None) -> Expr:
    """Recursively compute the Hermitian conjugate of an expression.

    Field applications retain their original transformation chain: antiquark
    realization conjugates the transformed basis itself. Standalone linear
    operators still expose their mathematical adjoints through ``dagger()``.
    """
    if used_labels is None:
        used_labels = {index.label for index in expr.indices}
    if isinstance(expr, Apply):
        ops: list[LinearOperator] = []
        curr = expr
        while isinstance(curr, Apply):
            ops.append(curr.operator)
            curr = curr.operand

        if isinstance(curr, QuarkField):
            field_operations = tuple(reversed(ops))
            dummy_spin = _fresh_index(SPIN, used_labels, "__dagger")
            if not curr.is_anti:
                q_target = QuarkField(flavor=curr.flavor, is_anti=True, pos=curr.pos, color=curr.color, spin=dummy_spin)
                g4 = GammaMatrix("gammaT", dummy_spin, curr.spin)
                res = q_target
                for op in field_operations:
                    res = Apply(op, res)
                return res * g4
            else:
                g4 = GammaMatrix("gammaT", curr.spin, dummy_spin)
                q_target = QuarkField(flavor=curr.flavor, is_anti=False, pos=curr.pos, color=curr.color, spin=dummy_spin)
                res = q_target
                for op in field_operations:
                    res = Apply(op, res)
                return g4 * res

        raise TypeError(
            "Field operations can only be Hermitian-conjugated when they act on a "
            f"quark field; got an operation chain acting on {type(curr).__name__}."
        )

    if isinstance(expr, QuarkField):
        dummy_spin = _fresh_index(SPIN, used_labels, "__dagger")

        if not expr.is_anti:
            qbar = QuarkField(
                flavor=expr.flavor, is_anti=True, pos=expr.pos,
                color=expr.color, spin=dummy_spin
            )
            g4 = GammaMatrix("gammaT", dummy_spin, expr.spin)
            return qbar * g4
        else:
            g4 = GammaMatrix("gammaT", expr.spin, dummy_spin)
            q = QuarkField(
                flavor=expr.flavor, is_anti=False, pos=expr.pos,
                color=expr.color, spin=dummy_spin
            )
            return g4 * q

    if isinstance(expr, GammaMatrix):
        kind = expr.kind.removesuffix("_dag") if expr.kind.endswith("_dag") else expr.kind + "_dag"
        return GammaMatrix(kind, expr.right, expr.left)

    if isinstance(expr, (ColorDelta, ColorEpsilon)):
        return expr

    if isinstance(expr, GaugeLink):
        # U(x, y)_{ab}^dagger = U(y, x)_{ba}: endpoints and color legs both swap.
        return GaugeLink(
            start_pos=expr.end_pos, end_pos=expr.start_pos,
            start_color=expr.end_color, end_color=expr.start_color,
        )

    if isinstance(expr, Propagator):
        return Propagator(
            flavor=expr.flavor,
            sink_pos=expr.source_pos,
            sink_color=expr.source_color,
            sink_spin=expr.source_spin,
            source_pos=expr.sink_pos,
            source_color=expr.sink_color,
            source_spin=expr.sink_spin,
        )

    if isinstance(expr, Product):
        dag_factors = [_dagger_expr(factor, used_labels) for factor in reversed(expr.factors)]
        result = dag_factors[0]
        for f in dag_factors[1:]:
            result = Product._make(result, f)
        return result

    raise TypeError(f"Cannot take dagger of {type(expr).__name__}")


def _reindex_expr(expr: Expr, namespace: str, mapping: dict[Index, Index] | None = None) -> Expr:
    """Return a structurally identical expression with all indices renamed."""
    if mapping is None:
        mapping = {}

    def remap(idx: Index) -> Index:
        if idx in mapping:
            return mapping[idx]

        new_label = f"{namespace}:{idx.label}"
        out = Index(new_label, idx.space)

        mapping[idx] = out
        return out

    if isinstance(expr, Apply):
        return Apply(expr.operator, _reindex_expr(expr.operand, namespace, mapping))

    if isinstance(expr, QuarkField):
        return QuarkField(
            flavor=expr.flavor,
            is_anti=expr.is_anti,
            pos=remap(expr.pos),
            color=remap(expr.color),
            spin=remap(expr.spin),
        )

    if isinstance(expr, GammaMatrix):
        return GammaMatrix(expr.kind, remap(expr.left), remap(expr.right))

    if isinstance(expr, ColorDelta):
        return ColorDelta(remap(expr.left), remap(expr.right))

    if isinstance(expr, ColorEpsilon):
        return ColorEpsilon(remap(expr.i), remap(expr.j), remap(expr.k))

    if isinstance(expr, GaugeLink):
        return GaugeLink(
            start_pos=remap(expr.start_pos),
            end_pos=remap(expr.end_pos),
            start_color=remap(expr.start_color),
            end_color=remap(expr.end_color),
        )

    if isinstance(expr, Propagator):
        return Propagator(
            flavor=expr.flavor,
            sink_pos=remap(expr.sink_pos),
            sink_color=remap(expr.sink_color),
            sink_spin=remap(expr.sink_spin),
            source_pos=remap(expr.source_pos),
            source_color=remap(expr.source_color),
            source_spin=remap(expr.source_spin),
        )

    if isinstance(expr, Product):
        rewritten = [_reindex_expr(f, namespace, mapping) for f in expr.factors]
        result = rewritten[0]
        for factor in rewritten[1:]:
            result = Product._make(result, factor)
        return result

    raise TypeError(f"Cannot reindex {type(expr).__name__}")


def _compose_exprs(left: Expr, right: Expr) -> Expr:
    """Compose independent dummy scopes while retaining explicit open-index couplings."""
    left_counts = Counter(left.indices)
    right_counts = Counter(right.indices)
    left_labels = {index.label for index in left_counts}
    right_labels = {index.label for index in right_counts}
    used_labels = left_labels | right_labels
    rewritten = []
    for expression, counts, other_labels in (
        (left, left_counts, right_labels),
        (right, right_counts, left_labels),
    ):
        mapping = {index: index for index in counts}
        for index, occurrences in counts.items():
            if index.space != POSITION and occurrences == 2 and index.label in other_labels:
                mapping[index] = _fresh_index(index.space, used_labels, "__bound")
        if any(original != replacement for original, replacement in mapping.items()):
            expression = _reindex_expr(expression, "", mapping)
        rewritten.append(expression)
    return Product._make(*rewritten)


class _CanonicalIndexMap:
    """Canonicalize summed indices without renaming exposed tensor legs.

    Spin/color indices occurring exactly twice are dummy names. All other
    spin/color labels retain their typed identity. Position labels describe
    locality groups rather than Einstein indices and remain canonicalized,
    with momentum assignments transformed through the same mapping.
    """

    __slots__ = ("_maps", "_bound_indices")

    def __init__(self, indices: tuple[Index, ...]) -> None:
        self._bound_indices = frozenset(
            index for index, occurrences in Counter(indices).items()
            if index.space != POSITION and occurrences == 2
        )
        self._maps: dict[str, dict[str, int]] = {
            POSITION: {},
            COLOR: {},
            SPIN: {},
        }

    def token(self, index: Index) -> tuple[str, str, str | int]:
        if index.space != POSITION and index not in self._bound_indices:
            return (index.space, "exposed", index.label)
        return self.token_for_label(index.space, index.label)

    def token_for_label(self, space: str, label: str) -> tuple[str, str, int]:
        mapping = self._maps[space]
        if label not in mapping:
            mapping[label] = len(mapping)
        return (space, "dummy", mapping[label])


def _canonical_expr_key(expr: Expr, indices: _CanonicalIndexMap) -> tuple:
    """Return a structural key with dummy renaming and fixed exposed labels."""
    if isinstance(expr, Apply):
        return (
            "Apply",
            expr.operator,
            _canonical_expr_key(expr.operand, indices),
        )

    if isinstance(expr, QuarkField):
        return (
            "QuarkField",
            expr.flavor,
            expr.is_anti,
            indices.token(expr.pos),
            indices.token(expr.color),
            indices.token(expr.spin),
        )

    if isinstance(expr, GammaMatrix):
        return (
            "GammaMatrix",
            expr.kind,
            indices.token(expr.left),
            indices.token(expr.right),
        )

    if isinstance(expr, ColorDelta):
        return (
            "ColorDelta",
            indices.token(expr.left),
            indices.token(expr.right),
        )

    if isinstance(expr, ColorEpsilon):
        return (
            "ColorEpsilon",
            indices.token(expr.i),
            indices.token(expr.j),
            indices.token(expr.k),
        )

    if isinstance(expr, GaugeLink):
        return (
            "GaugeLink",
            indices.token(expr.start_pos),
            indices.token(expr.end_pos),
            indices.token(expr.start_color),
            indices.token(expr.end_color),
        )

    if isinstance(expr, Propagator):
        return (
            "Propagator",
            expr.flavor,
            indices.token(expr.sink_pos),
            indices.token(expr.sink_color),
            indices.token(expr.sink_spin),
            indices.token(expr.source_pos),
            indices.token(expr.source_color),
            indices.token(expr.source_spin),
        )

    if isinstance(expr, Product):
        return (
            "Product",
            tuple(_canonical_expr_key(factor, indices) for factor in expr.factors),
        )

    # Extensibility fallback for custom Expr subclasses.  Built-in frontend
    # nodes are handled above; this keeps structural identity usable for a
    # custom hashable/representable leaf without guessing its physics.
    return (type(expr).__qualname__, repr(expr))


def _operator_structural_key(expr: Expr, momenta: dict[str, Momentum]) -> tuple:
    """Canonical mathematical identity of a basis Operator.

    Metadata is intentionally excluded: it is annotation, not symbolic physics.
    """
    index_map = _CanonicalIndexMap(expr.indices)
    expr_key = _canonical_expr_key(expr, index_map)
    momentum_key = tuple(sorted(
        (
            index_map.token_for_label(POSITION, label),
            tuple(momentum),
        )
        for label, momentum in momenta.items()
    ))
    return (expr_key, momentum_key)


class Operator:
    """A complete interpolating operator = expression + metadata."""

    __slots__ = ("_expr", "_momenta", "_metadata")

    def __init__(
        self,
        expr: Expr,
        momentum: Momentum | None = None,
        momenta: dict[str, Momentum] | None = None,
        **metadata,
    ) -> None:
        # Frontend construction records exactly the symbolic expression supplied
        # by the user. Compiler canonicalization is intentionally deferred to
        # ``concord_lqcd.compiler.normalization`` so the core layer has no dependency
        # on compiler implementation details.
        if not isinstance(expr, Expr):
            raise TypeError("Operator expression must be an Expr instance.")
        IndexRegistry(expr.indices)
        self._expr = expr
        if momentum is not None and momenta is not None:
            raise ValueError("Pass either `momentum` or `momenta`, not both.")
        if momenta is not None:
            labels = {index.label for index in expr.indices if index.space == POSITION}
            unknown = set(momenta) - labels
            if unknown:
                raise ValueError(
                    f"Momentum position label(s) {sorted(unknown, key=str)} do not occur in the expression. "
                    f"Available position labels: {sorted(labels)}."
                )
            self._momenta = {label: _normalize_momentum(mom) for label, mom in momenta.items()}
        elif momentum is not None:
            labels = sorted({idx.label for idx in expr.indices if idx.space == POSITION})
            if len(labels) != 1:
                raise ValueError(
                    "`momentum` is only unambiguous when the operator has exactly one "
                    f"position/locality label; this operator has {labels or 'none'}. "
                    "For composite operators with multiple locality groups, pass "
                    "`momenta={'label': (px,py,pz), ...}` instead, keyed by the actual "
                    "position label(s) used in the expression."
                )
            self._momenta = {labels[0]: _normalize_momentum(momentum)}
        else:
            self._momenta = {}

        self._metadata = metadata

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def momenta(self) -> dict:
        return dict(self._momenta)

    @property
    def momentum(self) -> Momentum | None:
        if not self._momenta:
            return None
        if len(self._momenta) > 1:
            raise ValueError(
                f"This operator has {len(self._momenta)} independent momentum groups "
                f"({sorted(self._momenta)}) -- use `.momenta` instead of `.momentum`."
            )
        return next(iter(self._momenta.values()))

    @property
    def total_momentum(self) -> Momentum:
        px = sum(p[0] for p in self._momenta.values())
        py = sum(p[1] for p in self._momenta.values())
        pz = sum(p[2] for p in self._momenta.values())
        return (px, py, pz)

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    @property
    def structural_key(self) -> tuple:
        """Hashable structural identity with a preserved open-index interface.

        Summed spin/color indices and locality-group labels are canonicalized.
        Exposed spin/color labels, momentum assignments, and ordered tensor
        attachments are preserved.
        """
        return _operator_structural_key(self._expr, self._momenta)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Operator):
            return NotImplemented
        return self.structural_key == other.structural_key

    def __hash__(self) -> int:
        return hash(("Operator", self.structural_key))

    def dagger(self) -> Operator:
        dag_expr = _dagger_expr(self._expr)
        dag_momenta = {k: tuple(-x for x in v) for k, v in self._momenta.items()}
        return Operator(dag_expr, momenta=dag_momenta, **self._metadata)

    def reindexed(self, namespace: str) -> Operator:
        if not namespace:
            raise ValueError("namespace must be a non-empty string.")
        mapping: dict[Index, Index] = {}
        new_expr = _reindex_expr(self._expr, namespace, mapping)
        label_map = {old.label: new.label for old, new in mapping.items() if old.space == POSITION}
        new_momenta = {label_map.get(label, label): mom for label, mom in self._momenta.items()}
        return Operator(new_expr, momenta=new_momenta or None, **self._metadata)

    def iter_terms(self):
        yield (1.0 + 0.0j, self)

    def __add__(self, other: object):
        from .operator_expression import OperatorCombination
        return OperatorCombination.from_element(self) + other

    def __radd__(self, other: object):
        from .operator_expression import OperatorCombination
        return OperatorCombination.from_element(self).__radd__(other)

    def __sub__(self, other: object):
        from .operator_expression import OperatorCombination
        return OperatorCombination.from_element(self) - other

    def __rsub__(self, other: object):
        from .operator_expression import OperatorCombination
        return OperatorCombination.from_element(self).__rsub__(other)

    def __neg__(self):
        from .operator_expression import OperatorCombination
        return -OperatorCombination.from_element(self)

    def __truediv__(self, scalar: object):
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide an Operator by zero.")
        from .operator_expression import OperatorCombination
        return OperatorCombination([(1.0 / complex(scalar), self)])

    def __mul__(self, other: object):
        if isinstance(other, (int, float, complex)):
            from .operator_expression import OperatorCombination
            return OperatorCombination([(complex(other), self)])
        if isinstance(other, Operator):
            overlap = set(self._momenta) & set(other._momenta)
            if overlap:
                raise ValueError(
                    f"Both operands already have a momentum group with label(s) {sorted(overlap)}. "
                    "Position labels must be unique across operands before combining -- "
                    "reindex one side first if it might collide."
                )
            merged_expr = _compose_exprs(self._expr, other._expr)
            merged_momenta = {**self._momenta, **other._momenta}
            merged_metadata = {**self._metadata, **other._metadata}
            return Operator(merged_expr, momenta=merged_momenta or None, **merged_metadata)
        if isinstance(other, Expr):
            if any(idx.space == POSITION for idx in other.indices):
                raise ValueError(
                    "Cannot multiply an Operator by a bare Expr containing position indices. "
                    "If this is a physical constituent, wrap it in an Operator first to "
                    "assign it a momentum."
                )
            merged_expr = _compose_exprs(self._expr, other)
            return Operator(merged_expr, momenta=self._momenta or None, **self._metadata)
        from .operator_expression import OperatorLike
        if isinstance(other, OperatorLike):
            from .operator_expression import OperatorCombination
            return OperatorCombination.from_element(self) * other
        return NotImplemented

    def __rmul__(self, other: object):
        if isinstance(other, (int, float, complex)):
            from .operator_expression import OperatorCombination
            return OperatorCombination([(complex(other), self)])
        if isinstance(other, Expr) and not isinstance(other, Operator):
            if any(idx.space == POSITION for idx in other.indices):
                raise ValueError(
                    "Cannot multiply an Operator by a bare Expr containing position indices. "
                    "If this is a physical constituent, wrap it in an Operator first to "
                    "assign it a momentum."
                )
            merged_expr = _compose_exprs(other, self._expr)
            return Operator(merged_expr, momenta=self._momenta or None, **self._metadata)
        from .operator_expression import OperatorLike
        if isinstance(other, OperatorLike):
            from .operator_expression import OperatorCombination
            return OperatorCombination.from_element(other) * self
        return NotImplemented

    @property
    def indices(self) -> tuple[Index, ...]:
        return self._expr.indices

    def __repr__(self) -> str:
        parts = [repr(self._expr)]
        if self._momenta:
            parts.append(f"p={self._momenta}")
        return f"Operator({', '.join(parts)})"
