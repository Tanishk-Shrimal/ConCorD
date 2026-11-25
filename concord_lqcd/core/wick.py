"""
Wick contraction engine.

Turns a basis Correlator into the list of WickTerms that sum to it: every
perfect matching of quark-type fields against antiquark-type fields of
the same flavor, with the correct fermionic sign. Each term retains the
exact pairing and a symbolic bookkeeping expression. Bare fields in that
expression are replaced by Propagator nodes; field-operation wrappers are
retained for later lowering with the original operator context.

Unlike a "spectroscopy-only" contraction rule that only pairs sink
fields against source fields, this generator also allows same-side
pairings (a sink field contracting with another sink field, or a source
field with another source field) -- that's exactly what a disconnected
/ vacuum-loop topology is, and excluding it would silently drop those
diagrams instead of computing and flagging them.

Each term is further analyzed as a graph: nodes are locality (position)
labels, edges are propagators. Its components retain which Correlator
insertions they touch. The connected / disconnected / mixed labels are
coarse insertion-spanning versus insertion-local metadata, not a test of
multihadron factorization. The compiler reconstructs exact open fermion
paths and closed cycles from the pairing and spin/index connectivity.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from .correlator import Correlator
from .expr import Expr, Product
from .fields import QuarkField, quark_fields
from .propagator import Propagator


@dataclass(frozen=True)
class DiagramComponent:
    """One connected piece of a Wick term's operator-locality graph.

    ``operator_indices`` identify Correlator insertions, not constituent
    hadrons within a composite insertion.
    """

    position_labels: frozenset
    operator_indices: frozenset = frozenset()

    @property
    def is_local(self) -> bool:
        return len(self.operator_indices) <= 1

    @property
    def is_pairwise(self) -> bool:
        return len(self.operator_indices) == 2

    @property
    def is_multi_operator(self) -> bool:
        return len(self.operator_indices) > 2


def _topology_from_components(components: tuple[DiagramComponent, ...]) -> str:
    spanning = [c for c in components if not c.is_local]
    loops = [c for c in components if c.is_local]
    if spanning and loops:
        return "mixed"
    return "connected" if spanning else "disconnected"


@dataclass(frozen=True)
class WickTerm:
    """One exact flavor-preserving matching and its symbolic bookkeeping.

    ``pairing`` references the quark and antiquark occurrences in the original
    namespaced basis correlator. Together with that operator context it defines
    quark lines and endpoint operations during PhysicsIR lowering. ``expr``
    retains spin/color/gauge tensors and field-operation wrappers; it is not
    necessarily a fully contracted expression. No numerical tensors are built
    at this layer.
    """

    sign: int
    coefficient: complex
    expr: Expr
    components: tuple[DiagramComponent, ...]
    pairing: tuple[tuple[QuarkField, QuarkField], ...]

    @property
    def topology(self) -> str:
        """Coarse insertion-spanning/local classification.

        ``connected`` means at least one spanning component and no local
        components; ``disconnected`` means only local components; ``mixed``
        means both. Multiple independent spanning components still report
        ``connected``. This is not a full fermion-connectivity classification.
        """
        return _topology_from_components(self.components)


def _fermionic_sign(all_fields: list[QuarkField], pairing: list[tuple[QuarkField, QuarkField]]) -> int:
    """Sign of the permutation needed to move each contracted pair next
    to its partner, starting from the fields' original left-to-right
    order."""
    index_of = {id(f): idx for idx, f in enumerate(all_fields)}
    flattened: list[QuarkField] = []
    for q_field, aq_field in pairing:
        flattened.extend((q_field, aq_field))

    permutation = [index_of[id(f)] for f in flattened]
    inversions = 0
    for i, lhs in enumerate(permutation):
        for rhs in permutation[i + 1:]:
            if lhs > rhs:
                inversions += 1
    return -1 if inversions % 2 else 1


def _generate_pairings(
    quark_type: list[QuarkField], antiquark_type: list[QuarkField]
) -> Iterator[list[tuple[QuarkField, QuarkField]]] | None:
    # Wick contractions conserve the explicit flavor carried by each field.
    # In particular, u and d are distinct QCD flavors even when they share
    # the same numerical light-quark perambulator.  Isospin combinations must
    # be represented explicitly by the operator/linear-combination layer;
    # they must not be introduced by the contraction engine.
    flavors = sorted({f.flavor for f in quark_type} | {f.flavor for f in antiquark_type})
    channel_options: list[list[list[tuple[QuarkField, QuarkField]]]] = []

    for flavor in flavors:
        q_side = [f for f in quark_type if f.flavor == flavor]
        aq_side = [f for f in antiquark_type if f.flavor == flavor]
        if len(q_side) != len(aq_side):
            return None
        channel_options.append([list(zip(q_side, perm)) for perm in itertools.permutations(aq_side)])

    if not channel_options:
        return iter([[]])

    return (
        [pair for channel in combo for pair in channel]
        for combo in itertools.product(*channel_options)
    )


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _analyze_components(
    pairing: list[tuple[QuarkField, QuarkField]], op_id_map: dict[int, int]
) -> tuple[DiagramComponent, ...]:
    uf = _UnionFind()
    label_ops: dict[str, set[int]] = defaultdict(set)

    for q_field, aq_field in pairing:
        q_label, aq_label = q_field.pos.label, aq_field.pos.label
        q_op = op_id_map.get(id(q_field), 0)
        aq_op = op_id_map.get(id(aq_field), 0)

        label_ops[q_label].add(q_op)
        label_ops[aq_label].add(aq_op)
        uf.union(q_label, aq_label)

    groups: dict[str, set[str]] = defaultdict(set)
    for label in label_ops:
        groups[uf.find(label)].add(label)

    components = []
    for labels in groups.values():
        ops = frozenset(op for l in labels for op in label_ops[l])
        components.append(DiagramComponent(
            position_labels=frozenset(labels),
            operator_indices=ops,
        ))
    return tuple(components)


def wick_contract(correlator: Correlator) -> list[WickTerm]:
    """All WickTerms for one basis Correlator.

    Linear-combination correlators must first be expanded with
    ``Correlator.iter_basis_terms()`` so that each Wick term retains the
    correct basis-operator context for later compiler lowering.
    """
    if not correlator.is_basis:
        raise ValueError(
            "wick_contract() requires a basis Correlator. Expand linear "
            "combinations with `correlator.iter_basis_terms()` first."
        )

    all_ops = correlator.symbolic_operators
    op_fields = correlator.operator_fields

    all_fields: list[QuarkField] = []
    op_id_map: dict[int, int] = {}
    for op_idx, fields in enumerate(op_fields):
        all_fields.extend(fields)
        for f in fields:
            op_id_map[id(f)] = op_idx

    quark_type = [f for f in all_fields if not f.is_anti]
    antiquark_type = [f for f in all_fields if f.is_anti]

    pairings = _generate_pairings(quark_type, antiquark_type)
    if pairings is None:
        return []

    # Operator expressions are general Expr nodes. A multi-factor operator
    # contributes its flat Product factors, while a singleton expression such
    # as Apply(D, q) contributes one top-level factor.
    all_factors: list[Expr] = []
    for op in all_ops:
        if isinstance(op.expr, Product):
            all_factors.extend(op.expr.factors)
        else:
            all_factors.append(op.expr)

    terms: list[WickTerm] = []
    for pairing in pairings:
        sign = _fermionic_sign(all_fields, pairing)
        components = _analyze_components(pairing, op_id_map)

        propagator_at: dict[int, Propagator] = {}
        drop: set[int] = set()
        for q_field, aq_field in pairing:
            propagator_at[id(q_field)] = Propagator(
                flavor=q_field.flavor,
                sink_pos=q_field.pos, sink_color=q_field.color, sink_spin=q_field.spin,
                source_pos=aq_field.pos, source_color=aq_field.color, source_spin=aq_field.spin,
            )
            drop.add(id(aq_field))

        new_factors = []
        for f in all_factors:
            if isinstance(f, QuarkField):
                fid = id(f)
                if fid in propagator_at:
                    new_factors.append(propagator_at[fid])
                    continue
                if fid in drop:
                    continue
            new_factors.append(f)

        terms.append(WickTerm(
            sign=sign,
            coefficient=1,
            expr=Product(tuple(new_factors)),
            components=components,
            pairing=tuple(pairing),
        ))

    return terms
