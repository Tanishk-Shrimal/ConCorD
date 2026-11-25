"""Bounded exact tensor-network matching with independently checked witnesses.

Resources and symmetry contracts are declarations supplied by the caller, not
inferred tensor identities. Coefficients stay outside this module. Invariants
only index candidates: neither a shared key nor graph matching alone certifies
a rewrite. Named outputs retain both their names and their order.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from math import isfinite
from time import monotonic
from typing import Literal

import networkx as nx


@dataclass(frozen=True, slots=True)
class Factor:
    resource: str
    labels: tuple[str, ...]
    kinds: tuple[str, ...]
    symmetry: Literal["ordered", "symmetric", "alternating"] = "ordered"


@dataclass(frozen=True, slots=True)
class Network:
    factors: tuple[Factor, ...]
    outputs: tuple[str, ...]


class SearchBudgetExceeded(RuntimeError):
    """No equivalence or inequivalence was established within the state budget."""


class SearchAborted(RuntimeError):
    """A hard wall-clock limit expired; callers must fail, never treat it as unproved."""


def _unlimited() -> None:
    pass


def _deadline_check(deadline: float | None) -> Callable[[], None]:
    if deadline is None:
        return _unlimited
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not isfinite(deadline)):
        raise ValueError("The deadline must be a finite monotonic timestamp")

    def check() -> None:
        if monotonic() >= deadline:
            raise SearchAborted("The tensor-isomorphism hard deadline expired")

    return check


def _index_kinds(value: Network, check: Callable[[], None]) -> dict[str, str]:
    check()
    if (not isinstance(value, Network) or not isinstance(value.factors, tuple)
            or not isinstance(value.outputs, tuple)):
        raise ValueError("A network requires tuples of factors and outputs")
    result = {}
    for factor in value.factors:
        check()
        if (not isinstance(factor, Factor) or not isinstance(factor.resource, str)
                or not isinstance(factor.labels, tuple) or not isinstance(factor.kinds, tuple)):
            raise ValueError("A factor requires a resource and tuples of labels and kinds")
        if factor.symmetry not in ("ordered", "symmetric", "alternating"):
            raise ValueError("Unknown tensor symmetry contract")
        if len(factor.labels) != len(factor.kinds):
            raise ValueError("Factor rank and port kinds disagree")
        for label, kind in zip(factor.labels, factor.kinds, strict=True):
            check()
            if not isinstance(label, str) or not isinstance(kind, str):
                raise ValueError("Index labels and kinds must be strings")
            if result.setdefault(label, kind) != kind:
                raise ValueError("One index cannot have different kinds")
    outputs = set()
    for label in value.outputs:
        check()
        if not isinstance(label, str) or label not in result:
            raise ValueError("An output index is missing")
        if label in outputs:
            raise ValueError("Output indices must be distinct")
        outputs.add(label)
    return result


def index_kinds(network: Network) -> dict[str, str]:
    """Validate a network and return the kind of every distinct index."""
    return _index_kinds(network, _unlimited)


def _parity(order: tuple[int, ...], check: Callable[[], None]) -> int:
    rank = len(order)
    seen = set()
    for destination in order:
        check()
        if type(destination) is not int or not 0 <= destination < rank or destination in seen:
            raise ValueError("The port map is not a permutation of the tensor rank")
        seen.add(destination)
    visited = set()
    cycles = 0
    for start in range(rank):
        check()
        if start in visited:
            continue
        cycles += 1
        position = start
        while position not in visited:
            check()
            visited.add(position)
            position = order[position]
    return -1 if (rank - cycles) % 2 else 1


def parity(order) -> int:
    """Return the sign of a permutation of ``range(len(order))``."""
    return _parity(tuple(order), _unlimited)


def _graph(value: Network, check: Callable[[], None]) -> nx.Graph:
    result = nx.Graph()
    kinds = _index_kinds(value, check)
    outputs = {label: ordinal for ordinal, label in enumerate(value.outputs)}
    for label, kind in kinds.items():
        check()
        boundary = (label, outputs[label]) if label in outputs else None
        result.add_node(("index", label), color=json.dumps(("index", kind, boundary)))
    for ordinal, factor in enumerate(value.factors):
        check()
        node = ("factor", ordinal)
        port_kinds = factor.kinds if factor.symmetry == "ordered" else tuple(sorted(factor.kinds))
        result.add_node(node, color=json.dumps(("factor", factor.resource, factor.symmetry, port_kinds)))
        for position, (label, kind) in enumerate(zip(factor.labels, factor.kinds, strict=True)):
            check()
            leg = ("leg", ordinal, position)
            slot = position if factor.symmetry == "ordered" else None
            result.add_node(leg, color=json.dumps(("leg", kind, slot)))
            result.add_edge(node, leg)
            result.add_edge(leg, ("index", label))
    check()
    return result


def graph(network: Network, *, deadline: float | None = None) -> nx.Graph:
    """Build a colored incidence graph, retaining repeated ports and factors."""
    return _graph(network, _deadline_check(deadline))


def _refine(graphs: tuple[nx.Graph, ...], check: Callable[[], None]):
    labels = []
    for value in graphs:
        check()
        if not isinstance(value, nx.Graph) or value.is_directed() or value.is_multigraph():
            raise ValueError("Expected an undirected simple incidence graph")
        current = {}
        for node in value:
            check()
            color = value.nodes[node].get("color")
            if not isinstance(color, str):
                raise ValueError("Graph nodes require string colors")
            current[node] = color
        labels.append(current)
    return _stabilize(graphs, labels, check)


def _stabilize(graphs: tuple[nx.Graph, ...], labels: list[dict], check: Callable[[], None]):
    """Jointly refine sortable initial colors to a stable partition with one palette."""
    palette = set()
    for current in labels:
        check()
        palette.update(current.values())
    lookup = {color: ordinal for ordinal, color in enumerate(sorted(palette))}
    colors = [{node: lookup[color] for node, color in current.items()} for current in labels]
    previous_count = len(palette)
    while True:
        check()
        signatures = []
        palette = set()
        for value, current in zip(graphs, colors, strict=True):
            refined = {}
            for node in value:
                check()
                neighbors = []
                for neighbor in value.neighbors(node):
                    check()
                    neighbors.append(current[neighbor])
                signature = (current[node], tuple(sorted(neighbors)))
                refined[node] = signature
                palette.add(signature)
            signatures.append(refined)
        lookup = {signature: ordinal for ordinal, signature in enumerate(sorted(palette))}
        colors = []
        for current in signatures:
            refined = {}
            for node, signature in current.items():
                check()
                refined[node] = lookup[signature]
            colors.append(refined)
        if len(palette) == previous_count:
            check()
            return colors
        previous_count = len(palette)


def invariant(value_graph: nx.Graph, *, deadline: float | None = None) -> tuple:
    """Return a hashable candidate key, never an equality certificate.

    Refinement uses exact color tuples rather than relying on hash equality.
    Distinct regular graphs can still have the same key and require matching.
    An optional absolute monotonic deadline bounds this preprocessing too.
    """
    check = _deadline_check(deadline)
    colors, = _refine((value_graph,), check)
    nodes = Counter()
    edges = Counter()
    for node in value_graph:
        check()
        nodes[value_graph.nodes[node]["color"], colors[node]] += 1
    for left, right in value_graph.edges:
        check()
        edges[tuple(sorted((colors[left], colors[right])))] += 1
    result = (tuple(sorted(nodes.items())), tuple(sorted(edges.items())))
    check()
    return result


def _verify_witness(left: Network, right: Network, witness, check: Callable[[], None]) -> int:
    left_kinds = _index_kinds(left, check)
    right_kinds = _index_kinds(right, check)
    if left.outputs != right.outputs:
        raise ValueError("The map changes named external axes or their order")
    if not isinstance(witness, Mapping):
        raise ValueError("A witness must contain index and factor maps")
    indices = witness.get("indices")
    records = witness.get("factors")
    if not isinstance(indices, Mapping) or set(indices) != set(left_kinds):
        raise ValueError("The index map is not a bijection")
    destinations = set()
    for label, target in indices.items():
        check()
        if not isinstance(target, str) or target not in right_kinds or target in destinations:
            raise ValueError("The index map is not a bijection")
        destinations.add(target)
        if left_kinds[label] != right_kinds[target]:
            raise ValueError("The index map changes index kinds")
    if destinations != set(right_kinds):
        raise ValueError("The index map is not a bijection")
    for label in left.outputs:
        check()
        if indices[label] != label:
            raise ValueError("The map moves a named external axis")
    if (not isinstance(records, (list, tuple)) or len(records) != len(left.factors)
            or len(records) != len(right.factors)):
        raise ValueError("The factor map is not a bijection")
    targets = set()
    phase = 1
    for factor, record in zip(left.factors, records, strict=True):
        check()
        if not isinstance(record, Mapping):
            raise ValueError("A factor record requires a target and ports")
        destination = record.get("target")
        if (type(destination) is not int or not 0 <= destination < len(right.factors)
                or destination in targets):
            raise ValueError("The factor map is not a bijection")
        targets.add(destination)
        target = right.factors[destination]
        ports = record.get("ports")
        if (not isinstance(ports, (list, tuple)) or len(ports) != len(factor.labels)
                or len(ports) != len(target.labels)):
            raise ValueError("The port map is not a permutation of the tensor rank")
        order = tuple(ports)
        local_phase = _parity(order, check)
        if (factor.resource, factor.symmetry) != (target.resource, target.symmetry):
            raise ValueError("A factor resource or symmetry contract changed")
        if factor.symmetry == "ordered" and order != tuple(range(len(order))):
            raise ValueError("An ordered tensor cannot have its ports permuted")
        for position, port in enumerate(order):
            check()
            if (factor.kinds[position] != target.kinds[port]
                    or indices[factor.labels[position]] != target.labels[port]):
                raise ValueError("A mapped tensor port does not preserve kinds and incidence")
        if factor.symmetry == "alternating":
            phase *= local_phase
    if "phase" in witness and (type(witness["phase"]) is not int or witness["phase"] != phase):
        raise ValueError("The claimed phase does not follow from the witness")
    check()
    return phase


def verify_witness(left: Network, right: Network, witness) -> int:
    """Check a complete index/factor/port bijection without any graph machinery.

    ``indices`` maps every left label to a right label. ``factors`` is ordered
    like the left factors; each record has an integer ``target`` and a ``ports``
    permutation mapping left positions to right positions. An optional integer
    ``phase`` must equal the product of alternating-factor permutation signs.
    Invalid witnesses raise ``ValueError``; valid ones return exactly +1 or -1.
    """
    return _verify_witness(left, right, witness, _unlimited)


def _matching_graph(value: Network, supplied: nx.Graph | None, check: Callable[[], None]) -> nx.Graph:
    expected = _graph(value, check)
    if supplied is None:
        return expected
    if (not isinstance(supplied, nx.Graph) or supplied.is_directed() or supplied.is_multigraph()
            or len(supplied) != len(expected) or supplied.number_of_edges() != expected.number_of_edges()):
        raise ValueError("A supplied graph does not describe its network")
    for node in expected:
        check()
        if node not in supplied or supplied.nodes[node].get("color") != expected.nodes[node]["color"]:
            raise ValueError("A supplied graph does not describe its network")
        for neighbor in expected.neighbors(node):
            check()
            if not supplied.has_edge(node, neighbor):
                raise ValueError("A supplied graph does not describe its network")
    return expected


def _cell(left_graph, right_graph, colors, check):
    """Return None, "discrete" or the first smallest nontrivial left cell's node and its right targets."""
    left_colors, right_colors = colors
    if Counter(left_colors.values()) != Counter(right_colors.values()):
        check()
        return None
    cells = defaultdict(list)
    for node in left_graph:
        check()
        cells[left_colors[node]].append(node)
    best = None
    for color, nodes in cells.items():
        check()
        if len(nodes) > 1 and (best is None or (len(nodes), color) < best[0]):
            best = ((len(nodes), color), nodes[0])
    if best is None:
        return "discrete"
    (_, color), source = best
    targets = []
    for node in right_graph:
        check()
        if right_colors[node] == color:
            targets.append(node)
    return source, iter(targets)


def _individualize(graphs, colors, source, target, check):
    marked = ({node: (color, node == source) for node, color in colors[0].items()},
              {node: (color, node == target) for node, color in colors[1].items()})
    return _stabilize(graphs, list(marked), check)


def _witness(left: Network, right: Network, mapping, check):
    indices = {}
    for label in _index_kinds(left, check):
        check()
        indices[label] = mapping[("index", label)][1]
    factors = []
    for ordinal, factor in enumerate(left.factors):
        check()
        ports = []
        for position in range(len(factor.labels)):
            check()
            ports.append(mapping[("leg", ordinal, position)][2])
        factors.append({"target": mapping[("factor", ordinal)][1], "ports": ports})
    result = {"indices": indices, "factors": factors}
    result["phase"] = _verify_witness(left, right, result, check)
    return result


def find_witness(
    left: Network,
    right: Network,
    left_graph: nx.Graph | None = None,
    right_graph: nx.Graph | None = None,
    *,
    max_states: int,
    deadline: float | None = None,
) -> dict | None:
    """Find a verified witness by individualization-refinement search.

    Both incidence graphs are refined jointly; one node of the smallest
    ambiguous left cell is then individualized against each right candidate
    in turn, iteratively rather than recursively. The search is exact and
    complete, so ``None`` is a proof that no witness exists. ``max_states``
    bounds the refined search nodes, the root included, and exhaustion raises
    ``SearchBudgetExceeded``. The outcome depends only on the networks and
    ``max_states``, never on timing. The optional absolute
    ``time.monotonic()`` ``deadline`` is a hard abort raising ``SearchAborted``.
    Supplied graphs are checked against their networks and cannot replace the
    independent incidence verifier.
    """
    if type(max_states) is not int or max_states < 0:
        raise ValueError("max_states must be a nonnegative integer")
    check = _deadline_check(deadline)
    states = 0

    def step() -> None:
        nonlocal states
        check()
        if states >= max_states:
            raise SearchBudgetExceeded("The tensor-isomorphism state budget was exhausted")
        states += 1

    step()
    left_graph = _matching_graph(left, left_graph, check)
    right_graph = _matching_graph(right, right_graph, check)
    if (left.outputs != right.outputs or len(left_graph) != len(right_graph)
            or left_graph.number_of_edges() != right_graph.number_of_edges()):
        check()
        return None
    graphs = (left_graph, right_graph)
    current = _refine(graphs, check)
    frames = []
    while True:
        if current is not None:
            outcome = _cell(left_graph, right_graph, current, check)
            if outcome == "discrete":
                left_colors, right_colors = current
                target = {color: node for node, color in right_colors.items()}
                try:
                    return _witness(left, right, {node: target[color] for node, color in left_colors.items()}, check)
                except ValueError:
                    pass
            elif outcome is not None:
                frames.append((current, *outcome))
            current = None
        if not frames:
            check()
            return None
        colors, source, candidates = frames[-1]
        candidate = next(candidates, None)
        if candidate is None:
            frames.pop()
            continue
        step()
        current = _individualize(graphs, colors, source, candidate, check)
