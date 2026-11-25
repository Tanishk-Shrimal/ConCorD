"""Generic fermion-line topology reconstruction for one LogicalCorrelator.

This layer is deliberately separate from propagation-resource lowering.
It uses exact Wick endpoints plus operator-local spin connectivity to expose
the full fermion-line components of a Wick term:

    * open fermion paths;
    * closed fermion cycles.

No hadron-type classification is performed.  Meson, baryon, multi-meson,
baryon-meson, tetraquark, and higher-point structures are handled by the same
endpoint/transition graph.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from concord_lqcd.compiler.physics_ir import (
    ColorTensorRecord,
    LogicalCorrelator,
    SpinTensorRecord,
)
from concord_lqcd.compiler.distillation_lowering import Endpoint


def _endpoint_operator_id(endpoint: Endpoint) -> int | None:
    if endpoint.record.operator_id is not None:
        return endpoint.record.operator_id
    for part in endpoint.record.pos_label.split(":"):
        if part.startswith("op") and part[2:].isdigit():
            return int(part[2:])
    return None


@dataclass(frozen=True)
class QuarkSegment:
    """One Wick-contracted propagator segment with exact endpoint identity."""

    source_endpoint: Endpoint
    sink_endpoint: Endpoint

    def __post_init__(self) -> None:
        source = self.source_endpoint.record
        sink = self.sink_endpoint.record
        if not source.is_anti or sink.is_anti:
            raise ValueError(
                "A QuarkSegment must run from an antiquark endpoint to a quark endpoint."
            )
        if source.flavor != sink.flavor:
            raise ValueError(
                "Flavor cannot change inside one Wick propagator segment; "
                "flavor changes belong to an insertion transition."
            )

    @property
    def source_label(self) -> str:
        return self.source_endpoint.pos_label

    @property
    def sink_label(self) -> str:
        return self.sink_endpoint.pos_label

    @property
    def flavor(self) -> str:
        return self.sink_endpoint.record.flavor


@dataclass(frozen=True)
class FermionVertexTransition:
    """Operator-local q -> qbar transition used only for fermion topology.

    Unlike Stage-5 generalized-perambulator insertions, these transitions may
    occur at source, sink, or intermediate operators.  They represent the
    operator's spin/index connectivity and therefore close mesonic fermion
    lines at boundary operators without changing propagation-resource
    semantics.
    """

    operator_id: int | None
    quark_endpoint: Endpoint
    antiquark_endpoint: Endpoint
    spin_tensors: tuple[SpinTensorRecord, ...] = ()
    color_tensors: tuple[ColorTensorRecord, ...] = ()

    @property
    def incoming_flavor(self) -> str:
        return self.quark_endpoint.record.flavor

    @property
    def outgoing_flavor(self) -> str:
        return self.antiquark_endpoint.record.flavor


@dataclass(frozen=True)
class FermionOpenPath:
    """One non-cyclic fermion-line component."""

    start_endpoint: Endpoint
    end_endpoint: Endpoint
    segments: tuple[QuarkSegment, ...]
    transitions: tuple[FermionVertexTransition, ...] = ()

    @property
    def flavors(self) -> tuple[str, ...]:
        return tuple(segment.flavor for segment in self.segments)

    @property
    def operator_ids(self) -> frozenset[int]:
        ids = {
            operator_id
            for segment in self.segments
            for operator_id in (
                _endpoint_operator_id(segment.source_endpoint),
                _endpoint_operator_id(segment.sink_endpoint),
            )
            if operator_id is not None
        }
        return frozenset(ids)


@dataclass(frozen=True)
class FermionClosedCycle:
    """One cyclic fermion-line component.

    ``transitions[i]`` follows ``segments[i]`` and joins it to the next segment;
    the final transition closes the last segment back to the first.  The cycle
    is deterministically anchored by the lowest original Wick-segment index,
    but no physical meaning is attached to that anchor.
    """

    segments: tuple[QuarkSegment, ...]
    transitions: tuple[FermionVertexTransition, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("A closed fermion cycle must contain at least one Wick segment.")
        if len(self.transitions) != len(self.segments):
            raise ValueError(
                "A closed fermion cycle must have exactly one operator transition "
                "after every Wick segment."
            )

    @property
    def flavors(self) -> tuple[str, ...]:
        return tuple(segment.flavor for segment in self.segments)

    @property
    def operator_ids(self) -> frozenset[int]:
        return frozenset(
            transition.operator_id
            for transition in self.transitions
            if transition.operator_id is not None
        )

    @property
    def is_local(self) -> bool:
        return len(self.operator_ids) <= 1


@dataclass(frozen=True)
class FermionTopology:
    """Exact fermion-line decomposition of one Wick term."""

    open_paths: tuple[FermionOpenPath, ...] = ()
    closed_cycles: tuple[FermionClosedCycle, ...] = ()

    @property
    def num_components(self) -> int:
        return len(self.open_paths) + len(self.closed_cycles)

    @property
    def num_open_paths(self) -> int:
        return len(self.open_paths)

    @property
    def num_closed_cycles(self) -> int:
        return len(self.closed_cycles)


def _segments_from_correlator(correlator: LogicalCorrelator) -> tuple[QuarkSegment, ...]:
    return tuple(
        QuarkSegment(
            source_endpoint=Endpoint(
                record=line.antiquark,
                operations=line.antiquark_operations,
            ),
            sink_endpoint=Endpoint(
                record=line.quark,
                operations=line.quark_operations,
            ),
        )
        for line in correlator.quark_lines
    )


def _all_operator_ids(segments: tuple[QuarkSegment, ...]) -> tuple[int, ...]:
    return tuple(sorted({
        operator_id
        for segment in segments
        for operator_id in (
            _endpoint_operator_id(segment.source_endpoint),
            _endpoint_operator_id(segment.sink_endpoint),
        )
        if operator_id is not None
    }))


def _candidate_spin_tensors(
    correlator: LogicalCorrelator,
    operator_id: int,
) -> tuple[SpinTensorRecord, ...]:
    return tuple(
        tensor
        for tensor in correlator.spin_tensors
        if tensor.operator_id in (None, operator_id)
    )


def _candidate_color_tensors(
    correlator: LogicalCorrelator,
    operator_id: int,
    endpoints: tuple[Endpoint, Endpoint],
) -> tuple[ColorTensorRecord, ...]:
    endpoint_colors = {endpoint.record.color_label for endpoint in endpoints}
    return tuple(
        tensor
        for tensor in correlator.color_tensors
        if tensor.operator_id in (None, operator_id)
        and bool(set(tensor.indices) & endpoint_colors)
    )


def _pair_operator_transitions(
    correlator: LogicalCorrelator,
    operator_id: int,
    segments: tuple[QuarkSegment, ...],
) -> tuple[tuple[int, int, FermionVertexTransition], ...]:
    """Resolve all unambiguous q -> qbar continuations at one operator.

    This differs intentionally from Stage-4 intermediate-current pairing:
    source/sink operators may contain baryon legs or other unmatched fermion
    endpoints.  Those endpoints remain boundaries of open paths while any
    bilinear substructure resolved by spin/index connectivity is joined.
    """

    incoming = [
        (idx, segment.sink_endpoint)
        for idx, segment in enumerate(segments)
        if _endpoint_operator_id(segment.sink_endpoint) == operator_id
    ]
    outgoing = [
        (idx, segment.source_endpoint)
        for idx, segment in enumerate(segments)
        if _endpoint_operator_id(segment.source_endpoint) == operator_id
    ]

    if not incoming or not outgoing:
        return ()

    candidate_tensors = _candidate_spin_tensors(correlator, operator_id)
    adjacency: dict[str, set[str]] = defaultdict(set)

    for _, endpoint in incoming + outgoing:
        adjacency[endpoint.record.spin_label]

    for tensor in candidate_tensors:
        left, right = tensor.left_spin, tensor.right_spin
        adjacency[left].add(right)
        adjacency[right].add(left)

    component_for_label: dict[str, int] = {}
    labels_by_component: dict[int, set[str]] = {}
    component_id = 0

    for label in adjacency:
        if label in component_for_label:
            continue
        stack = [label]
        labels: set[str] = set()
        while stack:
            current = stack.pop()
            if current in labels:
                continue
            labels.add(current)
            stack.extend(adjacency[current] - labels)
        for member in labels:
            component_for_label[member] = component_id
        labels_by_component[component_id] = labels
        component_id += 1

    incoming_by_component: dict[int, list[tuple[int, Endpoint]]] = defaultdict(list)
    outgoing_by_component: dict[int, list[tuple[int, Endpoint]]] = defaultdict(list)

    for ref in incoming:
        incoming_by_component[
            component_for_label[ref[1].record.spin_label]
        ].append(ref)
    for ref in outgoing:
        outgoing_by_component[
            component_for_label[ref[1].record.spin_label]
        ].append(ref)

    pairs: list[tuple[int, int, FermionVertexTransition]] = []
    paired_incoming: set[int] = set()
    paired_outgoing: set[int] = set()

    for cid in sorted(labels_by_component):
        ins = incoming_by_component.get(cid, [])
        outs = outgoing_by_component.get(cid, [])

        # A purely incoming or outgoing spin component is an open boundary.
        if not ins or not outs:
            continue

        if len(ins) != 1 or len(outs) != 1:
            raise ValueError(
                f"Operator op{operator_id} has ambiguous fermion-line spin "
                "connectivity inside one spin-index component."
            )

        in_idx, quark_endpoint = ins[0]
        out_idx, antiquark_endpoint = outs[0]
        labels = labels_by_component[cid]
        selected_spin = tuple(
            tensor
            for tensor in candidate_tensors
            if tensor.left_spin in labels or tensor.right_spin in labels
        )
        pairs.append((
            in_idx,
            out_idx,
            FermionVertexTransition(
                operator_id=operator_id,
                quark_endpoint=quark_endpoint,
                antiquark_endpoint=antiquark_endpoint,
                spin_tensors=selected_spin,
                color_tensors=_candidate_color_tensors(
                    correlator,
                    operator_id,
                    (quark_endpoint, antiquark_endpoint),
                ),
            ),
        ))
        paired_incoming.add(in_idx)
        paired_outgoing.add(out_idx)

    # Preserve the scalar/single-bilinear case when no explicit spin tensor is
    # present and the endpoint labels alone did not connect the two sides.
    unmatched_incoming = [ref for ref in incoming if ref[0] not in paired_incoming]
    unmatched_outgoing = [ref for ref in outgoing if ref[0] not in paired_outgoing]
    if len(incoming) == 1 and len(outgoing) == 1 and not pairs:
        in_idx, quark_endpoint = unmatched_incoming[0]
        out_idx, antiquark_endpoint = unmatched_outgoing[0]
        pairs.append((
            in_idx,
            out_idx,
            FermionVertexTransition(
                operator_id=operator_id,
                quark_endpoint=quark_endpoint,
                antiquark_endpoint=antiquark_endpoint,
                spin_tensors=(),
                color_tensors=_candidate_color_tensors(
                    correlator,
                    operator_id,
                    (quark_endpoint, antiquark_endpoint),
                ),
            ),
        ))

    return tuple(pairs)


def _transition_graph(
    correlator: LogicalCorrelator,
    segments: tuple[QuarkSegment, ...],
) -> tuple[
    dict[int, tuple[int, FermionVertexTransition]],
    dict[int, int],
]:
    """Build the exact segment-to-segment graph at every operator vertex."""

    next_segment: dict[int, tuple[int, FermionVertexTransition]] = {}
    previous_segment: dict[int, int] = {}

    for operator_id in _all_operator_ids(segments):
        for incoming_idx, outgoing_idx, transition in _pair_operator_transitions(
            correlator,
            operator_id,
            segments,
        ):
            if incoming_idx in next_segment:
                raise ValueError(
                    f"Wick segment {incoming_idx} has more than one fermion-line "
                    "continuation through operator-local spin connectivity."
                )
            if outgoing_idx in previous_segment:
                raise ValueError(
                    f"Wick segment {outgoing_idx} has more than one incoming "
                    "fermion-line continuation."
                )

            next_segment[incoming_idx] = (outgoing_idx, transition)
            previous_segment[outgoing_idx] = incoming_idx

    return next_segment, previous_segment


def reconstruct_fermion_topology(correlator: LogicalCorrelator) -> FermionTopology:
    """Decompose a Wick term into open fermion paths and closed cycles.

    Wick propagator segments are edges.  At every operator, q and qbar segment
    endpoints are joined only when the existing spin/index connectivity resolves
    an operator-local fermion transition.  Unmatched endpoints remain genuine
    open boundaries.  This naturally gives, for example:

      * baryon two-point: three open paths;
      * meson two-point: one closed cycle spanning source and sink;
      * factorized meson-meson: multiple independent closed cycles;
      * baryon-meson: open baryon paths plus mesonic cycle(s);
      * disconnected local current: an additional local closed cycle.

    The algorithm has no meson/baryon/multi-hadron branches.
    """

    segments = _segments_from_correlator(correlator)
    if not segments:
        return FermionTopology()

    next_segment, previous_segment = _transition_graph(correlator, segments)

    visited: set[int] = set()
    open_paths: list[FermionOpenPath] = []
    closed_cycles: list[FermionClosedCycle] = []

    # Any segment with no predecessor starts a non-cyclic component.
    for start_idx in range(len(segments)):
        if start_idx in previous_segment or start_idx in visited:
            continue

        chain_indices: list[int] = []
        transitions: list[FermionVertexTransition] = []
        current_idx = start_idx
        local_seen: set[int] = set()

        while True:
            if current_idx in local_seen:
                raise ValueError(
                    "A fermion component reached a cycle from an open-path start; "
                    "the operator transition graph is not one-to-one."
                )
            local_seen.add(current_idx)
            visited.add(current_idx)
            chain_indices.append(current_idx)

            transition = next_segment.get(current_idx)
            if transition is None:
                break
            current_idx, vertex_transition = transition
            transitions.append(vertex_transition)

        chain = tuple(segments[idx] for idx in chain_indices)
        open_paths.append(FermionOpenPath(
            start_endpoint=chain[0].source_endpoint,
            end_endpoint=chain[-1].sink_endpoint,
            segments=chain,
            transitions=tuple(transitions),
        ))

    # Every remaining one-in/one-out component is a closed cycle.  Iterating
    # segment indices in ascending order gives a deterministic cycle anchor.
    for anchor_idx in range(len(segments)):
        if anchor_idx in visited:
            continue

        cycle_indices: list[int] = []
        transitions: list[FermionVertexTransition] = []
        current_idx = anchor_idx
        local_seen: set[int] = set()

        while True:
            if current_idx in local_seen:
                if current_idx != anchor_idx:
                    raise ValueError(
                        "A closed fermion component entered a previously seen segment "
                        "before returning to its deterministic anchor."
                    )
                break

            if current_idx in visited:
                raise ValueError(
                    "Two fermion components unexpectedly share a Wick segment."
                )

            local_seen.add(current_idx)
            visited.add(current_idx)
            cycle_indices.append(current_idx)

            transition = next_segment.get(current_idx)
            if transition is None:
                raise ValueError(
                    "An unvisited fermion component has no continuation; it should "
                    "have been reconstructed as an open path."
                )

            current_idx, vertex_transition = transition
            transitions.append(vertex_transition)

        closed_cycles.append(FermionClosedCycle(
            segments=tuple(segments[idx] for idx in cycle_indices),
            transitions=tuple(transitions),
        ))

    if len(visited) != len(segments):
        raise RuntimeError("Fermion topology reconstruction did not consume every Wick segment.")

    return FermionTopology(
        open_paths=tuple(open_paths),
        closed_cycles=tuple(closed_cycles),
    )
