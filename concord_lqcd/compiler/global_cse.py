"""Global common-subexpression elimination for primitive contraction graphs.

This layer consumes the frozen per-Wick-term ``WickContractionGraph`` objects
and builds a backend-independent computation DAG across an arbitrary collection
of correlators.  It performs only exact structural reuse:

* primitive numerical resources are deduplicated globally by ``resource_id``;
* connected tensor-network components are canonicalized independent of local
  node ids and dummy contraction-index labels;
* identical connected components are represented by one computation node and
  reused by every Wick term that needs them;
* complete Wick terms with the same post-CSE component multiset and exact
  external-index bindings are combined, and exact scalar zero sums are removed.

No pairwise contraction order, memory planning, scheduling, device placement,
or runtime execution is introduced here.  Additional CSE of pairwise
intermediates belongs after a contraction-order representation exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from concord_lqcd.compiler.contraction_graph import (
    PrimitiveTensorKind,
    PrimitiveTensorNode,
    TensorLegBinding,
    WickContractionGraph,
    lower_physics_ir_to_contraction_graphs,
)
from concord_lqcd.compiler.distillation_lowering import KernelRecipe, kernel_resource_signature
from concord_lqcd.compiler.physics_ir import PhysicsIR, SpinTensorRecord
from concord_lqcd.compiler.propagation_lowering import PropagationRecipe


@dataclass(frozen=True)
class GlobalTensorResource:
    """One globally unique numerical tensor resource requirement."""

    resource_id: str
    kind: PrimitiveTensorKind
    payload: Any


@dataclass(frozen=True)
class ComponentTensorUse:
    """One occurrence of a global tensor resource inside a component computation."""

    slot: str
    kind: PrimitiveTensorKind
    resource_id: str
    placement: tuple[Any, ...]
    bindings: tuple[TensorLegBinding, ...]


@dataclass(frozen=True)
class ComponentComputation:
    """One reusable connected n-ary tensor contraction.

    ``tensors`` contains tensor occurrences, not unique resources.  Therefore a
    component may use the same perambulator resource multiple times without
    collapsing those distinct legs.  ``output_indices`` are canonical free
    indices; scalar components have none.
    """

    computation_id: str
    signature: str
    tensors: tuple[ComponentTensorUse, ...]
    output_indices: tuple[str, ...]

    @property
    def resource_ids(self) -> tuple[str, ...]:
        return tuple(tensor.resource_id for tensor in self.tensors)

    @property
    def is_scalar(self) -> bool:
        return not self.output_indices


@dataclass(frozen=True)
class ComponentUse:
    """Use of a canonical component computation by one Wick term."""

    computation_id: str
    output_bindings: tuple[TensorLegBinding, ...] = ()


@dataclass(frozen=True)
class GlobalWickTermUse:
    """One Wick term expressed as a coefficient times reusable components."""

    term_index: int
    coefficient: complex
    components: tuple[ComponentUse, ...]


@dataclass(frozen=True)
class GlobalCorrelatorUse:
    """All Wick terms associated with one user-supplied correlator key."""

    correlator_id: str
    terms: tuple[GlobalWickTermUse, ...]


@dataclass(frozen=True)
class GlobalCSEStatistics:
    """Raw occurrence counts and live unique counts after global optimization."""

    resource_occurrences: int
    unique_resources: int
    component_occurrences: int
    unique_components: int

    @property
    def resource_reuses(self) -> int:
        return self.resource_occurrences - self.unique_resources

    @property
    def component_reuses(self) -> int:
        return self.component_occurrences - self.unique_components


@dataclass(frozen=True)
class GlobalComputationDAG:
    """Global exact-reuse DAG for a collection of correlator contraction graphs."""

    resources: Mapping[str, GlobalTensorResource]
    components: Mapping[str, ComponentComputation]
    correlators: tuple[GlobalCorrelatorUse, ...]
    statistics: GlobalCSEStatistics
    aggregation_proof: Any = None

    def component(self, computation_id: str) -> ComponentComputation:
        return self.components[computation_id]

    def resource(self, resource_id: str) -> GlobalTensorResource:
        return self.resources[resource_id]


def _component_use_sort_key(
    use: ComponentUse,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        use.computation_id,
        tuple((binding.port, binding.index) for binding in use.output_bindings),
    )


def _public_output_index(index: str) -> str:
    """Remove only the private basis scope, preserving insertion and free labels."""
    parts = index.split(":")
    if (
        len(parts) >= 4
        and re.fullmatch(r"c[0-9]+|basis_[0-9a-f]{20}", parts[1])
        and parts[2].startswith("op")
        and parts[2][2:].isdigit()
    ):
        return ":".join((parts[0], *parts[2:]))
    return index


def _aggregate_correlator_terms(
    correlator: GlobalCorrelatorUse,
) -> GlobalCorrelatorUse:
    """Combine exactly equal complete Wick terms within one correlator sum.

    Component multiplicity and occurrence-specific output bindings are part of
    the key. Coefficients are accumulated in input order. A scalar group is
    removed only when its resulting coefficient compares exactly equal to zero;
    an open-index zero term is retained because it carries the correlator's
    output interface. This intentionally performs no tolerance-based
    simplification.
    """

    coefficients: dict[tuple[ComponentUse, ...], complex] = {}
    for term in correlator.terms:
        components = tuple(sorted(term.components, key=_component_use_sort_key))
        if components in coefficients:
            coefficients[components] = coefficients[components] + term.coefficient
        else:
            coefficients[components] = term.coefficient

    terms = tuple(
        GlobalWickTermUse(
            term_index=term_index,
            coefficient=coefficient,
            components=components,
        )
        for term_index, (components, coefficient) in enumerate(
            (
                item
                for item in coefficients.items()
                if item[1] != 0 or any(use.output_bindings for use in item[0])
            )
        )
    )
    return GlobalCorrelatorUse(
        correlator_id=correlator.correlator_id,
        terms=terms,
    )


def aggregate_global_wick_terms(dag: GlobalComputationDAG) -> GlobalComputationDAG:
    """Aggregate exact complete-term reuse and prune unreachable scalar work.

    Aggregation is local to each correlator sum; equal terms belonging to two
    different correlators remain distinct outputs. External-index bindings
    remain part of term identity, so this pass never assumes that named output
    indices may be alpha-renamed.
    """

    referenced_component_ids = {
        component_use.computation_id
        for correlator in dag.correlators
        for term in correlator.terms
        for component_use in term.components
    }
    missing_components = referenced_component_ids - set(dag.components)
    if missing_components:
        raise ValueError(
            "Wick terms reference component computations missing from the global "
            f"DAG: {tuple(sorted(missing_components))}."
        )

    referenced_resource_ids = {
        tensor.resource_id
        for computation_id in referenced_component_ids
        for tensor in dag.components[computation_id].tensors
    }
    missing_resources = referenced_resource_ids - set(dag.resources)
    if missing_resources:
        raise ValueError(
            "Component computations reference resources missing from the global "
            f"DAG: {tuple(sorted(missing_resources))}."
        )

    correlators = tuple(_aggregate_correlator_terms(use) for use in dag.correlators)
    live_component_ids = {
        component_use.computation_id
        for correlator in correlators
        for term in correlator.terms
        for component_use in term.components
    }
    components = {
        computation_id: component
        for computation_id, component in dag.components.items()
        if computation_id in live_component_ids
    }

    live_resource_ids = {
        tensor.resource_id
        for component in components.values()
        for tensor in component.tensors
    }
    resources = {
        resource_id: resource
        for resource_id, resource in dag.resources.items()
        if resource_id in live_resource_ids
    }

    return GlobalComputationDAG(
        resources=resources,
        components=components,
        correlators=correlators,
        statistics=GlobalCSEStatistics(
            resource_occurrences=dag.statistics.resource_occurrences,
            unique_resources=len(resources),
            component_occurrences=dag.statistics.component_occurrences,
            unique_components=len(components),
        ),
    )


def _kernel_operator_ids(kernel: KernelRecipe) -> tuple[int, ...]:
    return tuple(sorted({
        field.record.operator_id
        for field in kernel.projected_fields
        if field.record.operator_id is not None
    }))


def _operator_for_kernel_port(kernel: KernelRecipe, port: str) -> int | None:
    try:
        idx = kernel.open_ports.index(port)
    except ValueError as exc:
        raise ValueError(f"Kernel port {port!r} is not present in the kernel recipe.") from exc
    return kernel.projected_fields[idx].record.operator_id


def _node_placement(node: PrimitiveTensorNode) -> tuple[Any, ...]:
    """Return runtime-time-slot placement information for safe CSE.

    Resource ids intentionally omit source/sink times and alpha-renamed spatial
    labels.  That is correct for identifying reusable data/programs, but a
    computed tensor value at operator slot 0 must not be CSE'd with the same
    resource evaluated at operator slot 1.  Placement therefore participates in
    component identity while remaining separate from resource identity.
    """

    payload = node.payload
    if node.kind == "kernel":
        if not isinstance(payload, KernelRecipe):
            raise TypeError("Kernel graph node does not carry a KernelRecipe payload.")
        return ("kernel_ops", _kernel_operator_ids(payload))

    if node.kind == "spin":
        if not isinstance(payload, SpinTensorRecord):
            raise TypeError("Spin graph node does not carry a SpinTensorRecord payload.")
        return ("spin_op", payload.operator_id)

    if node.kind == "propagation":
        if not isinstance(payload, PropagationRecipe):
            raise TypeError("Propagation graph node does not carry a PropagationRecipe payload.")
        source_op = _operator_for_kernel_port(payload.start_kernel, payload.start_port)
        sink_op = _operator_for_kernel_port(payload.end_kernel, payload.end_port)
        insertion_ops = tuple(
            _operator_for_kernel_port(spec.kernel, spec.incoming_port)
            for spec in payload.insertion_specs
        )
        return ("path_ops", source_op, insertion_ops, sink_op)

    raise ValueError(f"Unsupported primitive tensor kind {node.kind!r}.")


def _graph_operator_arity(graph: WickContractionGraph) -> int:
    ids: set[int] = set()
    for node in graph.nodes:
        placement = _node_placement(node)
        stack = [placement]
        while stack:
            value = stack.pop()
            if isinstance(value, tuple):
                stack.extend(value)
            elif isinstance(value, int):
                ids.add(value)
    inferred = max(ids) + 1 if ids else 0
    if graph.operator_arity is None:
        return inferred
    if graph.operator_arity < inferred:
        raise ValueError("Contraction graph operator_arity omits a primitive operator slot.")
    return graph.operator_arity


def _index_endpoints(
    graph: WickContractionGraph,
) -> dict[str, list[tuple[int, str]]]:
    endpoints: dict[str, list[tuple[int, str]]] = {}
    for node_idx, node in enumerate(graph.nodes):
        for binding in node.bindings:
            endpoints.setdefault(binding.index, []).append((node_idx, binding.port))
    return endpoints


def _connected_node_components(graph: WickContractionGraph) -> tuple[tuple[int, ...], ...]:
    endpoints = _index_endpoints(graph)
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(graph.nodes))}
    for refs in endpoints.values():
        if len(refs) == 2:
            left, right = refs
            if left[0] != right[0]:
                adjacency[left[0]].add(right[0])
                adjacency[right[0]].add(left[0])
        elif len(refs) > 2:
            raise ValueError("Global CSE received an overconnected contraction index.")

    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for start in range(len(graph.nodes)):
        if start in seen:
            continue
        stack = [start]
        members: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            members.append(current)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        components.append(tuple(sorted(members)))
    return tuple(components)


def _component_index_endpoints(
    graph: WickContractionGraph,
    members: Sequence[int],
) -> dict[str, list[tuple[int, str]]]:
    member_set = set(members)
    endpoints: dict[str, list[tuple[int, str]]] = {}
    for node_idx in members:
        node = graph.nodes[node_idx]
        for binding in node.bindings:
            endpoints.setdefault(binding.index, []).append((node_idx, binding.port))

    for index, refs in endpoints.items():
        if len(refs) > 2:
            raise ValueError(f"Contraction index {index!r} has degree greater than two.")
        if any(node_idx not in member_set for node_idx, _ in refs):
            raise RuntimeError("Component index endpoint escaped its node component.")
    return endpoints


def _other_endpoint(
    refs: Sequence[tuple[int, str]],
    node_idx: int,
    port: str,
) -> tuple[int, str] | None:
    if len(refs) == 1:
        return None
    if len(refs) != 2:
        raise ValueError("Tensor-network index must have degree one or two.")
    if refs[0] == (node_idx, port):
        return refs[1]
    if refs[1] == (node_idx, port):
        return refs[0]
    raise RuntimeError("Tensor binding is missing from its contraction-index endpoint list.")


def _canonical_traversal(
    graph: WickContractionGraph,
    members: Sequence[int],
    arity: int,
) -> tuple[tuple[int, ...], str]:
    """Choose a node-id/index-label independent traversal of one connected component."""

    endpoints = _component_index_endpoints(graph, members)
    member_set = set(members)
    candidates: list[tuple[str, tuple[int, ...]]] = []

    for root in members:
        assigned: dict[int, int] = {}
        order: list[int] = []

        def visit(node_idx: int) -> None:
            if node_idx in assigned:
                return
            assigned[node_idx] = len(order)
            order.append(node_idx)
            node = graph.nodes[node_idx]
            for binding in node.bindings:
                other = _other_endpoint(endpoints[binding.index], node_idx, binding.port)
                if other is not None and other[0] in member_set:
                    visit(other[0])

        visit(root)
        if len(order) != len(members):
            raise RuntimeError("Canonical traversal did not cover the connected component.")

        descriptors = []
        for node_idx in order:
            node = graph.nodes[node_idx]
            ports = []
            for binding in node.bindings:
                other = _other_endpoint(endpoints[binding.index], node_idx, binding.port)
                if other is None:
                    ports.append((binding.port, "free"))
                else:
                    ports.append((
                        binding.port,
                        "edge",
                        assigned[other[0]],
                        other[1],
                    ))
            descriptors.append({
                "kind": node.kind,
                "resource": node.resource_id,
                "placement": _node_placement(node),
                "ports": ports,
            })

        encoded = json.dumps(
            {"arity": arity, "nodes": descriptors},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        candidates.append((encoded, tuple(order)))

    return min(candidates, key=lambda item: item[0])[1], min(candidates, key=lambda item: item[0])[0]


def _canonicalize_component(
    graph: WickContractionGraph,
    members: Sequence[int],
    arity: int,
) -> tuple[ComponentComputation, tuple[TensorLegBinding, ...]]:
    order, traversal_signature = _canonical_traversal(graph, members, arity)
    canonical_node = {node_idx: idx for idx, node_idx in enumerate(order)}
    endpoints = _component_index_endpoints(graph, members)

    internal_specs: list[tuple[tuple[int, str], tuple[int, str], str]] = []
    free_specs: list[tuple[tuple[int, str], str]] = []

    for original_index, refs in endpoints.items():
        if len(refs) == 1:
            node_idx, port = refs[0]
            free_specs.append(((canonical_node[node_idx], port), original_index))
        else:
            lhs = (canonical_node[refs[0][0]], refs[0][1])
            rhs = (canonical_node[refs[1][0]], refs[1][1])
            if rhs < lhs:
                lhs, rhs = rhs, lhs
            internal_specs.append((lhs, rhs, original_index))

    internal_specs.sort(key=lambda item: (item[0], item[1]))
    free_specs.sort(key=lambda item: item[0])

    canonical_index: dict[str, str] = {}
    for idx, (_, _, original_index) in enumerate(internal_specs):
        canonical_index[original_index] = f"i{idx}"
    for idx, (_, original_index) in enumerate(free_specs):
        canonical_index[original_index] = f"o{idx}"

    tensor_uses: list[ComponentTensorUse] = []
    for slot_idx, node_idx in enumerate(order):
        node = graph.nodes[node_idx]
        tensor_uses.append(ComponentTensorUse(
            slot=f"t{slot_idx}",
            kind=node.kind,
            resource_id=node.resource_id,
            placement=_node_placement(node),
            bindings=tuple(
                TensorLegBinding(binding.port, canonical_index[binding.index])
                for binding in node.bindings
            ),
        ))

    output_indices = tuple(f"o{idx}" for idx in range(len(free_specs)))
    payload = {
        "arity": arity,
        "traversal": traversal_signature,
        "tensors": [
            {
                "slot": tensor.slot,
                "kind": tensor.kind,
                "resource": tensor.resource_id,
                "placement": tensor.placement,
                "bindings": [(b.port, b.index) for b in tensor.bindings],
            }
            for tensor in tensor_uses
        ],
        "outputs": output_indices,
    }
    signature = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
    computation = ComponentComputation(
        computation_id=f"component_{digest}",
        signature=signature,
        tensors=tuple(tensor_uses),
        output_indices=output_indices,
    )
    output_bindings = tuple(
        TensorLegBinding(f"o{idx}", original_index)
        for idx, (_, original_index) in enumerate(free_specs)
    )
    return computation, output_bindings


def _register_resource(
    resources: dict[str, GlobalTensorResource],
    node: PrimitiveTensorNode,
    signatures: dict[str, str],
) -> None:
    if node.kind == "kernel":
        signature = kernel_resource_signature(node.payload)
    elif node.kind == "propagation":
        signature = node.payload.resource_signature
    elif node.kind == "spin":
        signature = node.payload.kind
    else:
        raise ValueError(f"Unsupported primitive tensor kind {node.kind!r}.")

    existing = resources.get(node.resource_id)
    if existing is None:
        resources[node.resource_id] = GlobalTensorResource(
            resource_id=node.resource_id,
            kind=node.kind,
            payload=node.payload,
        )
        signatures[node.resource_id] = signature
        return
    if existing.kind != node.kind:
        raise ValueError(
            f"Resource id {node.resource_id!r} is used by both {existing.kind!r} "
            f"and {node.kind!r} tensor nodes."
        )
    if signatures[node.resource_id] != signature:
        raise ValueError(
            f"Resource id {node.resource_id!r} has incompatible {node.kind!r} "
            "payload signatures. Distinct tensors must use distinct resource ids."
        )


def build_global_cse(
    graphs_by_correlator: Mapping[str, Sequence[WickContractionGraph]],
) -> GlobalComputationDAG:
    """Build exact global resource, component, and term CSE for all correlators."""

    resources: dict[str, GlobalTensorResource] = {}
    resource_signatures: dict[str, str] = {}
    components: dict[str, ComponentComputation] = {}
    correlator_uses: list[GlobalCorrelatorUse] = []
    resource_occurrences = 0
    component_occurrences = 0

    for correlator_id, graphs in graphs_by_correlator.items():
        terms: list[GlobalWickTermUse] = []
        for term_index, graph in enumerate(graphs):
            graph.validate()
            arity = _graph_operator_arity(graph)

            for node in graph.nodes:
                resource_occurrences += 1
                _register_resource(resources, node, resource_signatures)

            component_uses: list[ComponentUse] = []
            for members in _connected_node_components(graph):
                component_occurrences += 1
                computation, output_bindings = _canonicalize_component(
                    graph, members, arity
                )
                existing = components.get(computation.computation_id)
                if existing is None:
                    components[computation.computation_id] = computation
                elif existing.signature != computation.signature:
                    raise RuntimeError(
                        "Component hash collision detected between distinct canonical "
                        "tensor-network signatures."
                    )
                component_uses.append(ComponentUse(
                    computation_id=computation.computation_id,
                    output_bindings=tuple(
                        TensorLegBinding(binding.port, _public_output_index(binding.index))
                        for binding in output_bindings
                    ),
                ))

            # Multiplication of disconnected scalar components is commutative.
            # Sorting uses gives deterministic output without changing the component
            # computations themselves.  For non-scalars, bindings retain the mapping
            # back to the Wick-term external indices.
            component_uses.sort(key=_component_use_sort_key)
            terms.append(GlobalWickTermUse(
                term_index=term_index,
                coefficient=graph.coefficient,
                components=tuple(component_uses),
            ))

        correlator_uses.append(GlobalCorrelatorUse(
            correlator_id=str(correlator_id),
            terms=tuple(terms),
        ))

    dag = GlobalComputationDAG(
        resources=dict(resources),
        components=dict(components),
        correlators=tuple(correlator_uses),
        statistics=GlobalCSEStatistics(
            resource_occurrences=resource_occurrences,
            unique_resources=len(resources),
            component_occurrences=component_occurrences,
            unique_components=len(components),
        ),
    )
    return aggregate_global_wick_terms(dag)


def lower_physics_ir_collection_to_global_cse(
    physics_ir_by_correlator: Mapping[str, PhysicsIR],
    *,
    mode_size: int | None = None,
    shared_orthonormal_laph_basis: bool = False,
) -> GlobalComputationDAG:
    """Lower correlators with only explicitly declared LapH mode identities.

    Mode size alone does not prove orthonormality or equality across flavors.
    Identity elimination requires both a mode size and the shared orthonormal
    basis contract. Without that declaration, overlap kernels remain explicit.
    """
    if type(shared_orthonormal_laph_basis) is not bool:
        raise TypeError("shared_orthonormal_laph_basis must be a boolean.")
    if shared_orthonormal_laph_basis and mode_size is None:
        raise ValueError("The shared LapH basis contract requires mode_size.")

    graphs = {
        str(correlator_id): tuple(
            result.graph
            if not shared_orthonormal_laph_basis
            else result.graph.eliminate_exact_mode_identity_kernels(
                mode_size=mode_size
            )
            for result in lower_physics_ir_to_contraction_graphs(physics_ir)
        )
        for correlator_id, physics_ir in physics_ir_by_correlator.items()
    }
    return build_global_cse(graphs)
