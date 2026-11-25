"""Primitive tensor-contraction graph construction for one PhysicsIR Wick term.

This layer converts the verified symbolic/distillation objects into one
backend-independent tensor graph per ``LogicalCorrelator``.  It does not choose
contraction order, perform CSE, schedule work, manage memory, or execute tensors.

Full fermion topology is authoritative.  LapH operator transitions remain
explicit kernel/spin contractions.  Explicit POINT bilinear transitions may be
absorbed into generalized propagation resources; their spatial kernel and spin
tensors then do not appear a second time as standalone graph nodes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from itertools import permutations
import json
from math import factorial
from typing import Any, Literal, Mapping, Sequence

from concord_lqcd.compiler.physics_ir import (
    LogicalCorrelator,
    PhysicsIR,
    Representation,
    SpinTensorRecord,
    _validate_coefficient,
)
from concord_lqcd.compiler.distillation_lowering import (
    Endpoint,
    KernelRecipe,
    ProjectionContribution,
    construct_kernel_recipe,
    disconnected_color_scalar,
    identify_kernel_resource,
    identify_operator_vertices,
    kernel_resource_signature,
    realize_projected_fields,
)
from concord_lqcd.compiler.fermion_topology import (
    FermionClosedCycle,
    FermionOpenPath,
    FermionTopology,
    FermionVertexTransition,
    QuarkSegment,
    reconstruct_fermion_topology,
)
from concord_lqcd.compiler.propagation_lowering import (
    PropagationRecipe,
    _infer_propagation_recipe,
)
from concord_lqcd.compiler.closed_cycle_lowering import lower_closed_cycle


PrimitiveTensorKind = Literal["kernel", "propagation", "spin"]

_MAX_KERNEL_AXIS_CANONICALIZATIONS = 720


@dataclass(frozen=True)
class TensorLegBinding:
    """Bind one local tensor port to one global contraction index."""

    port: str
    index: str


@dataclass(frozen=True)
class PrimitiveTensorNode:
    """One tensor occurrence in a Wick-term contraction graph.

    ``resource_id`` identifies reusable numerical data; ``node_id`` identifies
    this occurrence.  Equal resource ids are intentionally not deduplicated here.
    """

    node_id: str
    kind: PrimitiveTensorKind
    resource_id: str
    bindings: tuple[TensorLegBinding, ...]
    payload: Any

    @property
    def legs(self) -> tuple[str, ...]:
        return tuple(binding.index for binding in self.bindings)

    def index_for_port(self, port: str) -> str:
        matches = [binding.index for binding in self.bindings if binding.port == port]
        if len(matches) != 1:
            raise KeyError(
                f"Tensor node {self.node_id!r} does not expose exactly one port {port!r}."
            )
        return matches[0]


@dataclass(frozen=True)
class WickContractionGraph:
    """Primitive tensor graph for one complete PhysicsIR Wick term."""

    coefficient: complex
    nodes: tuple[PrimitiveTensorNode, ...]
    # Preserve insertion provenance when constant factors or exact identities
    # disappear. Hand-built graphs may continue to infer arity from their nodes.
    operator_arity: int | None = None

    @property
    def index_degrees(self) -> Mapping[str, int]:
        return dict(Counter(index for node in self.nodes for index in node.legs))

    @property
    def free_indices(self) -> tuple[str, ...]:
        return tuple(index for index, degree in self.index_degrees.items() if degree == 1)

    @property
    def overconnected_indices(self) -> tuple[str, ...]:
        return tuple(index for index, degree in self.index_degrees.items() if degree > 2)

    @property
    def is_closed_scalar(self) -> bool:
        degrees = self.index_degrees
        return bool(self.nodes) and all(degree == 2 for degree in degrees.values())

    def nodes_of_kind(self, kind: PrimitiveTensorKind) -> tuple[PrimitiveTensorNode, ...]:
        return tuple(node for node in self.nodes if node.kind == kind)

    def eliminate_exact_mode_identity_kernels(
        self,
        *,
        mode_size: int,
    ) -> "WickContractionGraph":
        """Remove mode identities under a declared shared orthonormal basis.

        Calling this rewrite directly asserts that all flavors use the same
        orthonormal modes at each insertion. Normal compilation gates this
        rewrite on its explicit basis contract.

        Replacing ``delta(i, j)`` by one shared contraction index is exact when
        both mode legs are contracted elsewhere.  A redundant identity edge
        closes an independent mode loop and therefore contributes
        ``mode_size``.  Kernels carrying free indices remain explicit because
        they represent an identity-valued output rather than a contraction.
        """
        if not isinstance(mode_size, int) or isinstance(mode_size, bool):
            raise TypeError("mode_size must be a positive integer.")
        if mode_size <= 0:
            raise ValueError("mode_size must be a positive integer.")

        degrees = self.index_degrees
        identity_nodes = tuple(
            node
            for node in self.nodes
            if node.kind == "kernel"
            and isinstance(node.payload, KernelRecipe)
            and node.payload.is_exact_mode_identity
            and len(node.bindings) == 2
            and all(degrees[binding.index] == 2 for binding in node.bindings)
        )
        if not identity_nodes:
            return self

        parents: dict[str, str] = {}

        def representative(index: str) -> str:
            parent = parents.get(index, index)
            if parent != index:
                parent = representative(parent)
                parents[index] = parent
            return parent

        def merge(first_index: str, second_index: str) -> bool:
            first_root = representative(first_index)
            second_root = representative(second_index)
            if first_root == second_root:
                return False
            retained_root, replaced_root = sorted((first_root, second_root))
            parents[replaced_root] = retained_root
            return True

        coefficient = self.coefficient
        removed_node_ids: set[str] = set()
        for node in identity_nodes:
            removed_node_ids.add(node.node_id)
            first_index = node.bindings[0].index
            second_index = node.bindings[1].index
            if not merge(first_index, second_index):
                coefficient *= mode_size

        rewritten_nodes = tuple(
            PrimitiveTensorNode(
                node_id=node.node_id,
                kind=node.kind,
                resource_id=node.resource_id,
                bindings=tuple(
                    TensorLegBinding(binding.port, representative(binding.index))
                    for binding in node.bindings
                ),
                payload=node.payload,
            )
            for node in self.nodes
            if node.node_id not in removed_node_ids
        )
        rewritten = WickContractionGraph(
            coefficient=coefficient,
            nodes=rewritten_nodes,
            operator_arity=self.operator_arity,
        )
        rewritten.validate()
        return rewritten

    def validate(self) -> None:
        if self.operator_arity is not None and (
            type(self.operator_arity) is not int or self.operator_arity < 0
        ):
            raise ValueError("Contraction graph operator_arity must be a non-negative integer.")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Contraction graph contains duplicate tensor node ids.")
        for node in self.nodes:
            ports = [binding.port for binding in node.bindings]
            if len(ports) != len(set(ports)):
                raise ValueError(
                    f"Tensor node {node.node_id!r} contains duplicate local port names."
                )
            if any(not binding.index for binding in node.bindings):
                raise ValueError(
                    f"Tensor node {node.node_id!r} contains an empty contraction index."
                )
        if self.overconnected_indices:
            raise ValueError(
                "Contraction graph contains indices attached to more than two primitive "
                f"tensor legs: {self.overconnected_indices}."
            )


@dataclass(frozen=True)
class ContractionGraphLoweringResult:
    """Debuggable result of lowering one LogicalCorrelator to primitive tensors."""

    kernels: tuple[KernelRecipe, ...]
    topology: FermionTopology
    graph: WickContractionGraph


@dataclass(frozen=True)
class _PropagationUse:
    recipe: PropagationRecipe
    start_endpoint: Endpoint
    end_endpoint: Endpoint


def _spin_resource_id(tensor: SpinTensorRecord) -> str:
    digest = hashlib.sha256(
        json.dumps({"kind": tensor.kind}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"spin_tensor_{digest}"


def _canonicalize_kernel_axes(
    kernel: KernelRecipe,
) -> tuple[KernelRecipe, tuple[int, ...]]:
    """Canonicalize field operands and return their original local axes.

    The projected fields multiply as scalar operands before their open mode
    axes are exposed. Reordering them, including fields of different flavors
    or transformations, therefore permutes only the open axes. The full
    resource signature retains those field attributes and color incidence.
    Large permutation spaces retain the original recipe and graph exactly.
    """

    expected_ports = tuple(f"mode_{index}" for index in range(kernel.num_quarks))
    if kernel.open_ports != expected_ports:
        return kernel, tuple(range(kernel.num_quarks))

    if factorial(kernel.num_quarks) > _MAX_KERNEL_AXIS_CANONICALIZATIONS:
        return kernel, tuple(range(kernel.num_quarks))

    best: tuple[str, tuple[int, ...], KernelRecipe] | None = None
    for axis_order in permutations(range(kernel.num_quarks)):
        candidate = replace(
            kernel,
            projected_fields=tuple(
                kernel.projected_fields[index] for index in axis_order
            ),
        )
        item = (
            kernel_resource_signature(candidate),
            axis_order,
            candidate,
        )
        if best is None or item[:2] < best[:2]:
            best = item
    assert best is not None
    return best[2], best[1]


def _is_point_transition(transition: FermionVertexTransition) -> bool:
    return (
        transition.quark_endpoint.representation == Representation.POINT
        and transition.antiquark_endpoint.representation == Representation.POINT
    )


def _require_explicit_laph_transition(transition: FermionVertexTransition) -> None:
    if (
        transition.quark_endpoint.representation != Representation.LAPH
        or transition.antiquark_endpoint.representation != Representation.LAPH
    ):
        raise NotImplementedError(
            "A non-absorbed fermion transition currently requires LapH endpoints. "
            "Mixed POINT/LapH operator transitions need an additional numerical "
            "resource realization before entering the primitive tensor graph."
        )


def _infer_chain_recipe(
    segments: Sequence[QuarkSegment],
    absorbed_transitions: Sequence[FermionVertexTransition],
    kernel_map: Mapping[str, KernelRecipe],
) -> _PropagationUse:
    if not segments:
        raise ValueError("Cannot construct a propagation chain without Wick segments.")
    if len(segments) != len(absorbed_transitions) + 1:
        raise ValueError(
            "A propagation chain must contain exactly one more Wick segment than "
            "absorbed POINT transitions."
        )
    return _PropagationUse(
        recipe=_infer_propagation_recipe(
            start_endpoint=segments[0].source_endpoint,
            end_endpoint=segments[-1].sink_endpoint,
            segments=segments,
            insertions=absorbed_transitions,
            kernel_map=kernel_map,
        ),
        start_endpoint=segments[0].source_endpoint,
        end_endpoint=segments[-1].sink_endpoint,
    )


def _partition_open_path(
    path: FermionOpenPath,
    kernel_map: Mapping[str, KernelRecipe],
) -> tuple[_PropagationUse, ...]:
    segments = path.segments
    transitions = path.transitions
    if not segments:
        return ()
    if len(transitions) != len(segments) - 1:
        raise ValueError(
            "An open fermion path must have one fewer operator transition than Wick segments."
        )

    uses: list[_PropagationUse] = []
    chain_segments = [segments[0]]
    absorbed: list[FermionVertexTransition] = []

    for idx, transition in enumerate(transitions):
        if _is_point_transition(transition):
            absorbed.append(transition)
            chain_segments.append(segments[idx + 1])
        else:
            _require_explicit_laph_transition(transition)
            uses.append(_infer_chain_recipe(chain_segments, absorbed, kernel_map))
            chain_segments = [segments[idx + 1]]
            absorbed = []

    uses.append(_infer_chain_recipe(chain_segments, absorbed, kernel_map))
    return tuple(uses)


def _partition_closed_cycle_with_point_insertions(
    cycle: FermionClosedCycle,
    kernel_map: Mapping[str, KernelRecipe],
) -> tuple[_PropagationUse, ...]:
    """Cut a closed cycle at LapH transitions and absorb connected POINT currents."""

    segments = cycle.segments
    transitions = cycle.transitions
    n = len(segments)
    explicit_indices = [
        idx for idx, transition in enumerate(transitions)
        if not _is_point_transition(transition)
    ]
    if not explicit_indices:
        raise NotImplementedError(
            "A closed fermion cycle made entirely from POINT transitions cannot be "
            "reconstructed from ordinary distillation perambulators. POINT "
            "disconnected loops require additional all-to-all/stochastic/precomputed "
            "loop information."
        )
    for idx in explicit_indices:
        _require_explicit_laph_transition(transitions[idx])

    uses: list[_PropagationUse] = []
    for cut_idx in explicit_indices:
        current = (cut_idx + 1) % n
        chain_segments = [segments[current]]
        absorbed: list[FermionVertexTransition] = []
        while True:
            transition = transitions[current]
            if not _is_point_transition(transition):
                break
            absorbed.append(transition)
            current = (current + 1) % n
            chain_segments.append(segments[current])
        uses.append(_infer_chain_recipe(chain_segments, absorbed, kernel_map))
    return tuple(uses)


def _lower_topology_to_propagation_uses(
    topology: FermionTopology,
    kernel_map: Mapping[str, KernelRecipe],
) -> tuple[_PropagationUse, ...]:
    uses: list[_PropagationUse] = []

    for path in topology.open_paths:
        uses.extend(_partition_open_path(path, kernel_map))

    for cycle in topology.closed_cycles:
        if any(_is_point_transition(t) for t in cycle.transitions):
            uses.extend(_partition_closed_cycle_with_point_insertions(cycle, kernel_map))
            continue

        # Frozen exact-distillation closed-cycle lowering is authoritative for
        # cycles containing only explicit LapH transitions.
        lowered = lower_closed_cycle(cycle, kernel_map)
        uses.extend(
            _PropagationUse(
                recipe=recipe,
                start_endpoint=segment.source_endpoint,
                end_endpoint=segment.sink_endpoint,
            )
            for recipe, segment in zip(lowered.propagations, cycle.segments)
        )

    return tuple(uses)


def _lower_correlator_components(
    correlator: LogicalCorrelator,
) -> tuple[tuple[KernelRecipe, ...], FermionTopology, tuple[_PropagationUse, ...]]:
    kernels: list[KernelRecipe] = []
    kernel_map: dict[str, KernelRecipe] = {}
    for vertex in identify_operator_vertices(correlator):
        recipe = construct_kernel_recipe(realize_projected_fields(vertex))
        kernels.append(recipe)
        kernel_map[vertex.vertex_id] = recipe

    topology = reconstruct_fermion_topology(correlator)
    propagation_uses = _lower_topology_to_propagation_uses(topology, kernel_map)
    return tuple(kernels), topology, propagation_uses


def _validate_absorbed_point_kernel(recipe: PropagationRecipe) -> None:
    for spec in recipe.insertion_specs:
        if set(spec.kernel.open_ports) != {spec.incoming_port, spec.outgoing_port}:
            raise NotImplementedError(
                "A generalized POINT insertion currently requires its Stage-3 spatial "
                "component to contain exactly the two current legs. A larger shared "
                "spatial component cannot be removed as one absorbed tensor without "
                "also removing unrelated operator legs."
            )


def _mode_index(node_id: str, port: str) -> str:
    return f"mode:{node_id}:{port}"


def _spin_index(label: str) -> str:
    return f"spin:{label}"


def lower_logical_correlator_to_contraction_graph(
    correlator: LogicalCorrelator,
) -> ContractionGraphLoweringResult:
    """Construct one primitive tensor graph for one complete Wick term."""

    scalar = disconnected_color_scalar(correlator)
    try:
        coefficient = correlator.coefficient * scalar
    except OverflowError as error:
        raise ValueError("Folded color coefficient must be finite and representable.") from error
    _validate_coefficient(coefficient, "Folded color Wick term")
    operator_ids = {
        record.operator_id
        for line in correlator.quark_lines
        for record in (line.quark, line.antiquark)
        if record.operator_id is not None
    }
    operator_ids.update(
        tensor.operator_id
        for tensors in (correlator.spin_tensors, correlator.color_tensors, correlator.gauge_tensors)
        for tensor in tensors if tensor.operator_id is not None
    )

    kernels, topology, propagation_uses = _lower_correlator_components(correlator)

    absorbed_kernel_ids: set[int] = set()
    absorbed_spin_tensors: set[SpinTensorRecord] = set()
    for use in propagation_uses:
        _validate_absorbed_point_kernel(use.recipe)
        for spec in use.recipe.insertion_specs:
            absorbed_kernel_ids.add(id(spec.kernel))
            absorbed_spin_tensors.update(spec.spin_tensors)

    nodes: list[PrimitiveTensorNode] = []
    kernel_node_ids: dict[int, str] = {}
    mode_indices: dict[tuple[int, str], str] = {}

    # Kernel occurrences stay distinct even when their reusable resource id is equal.
    for kernel in kernels:
        if id(kernel) in absorbed_kernel_ids:
            continue
        if any(
            field.contribution == ProjectionContribution.POINT
            for field in kernel.projected_fields
        ):
            raise NotImplementedError(
                "A standalone POINT spatial kernel is not numerically available in the "
                "current exact-distillation graph. Connected POINT bilinears must be "
                "absorbed into generalized propagation resources."
            )

        node_id = f"kernel_{len(kernel_node_ids)}"
        kernel_node_ids[id(kernel)] = node_id
        canonical_kernel, canonical_to_local = _canonicalize_kernel_axes(kernel)
        bindings_list: list[TensorLegBinding] = []
        for canonical_port, local_axis in zip(
            canonical_kernel.open_ports,
            canonical_to_local,
            strict=True,
        ):
            local_port = kernel.open_ports[local_axis]
            index = _mode_index(node_id, local_port)
            bindings_list.append(TensorLegBinding(canonical_port, index))
            mode_indices[(id(kernel), local_port)] = index
        bindings = tuple(bindings_list)
        nodes.append(PrimitiveTensorNode(
            node_id=node_id,
            kind="kernel",
            resource_id=identify_kernel_resource(canonical_kernel),
            bindings=bindings,
            payload=canonical_kernel,
        ))

    # Propagation tensor layout is sink mode/spin followed by source mode/spin.
    for use_idx, use in enumerate(propagation_uses):
        recipe = use.recipe
        try:
            sink_mode = mode_indices[(id(recipe.end_kernel), recipe.end_port)]
            source_mode = mode_indices[(id(recipe.start_kernel), recipe.start_port)]
        except KeyError as exc:
            raise ValueError(
                "Propagation boundary refers to a kernel port that is not an explicit "
                "tensor node in this contraction graph."
            ) from exc

        nodes.append(PrimitiveTensorNode(
            node_id=f"propagation_{use_idx}",
            kind="propagation",
            resource_id=recipe.resource_id,
            bindings=(
                TensorLegBinding("sink_mode", sink_mode),
                TensorLegBinding(
                    "sink_spin", _spin_index(use.end_endpoint.record.spin_label)
                ),
                TensorLegBinding("source_mode", source_mode),
                TensorLegBinding(
                    "source_spin", _spin_index(use.start_endpoint.record.spin_label)
                ),
            ),
            payload=recipe,
        ))

    # Spin tensors inside generalized POINT resources are already absorbed.
    spin_idx = 0
    for tensor in correlator.spin_tensors:
        if tensor in absorbed_spin_tensors:
            continue
        nodes.append(PrimitiveTensorNode(
            node_id=f"spin_{spin_idx}",
            kind="spin",
            resource_id=_spin_resource_id(tensor),
            bindings=(
                TensorLegBinding("left", _spin_index(tensor.left_spin)),
                TensorLegBinding("right", _spin_index(tensor.right_spin)),
            ),
            payload=tensor,
        ))
        spin_idx += 1

    graph = WickContractionGraph(
        coefficient=coefficient,
        nodes=tuple(nodes),
        operator_arity=max(operator_ids) + 1 if operator_ids else 0,
    )
    graph.validate()
    return ContractionGraphLoweringResult(
        kernels=kernels,
        topology=topology,
        graph=graph,
    )


def lower_correlator_to_kernel_recipes_and_propagations(
    correlator: LogicalCorrelator,
) -> tuple[tuple[KernelRecipe, ...], tuple[PropagationRecipe, ...]]:
    """Return kernel and propagation recipes using canonical fermion topology."""
    kernels, _, propagation_uses = _lower_correlator_components(correlator)
    return kernels, tuple(use.recipe for use in propagation_uses)


def lower_physics_ir_to_contraction_graphs(
    physics_ir: PhysicsIR,
) -> tuple[ContractionGraphLoweringResult, ...]:
    """Construct one primitive contraction graph for every PhysicsIR Wick term."""
    return tuple(
        lower_logical_correlator_to_contraction_graph(correlator)
        for correlator in physics_ir.correlators
    )
