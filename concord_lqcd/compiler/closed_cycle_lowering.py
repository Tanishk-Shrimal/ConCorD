"""Lower closed fermion cycles to exact-distillation contraction resources.

This layer consumes ``FermionClosedCycle`` topology and describes the cyclic
contraction needed for exact LapH/distillation evaluation.  It deliberately
introduces no special "loop tensor" resource.  Every Wick propagator segment
requests an ordinary precomputed perambulator, while operator-local transitions
supply the spatial kernels and spin tensors that close the trace.

Current scope:
  * exact LapH/distillation closed cycles;
  * ordinary all-time perambulators on disk;
  * no stochastic estimator or stochastic-distillation realization;
  * genuinely POINT disconnected loops are rejected because ordinary
    perambulators do not contain the full point-space all-to-all propagator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from concord_lqcd.compiler.physics_ir import LogicalCorrelator, Representation, SpinTensorRecord
from concord_lqcd.compiler.distillation_lowering import KernelRecipe
from concord_lqcd.compiler.fermion_topology import (
    FermionClosedCycle,
    FermionTopology,
    FermionVertexTransition,
    reconstruct_fermion_topology,
)
from concord_lqcd.compiler.propagation_lowering import (
    PropagationRecipe,
    _kernel_and_port_for_endpoint,
    _infer_propagation_recipe,
    _unique_kernels,
)
from concord_lqcd.compiler.distillation_lowering import (
    construct_kernel_recipe,
    identify_operator_vertices,
    realize_projected_fields,
)


@dataclass(frozen=True)
class ClosedCycleTransitionSpec:
    """One operator-local tensor transition inside a closed fermion trace.

    The spatial/color information stays in the Stage-3 kernel recipe(s); spin
    tensors remain independent.  Local bilinears normally have the same kernel
    on both sides, but the representation allows distinct spatial components so
    topology lowering does not assume a hadron/operator class.
    """

    incoming_kernel: KernelRecipe
    outgoing_kernel: KernelRecipe
    incoming_port: str
    outgoing_port: str
    incoming_flavor: str
    outgoing_flavor: str
    spin_tensors: tuple[SpinTensorRecord, ...] = ()

    @property
    def is_single_kernel_transition(self) -> bool:
        return self.incoming_kernel is self.outgoing_kernel


@dataclass(frozen=True)
class ClosedCycleRecipe:
    """Cyclic contraction recipe for one exact-distillation fermion loop.

    ``propagations[i]`` is the ordinary perambulator associated with Wick
    segment ``i``.  ``transitions[i]`` follows that propagation and joins its
    sink endpoint to the source endpoint of propagation ``i+1`` (cyclically).
    No separately stored disconnected-loop tensor is required.
    """

    propagations: tuple[PropagationRecipe, ...]
    transitions: tuple[ClosedCycleTransitionSpec, ...]
    operator_ids: frozenset[int]

    def __post_init__(self) -> None:
        if not self.propagations:
            raise ValueError("A closed-cycle recipe requires at least one propagation segment.")
        if len(self.propagations) != len(self.transitions):
            raise ValueError(
                "A closed cycle must alternate one ordinary propagation with one "
                "operator-local transition."
            )
        if any(propagation.is_generalized for propagation in self.propagations):
            raise ValueError(
                "Closed-cycle lowering must use ordinary perambulators for every "
                "Wick segment; generalized propagation belongs to open-path lowering."
            )

    @property
    def propagation_resource_ids(self) -> tuple[str, ...]:
        return tuple(propagation.resource_id for propagation in self.propagations)

    @property
    def flavors(self) -> tuple[str, ...]:
        return tuple(propagation.flavor for propagation in self.propagations)

    @property
    def is_local(self) -> bool:
        return len(self.operator_ids) <= 1


def _same_endpoint(lhs, rhs) -> bool:
    return lhs.record == rhs.record and lhs.operations == rhs.operations


def _require_laph_cycle(cycle: FermionClosedCycle) -> None:
    """Reject point-space loops that cannot be reconstructed from tau alone."""
    for segment in cycle.segments:
        if (
            segment.source_endpoint.representation != Representation.LAPH
            or segment.sink_endpoint.representation != Representation.LAPH
        ):
            raise NotImplementedError(
                "Disconnected closed cycles are currently evaluated only in the "
                "LapH/distillation subspace using ordinary all-time perambulators. "
                "A genuinely POINT disconnected loop requires additional all-to-all, "
                "stochastic, or precomputed-loop data."
            )


def _ordinary_propagation_for_segment(
    segment,
    kernel_map: Mapping[str, KernelRecipe],
) -> PropagationRecipe:
    recipe = _infer_propagation_recipe(
        start_endpoint=segment.source_endpoint,
        end_endpoint=segment.sink_endpoint,
        segments=(segment,),
        insertions=(),
        kernel_map=kernel_map,
    )
    if recipe.is_generalized or recipe.resource_kind != "perambulator":
        raise ValueError(
            "A closed-cycle Wick segment must lower to an ordinary perambulator resource."
        )
    return recipe


def _transition_spec(
    transition: FermionVertexTransition,
    kernels: tuple[KernelRecipe, ...],
) -> ClosedCycleTransitionSpec:
    incoming_kernel, incoming_port = _kernel_and_port_for_endpoint(
        transition.quark_endpoint,
        kernels,
    )
    outgoing_kernel, outgoing_port = _kernel_and_port_for_endpoint(
        transition.antiquark_endpoint,
        kernels,
    )

    return ClosedCycleTransitionSpec(
        incoming_kernel=incoming_kernel,
        outgoing_kernel=outgoing_kernel,
        incoming_port=incoming_port,
        outgoing_port=outgoing_port,
        incoming_flavor=transition.incoming_flavor,
        outgoing_flavor=transition.outgoing_flavor,
        spin_tensors=transition.spin_tensors,
    )


def lower_closed_cycle(
    cycle: FermionClosedCycle,
    kernel_map: Mapping[str, KernelRecipe],
) -> ClosedCycleRecipe:
    """Lower one closed fermion cycle to ordinary perambulators + kernel transitions."""
    _require_laph_cycle(cycle)
    kernels = _unique_kernels(kernel_map)

    propagations = tuple(
        _ordinary_propagation_for_segment(segment, kernel_map)
        for segment in cycle.segments
    )
    transitions = tuple(
        _transition_spec(transition, kernels)
        for transition in cycle.transitions
    )

    n = len(cycle.segments)
    for idx, transition in enumerate(cycle.transitions):
        current_segment = cycle.segments[idx]
        next_segment = cycle.segments[(idx + 1) % n]

        if not _same_endpoint(transition.quark_endpoint, current_segment.sink_endpoint):
            raise ValueError(
                "Closed-cycle transition does not originate from the preceding "
                "Wick segment sink endpoint."
            )
        if not _same_endpoint(transition.antiquark_endpoint, next_segment.source_endpoint):
            raise ValueError(
                "Closed-cycle transition does not terminate on the following "
                "Wick segment source endpoint."
            )

        if propagations[idx].outgoing_flavor != transition.incoming_flavor:
            raise ValueError(
                "Closed-cycle transition incoming flavor does not match the "
                "preceding perambulator segment."
            )
        if propagations[(idx + 1) % n].incoming_flavor != transition.outgoing_flavor:
            raise ValueError(
                "Closed-cycle transition outgoing flavor does not match the "
                "following perambulator segment."
            )

    return ClosedCycleRecipe(
        propagations=propagations,
        transitions=transitions,
        operator_ids=cycle.operator_ids,
    )


@dataclass(frozen=True)
class ClosedCycleLoweringResult:
    """Stage-3 kernels, full fermion topology, and all closed-cycle recipes."""

    kernels: tuple[KernelRecipe, ...]
    topology: FermionTopology
    closed_cycles: tuple[ClosedCycleRecipe, ...]


def lower_correlator_closed_cycles(correlator: LogicalCorrelator) -> ClosedCycleLoweringResult:
    """Lower every closed fermion component of one LogicalCorrelator.

    Open components remain available in ``topology.open_paths`` and continue to
    use the Stage-5 open-path propagation lowering.  This function only lowers
    closed components and never invents a dedicated disconnected-loop resource.
    """
    vertices = identify_operator_vertices(correlator)
    kernels: list[KernelRecipe] = []
    kernel_map: dict[str, KernelRecipe] = {}

    for vertex in vertices:
        recipe = construct_kernel_recipe(realize_projected_fields(vertex))
        kernels.append(recipe)
        kernel_map[vertex.vertex_id] = recipe

    topology = reconstruct_fermion_topology(correlator)
    closed_cycles = tuple(
        lower_closed_cycle(cycle, kernel_map)
        for cycle in topology.closed_cycles
    )

    return ClosedCycleLoweringResult(
        kernels=tuple(kernels),
        topology=topology,
        closed_cycles=closed_cycles,
    )
