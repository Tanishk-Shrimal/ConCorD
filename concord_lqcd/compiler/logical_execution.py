"""Backend-independent logical execution DAG construction.

This layer consumes the algebraically optimized ``GlobalIntermediateDAG`` and
makes every remaining operation explicit:

* evaluate each unique primitive tensor value;
* compute each unique contraction intermediate;
* assemble each Wick term as a coefficient times the product of its connected
  component results;
* sum Wick terms into each requested correlator output.

The DAG answers only *what depends on what*.  It deliberately contains no
execution schedule, device placement, buffer allocation, residency policy,
prefetching, communication, or runtime tensor loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from types import MappingProxyType
from typing import Mapping, TypeAlias

from concord_lqcd.compiler._validation import _has_static_validation
from concord_lqcd.compiler.contraction_graph import PrimitiveTensorKind, TensorLegBinding
from concord_lqcd.compiler.partial_cse import GlobalIntermediateDAG


@dataclass(frozen=True)
class LogicalValueUse:
    """One occurrence of an existing value in a logical operation.

    ``bindings`` map the value's public interface ports to indices local to the
    consuming operation.  Repeated uses of the same ``value_id`` remain distinct
    occurrences; this is required, for example, for a Wick term containing the
    same disconnected scalar component twice.
    """

    value_id: str
    bindings: tuple[TensorLegBinding, ...]


@dataclass(frozen=True)
class LogicalPrimitiveEvaluation:
    """Evaluate one unique primitive numerical tensor value."""

    node_id: str
    value_id: str
    resource_id: str
    kind: PrimitiveTensorKind
    placement: tuple[object, ...]
    output_ports: tuple[str, ...]

    @property
    def dependencies(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class LogicalContraction:
    """Compute one globally unique contraction intermediate."""

    node_id: str
    inputs: tuple[LogicalValueUse, ...]
    contracted_indices: tuple[str, ...]
    output_indices: tuple[str, ...]
    estimated_flops: int
    output_elements: int

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(use.value_id for use in self.inputs)


@dataclass(frozen=True)
class LogicalWickTermProduct:
    """Assemble one Wick term from its disconnected component results.

    Algebraically this node represents

        coefficient * product(component_values)

    where a zero-input product is the scalar one.  Input occurrences are not
    deduplicated because multiplicity is part of the Wick-term algebra.
    """

    node_id: str
    correlator_id: str
    term_index: int
    inputs: tuple[LogicalValueUse, ...]
    coefficient: complex
    output_indices: tuple[str, ...]

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(use.value_id for use in self.inputs)


@dataclass(frozen=True)
class LogicalCorrelatorSum:
    """Sum all Wick terms contributing to one requested correlator."""

    node_id: str
    correlator_id: str
    term_ids: tuple[str, ...]
    output_indices: tuple[str, ...]

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.term_ids


LogicalExecutionNode: TypeAlias = (
    LogicalPrimitiveEvaluation
    | LogicalContraction
    | LogicalWickTermProduct
    | LogicalCorrelatorSum
)


@dataclass(frozen=True)
class LogicalExecutionStatistics:
    primitive_evaluations: int
    contractions: int
    wick_term_products: int
    correlator_sums: int
    dependency_occurrences: int


@dataclass(frozen=True)
class LogicalExecutionDAG:
    """Complete hardware-independent execution dependency DAG."""

    source_dag: GlobalIntermediateDAG
    primitives: Mapping[str, LogicalPrimitiveEvaluation]
    contractions: Mapping[str, LogicalContraction]
    terms: tuple[LogicalWickTermProduct, ...]
    correlators: tuple[LogicalCorrelatorSum, ...]
    outputs: Mapping[str, str]
    statistics: LogicalExecutionStatistics

    @cached_property
    def _assembly_nodes(self) -> Mapping[str, LogicalWickTermProduct | LogicalCorrelatorSum]:
        """Index immutable assembly records without caching mutable node maps."""

        nodes: dict[str, LogicalWickTermProduct | LogicalCorrelatorSum] = {}
        for node in (*self.terms, *self.correlators):
            nodes.setdefault(node.node_id, node)
        return MappingProxyType(nodes)

    def node(self, node_id: str) -> LogicalExecutionNode:
        if node_id in self.primitives:
            return self.primitives[node_id]
        if node_id in self.contractions:
            return self.contractions[node_id]
        if isinstance(self.terms, tuple) and isinstance(self.correlators, tuple):
            return self._assembly_nodes[node_id]
        for node in (*self.terms, *self.correlators):
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def dependencies(self, node_id: str) -> tuple[str, ...]:
        return self.node(node_id).dependencies

    @property
    def node_ids(self) -> tuple[str, ...]:
        return (
            tuple(self.primitives)
            + tuple(self.contractions)
            + tuple(term.node_id for term in self.terms)
            + tuple(correlator.node_id for correlator in self.correlators)
        )

    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic dependency-first ordering of all DAG nodes."""

        all_ids = set(self.node_ids)
        state: dict[str, int] = {}
        ordered: list[str] = []

        def visit(node_id: str) -> None:
            marker = state.get(node_id, 0)
            if marker == 2:
                return
            if marker == 1:
                raise ValueError("Logical execution graph contains a dependency cycle.")
            if node_id not in all_ids:
                raise ValueError(
                    f"Logical execution node references missing dependency {node_id!r}."
                )
            state[node_id] = 1
            for dependency in self.dependencies(node_id):
                visit(dependency)
            state[node_id] = 2
            ordered.append(node_id)

        for node_id in self.node_ids:
            visit(node_id)
        return tuple(ordered)

    def validate(self) -> None:
        """Validate structure and exact agreement with the attached compiler source."""

        if _has_static_validation(self):
            return
        ids = self.node_ids
        if len(ids) != len(set(ids)):
            raise ValueError("Logical execution DAG contains duplicate node ids.")

        global_resources = self.source_dag.source_plan.source_dag.resources
        for node_id, primitive in self.primitives.items():
            if node_id != primitive.node_id:
                raise ValueError("Primitive logical map key disagrees with its record identity.")
            if primitive.node_id != primitive.value_id:
                raise ValueError("Primitive logical node id must equal its value id.")
            if primitive.resource_id not in global_resources:
                raise ValueError(
                    f"Primitive value {primitive.value_id!r} references unknown resource "
                    f"{primitive.resource_id!r}."
                )
            if len(primitive.output_ports) != len(set(primitive.output_ports)):
                raise ValueError("Primitive value exposes duplicate output ports.")

        for node_id, contraction in self.contractions.items():
            if node_id != contraction.node_id:
                raise ValueError("Contraction logical map key disagrees with its record identity.")

        value_ports: dict[str, tuple[str, ...]] = {
            node_id: primitive.output_ports
            for node_id, primitive in self.primitives.items()
        }

        # Intermediates are already recursively unique; validate their explicit
        # input interfaces in dependency order before exposing their output ports.
        remaining = dict(self.contractions)
        while remaining:
            progressed = False
            for node_id, contraction in list(remaining.items()):
                if any(use.value_id not in value_ports for use in contraction.inputs):
                    continue
                self._validate_value_uses(contraction.inputs, value_ports)
                local_indices = [
                    binding.index
                    for use in contraction.inputs
                    for binding in use.bindings
                ]
                present = set(local_indices)
                contracted = set(contraction.contracted_indices)
                outputs = set(contraction.output_indices)
                if contracted & outputs:
                    raise ValueError(
                        f"Contraction {node_id!r} marks an index as both contracted and output."
                    )
                if contracted | outputs != present:
                    raise ValueError(
                        f"Contraction {node_id!r} input indices do not match its declared "
                        "contracted/output interface."
                    )
                if len(contraction.output_indices) != len(outputs):
                    raise ValueError(
                        f"Contraction {node_id!r} exposes duplicate output indices."
                    )
                value_ports[node_id] = contraction.output_indices
                del remaining[node_id]
                progressed = True
            if not progressed:
                missing = sorted(remaining)
                raise ValueError(
                    "Logical contraction dependencies are cyclic or reference missing values: "
                    f"{missing}"
                )

        term_by_id = {term.node_id: term for term in self.terms}
        term_keys: set[tuple[str, int]] = set()
        for term in self.terms:
            term_key = (term.correlator_id, term.term_index)
            if term_key in term_keys:
                raise ValueError("Logical DAG contains duplicate Wick term indices for a correlator.")
            term_keys.add(term_key)
            try:
                finite_coefficient = isfinite(term.coefficient.real) and isfinite(term.coefficient.imag)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Wick term coefficient must be a finite numeric scalar.") from exc
            if not finite_coefficient:
                raise ValueError("Wick term coefficient must be a finite numeric scalar.")
            self._validate_value_uses(term.inputs, value_ports)
            exposed = [
                binding.index
                for use in term.inputs
                for binding in use.bindings
            ]
            if len(exposed) != len(set(exposed)):
                raise ValueError(
                    f"Wick term {term.node_id!r} has the same external index exposed by "
                    "more than one disconnected component."
                )
            if tuple(sorted(exposed)) != term.output_indices:
                raise ValueError(
                    f"Wick term {term.node_id!r} output indices do not match its inputs."
                )

        correlator_ids: set[str] = set()
        consumed_terms: set[str] = set()
        for correlator in self.correlators:
            if correlator.correlator_id in correlator_ids:
                raise ValueError(
                    f"Duplicate correlator id {correlator.correlator_id!r} in logical DAG."
                )
            correlator_ids.add(correlator.correlator_id)
            if len(correlator.term_ids) != len(set(correlator.term_ids)):
                raise ValueError("Correlator sum references a Wick term more than once.")
            for term_id in correlator.term_ids:
                try:
                    term = term_by_id[term_id]
                except KeyError as exc:
                    raise ValueError(
                        f"Correlator {correlator.correlator_id!r} references unknown Wick "
                        f"term {term_id!r}."
                    ) from exc
                if term.correlator_id != correlator.correlator_id:
                    raise ValueError(
                        "Correlator sum references a Wick term owned by another correlator."
                    )
                if term.output_indices != correlator.output_indices:
                    raise ValueError(
                        f"Wick terms for correlator {correlator.correlator_id!r} do not "
                        "share one output interface."
                    )
                consumed_terms.add(term_id)
            if self.outputs.get(correlator.correlator_id) != correlator.node_id:
                raise ValueError(
                    f"Logical output mapping for {correlator.correlator_id!r} is inconsistent."
                )

        if set(self.outputs) != correlator_ids:
            raise ValueError("Logical output map does not exactly match correlator ids.")
        if consumed_terms != set(term_by_id):
            raise ValueError("Every logical Wick term must occur in its owning correlator sum.")

        self._validate_source_agreement()

        # This also verifies all assembly dependencies and catches any cycle.
        self.topological_order()

    def _validate_source_agreement(self) -> None:
        """Compare frozen records without rerunning Wick contraction or CSE."""

        partial = self.source_dag
        global_dag = partial.source_plan.source_dag
        if set(self.primitives) != set(partial.primitive_values):
            raise ValueError("Logical primitives do not match the compiler source values.")
        for value_id, value in partial.primitive_values.items():
            if value.value_id != value_id:
                raise ValueError("Compiler source primitive key disagrees with its record identity.")
            node = self.primitives[value_id]
            if (node.resource_id, node.kind, node.placement, node.output_ports) != (
                value.resource_id, value.kind, value.placement, value.ports,
            ):
                raise ValueError("Logical primitive metadata disagrees with the compiler source.")

        if set(self.contractions) != set(partial.intermediates):
            raise ValueError("Logical contractions do not match the compiler source intermediates.")
        for intermediate_id, intermediate in partial.intermediates.items():
            if intermediate.intermediate_id != intermediate_id:
                raise ValueError("Compiler source intermediate key disagrees with its record identity.")
            node = self.contractions[intermediate_id]
            if (
                tuple((use.value_id, use.bindings) for use in node.inputs),
                node.contracted_indices, node.output_indices,
                node.estimated_flops, node.output_elements,
            ) != (
                tuple((use.value_id, use.bindings) for use in intermediate.inputs),
                intermediate.contracted_indices, intermediate.output_indices,
                intermediate.estimated_flops, intermediate.output_elements,
            ):
                raise ValueError("Logical contraction disagrees with the compiler source.")

        if set(partial.component_plans) != set(global_dag.components):
            raise ValueError("Compiler source components do not exactly match their partial-CSE plans.")
        for component_id, component in global_dag.components.items():
            if (
                component.computation_id != component_id
                or partial.component_plans[component_id].computation_id != component_id
            ):
                raise ValueError("Compiler source component key disagrees with its record identity.")

        source_correlators = {}
        source_terms = {}
        for correlator in global_dag.correlators:
            correlator_id = str(correlator.correlator_id)
            if correlator_id in source_correlators:
                raise ValueError("Compiler source contains duplicate correlator identities.")
            source_correlators[correlator_id] = correlator
            for term in correlator.terms:
                term_key = (correlator_id, term.term_index)
                if term_key in source_terms:
                    raise ValueError("Compiler source contains duplicate Wick term indices.")
                source_terms[term_key] = term

        logical_terms = {(term.correlator_id, term.term_index): term for term in self.terms}
        if set(logical_terms) != set(source_terms):
            raise ValueError("Logical Wick terms do not exactly match the compiler source.")
        for term_key, source_term in source_terms.items():
            term = logical_terms[term_key]
            expected_inputs = tuple(
                _compose_component_result_use(partial, component_use)
                for component_use in source_term.components
            )
            if term.coefficient != source_term.coefficient or term.inputs != expected_inputs:
                raise ValueError("Logical Wick term algebra disagrees with the compiler source.")

        if {correlator.correlator_id for correlator in self.correlators} != set(source_correlators):
            raise ValueError("Logical correlators do not exactly match the compiler source.")
        for correlator in self.correlators:
            source_correlator = source_correlators[correlator.correlator_id]
            expected_terms = tuple(
                logical_terms[(correlator.correlator_id, term.term_index)].node_id
                for term in source_correlator.terms
            )
            if correlator.term_ids != expected_terms:
                raise ValueError("Logical correlator sum order disagrees with the compiler source.")
            if not expected_terms and correlator.output_indices:
                raise ValueError("Empty compiler source correlator must have a scalar output interface.")

    @staticmethod
    def _validate_value_uses(
        uses: tuple[LogicalValueUse, ...],
        value_ports: Mapping[str, tuple[str, ...]],
    ) -> None:
        for use in uses:
            try:
                expected_ports = value_ports[use.value_id]
            except KeyError as exc:
                raise ValueError(
                    f"Logical operation references unavailable value {use.value_id!r}."
                ) from exc
            actual_ports = tuple(binding.port for binding in use.bindings)
            if actual_ports != expected_ports:
                raise ValueError(
                    f"Logical use of value {use.value_id!r} has interface ports "
                    f"{actual_ports!r}, expected {expected_ports!r}."
                )


def _compose_component_result_use(
    intermediate_dag: GlobalIntermediateDAG,
    component_use,
) -> LogicalValueUse:
    """Map a reusable component result back to one Wick-term output interface."""

    computation_id = component_use.computation_id
    try:
        component = intermediate_dag.source_plan.source_dag.components[computation_id]
        component_plan = intermediate_dag.component_plans[computation_id]
    except KeyError as exc:
        raise ValueError(
            f"Wick term references component {computation_id!r} without a partial-CSE plan."
        ) from exc

    component_to_term = {
        binding.port: binding.index
        for binding in component_use.output_bindings
    }
    if len(component_to_term) != len(component_use.output_bindings):
        raise ValueError("Component use contains duplicate output binding ports.")
    if set(component_to_term) != set(component.output_indices):
        raise ValueError(
            f"Component use {computation_id!r} does not bind exactly the component outputs."
        )

    result_bindings: list[TensorLegBinding] = []
    seen_result_ports: set[str] = set()
    for binding in component_plan.result_bindings:
        if binding.port in seen_result_ports:
            raise ValueError("Component result exposes a duplicate interface port.")
        seen_result_ports.add(binding.port)
        try:
            term_index = component_to_term[binding.index]
        except KeyError as exc:
            raise ValueError(
                f"Component result binding {binding.index!r} is not one of the component "
                "output indices."
            ) from exc
        result_bindings.append(TensorLegBinding(binding.port, term_index))

    if {
        binding.index for binding in component_plan.result_bindings
    } != set(component.output_indices):
        raise ValueError(
            f"Partial-CSE result for component {computation_id!r} does not expose exactly "
            "the component output interface."
        )

    return LogicalValueUse(
        value_id=component_plan.result_value_id,
        bindings=tuple(result_bindings),
    )


def build_logical_execution_dag(
    intermediate_dag: GlobalIntermediateDAG,
) -> LogicalExecutionDAG:
    """Build the complete hardware-independent execution dependency DAG."""

    global_dag = intermediate_dag.source_plan.source_dag

    primitives = {
        value_id: LogicalPrimitiveEvaluation(
            node_id=value_id,
            value_id=value_id,
            resource_id=value.resource_id,
            kind=value.kind,
            placement=value.placement,
            output_ports=value.ports,
        )
        for value_id, value in intermediate_dag.primitive_values.items()
    }

    contractions = {
        intermediate_id: LogicalContraction(
            node_id=intermediate_id,
            inputs=tuple(
                LogicalValueUse(
                    value_id=input_use.value_id,
                    bindings=input_use.bindings,
                )
                for input_use in intermediate.inputs
            ),
            contracted_indices=intermediate.contracted_indices,
            output_indices=intermediate.output_indices,
            estimated_flops=intermediate.estimated_flops,
            output_elements=intermediate.output_elements,
        )
        for intermediate_id, intermediate in intermediate_dag.intermediates.items()
    }

    terms: list[LogicalWickTermProduct] = []
    correlators: list[LogicalCorrelatorSum] = []
    outputs: dict[str, str] = {}

    seen_correlator_ids: set[str] = set()
    for correlator in global_dag.correlators:
        correlator_id = str(correlator.correlator_id)
        if correlator_id in seen_correlator_ids:
            raise ValueError(f"Duplicate correlator id {correlator_id!r} in source DAG.")
        seen_correlator_ids.add(correlator_id)

        term_ids: list[str] = []
        correlator_output_indices: tuple[str, ...] | None = None
        seen_term_indices: set[int] = set()

        for term in correlator.terms:
            if term.term_index in seen_term_indices:
                raise ValueError(
                    f"Correlator {correlator_id!r} contains duplicate Wick term index "
                    f"{term.term_index}."
                )
            seen_term_indices.add(term.term_index)

            input_uses = tuple(
                _compose_component_result_use(intermediate_dag, component_use)
                for component_use in term.components
            )
            exposed_indices = tuple(sorted(
                binding.index
                for use in input_uses
                for binding in use.bindings
            ))
            if len(exposed_indices) != len(set(exposed_indices)):
                raise ValueError(
                    f"Wick term {term.term_index} of correlator {correlator_id!r} exposes "
                    "one external index from multiple disconnected components."
                )

            if correlator_output_indices is None:
                correlator_output_indices = exposed_indices
            elif exposed_indices != correlator_output_indices:
                raise ValueError(
                    f"Wick terms of correlator {correlator_id!r} do not share one "
                    "external output interface."
                )

            node_id = f"wick_term:{correlator_id}:{term.term_index}"
            terms.append(LogicalWickTermProduct(
                node_id=node_id,
                correlator_id=correlator_id,
                term_index=term.term_index,
                inputs=input_uses,
                coefficient=term.coefficient,
                output_indices=exposed_indices,
            ))
            term_ids.append(node_id)

        if correlator_output_indices is None:
            # An empty Wick expansion represents a scalar zero at this layer.
            correlator_output_indices = ()

        correlator_node_id = f"correlator_sum:{correlator_id}"
        correlators.append(LogicalCorrelatorSum(
            node_id=correlator_node_id,
            correlator_id=correlator_id,
            term_ids=tuple(term_ids),
            output_indices=correlator_output_indices,
        ))
        outputs[correlator_id] = correlator_node_id

    dependency_occurrences = (
        sum(len(node.inputs) for node in contractions.values())
        + sum(len(term.inputs) for term in terms)
        + sum(len(correlator.term_ids) for correlator in correlators)
    )

    dag = LogicalExecutionDAG(
        source_dag=intermediate_dag,
        primitives=dict(primitives),
        contractions=dict(contractions),
        terms=tuple(terms),
        correlators=tuple(correlators),
        outputs=dict(outputs),
        statistics=LogicalExecutionStatistics(
            primitive_evaluations=len(primitives),
            contractions=len(contractions),
            wick_term_products=len(terms),
            correlator_sums=len(correlators),
            dependency_occurrences=dependency_occurrences,
        ),
    )
    dag.validate()
    return dag
