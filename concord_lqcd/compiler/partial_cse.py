"""Global CSE for partial intermediates in optimized contraction trees.

This layer consumes the frozen ``GlobalContractionOrderPlan`` and identifies
identical contraction subtrees across all unique component plans.  It does not
change the selected contraction order, schedule execution, allocate memory, or
choose devices.

Two intermediate occurrences are reusable only when their recursively-defined
input values, input-interface connectivity, contracted-index structure, and
output-interface structure are identical after alpha-renaming local indices.
Temporary component step ids and canonical component index names therefore do
not participate in intermediate identity.

Input permutations are searched only for steps with at most six operands.
Larger steps retain their operand order: alpha-renaming still permits reuse,
but reordered inputs may conservatively remain separate computations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Mapping, Sequence

from concord_lqcd.compiler.contraction_graph import PrimitiveTensorKind, TensorLegBinding
from concord_lqcd.compiler.contraction_order import (
    ComponentContractionPlan,
    ContractionStep,
    GlobalContractionOrderPlan,
)
from concord_lqcd.compiler.global_cse import ComponentComputation, ComponentTensorUse


@dataclass(frozen=True)
class PrimitiveTensorValue:
    """One evaluated primitive tensor value at an exact logical placement."""

    value_id: str
    kind: PrimitiveTensorKind
    resource_id: str
    placement: tuple[object, ...]
    ports: tuple[str, ...]


@dataclass(frozen=True)
class IntermediateInputUse:
    """Use of one primitive/intermediate value by a canonical intermediate."""

    value_id: str
    bindings: tuple[TensorLegBinding, ...]


@dataclass(frozen=True)
class IntermediateComputation:
    """One globally reusable contraction-tree intermediate."""

    intermediate_id: str
    signature: str
    inputs: tuple[IntermediateInputUse, ...]
    contracted_indices: tuple[str, ...]
    output_indices: tuple[str, ...]
    estimated_flops: int
    output_elements: int


@dataclass(frozen=True)
class IntermediateUse:
    """Occurrence of a reusable intermediate in one component plan."""

    source_step_id: str
    intermediate_id: str
    output_bindings: tuple[TensorLegBinding, ...]


@dataclass(frozen=True)
class ComponentIntermediatePlan:
    """Map one optimized component tree onto globally reusable values."""

    computation_id: str
    step_uses: tuple[IntermediateUse, ...]
    result_value_id: str
    result_bindings: tuple[TensorLegBinding, ...]


@dataclass(frozen=True)
class PartialCSEStatistics:
    primitive_value_occurrences: int
    unique_primitive_values: int
    intermediate_occurrences: int
    unique_intermediates: int
    baseline_step_flops: int
    unique_step_flops: int

    @property
    def primitive_value_reuses(self) -> int:
        return self.primitive_value_occurrences - self.unique_primitive_values

    @property
    def intermediate_reuses(self) -> int:
        return self.intermediate_occurrences - self.unique_intermediates

    @property
    def estimated_flops_saved(self) -> int:
        return self.baseline_step_flops - self.unique_step_flops


@dataclass(frozen=True)
class GlobalIntermediateDAG:
    """Global DAG after exact subtree CSE of selected contraction trees."""

    source_plan: GlobalContractionOrderPlan
    primitive_values: Mapping[str, PrimitiveTensorValue]
    intermediates: Mapping[str, IntermediateComputation]
    component_plans: Mapping[str, ComponentIntermediatePlan]
    statistics: PartialCSEStatistics

    def value_is_primitive(self, value_id: str) -> bool:
        return value_id in self.primitive_values

    def value_is_intermediate(self, value_id: str) -> bool:
        return value_id in self.intermediates


@dataclass(frozen=True)
class _ActiveValue:
    value_id: str
    interface_ports: tuple[str, ...]
    bindings: tuple[TensorLegBinding, ...]
    plan_indices: tuple[str, ...]

    def index_for_port(self, port: str) -> str:
        matches = [binding.index for binding in self.bindings if binding.port == port]
        if len(matches) != 1:
            raise ValueError(
                f"Active value {self.value_id!r} does not expose exactly one port {port!r}."
            )
        return matches[0]


def _hash_payload(prefix: str, payload: object, *, length: int = 20) -> tuple[str, str]:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}", encoded


def _component_operator_arity(component: ComponentComputation) -> int:
    """Recover the operator-slot convention carried by a canonical component.

    Production ``ComponentComputation`` signatures are emitted by ``global_cse``
    and contain the graph operator arity explicitly.  Partial-value identity must
    preserve this scope: operator slot 1 in a two-point graph is not the same
    runtime slot convention as operator slot 1 in a three-point graph.

    A placement-based fallback is retained for hand-constructed component
    objects used by focused tests and external callers that do not carry a
    canonical Global-CSE signature.
    """

    operator_ids: set[int] = set()
    for tensor in component.tensors:
        stack: list[object] = [tensor.placement]
        while stack:
            value = stack.pop()
            if isinstance(value, tuple):
                stack.extend(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                operator_ids.add(value)
    minimum_arity = max(operator_ids) + 1 if operator_ids else 0

    try:
        payload = json.loads(component.signature)
    except (TypeError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, dict) and "arity" in payload:
        arity = payload["arity"]
        if (
            not isinstance(arity, int)
            or isinstance(arity, bool)
            or arity < 0
            or arity < minimum_arity
        ):
            raise ValueError(
                f"Component {component.computation_id!r} operator arity must be a "
                f"non-negative integer covering its placements (at least {minimum_arity})."
            )
        return arity

    return minimum_arity


def _primitive_value_from_tensor(
    tensor: ComponentTensorUse,
    *,
    operator_arity: int,
) -> PrimitiveTensorValue:
    ports = tuple(binding.port for binding in tensor.bindings)
    if len(ports) != len(set(ports)):
        raise ValueError("Primitive tensor occurrence contains duplicate interface ports.")

    scoped_placement: tuple[object, ...] = (
        "operator_arity",
        operator_arity,
        tensor.placement,
    )
    payload = {
        "kind": tensor.kind,
        "resource": tensor.resource_id,
        "placement": scoped_placement,
        "ports": ports,
    }
    value_id, _ = _hash_payload("primitive_value", payload)
    return PrimitiveTensorValue(
        value_id=value_id,
        kind=tensor.kind,
        resource_id=tensor.resource_id,
        placement=scoped_placement,
        ports=ports,
    )


def _initial_active_value(tensor: ComponentTensorUse, value_id: str) -> _ActiveValue:
    return _ActiveValue(
        value_id=value_id,
        interface_ports=tuple(binding.port for binding in tensor.bindings),
        bindings=tensor.bindings,
        plan_indices=tuple(binding.index for binding in tensor.bindings),
    )


def _validate_step_inputs(
    step: ContractionStep,
    selected: Sequence[_ActiveValue],
) -> None:
    if len(step.input_refs) != len(selected):
        raise ValueError("Contraction step input reference count is inconsistent.")
    if len(step.input_indices) != len(selected):
        raise ValueError("Contraction step input index metadata is inconsistent.")
    for occurrence, expected_indices in zip(selected, step.input_indices):
        if occurrence.plan_indices != expected_indices:
            raise ValueError(
                f"Contraction step {step.step_id!r} has stale active-value index metadata."
            )


def _candidate_for_permutation(
    step: ContractionStep,
    selected: Sequence[_ActiveValue],
    permutation: Sequence[int],
) -> tuple[
    str,
    tuple[IntermediateInputUse, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[TensorLegBinding, ...],
]:
    ordered = [selected[idx] for idx in permutation]

    endpoints: dict[str, list[tuple[int, str]]] = {}
    for child_pos, child in enumerate(ordered):
        seen_ports: set[str] = set()
        for binding in child.bindings:
            if binding.port in seen_ports:
                raise ValueError("Child value contains duplicate interface ports.")
            seen_ports.add(binding.port)
            endpoints.setdefault(binding.index, []).append((child_pos, binding.port))

    output_set = set(step.output_indices)
    missing_outputs = output_set - set(endpoints)
    if missing_outputs:
        raise ValueError(
            f"Contraction step {step.step_id!r} outputs absent indices {sorted(missing_outputs)}."
        )

    contracted_actual = set(endpoints) - output_set
    if contracted_actual != set(step.contracted_indices):
        raise ValueError(
            f"Contraction step {step.step_id!r} contracted-index metadata disagrees "
            "with its input/output interface."
        )

    internal_specs = sorted(
        (tuple(sorted(refs)), actual_index)
        for actual_index, refs in endpoints.items()
        if actual_index not in output_set
    )
    output_specs = sorted(
        (tuple(sorted(refs)), actual_index)
        for actual_index, refs in endpoints.items()
        if actual_index in output_set
    )

    local_name: dict[str, str] = {}
    for idx, (_, actual_index) in enumerate(internal_specs):
        local_name[actual_index] = f"i{idx}"
    for idx, (_, actual_index) in enumerate(output_specs):
        local_name[actual_index] = f"o{idx}"

    inputs: list[IntermediateInputUse] = []
    for child in ordered:
        binding_by_port = {binding.port: binding.index for binding in child.bindings}
        if set(binding_by_port) != set(child.interface_ports):
            raise ValueError("Active child interface ports disagree with its bindings.")
        inputs.append(IntermediateInputUse(
            value_id=child.value_id,
            bindings=tuple(
                TensorLegBinding(port, local_name[binding_by_port[port]])
                for port in child.interface_ports
            ),
        ))

    contracted_indices = tuple(f"i{idx}" for idx in range(len(internal_specs)))
    output_indices = tuple(f"o{idx}" for idx in range(len(output_specs)))
    output_bindings = tuple(
        TensorLegBinding(f"o{idx}", actual_index)
        for idx, (_, actual_index) in enumerate(output_specs)
    )

    payload = {
        "inputs": [
            {
                "value": input_use.value_id,
                "bindings": [(b.port, b.index) for b in input_use.bindings],
            }
            for input_use in inputs
        ],
        "contracted": contracted_indices,
        "outputs": output_indices,
    }
    _, signature = _hash_payload("unused", payload)
    return (
        signature,
        tuple(inputs),
        contracted_indices,
        output_indices,
        output_bindings,
    )


def _canonicalize_step(
    step: ContractionStep,
    selected: Sequence[_ActiveValue],
) -> tuple[IntermediateComputation, tuple[TensorLegBinding, ...]]:
    if not selected:
        raise ValueError("Cannot canonicalize an intermediate without inputs.")

    _validate_step_inputs(step, selected)

    permutations = (
        itertools.permutations(range(len(selected)))
        if len(selected) <= 6
        else (tuple(range(len(selected))),)
    )
    candidates = [
        _candidate_for_permutation(step, selected, permutation)
        for permutation in permutations
    ]
    signature, inputs, contracted_indices, output_indices, output_bindings = min(
        candidates, key=lambda item: item[0]
    )

    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
    computation = IntermediateComputation(
        intermediate_id=f"intermediate_{digest}",
        signature=signature,
        inputs=inputs,
        contracted_indices=contracted_indices,
        output_indices=output_indices,
        estimated_flops=step.estimated_flops,
        output_elements=step.output_elements,
    )
    return computation, output_bindings


def _register_intermediate(
    registry: dict[str, IntermediateComputation],
    computation: IntermediateComputation,
) -> None:
    existing = registry.get(computation.intermediate_id)
    if existing is None:
        registry[computation.intermediate_id] = computation
        return
    if existing.signature != computation.signature:
        raise RuntimeError("Intermediate hash collision detected.")
    if (
        existing.estimated_flops != computation.estimated_flops
        or existing.output_elements != computation.output_elements
    ):
        raise RuntimeError(
            "Structurally identical intermediates received inconsistent cost metadata "
            "inside one contraction-order plan."
        )


def _lower_component_plan(
    component: ComponentComputation,
    plan: ComponentContractionPlan,
    primitive_values: dict[str, PrimitiveTensorValue],
    intermediates: dict[str, IntermediateComputation],
) -> tuple[ComponentIntermediatePlan, int, int]:
    plan.validate(component)

    active: dict[str, _ActiveValue] = {}
    primitive_occurrences = 0
    operator_arity = _component_operator_arity(component)

    for tensor in component.tensors:
        primitive = _primitive_value_from_tensor(
            tensor,
            operator_arity=operator_arity,
        )
        primitive_occurrences += 1
        existing = primitive_values.get(primitive.value_id)
        if existing is None:
            primitive_values[primitive.value_id] = primitive
        elif existing != primitive:
            raise RuntimeError("Primitive value hash collision detected.")
        active[tensor.slot] = _initial_active_value(tensor, primitive.value_id)

    step_uses: list[IntermediateUse] = []
    baseline_flops = 0

    for step in plan.steps:
        selected: list[_ActiveValue] = []
        for ref in step.input_refs:
            try:
                selected.append(active[ref])
            except KeyError as exc:
                raise ValueError(
                    f"Contraction step {step.step_id!r} references unavailable value {ref!r}."
                ) from exc

        computation, output_bindings = _canonicalize_step(step, selected)
        _register_intermediate(intermediates, computation)
        baseline_flops += step.estimated_flops

        for ref in step.input_refs:
            del active[ref]

        active[step.step_id] = _ActiveValue(
            value_id=computation.intermediate_id,
            interface_ports=computation.output_indices,
            bindings=output_bindings,
            plan_indices=step.output_indices,
        )
        step_uses.append(IntermediateUse(
            source_step_id=step.step_id,
            intermediate_id=computation.intermediate_id,
            output_bindings=output_bindings,
        ))

    if set(active) != {plan.result_ref}:
        raise ValueError(
            "Partial-CSE lowering must leave exactly the contraction plan's final result."
        )
    result = active[plan.result_ref]
    if result.plan_indices != plan.output_indices:
        raise ValueError("Partial-CSE result indices do not match the component plan output.")

    return (
        ComponentIntermediatePlan(
            computation_id=component.computation_id,
            step_uses=tuple(step_uses),
            result_value_id=result.value_id,
            result_bindings=result.bindings,
        ),
        primitive_occurrences,
        baseline_flops,
    )


def build_partial_intermediate_cse(
    order_plan: GlobalContractionOrderPlan,
) -> GlobalIntermediateDAG:
    """Deduplicate exact contraction subtrees across all unique component plans."""

    primitive_values: dict[str, PrimitiveTensorValue] = {}
    intermediates: dict[str, IntermediateComputation] = {}
    component_plans: dict[str, ComponentIntermediatePlan] = {}

    primitive_occurrences = 0
    intermediate_occurrences = 0
    baseline_step_flops = 0

    for computation_id, component in order_plan.source_dag.components.items():
        if component.computation_id != computation_id:
            raise ValueError(
                f"Contraction component key {computation_id!r} disagrees with its "
                f"record identity {component.computation_id!r}."
            )
        try:
            component_plan = order_plan.component_plans[computation_id]
        except KeyError as exc:
            raise ValueError(
                f"Missing contraction-order plan for component {computation_id!r}."
            ) from exc

        lowered, primitive_count, component_baseline_flops = _lower_component_plan(
            component,
            component_plan,
            primitive_values,
            intermediates,
        )
        component_plans[computation_id] = lowered
        primitive_occurrences += primitive_count
        intermediate_occurrences += len(component_plan.steps)
        baseline_step_flops += component_baseline_flops

    extra_plans = set(order_plan.component_plans) - set(order_plan.source_dag.components)
    if extra_plans:
        raise ValueError(
            "Contraction-order plan contains entries for unknown components: "
            f"{sorted(extra_plans)}"
        )

    unique_step_flops = sum(
        computation.estimated_flops for computation in intermediates.values()
    )

    return GlobalIntermediateDAG(
        source_plan=order_plan,
        primitive_values=dict(primitive_values),
        intermediates=dict(intermediates),
        component_plans=dict(component_plans),
        statistics=PartialCSEStatistics(
            primitive_value_occurrences=primitive_occurrences,
            unique_primitive_values=len(primitive_values),
            intermediate_occurrences=intermediate_occurrences,
            unique_intermediates=len(intermediates),
            baseline_step_flops=baseline_step_flops,
            unique_step_flops=unique_step_flops,
        ),
    )
