"""Physical buffer planning for the new ConCorD logical execution pipeline.

This module consumes a frozen :class:`LogicalSchedule` and maps every scheduled
logical value onto an abstract reusable memory buffer. It is intentionally the
first physical-planning layer, not a runtime implementation:

* resource ids remain logical resource requirements -- no filenames/HDF5 paths;
* buffers are abstract memory slots -- no CPU/GPU assignment yet;
* the logical schedule/order is preserved exactly;
* output materialization happens before final-use inputs are released;
* non-overlapping logical lifetimes may reuse the same physical buffer.

The central identity boundary is therefore

    resource id != logical value id != physical buffer id.

Later backend-specific lowering can attach devices, storage locations, streams,
transfers, and concrete allocator/runtime operations without changing the
algebraic compiler layers.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Mapping, TypeAlias

from concord_lqcd.compiler._validation import _has_static_validation
from concord_lqcd.compiler.contraction_graph import TensorLegBinding
from concord_lqcd.compiler.logical_execution import (
    LogicalContraction,
    LogicalCorrelatorSum,
    LogicalPrimitiveEvaluation,
    LogicalWickTermProduct,
)
from concord_lqcd.compiler.logical_scheduling import LogicalSchedule


BufferReuseStrategy = Literal["best_fit", "size_matched", "none"]


def _require_integer(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Physical {name} must be an integer greater than or equal to {minimum}.")


@dataclass(frozen=True)
class PhysicalBufferSlot:
    """One reusable abstract memory slot."""

    buffer_id: str
    capacity_bytes: int
    scalar_type: str
    memory_space: str = "default"


@dataclass(frozen=True)
class PhysicalValueBinding:
    """Bind one logical value to one physical buffer for its scheduled lifetime."""

    value_id: str
    buffer_id: str
    shape: tuple[int, ...]
    axis_labels: tuple[str, ...]
    axis_spaces: tuple[str, ...]
    scalar_type: str
    size_bytes: int
    produced_at: int
    release_after_step: int | None
    retained_output: bool


@dataclass(frozen=True)
class PhysicalInputUse:
    """One physical input occurrence, preserving logical multiplicity/bindings."""

    value_id: str
    buffer_id: str
    bindings: tuple[TensorLegBinding, ...] = ()


@dataclass(frozen=True)
class MaterializePrimitiveOperation:
    step_index: int
    node_id: str
    value_id: str
    resource_id: str
    kind: str
    placement: tuple[object, ...]
    output_ports: tuple[str, ...]
    output_buffer_id: str


@dataclass(frozen=True)
class ContractOperation:
    step_index: int
    node_id: str
    inputs: tuple[PhysicalInputUse, ...]
    contracted_indices: tuple[str, ...]
    output_indices: tuple[str, ...]
    estimated_flops: int
    output_buffer_id: str


@dataclass(frozen=True)
class WickTermProductOperation:
    step_index: int
    node_id: str
    correlator_id: str
    term_index: int
    inputs: tuple[PhysicalInputUse, ...]
    coefficient: complex
    output_indices: tuple[str, ...]
    output_buffer_id: str


@dataclass(frozen=True)
class CorrelatorSumOperation:
    step_index: int
    node_id: str
    correlator_id: str
    inputs: tuple[PhysicalInputUse, ...]
    output_indices: tuple[str, ...]
    output_buffer_id: str


PhysicalOperation: TypeAlias = (
    MaterializePrimitiveOperation
    | ContractOperation
    | WickTermProductOperation
    | CorrelatorSumOperation
)


@dataclass(frozen=True)
class PhysicalPlanStep:
    """One physical operation followed by logical-value releases."""

    step_index: int
    operation: PhysicalOperation
    released_values: tuple[str, ...]
    released_buffer_ids: tuple[str, ...]
    active_capacity_before: int
    peak_active_capacity_during_step: int
    active_capacity_after: int


@dataclass(frozen=True)
class PhysicalOutputBinding:
    correlator_id: str
    value_id: str
    buffer_id: str
    shape: tuple[int, ...]
    axis_labels: tuple[str, ...]
    axis_spaces: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalPlanningStatistics:
    scheduled_operations: int
    unique_buffers: int
    buffer_reuses: int
    total_buffer_capacity_bytes: int
    peak_active_capacity_bytes: int
    final_active_capacity_bytes: int
    requested_output_bytes: int


@dataclass(frozen=True)
class PhysicalExecutionPlan:
    """Abstract single-memory-space physical plan for the new compiler pipeline."""

    source_schedule: LogicalSchedule
    memory_space: str
    reuse_strategy: BufferReuseStrategy
    buffers: Mapping[str, PhysicalBufferSlot]
    value_bindings: Mapping[str, PhysicalValueBinding]
    steps: tuple[PhysicalPlanStep, ...]
    outputs: Mapping[str, PhysicalOutputBinding]
    statistics: PhysicalPlanningStatistics

    def buffer(self, buffer_id: str) -> PhysicalBufferSlot:
        return self.buffers[buffer_id]

    def binding(self, value_id: str) -> PhysicalValueBinding:
        return self.value_bindings[value_id]

    def validate(self) -> None:
        if _has_static_validation(self):
            return
        schedule = self.source_schedule
        schedule.validate()
        analysis = schedule.source_analysis
        order = schedule.order

        if not isinstance(self.memory_space, str) or not self.memory_space:
            raise ValueError("Physical memory_space must be a non-empty string.")
        if self.reuse_strategy not in ("best_fit", "size_matched", "none"):
            raise ValueError(f"Unsupported physical buffer reuse strategy {self.reuse_strategy!r}.")

        if tuple(step.operation.node_id for step in self.steps) != order:
            raise ValueError("Physical plan does not preserve the logical schedule order.")
        if tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("Physical plan step indices are inconsistent.")
        if set(self.value_bindings) != set(order):
            raise ValueError("Physical value-binding map does not match scheduled values.")

        for buffer_id, slot in self.buffers.items():
            if slot.buffer_id != buffer_id:
                raise ValueError("Physical buffer is stored under the wrong id.")
            _require_integer(slot.capacity_bytes, "buffer capacity_bytes", minimum=1)
            if slot.memory_space != self.memory_space:
                raise ValueError("Physical buffer belongs to the wrong memory space.")

        for value_id in order:
            descriptor = analysis.value(value_id)
            lifetime = schedule.lifetime(value_id)
            binding = self.value_bindings[value_id]
            for dimension in binding.shape:
                _require_integer(dimension, "binding shape dimension", minimum=1)
            _require_integer(binding.size_bytes, "binding size_bytes", minimum=1)
            _require_integer(binding.produced_at, "binding produced_at")
            if binding.release_after_step is not None:
                _require_integer(binding.release_after_step, "binding release_after_step")
            if type(binding.retained_output) is not bool:
                raise ValueError("Physical binding retained_output must be a boolean.")
            if binding.value_id != value_id:
                raise ValueError("Physical value binding is stored under the wrong value id.")
            if binding.buffer_id not in self.buffers:
                raise ValueError(f"Physical value {value_id!r} references an unknown buffer.")
            slot = self.buffers[binding.buffer_id]
            if slot.capacity_bytes < descriptor.size_bytes:
                raise ValueError(f"Physical buffer for {value_id!r} is too small.")
            if self.reuse_strategy == "size_matched" and slot.capacity_bytes != descriptor.size_bytes:
                raise ValueError(f"Physical size_matched buffer for {value_id!r} has excess capacity.")
            if slot.scalar_type != descriptor.scalar_type:
                raise ValueError(f"Physical buffer for {value_id!r} has incompatible scalar type.")
            if (
                binding.shape != descriptor.shape
                or binding.axis_labels != descriptor.axis_labels
                or binding.axis_spaces != tuple(descriptor.axis_spaces)
                or binding.scalar_type != descriptor.scalar_type
                or binding.size_bytes != descriptor.size_bytes
            ):
                raise ValueError(f"Physical metadata for {value_id!r} disagrees with static analysis.")
            if binding.produced_at != lifetime.produced_at:
                raise ValueError(f"Physical production step for {value_id!r} is inconsistent.")
            if binding.release_after_step != lifetime.release_after_step:
                raise ValueError(f"Physical release step for {value_id!r} is inconsistent.")
            if binding.retained_output != lifetime.retained_output:
                raise ValueError(f"Physical output retention for {value_id!r} is inconsistent.")

        values_by_buffer: dict[str, list[PhysicalValueBinding]] = {
            buffer_id: [] for buffer_id in self.buffers
        }
        final_step = len(self.steps)
        for binding in self.value_bindings.values():
            values_by_buffer[binding.buffer_id].append(binding)
        for buffer_id, bindings in values_by_buffer.items():
            if not bindings:
                raise ValueError(f"Physical buffer {buffer_id!r} has no scheduled owner.")
            if self.reuse_strategy == "none" and len(bindings) != 1:
                raise ValueError("Physical no-reuse policy assigns multiple values to one buffer.")
            ordered = sorted(bindings, key=lambda b: (b.produced_at, b.value_id))
            previous_end = -1
            for binding in ordered:
                end = (
                    binding.release_after_step
                    if binding.release_after_step is not None
                    else final_step
                )
                if binding.produced_at <= previous_end:
                    raise ValueError(
                        f"Physical buffer {buffer_id!r} aliases overlapping logical lifetimes."
                    )
                previous_end = end

        output_value_ids = set(schedule.output_nodes)
        if set(self.outputs) != set(analysis.source_dag.outputs):
            raise ValueError("Physical output map does not match requested correlators.")
        for correlator_id, logical_value_id in analysis.source_dag.outputs.items():
            output = self.outputs[correlator_id]
            binding = self.value_bindings[logical_value_id]
            if (output.correlator_id != correlator_id or output.value_id != logical_value_id
                    or output.buffer_id != binding.buffer_id):
                raise ValueError("Physical correlator output binding is inconsistent.")
            if logical_value_id not in output_value_ids:
                raise ValueError("Physical correlator output is not retained by the schedule.")
            if (
                output.shape != binding.shape
                or output.axis_labels != binding.axis_labels
                or output.axis_spaces != binding.axis_spaces
            ):
                raise ValueError("Physical correlator output metadata is inconsistent.")

        active_owners: dict[str, str] = {}
        active_capacity = 0
        peak_active = 0
        for plan_step in self.steps:
            node_id = plan_step.operation.node_id
            output_binding = self.value_bindings[node_id]
            operation = plan_step.operation
            for name in ("step_index", "active_capacity_before",
                         "peak_active_capacity_during_step", "active_capacity_after"):
                _require_integer(getattr(plan_step, name), name)
            _require_integer(operation.step_index, "operation step_index")
            if isinstance(operation, ContractOperation):
                _require_integer(operation.estimated_flops, "operation estimated_flops")
            if isinstance(operation, WickTermProductOperation):
                _require_integer(operation.term_index, "operation term_index")
            if operation != _make_operation(schedule, plan_step.step_index, self.value_bindings):
                raise ValueError(f"Physical operation metadata for {node_id!r} disagrees with its logical source.")

            before = active_capacity
            if plan_step.active_capacity_before != before:
                raise ValueError("Physical active capacity before step is inconsistent.")
            for use in getattr(operation, "inputs", ()):
                if active_owners.get(use.buffer_id) != use.value_id:
                    raise ValueError("Physical operation reads an inactive or overwritten input.")
            if output_binding.buffer_id in active_owners:
                raise ValueError("Physical output aliases an active input or retained value.")
            active_owners[output_binding.buffer_id] = node_id
            during = before + self.buffers[output_binding.buffer_id].capacity_bytes
            active_capacity = during
            if plan_step.peak_active_capacity_during_step != during:
                raise ValueError("Physical peak active capacity during step is inconsistent.")
            peak_active = max(peak_active, during)

            expected_released = schedule.steps[plan_step.step_index].released_values
            if plan_step.released_values != expected_released:
                raise ValueError("Physical release list disagrees with logical scheduling.")
            if plan_step.released_buffer_ids != tuple(
                self.value_bindings[value_id].buffer_id for value_id in expected_released
            ):
                raise ValueError("Physical released-buffer list is inconsistent.")
            for value_id, buffer_id in zip(plan_step.released_values, plan_step.released_buffer_ids, strict=True):
                if active_owners.get(buffer_id) != value_id:
                    raise ValueError("Physical release targets an inactive or overwritten value.")
                del active_owners[buffer_id]
                active_capacity -= self.buffers[buffer_id].capacity_bytes
            after = active_capacity
            if plan_step.active_capacity_after != after:
                raise ValueError("Physical active capacity after step is inconsistent.")

        expected_owners = {self.value_bindings[value_id].buffer_id: value_id
                           for value_id in schedule.output_nodes}
        if active_owners != expected_owners:
            raise ValueError("Physical final buffer owners disagree with retained outputs.")
        final_active_capacity = sum(self.buffers[buffer_id].capacity_bytes for buffer_id in active_owners)
        if active_capacity != final_active_capacity:
            raise ValueError("Physical active capacity disagrees with the final buffer ownership ledger.")
        total_capacity = sum(slot.capacity_bytes for slot in self.buffers.values())
        statistics = PhysicalPlanningStatistics(
            scheduled_operations=len(self.steps),
            unique_buffers=len(self.buffers),
            buffer_reuses=len(self.steps) - len(self.buffers),
            total_buffer_capacity_bytes=total_capacity,
            peak_active_capacity_bytes=peak_active,
            final_active_capacity_bytes=final_active_capacity,
            requested_output_bytes=sum(analysis.value(value_id).size_bytes
                                       for value_id in schedule.output_nodes),
        )
        for name in statistics.__dataclass_fields__:
            _require_integer(getattr(self.statistics, name), f"statistic {name}")
        if self.statistics != statistics:
            raise ValueError("Physical planning statistics disagree with the buffer ownership ledger.")


def _physical_inputs(logical_uses, bindings: Mapping[str, PhysicalValueBinding]) -> tuple[PhysicalInputUse, ...]:
    return tuple(
        PhysicalInputUse(
            value_id=use.value_id,
            buffer_id=bindings[use.value_id].buffer_id,
            bindings=use.bindings,
        )
        for use in logical_uses
    )


def _make_operation(schedule: LogicalSchedule, step_index: int, bindings: Mapping[str, PhysicalValueBinding]) -> PhysicalOperation:
    dag = schedule.source_analysis.source_dag
    node_id = schedule.steps[step_index].node_id
    node = dag.node(node_id)
    out_buffer = bindings[node_id].buffer_id

    if isinstance(node, LogicalPrimitiveEvaluation):
        return MaterializePrimitiveOperation(
            step_index=step_index,
            node_id=node_id,
            value_id=node.value_id,
            resource_id=node.resource_id,
            kind=node.kind,
            placement=node.placement,
            output_ports=node.output_ports,
            output_buffer_id=out_buffer,
        )
    if isinstance(node, LogicalContraction):
        return ContractOperation(
            step_index=step_index,
            node_id=node_id,
            inputs=_physical_inputs(node.inputs, bindings),
            contracted_indices=node.contracted_indices,
            output_indices=node.output_indices,
            estimated_flops=node.estimated_flops,
            output_buffer_id=out_buffer,
        )
    if isinstance(node, LogicalWickTermProduct):
        return WickTermProductOperation(
            step_index=step_index,
            node_id=node_id,
            correlator_id=node.correlator_id,
            term_index=node.term_index,
            inputs=_physical_inputs(node.inputs, bindings),
            coefficient=node.coefficient,
            output_indices=node.output_indices,
            output_buffer_id=out_buffer,
        )
    if isinstance(node, LogicalCorrelatorSum):
        return CorrelatorSumOperation(
            step_index=step_index,
            node_id=node_id,
            correlator_id=node.correlator_id,
            inputs=tuple(
                PhysicalInputUse(term_id, bindings[term_id].buffer_id, ())
                for term_id in node.term_ids
            ),
            output_indices=node.output_indices,
            output_buffer_id=out_buffer,
        )
    raise TypeError(type(node).__name__)


@dataclass(frozen=True)
class _LifetimeAllocationRequest:
    value_id: str
    size_bytes: int
    scalar_type: str
    produced_at: int
    end_at: int


def _allocate_online_buffers(
    schedule: LogicalSchedule,
    *,
    memory_space: str,
    reuse_strategy: BufferReuseStrategy,
) -> tuple[dict[str, PhysicalBufferSlot], dict[str, str]]:
    """Preserve the schedule-order allocation of all existing reuse policies."""

    buffers: dict[str, PhysicalBufferSlot] = {}
    assignments: dict[str, str] = {}
    # Free buffers of each scalar type, sorted by (capacity, buffer id). The first
    # entry at or above the requested size is the smallest compatible buffer.
    free_buffers: set[str] = set()
    free_by_type: dict[str, list[tuple[int, str]]] = {}
    for step in schedule.steps:
        descriptor = schedule.source_analysis.value(step.node_id)
        buffer_id = None
        if reuse_strategy != "none":
            free = free_by_type.get(descriptor.scalar_type, [])
            position = bisect_left(free, (descriptor.size_bytes, ""))
            if position < len(free) and (
                reuse_strategy != "size_matched" or free[position][0] == descriptor.size_bytes
            ):
                _, buffer_id = free.pop(position)
                free_buffers.remove(buffer_id)
        if buffer_id is None:
            buffer_id = f"buffer_{len(buffers)}"
            buffers[buffer_id] = PhysicalBufferSlot(
                buffer_id=buffer_id,
                capacity_bytes=descriptor.size_bytes,
                scalar_type=descriptor.scalar_type,
                memory_space=memory_space,
            )
        assignments[step.node_id] = buffer_id
        for value_id in step.released_values:
            released = assignments[value_id]
            if released not in free_buffers:
                free_buffers.add(released)
                slot = buffers[released]
                insort(free_by_type.setdefault(slot.scalar_type, []), (slot.capacity_bytes, released))
    return buffers, assignments


def _online_buffer_capacity_bytes(
    schedule: LogicalSchedule,
    *,
    reuse_strategy: BufferReuseStrategy,
) -> int:
    """Return the schedule-order pool ceiling for an already validated schedule."""

    buffers, _ = _allocate_online_buffers(
        schedule, memory_space="default", reuse_strategy=reuse_strategy
    )
    return sum(slot.capacity_bytes for slot in buffers.values())


def _allocate_lifetime_requests(
    requests: tuple[_LifetimeAllocationRequest, ...],
    *,
    memory_space: str,
) -> tuple[dict[str, PhysicalBufferSlot], dict[str, str]]:
    """Pack closed lifetime intervals largest-first without same-step aliasing."""

    buffers: dict[str, PhysicalBufferSlot] = {}
    assignments: dict[str, str] = {}
    occupants: dict[str, list[tuple[int, int]]] = {}
    # All buffers of each scalar type, sorted by (capacity, buffer id). The first
    # buffer in this order whose occupants leave the interval free is the
    # smallest compatible one, so the walk stops there.
    by_type: dict[str, list[tuple[int, str]]] = {}
    ordered = sorted(
        requests,
        key=lambda request: (
            -request.size_bytes,
            request.produced_at,
            request.end_at,
            request.value_id,
        ),
    )
    for request in ordered:
        interval = (request.produced_at, request.end_at)
        candidates = by_type.setdefault(request.scalar_type, [])
        buffer_id = None
        for index in range(bisect_left(candidates, (request.size_bytes, "")), len(candidates)):
            candidate_id = candidates[index][1]
            intervals = occupants[candidate_id]
            position = bisect_left(intervals, interval)
            if position and intervals[position - 1][1] >= request.produced_at:
                continue
            if position < len(intervals) and request.end_at >= intervals[position][0]:
                continue
            buffer_id = candidate_id
            break
        if buffer_id is None:
            buffer_id = f"buffer_{len(buffers)}"
            buffers[buffer_id] = PhysicalBufferSlot(
                buffer_id=buffer_id,
                capacity_bytes=request.size_bytes,
                scalar_type=request.scalar_type,
                memory_space=memory_space,
            )
            occupants[buffer_id] = []
            insort(candidates, (request.size_bytes, buffer_id))
            position = 0
        assignments[request.value_id] = buffer_id
        occupants[buffer_id].insert(position, interval)
    return buffers, assignments


def _allocation_peak_capacity(
    schedule: LogicalSchedule,
    buffers: Mapping[str, PhysicalBufferSlot],
    assignments: Mapping[str, str],
) -> int:
    active_capacity = 0
    peak_capacity = 0
    for step in schedule.steps:
        active_capacity += buffers[assignments[step.node_id]].capacity_bytes
        peak_capacity = max(peak_capacity, active_capacity)
        active_capacity -= sum(
            buffers[assignments[value_id]].capacity_bytes
            for value_id in step.released_values
        )
    return peak_capacity


def _select_buffer_allocation(
    schedule: LogicalSchedule,
    *,
    memory_space: str,
    reuse_strategy: BufferReuseStrategy,
    memory_limit_bytes: int | None,
) -> tuple[dict[str, PhysicalBufferSlot], dict[str, str]]:
    """Keep online allocation unless a safe improvement or hard-budget fit exists."""

    chosen = _allocate_online_buffers(
        schedule, memory_space=memory_space, reuse_strategy=reuse_strategy
    )
    chosen_total = sum(slot.capacity_bytes for slot in chosen[0].values())
    smallest_considered = chosen_total
    if reuse_strategy == "best_fit" and chosen_total > schedule.statistics.peak_live_bytes:
        requests: list[_LifetimeAllocationRequest] = []
        for value_id in schedule.order:
            descriptor = schedule.source_analysis.value(value_id)
            lifetime = schedule.lifetime(value_id)
            requests.append(_LifetimeAllocationRequest(
                value_id=value_id,
                size_bytes=descriptor.size_bytes,
                scalar_type=descriptor.scalar_type,
                produced_at=lifetime.produced_at,
                end_at=(
                    lifetime.release_after_step
                    if lifetime.release_after_step is not None
                    else len(schedule.steps)
                ),
            ))
        lifetime_allocation = _allocate_lifetime_requests(tuple(requests), memory_space=memory_space)
        lifetime_total = sum(slot.capacity_bytes for slot in lifetime_allocation[0].values())
        smallest_considered = min(chosen_total, lifetime_total)
        if lifetime_total <= chosen_total:
            chosen_peak = _allocation_peak_capacity(schedule, *chosen)
            lifetime_peak = _allocation_peak_capacity(schedule, *lifetime_allocation)
            improves_memory = (
                lifetime_peak <= chosen_peak
                and (lifetime_total < chosen_total or lifetime_peak < chosen_peak)
            )
            fits_required_budget = (
                memory_limit_bytes is not None
                and chosen_total > memory_limit_bytes >= lifetime_total
            )
            if improves_memory or fits_required_budget:
                chosen = lifetime_allocation
                chosen_total = lifetime_total
    if memory_limit_bytes is not None and chosen_total > memory_limit_bytes:
        raise MemoryError(
            "Physical buffer pool exceeds memory_limit_bytes: "
            f"smallest considered pool requires {smallest_considered}, limit {memory_limit_bytes}."
        )
    return chosen


def build_physical_execution_plan(
    schedule: LogicalSchedule,
    *,
    memory_space: str = "default",
    reuse_strategy: BufferReuseStrategy = "best_fit",
    memory_limit_bytes: int | None = None,
) -> PhysicalExecutionPlan:
    """Lower a schedule, comparing safe lifetime packing for best-fit reuse.

    Equal pool/peak metrics preserve the online assignment. A hard pool budget
    may admit a smaller lifetime-packed pool even when its active peak is higher.
    Size-matched and no-reuse policies retain their schedule-order allocation.
    """

    schedule.validate()
    if not isinstance(memory_space, str) or not memory_space:
        raise ValueError("memory_space must be a non-empty string.")
    if reuse_strategy not in ("best_fit", "size_matched", "none"):
        raise ValueError(f"Unsupported physical buffer reuse strategy {reuse_strategy!r}.")
    if memory_limit_bytes is not None:
        _require_integer(memory_limit_bytes, "memory_limit_bytes", minimum=1)

    analysis = schedule.source_analysis
    buffers, buffer_assignments = _select_buffer_allocation(
        schedule,
        memory_space=memory_space,
        reuse_strategy=reuse_strategy,
        memory_limit_bytes=memory_limit_bytes,
    )
    bindings: dict[str, PhysicalValueBinding] = {}
    active_values: set[str] = set()
    active_buffer_ids: set[str] = set()
    active_capacity = 0
    total_capacity = sum(slot.capacity_bytes for slot in buffers.values())
    peak_active_capacity = 0
    buffer_reuses = len(schedule.steps) - len(buffers)
    plan_steps: list[PhysicalPlanStep] = []

    for logical_step in schedule.steps:
        node_id = logical_step.node_id
        descriptor = analysis.value(node_id)
        lifetime = schedule.lifetime(node_id)

        buffer_id = buffer_assignments[node_id]
        if buffer_id in active_buffer_ids:
            raise RuntimeError("Physical planner selected an active output buffer.")

        binding = PhysicalValueBinding(
            value_id=node_id,
            buffer_id=buffer_id,
            shape=descriptor.shape,
            axis_labels=descriptor.axis_labels,
            axis_spaces=tuple(descriptor.axis_spaces),
            scalar_type=descriptor.scalar_type,
            size_bytes=descriptor.size_bytes,
            produced_at=lifetime.produced_at,
            release_after_step=lifetime.release_after_step,
            retained_output=lifetime.retained_output,
        )
        bindings[node_id] = binding

        active_before = active_capacity
        active_values.add(node_id)
        active_buffer_ids.add(buffer_id)
        active_capacity += buffers[buffer_id].capacity_bytes
        peak_during = active_capacity
        peak_active_capacity = max(peak_active_capacity, peak_during)

        released_buffer_ids: list[str] = []
        for released_value in logical_step.released_values:
            if released_value not in active_values:
                raise RuntimeError(
                    f"Logical schedule releases non-active value {released_value!r}."
                )
            released_buffer = bindings[released_value].buffer_id
            active_values.remove(released_value)
            active_buffer_ids.remove(released_buffer)
            active_capacity -= buffers[released_buffer].capacity_bytes
            released_buffer_ids.append(released_buffer)

        active_after = active_capacity
        operation = _make_operation(schedule, logical_step.step_index, bindings)
        plan_steps.append(PhysicalPlanStep(
            step_index=logical_step.step_index,
            operation=operation,
            released_values=logical_step.released_values,
            released_buffer_ids=tuple(released_buffer_ids),
            active_capacity_before=active_before,
            peak_active_capacity_during_step=peak_during,
            active_capacity_after=active_after,
        ))

    outputs: dict[str, PhysicalOutputBinding] = {}
    for correlator_id, value_id in analysis.source_dag.outputs.items():
        binding = bindings[value_id]
        outputs[correlator_id] = PhysicalOutputBinding(
            correlator_id=correlator_id,
            value_id=value_id,
            buffer_id=binding.buffer_id,
            shape=binding.shape,
            axis_labels=binding.axis_labels,
            axis_spaces=binding.axis_spaces,
        )

    final_active_capacity = plan_steps[-1].active_capacity_after if plan_steps else 0
    requested_output_bytes = sum(
        analysis.value(value_id).size_bytes for value_id in schedule.output_nodes
    )
    plan = PhysicalExecutionPlan(
        source_schedule=schedule,
        memory_space=memory_space,
        reuse_strategy=reuse_strategy,
        buffers=dict(buffers),
        value_bindings=dict(bindings),
        steps=tuple(plan_steps),
        outputs=outputs,
        statistics=PhysicalPlanningStatistics(
            scheduled_operations=len(plan_steps),
            unique_buffers=len(buffers),
            buffer_reuses=buffer_reuses,
            total_buffer_capacity_bytes=total_capacity,
            peak_active_capacity_bytes=peak_active_capacity,
            final_active_capacity_bytes=final_active_capacity,
            requested_output_bytes=requested_output_bytes,
        ),
    )
    plan.validate()
    return plan
