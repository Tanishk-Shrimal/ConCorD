"""Focused tests for the new abstract physical buffer planner."""

from bisect import bisect_left
from itertools import count
import random

import pytest

from concord_lqcd.backend.runtime.pipeline import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.backend.planner import (
    ContractOperation,
    CorrelatorSumOperation,
    MaterializePrimitiveOperation,
    WickTermProductOperation,
    build_physical_execution_plan,
)
from concord_lqcd.backend.planner.physical import (
    PhysicalBufferSlot,
    _LifetimeAllocationRequest,
    _allocate_lifetime_requests,
    _allocate_online_buffers,
    _make_operation,
)
from concord_lqcd.compiler.logical_scheduling import LogicalSchedule
from concord_lqcd.compiler import (
    ContractionSizeModel,
    analyze_static_execution,
    build_logical_execution_dag,
    build_logical_schedule,
    build_partial_intermediate_cse,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import (
    CorrMatrix,
    Correlator,
    Operator,
    c,
    cbar,
    d,
    dbar,
    gamma5,
    gamma_identity,
    reset_registry,
    u,
    ubar,
)
from concord_lqcd.core.wick import wick_contract


def _physics_ir():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * u("y", "a", "s2")
    )
    source = Operator(
        ubar("x", "b", "s3") * gamma5("s3", "s4") * c("x", "b", "s4")
    )
    correlator = Correlator(sink, source)
    return lower_to_physics_ir(correlator, wick_contract(correlator))


def _schedule(correlator_ids=("C00",), mode_size=5):
    physics_ir = _physics_ir()
    global_dag = lower_physics_ir_collection_to_global_cse({
        correlator_id: physics_ir for correlator_id in correlator_ids
    })
    order = optimize_global_contraction_order(
        global_dag,
        ContractionSizeModel(mode_size=mode_size),
        strategy="greedy",
    )
    logical = build_logical_execution_dag(build_partial_intermediate_cse(order))
    analysis = analyze_static_execution(logical)
    return build_logical_schedule(analysis, strategy="memory_aware")


def test_frontend_physical_plan_preserves_schedule_and_operation_semantics():
    schedule = _schedule(mode_size=6)
    plan = build_physical_execution_plan(schedule)

    assert tuple(step.operation.node_id for step in plan.steps) == schedule.order
    assert plan.statistics.scheduled_operations == len(schedule.steps)
    assert set(plan.value_bindings) == set(schedule.order)
    assert set(plan.outputs) == {"C00"}

    seen_types = set()
    resources = schedule.source_analysis.source_dag.source_dag.source_plan.source_dag.resources
    for step in plan.steps:
        operation = step.operation
        binding = plan.binding(operation.node_id)
        assert operation.output_buffer_id == binding.buffer_id
        assert plan.buffer(binding.buffer_id).capacity_bytes >= binding.size_bytes
        seen_types.add(type(operation))
        if isinstance(operation, MaterializePrimitiveOperation):
            assert operation.resource_id in resources
            logical = schedule.source_analysis.source_dag.primitives[operation.node_id]
            assert operation.resource_id == logical.resource_id
            assert operation.value_id == logical.value_id

    assert MaterializePrimitiveOperation in seen_types
    assert ContractOperation in seen_types
    assert WickTermProductOperation in seen_types
    assert CorrelatorSumOperation in seen_types

    output = plan.outputs["C00"]
    logical_output = schedule.source_analysis.source_dag.outputs["C00"]
    assert output.value_id == logical_output
    assert output.buffer_id == plan.binding(logical_output).buffer_id
    assert plan.binding(logical_output).retained_output


@pytest.mark.parametrize("reuse_strategy", ("best_fit", "size_matched", "none"))
def test_operation_construction_does_not_rebuild_complete_schedule_order(monkeypatch, reuse_strategy):
    schedule = _schedule(("C0", "C1"), mode_size=3)
    plan = build_physical_execution_plan(schedule, reuse_strategy=reuse_strategy)
    original_order = LogicalSchedule.order.fget
    order_calls = []

    def counted_order(current):
        order_calls.append(current)
        return original_order(current)

    monkeypatch.setattr(LogicalSchedule, "order", property(counted_order))
    operations = tuple(_make_operation(schedule, step.step_index, plan.value_bindings)
                       for step in plan.steps)

    assert operations == tuple(step.operation for step in plan.steps)
    assert {type(operation) for operation in operations} == {
        MaterializePrimitiveOperation, ContractOperation,
        WickTermProductOperation, CorrelatorSumOperation,
    }
    assert not order_calls


def test_best_fit_reuses_only_nonoverlapping_released_buffers():
    # Two equal correlators share the expensive component computation but retain
    # distinct term/sum assembly values, giving real production lifetimes that
    # can reuse released scalar buffers.
    schedule = _schedule(("C0", "C1"))
    plan = build_physical_execution_plan(schedule, reuse_strategy="best_fit")

    assert plan.statistics.buffer_reuses > 0
    assert plan.statistics.unique_buffers < plan.statistics.scheduled_operations

    values_by_buffer = {}
    for binding in plan.value_bindings.values():
        values_by_buffer.setdefault(binding.buffer_id, []).append(binding)

    reused = [bindings for bindings in values_by_buffer.values() if len(bindings) > 1]
    assert reused
    final_step = len(plan.steps)
    for bindings in reused:
        ordered = sorted(bindings, key=lambda item: item.produced_at)
        for previous, current in zip(ordered, ordered[1:]):
            previous_end = (
                previous.release_after_step
                if previous.release_after_step is not None
                else final_step
            )
            assert previous_end < current.produced_at

    # A value released by a step cannot donate its buffer to that same step's
    # output: output materialization precedes release in the conservative model.
    releasing_step = next(step for step in plan.steps if step.released_values)
    released_buffers = set(releasing_step.released_buffer_ids)
    assert releasing_step.operation.output_buffer_id not in released_buffers


def test_no_reuse_policy_allocates_one_buffer_per_logical_value():
    schedule = _schedule(("C0", "C1"))
    plan = build_physical_execution_plan(schedule, reuse_strategy="none")

    assert plan.statistics.buffer_reuses == 0
    assert plan.statistics.unique_buffers == len(schedule.order)
    assert len({binding.buffer_id for binding in plan.value_bindings.values()}) == len(schedule.order)
    expected_capacity = sum(
        schedule.source_analysis.value(value_id).size_bytes
        for value_id in schedule.order
    )
    assert plan.statistics.total_buffer_capacity_bytes == expected_capacity


def test_size_matched_reuse_preserves_logical_peak_capacity():
    schedule = _schedule(("C0", "C1"))
    plan = build_physical_execution_plan(schedule, reuse_strategy="size_matched")

    assert plan.statistics.buffer_reuses > 0
    assert all(
        plan.buffer(binding.buffer_id).capacity_bytes == binding.size_bytes
        for binding in plan.value_bindings.values()
    )
    assert (
        plan.statistics.peak_active_capacity_bytes
        == schedule.statistics.peak_live_bytes
    )


def test_physical_memory_limit_applies_to_buffer_pool_capacity():
    schedule = _schedule(("C0", "C1"))
    required_without_reuse = sum(
        schedule.source_analysis.value(value_id).size_bytes
        for value_id in schedule.order
    )

    with pytest.raises(MemoryError, match="exceeds memory_limit_bytes"):
        build_physical_execution_plan(
            schedule,
            reuse_strategy="none",
            memory_limit_bytes=required_without_reuse - 1,
        )


def test_physical_capacity_statistics_track_slots_not_logical_value_sizes():
    schedule = _schedule(("C0", "C1"))
    plan = build_physical_execution_plan(schedule, reuse_strategy="best_fit")

    assert plan.statistics.total_buffer_capacity_bytes == sum(
        buffer.capacity_bytes for buffer in plan.buffers.values()
    )
    assert plan.statistics.peak_active_capacity_bytes == max(
        step.peak_active_capacity_during_step for step in plan.steps
    )
    assert plan.statistics.final_active_capacity_bytes == sum(
        plan.buffer(plan.binding(value_id).buffer_id).capacity_bytes
        for value_id in schedule.output_nodes
    )
    assert plan.statistics.requested_output_bytes == sum(
        schedule.source_analysis.value(value_id).size_bytes
        for value_id in schedule.output_nodes
    )


def _new_slot(buffers, size_bytes, scalar_type):
    buffer_id = f"buffer_{len(buffers)}"
    buffers[buffer_id] = PhysicalBufferSlot(
        buffer_id=buffer_id,
        capacity_bytes=size_bytes,
        scalar_type=scalar_type,
        memory_space="default",
    )
    return buffer_id


def _full_scan_online(schedule, reuse_strategy):
    """Reference: compare every free buffer and take the smallest compatible one."""
    buffers, assignments, free = {}, {}, set()
    for step in schedule.steps:
        descriptor = schedule.source_analysis.value(step.node_id)
        compatible = [
            buffers[buffer_id]
            for buffer_id in free
            if reuse_strategy != "none"
            and buffers[buffer_id].scalar_type == descriptor.scalar_type
            and buffers[buffer_id].capacity_bytes >= descriptor.size_bytes
            and (
                reuse_strategy != "size_matched"
                or buffers[buffer_id].capacity_bytes == descriptor.size_bytes
            )
        ]
        if compatible:
            buffer_id = min(compatible, key=lambda slot: (slot.capacity_bytes, slot.buffer_id)).buffer_id
            free.remove(buffer_id)
        else:
            buffer_id = _new_slot(buffers, descriptor.size_bytes, descriptor.scalar_type)
        assignments[step.node_id] = buffer_id
        free.update(assignments[value_id] for value_id in step.released_values)
    return buffers, assignments


def _full_scan_lifetime(requests):
    """Reference: largest first, compare every buffer, take the smallest free one."""
    buffers, assignments, occupants = {}, {}, {}
    ordered = sorted(
        requests,
        key=lambda request: (-request.size_bytes, request.produced_at, request.end_at, request.value_id),
    )
    for request in ordered:
        interval = (request.produced_at, request.end_at)
        compatible = []
        for buffer_id, slot in buffers.items():
            if slot.scalar_type != request.scalar_type or slot.capacity_bytes < request.size_bytes:
                continue
            intervals = occupants[buffer_id]
            position = bisect_left(intervals, interval)
            if position and intervals[position - 1][1] >= request.produced_at:
                continue
            if position < len(intervals) and request.end_at >= intervals[position][0]:
                continue
            compatible.append((slot, position))
        if compatible:
            slot, position = min(compatible, key=lambda item: (item[0].capacity_bytes, item[0].buffer_id))
            buffer_id = slot.buffer_id
        else:
            buffer_id = _new_slot(buffers, request.size_bytes, request.scalar_type)
            occupants[buffer_id] = []
            position = 0
        assignments[request.value_id] = buffer_id
        occupants[buffer_id].insert(position, interval)
    return buffers, assignments


def _scalar_channel_schedule():
    """sigma and a two-pion operator: many values of equal size, released in varied order."""
    reset_registry()
    labels = count()

    def meson(antiquark, quark, gamma, position):
        color, left, right = (f"m{next(labels)}" for _ in range(3))
        return Operator(
            antiquark(position, color, left) * gamma(left, right) * quark(position, color, right)
        )

    sigma = meson(ubar, u, gamma_identity, "x") + meson(dbar, d, gamma_identity, "x")
    pipi = (
        meson(dbar, u, gamma5, "x") * meson(ubar, d, gamma5, "y")
        + meson(ubar, d, gamma5, "x") * meson(dbar, u, gamma5, "y")
    )
    return compile_correlators(
        CorrMatrix([sigma, pipi]), RuntimeCompilationConfig(mode_size=3)
    ).schedule


@pytest.mark.parametrize("reuse_strategy", ["best_fit", "size_matched", "none"])
def test_online_allocation_matches_a_full_scan_of_free_buffers(reuse_strategy):
    schedule = _scalar_channel_schedule()
    result = _allocate_online_buffers(schedule, memory_space="default", reuse_strategy=reuse_strategy)

    assert len(result[0]) > 1
    assert result == _full_scan_online(schedule, reuse_strategy)


@pytest.mark.parametrize("seed", range(5))
def test_lifetime_packing_matches_a_full_scan_of_all_buffers(seed):
    generator = random.Random(seed)
    requests = []
    for index in range(400):
        produced_at = generator.randrange(200)
        requests.append(_LifetimeAllocationRequest(
            value_id=f"v{index}",
            size_bytes=generator.choice((16, 64, 64, 256, 1024)),
            scalar_type=generator.choice(("complex128", "complex64")),
            produced_at=produced_at,
            end_at=produced_at + generator.randrange(1, 40),
        ))
    result = _allocate_lifetime_requests(tuple(requests), memory_space="default")

    assert len(result[0]) > 10
    assert result == _full_scan_lifetime(requests)
