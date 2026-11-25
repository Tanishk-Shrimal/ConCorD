"""Independent physical-slot and concrete pool ownership checks."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from concord_lqcd.backend.planner.physical import (
    ContractOperation,
    CorrelatorSumOperation,
    MaterializePrimitiveOperation,
    PhysicalBufferSlot,
    WickTermProductOperation,
    build_physical_execution_plan,
)
from concord_lqcd.backend.runtime.batched_executor import (
    _execute_batched_contraction,
    _execute_batched_contraction_into_if_possible,
    _execute_batched_correlator_sum,
    _execute_batched_wick_term,
    _private_batch_index,
)
from concord_lqcd.backend.runtime.numerical_executor import (
    RuntimeBufferPool,
    _execute_contraction,
    _execute_contraction_into_if_possible,
    _execute_correlator_sum,
    _execute_wick_term,
)
from concord_lqcd.compiler import (
    analyze_static_execution,
    build_logical_schedule,
)
from tests.test_static_lifetime_numerical import _arrays, _direct, _source


def _plan(*, mode_size=2, dtype="complex128", strategy="memory_aware",
          include_dead=True, reuse_strategy="best_fit", element_bytes=None):
    logical = _source(mode_size)
    analysis = analyze_static_execution(
        logical, scalar_type=dtype,
        element_bytes=np.dtype(dtype).itemsize if element_bytes is None else element_bytes,
    )
    schedule = build_logical_schedule(analysis, strategy=strategy, include_dead=include_dead)
    return build_physical_execution_plan(schedule, reuse_strategy=reuse_strategy)


@pytest.mark.parametrize("mode_size", (2, 4))
@pytest.mark.parametrize("dtype", ("complex64", "complex128"))
@pytest.mark.parametrize("strategy", ("topological", "critical_path", "memory_aware"))
@pytest.mark.parametrize("include_dead", (False, True))
@pytest.mark.parametrize("reuse_strategy", ("best_fit", "size_matched", "none"))
def test_independent_pool_replay_and_resident_slot_ledger(
    mode_size, dtype, strategy, include_dead, reuse_strategy,
):
    arrays = _arrays(mode_size, dtype)
    references = [_direct({name: array + offset for name, array in arrays.items()})
                  for offset in (0, 0.25, 0.5)]
    assert np.max(np.abs(references[0]["matrix"])) > 0.1
    plan = _plan(mode_size=mode_size, dtype=dtype, strategy=strategy,
                 include_dead=include_dead, reuse_strategy=reuse_strategy)
    logical = plan.source_schedule.source_analysis.source_dag
    batch_index = _private_batch_index(plan)
    pool = RuntimeBufferPool(plan, batch_size=3)
    pointers = {identity: storage.data_ptr() for identity, storage in pool.storage.items()}
    assert pool.allocation_bytes == 3 * sum(slot.capacity_bytes for slot in plan.buffers.values())
    observed_reuses = 0
    used_slots = set()
    snapshots = []

    for batch_size in (3, 1):
        pool.reset(batch_size=batch_size)
        owners, resident = {}, set()
        peak = 0
        for number, step in enumerate(plan.steps):
            node = logical.node(step.operation.node_id)
            binding = plan.binding(node.node_id)
            before = sum(plan.buffer(identity).capacity_bytes for identity in owners)
            assert step.active_capacity_before == before
            assert pool.active_capacity_bytes == batch_size * before
            assert set(node.dependencies) <= resident
            assert binding.buffer_id not in owners
            if binding.buffer_id in used_slots:
                observed_reuses += 1
            pool.storage[binding.buffer_id].fill_(complex(91, -73))
            operation = step.operation
            wrote_directly = False
            if isinstance(operation, MaterializePrimitiveOperation):
                results = [arrays[operation.resource_id] + 0.25 * position for position in range(batch_size)]
                value = torch.from_numpy(results[0] if batch_size == 1 else np.stack(results))
            elif isinstance(operation, ContractOperation):
                value = (_execute_contraction_into_if_possible(plan, pool, operation) if batch_size == 1
                         else _execute_batched_contraction_into_if_possible(plan, pool, operation, batch_index))
                wrote_directly = value is not None
                if value is None:
                    value = (_execute_contraction(plan, pool, operation) if batch_size == 1
                             else _execute_batched_contraction(plan, pool, operation, batch_index))
            elif isinstance(operation, WickTermProductOperation):
                value = (_execute_wick_term(plan, pool, operation) if batch_size == 1
                         else _execute_batched_wick_term(plan, pool, operation, batch_index))
            elif isinstance(operation, CorrelatorSumOperation):
                value = (_execute_correlator_sum(plan, pool, operation) if batch_size == 1
                         else _execute_batched_correlator_sum(plan, pool, operation))
            else:
                raise AssertionError(type(operation))
            view = pool.read(node.node_id) if wrote_directly else pool.write(node.node_id, value)
            assert view.data_ptr() == pointers[binding.buffer_id]
            owners[binding.buffer_id] = node.node_id
            resident.add(node.node_id)
            used_slots.add(binding.buffer_id)
            during = sum(plan.buffer(identity).capacity_bytes for identity in owners)
            peak = max(peak, during)
            assert step.peak_active_capacity_during_step == during
            assert pool.active_capacity_bytes == batch_size * during
            future = Counter(dependency for future_step in plan.steps[number + 1:]
                             for dependency in logical.node(future_step.operation.node_id).dependencies)
            releases = tuple(dependency for dependency in dict.fromkeys(node.dependencies)
                             if dependency not in logical.outputs.values() and not future[dependency])
            if node.node_id not in logical.outputs.values() and not future[node.node_id]:
                releases += (node.node_id,)
            assert step.released_values == releases
            for identity in releases:
                del owners[plan.binding(identity).buffer_id]
                resident.remove(identity)
                pool.release(identity)
                with pytest.raises(RuntimeError, match="inactive"):
                    pool.read(identity)
            after = sum(plan.buffer(identity).capacity_bytes for identity in owners)
            assert step.active_capacity_after == after
            assert pool.active_capacity_bytes == batch_size * after
            assert pool.active_owner == owners and pool.active_values == resident
        assert resident == set(logical.outputs.values())
        assert plan.statistics.peak_active_capacity_bytes == peak
        assert plan.statistics.final_active_capacity_bytes == after
        assert plan.statistics.buffer_reuses == len(plan.steps) - len(plan.buffers)
        outputs = {name: pool.read(identity).numpy().copy() for name, identity in logical.outputs.items()}
        snapshots.append(outputs)
        for position in range(batch_size):
            for name, actual in outputs.items():
                selected = actual if batch_size == 1 else actual[position]
                np.testing.assert_allclose(selected, references[position][name],
                                           rtol=5e-5 if dtype == "complex64" else 1e-12, atol=1e-6)
        assert pool.allocation_bytes == 3 * plan.statistics.total_buffer_capacity_bytes
    assert observed_reuses > 0
    for name in snapshots[0]:
        np.testing.assert_allclose(snapshots[0][name][0], snapshots[1][name],
                                   rtol=5e-5 if dtype == "complex64" else 1e-12, atol=1e-6)
    pool.reset()
    pool.close()
    assert pool.allocation_bytes == 0


def test_direct_physical_validation_checks_its_source_schedule():
    plan = _plan()
    steps = list(plan.source_schedule.steps)
    steps[0] = replace(steps[0], ready_width=steps[0].ready_width + 1)
    altered = replace(plan, source_schedule=replace(plan.source_schedule, steps=tuple(steps)))
    with pytest.raises(ValueError, match="ready width"):
        altered.validate()


@pytest.mark.parametrize("field", ("active_capacity_before", "peak_active_capacity_during_step",
                                  "active_capacity_after"))
def test_direct_physical_validation_replays_capacity_ledger(field):
    plan = _plan()
    steps = list(plan.steps)
    steps[0] = replace(steps[0], **{field: getattr(steps[0], field) + 16})
    statistics = replace(plan.statistics,
                         peak_active_capacity_bytes=max(step.peak_active_capacity_during_step for step in steps),
                         final_active_capacity_bytes=steps[-1].active_capacity_after)
    with pytest.raises(ValueError, match="capacity"):
        replace(plan, steps=tuple(steps), statistics=statistics).validate()


@pytest.mark.parametrize("field", ("scheduled_operations", "unique_buffers", "buffer_reuses",
                                  "requested_output_bytes"))
def test_direct_physical_validation_checks_all_summary_fields(field):
    plan = _plan()
    with pytest.raises(ValueError, match="statistic"):
        replace(plan, statistics=replace(plan.statistics, **{field: getattr(plan.statistics, field) + 1})).validate()


@pytest.mark.parametrize("metadata", ("primitive_value", "output_correlator", "unused_slot",
                                     "memory_space", "strategy", "none_reuse", "size_matched"))
def test_physical_validation_rejects_inconsistent_identity_and_policy(metadata):
    plan = _plan()
    if metadata == "primitive_value":
        steps = list(plan.steps)
        position = next(number for number, step in enumerate(steps)
                        if isinstance(step.operation, MaterializePrimitiveOperation))
        steps[position] = replace(steps[position], operation=replace(steps[position].operation, value_id="wrong"))
        altered = replace(plan, steps=tuple(steps))
    elif metadata == "output_correlator":
        key = next(iter(plan.outputs))
        altered = replace(plan, outputs={**plan.outputs, key: replace(plan.outputs[key], correlator_id="wrong")})
    elif metadata == "unused_slot":
        slot = PhysicalBufferSlot("unused", 16, "complex128")
        altered = replace(plan, buffers={**plan.buffers, slot.buffer_id: slot},
                          statistics=replace(plan.statistics, total_buffer_capacity_bytes=
                                             plan.statistics.total_buffer_capacity_bytes + slot.capacity_bytes))
    elif metadata == "memory_space":
        altered = replace(plan, memory_space="", buffers={key: replace(slot, memory_space="")
                                                         for key, slot in plan.buffers.items()})
    elif metadata == "size_matched":
        slot = next(iter(plan.buffers.values()))
        altered = replace(plan, reuse_strategy="size_matched",
                          buffers={**plan.buffers, slot.buffer_id: replace(slot, capacity_bytes=slot.capacity_bytes + 16)})
    else:
        altered = replace(plan, reuse_strategy="none" if metadata == "none_reuse" else "unknown")
    with pytest.raises(ValueError):
        altered.validate()


@pytest.mark.parametrize("value", (True, 128.0, float("nan"), float("inf")))
def test_physical_capacity_requires_an_integer(value):
    plan = _plan()
    slot = next(iter(plan.buffers.values()))
    altered = replace(plan, buffers={**plan.buffers, slot.buffer_id: replace(slot, capacity_bytes=value)})
    with pytest.raises(ValueError, match="capacity"):
        altered.validate()


@pytest.mark.parametrize("value", (True, 100000.0, float("nan"), float("inf")))
def test_physical_memory_budget_requires_an_integer(value):
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        build_physical_execution_plan(_plan().source_schedule, memory_limit_bytes=value)


def test_reused_slot_cannot_be_claimed_until_previous_owner_releases():
    plan = _plan()
    pool = RuntimeBufferPool(plan)
    grouped = {}
    for binding in plan.value_bindings.values():
        grouped.setdefault(binding.buffer_id, []).append(binding)
    pair = next(items[:2] for items in grouped.values() if len(items) > 1)
    first, second = pair
    pool.reserve(first.value_id).fill_(2 + 3j)
    with pytest.raises(RuntimeError, match="overwrite active"):
        pool.reserve(second.value_id)
    with pytest.raises(RuntimeError, match="values are active"):
        pool.activate_batch_size(1)
    with pytest.raises(RuntimeError, match="active values"):
        pool.close()
    np.testing.assert_array_equal(pool.read(first.value_id).numpy(), np.full(first.shape, 2 + 3j))
    pool.release(first.value_id)
    with pytest.raises(RuntimeError, match="inactive"):
        pool.release(first.value_id)
    pool.reserve(second.value_id)
    pool.reset()
    pool.close()


def test_writing_the_whole_pool_keeps_its_storage():
    pool = RuntimeBufferPool(_plan(), batch_size=2)
    pointers = {identity: storage.data_ptr() for identity, storage in pool.storage.items()}
    for storage in pool.storage.values():
        storage.fill_(1)
    pool.write_all()
    assert {identity: storage.data_ptr() for identity, storage in pool.storage.items()} == pointers
    assert all(not storage.any() for storage in pool.storage.values())
    assert pool.is_idle


@pytest.mark.parametrize("damage", ("overlap", "release", "input"))
def test_physical_alias_and_release_negative_controls(damage):
    plan = _plan()
    steps = list(plan.steps)
    if damage == "overlap":
        outputs = list(plan.outputs.values())
        binding = plan.binding(outputs[1].value_id)
        altered = replace(plan, value_bindings={**plan.value_bindings,
                          binding.value_id: replace(binding, buffer_id=outputs[0].buffer_id)})
    elif damage == "release":
        position = next(number for number, step in enumerate(steps) if step.released_buffer_ids)
        steps[position] = replace(steps[position], released_buffer_ids=("unknown",))
        altered = replace(plan, steps=tuple(steps))
    else:
        position = next(number for number, step in enumerate(steps) if getattr(step.operation, "inputs", ()))
        operation = steps[position].operation
        inputs = (replace(operation.inputs[0], buffer_id="unknown"), *operation.inputs[1:])
        steps[position] = replace(steps[position], operation=replace(operation, inputs=inputs))
        altered = replace(plan, steps=tuple(steps))
    with pytest.raises(ValueError):
        altered.validate()


@pytest.mark.parametrize("element_bytes", (8, 32))
def test_runtime_byte_model_is_checked_before_allocating_pool_storage(monkeypatch, element_bytes):
    plan = _plan(element_bytes=element_bytes)
    allocations = []
    original_empty = torch.empty

    def tracked_empty(*args, **kwargs):
        if args and args[0] != ():
            allocations.append(args[0])
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", tracked_empty)
    with pytest.raises(ValueError, match="runtime (dtype|element size)"):
        RuntimeBufferPool(plan)
    assert allocations == []


@pytest.mark.parametrize("record", ("slot_capacity", "binding_shape", "binding_size",
                                   "binding_production", "binding_release", "binding_retention",
                                   "step_index", "operation_step", "statistics"))
def test_numerically_equal_nonintegral_metadata_is_rejected(record):
    plan = _plan()
    if record == "slot_capacity":
        slot = next(iter(plan.buffers.values()))
        altered = replace(plan, buffers={**plan.buffers, slot.buffer_id:
                          replace(slot, capacity_bytes=float(slot.capacity_bytes))})
    elif record.startswith("binding_"):
        binding = next(value for value in plan.value_bindings.values()
                       if value.shape and value.release_after_step is not None)
        changes = {
            "binding_shape": {"shape": tuple(float(value) for value in binding.shape)},
            "binding_size": {"size_bytes": float(binding.size_bytes)},
            "binding_production": {"produced_at": float(binding.produced_at)},
            "binding_release": {"release_after_step": float(binding.release_after_step)},
            "binding_retention": {"retained_output": int(binding.retained_output)},
        }
        altered = replace(plan, value_bindings={**plan.value_bindings,
                          binding.value_id: replace(binding, **changes[record])})
    elif record == "statistics":
        altered = replace(plan, statistics=replace(plan.statistics,
                          scheduled_operations=float(plan.statistics.scheduled_operations)))
    else:
        steps = list(plan.steps)
        step = steps[0]
        changes = {"step_index": float(step.step_index)} if record == "step_index" else {
            "operation": replace(step.operation, step_index=float(step.operation.step_index))}
        steps[0] = replace(step, **changes)
        altered = replace(plan, steps=tuple(steps))
    with pytest.raises(ValueError, match="integer|boolean"):
        altered.validate()
