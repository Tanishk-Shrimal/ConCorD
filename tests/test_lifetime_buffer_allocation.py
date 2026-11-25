"""Independent interval packing and conservative allocation selection checks."""

from types import SimpleNamespace

import numpy as np
import pytest

import concord_lqcd.backend.planner.physical as physical
from concord_lqcd.backend.planner.physical import (
    PhysicalBufferSlot,
    _LifetimeAllocationRequest,
    _allocate_lifetime_requests,
    _allocate_online_buffers,
    _allocation_peak_capacity,
    _select_buffer_allocation,
)


def _request(value_id, size, start, end, dtype="complex128"):
    return _LifetimeAllocationRequest(value_id, size, dtype, start, end)


def _schedule(requests):
    """Supply only allocator metadata, not a mock-certified compiler DAG."""

    descriptors = {request.value_id: request for request in requests}
    lifetimes = {
        request.value_id: SimpleNamespace(
            produced_at=request.produced_at,
            release_after_step=request.end_at if request.end_at < len(requests) else None,
        )
        for request in requests
    }
    return SimpleNamespace(
        order=tuple(request.value_id for request in requests),
        steps=tuple(
            SimpleNamespace(
                node_id=request.value_id,
                released_values=tuple(value.value_id for value in requests if value.end_at == step),
            )
            for step, request in enumerate(requests)
        ),
        source_analysis=SimpleNamespace(value=descriptors.__getitem__),
        lifetime=lifetimes.__getitem__,
        statistics=SimpleNamespace(peak_live_bytes=max((
            sum(value.size_bytes for value in requests if value.produced_at <= step <= value.end_at)
            for step in range(len(requests))
        ), default=0)),
    )


def _reference_packing(requests):
    """Scan every prior occupant independently of the binary-neighbor implementation."""

    slots = {}
    occupants = {}
    assignments = {}
    for request in sorted(requests, key=lambda item: (-item.size_bytes, item.produced_at, item.end_at, item.value_id)):
        available = [
            identity for identity, (capacity, dtype) in slots.items()
            if dtype == request.scalar_type and capacity >= request.size_bytes
            and all(request.end_at < previous.produced_at or previous.end_at < request.produced_at
                    for previous in occupants[identity])
        ]
        if available:
            identity = min(available, key=lambda item: (slots[item][0], item))
        else:
            identity = f"buffer_{len(slots)}"
            slots[identity] = (request.size_bytes, request.scalar_type)
            occupants[identity] = []
        occupants[identity].append(request)
        assignments[request.value_id] = identity
    return slots, assignments


def _metrics(requests, allocation):
    buffers, assignments = allocation
    total = sum(slot.capacity_bytes for slot in buffers.values())
    peak = max((
        sum(buffers[identity].capacity_bytes for identity in {
            assignments[request.value_id] for request in requests
            if request.produced_at <= step <= request.end_at
        })
        for step in range(len(requests))
    ), default=0)
    return total, peak


def test_largest_first_packing_avoids_small_long_lived_value_fragmentation():
    requests = (
        _request("first_large", 64, 0, 0),
        _request("small_live", 16, 1, 2),
        _request("last_large", 64, 2, 2),
    )
    schedule = _schedule(requests)
    online = _allocate_online_buffers(schedule, memory_space="cpu", reuse_strategy="best_fit")
    selected = _select_buffer_allocation(schedule, memory_space="cpu", reuse_strategy="best_fit", memory_limit_bytes=None)
    assert _metrics(requests, online) == (128, 128)
    assert _metrics(requests, selected) == (80, 80)
    assert selected[1]["first_large"] == selected[1]["last_large"]
    assert selected[1]["small_live"] != selected[1]["last_large"]
    assert all(slot.memory_space == "cpu" for slot in selected[0].values())


def test_closed_intervals_keep_same_step_output_separate_from_released_input():
    requests = (_request("input", 64, 0, 1), _request("output", 64, 1, 2))
    buffers, assignments = _allocate_lifetime_requests(requests, memory_space="cuda")
    assert len(buffers) == 2
    assert assignments["input"] != assignments["output"]


def test_packing_keeps_retained_outputs_and_different_scalar_types_separate():
    requests = (
        _request("retained", 64, 0, 5),
        _request("temporary", 64, 1, 1),
        _request("float_temporary", 64, 2, 2, "complex64"),
        _request("later", 64, 3, 3),
    )
    buffers, assignments = _allocate_lifetime_requests(requests, memory_space="default")
    assert assignments["temporary"] == assignments["later"]
    assert assignments["retained"] != assignments["later"]
    assert assignments["float_temporary"] != assignments["later"]
    assert len(buffers) == 3


@pytest.mark.parametrize("seed", range(24))
def test_binary_interval_insertion_matches_independent_all_occupant_reference(seed):
    random = np.random.default_rng(seed)
    requests = tuple(
        _request(
            f"value_{number}", int(random.integers(1, 12)) * 16,
            start := int(random.integers(0, 18)), int(random.integers(start, 22)),
            "complex128" if number % 3 else "complex64",
        )
        for number in range(30)
    )
    expected_slots, expected_assignments = _reference_packing(requests)
    buffers, assignments = _allocate_lifetime_requests(requests, memory_space="default")
    assert assignments == expected_assignments
    assert {identity: (slot.capacity_bytes, slot.scalar_type) for identity, slot in buffers.items()} == expected_slots
    reversed_allocation = _allocate_lifetime_requests(tuple(reversed(requests)), memory_space="default")
    assert reversed_allocation == (buffers, assignments)


@pytest.mark.parametrize("seed", range(24))
def test_default_selection_never_increases_pool_or_active_peak_and_keeps_ties(seed):
    random = np.random.default_rng(seed)
    requests = tuple(_request(f"value_{step}", int(random.integers(1, 12)) * 16, step,
                             int(random.integers(step, 19))) for step in range(18))
    schedule = _schedule(requests)
    online = _allocate_online_buffers(schedule, memory_space="default", reuse_strategy="best_fit")
    selected = _select_buffer_allocation(schedule, memory_space="default", reuse_strategy="best_fit", memory_limit_bytes=None)
    original_total, original_peak = _metrics(requests, online)
    selected_total, selected_peak = _metrics(requests, selected)
    assert selected_total <= original_total
    assert selected_peak <= original_peak
    assert _allocation_peak_capacity(schedule, *selected) == selected_peak
    if (selected_total, selected_peak) == (original_total, original_peak):
        assert selected == online


@pytest.mark.parametrize("policy", ("size_matched", "none"))
def test_other_reuse_policies_never_invoke_lifetime_packing(monkeypatch, policy):
    requests = (_request("large", 64, 0, 0), _request("small", 16, 1, 1))
    schedule = _schedule(requests)
    expected = _allocate_online_buffers(schedule, memory_space="default", reuse_strategy=policy)

    def forbidden(*arguments, **keywords):
        raise AssertionError("lifetime candidate should not be constructed")

    monkeypatch.setattr(physical, "_allocate_lifetime_requests", forbidden)
    actual = _select_buffer_allocation(schedule, memory_space="default", reuse_strategy=policy, memory_limit_bytes=None)
    assert actual == expected


def test_hard_budget_accepts_lifetime_pool_when_online_pool_does_not_fit():
    requests = (_request("large", 64, 0, 0), _request("small", 16, 1, 2), _request("later", 64, 2, 2))
    selected = _select_buffer_allocation(_schedule(requests), memory_space="default", reuse_strategy="best_fit", memory_limit_bytes=80)
    assert _metrics(requests, selected)[0] == 80
    with pytest.raises(MemoryError, match="smallest considered pool requires 80"):
        _select_buffer_allocation(_schedule(requests), memory_space="default", reuse_strategy="best_fit", memory_limit_bytes=79)


def test_logical_peak_lower_bound_skips_unnecessary_lifetime_search(monkeypatch):
    requests = tuple(_request(f"retained_{step}", 64, step, 12) for step in range(12))
    schedule = _schedule(requests)
    online = _allocate_online_buffers(schedule, memory_space="default", reuse_strategy="best_fit")
    assert _metrics(requests, online) == (768, 768)

    def forbidden(*arguments, **keywords):
        raise AssertionError("online pool already meets the logical peak lower bound")

    monkeypatch.setattr(physical, "_allocate_lifetime_requests", forbidden)
    selected = _select_buffer_allocation(schedule, memory_space="default", reuse_strategy="best_fit", memory_limit_bytes=768)
    assert selected == online
    with pytest.raises(MemoryError, match="smallest considered pool requires 768"):
        _select_buffer_allocation(schedule, memory_space="default", reuse_strategy="best_fit", memory_limit_bytes=767)


def test_compiled_tetraquark_pool_fits_a_budget_the_online_assignment_exceeds():
    from tests.plan_costs import _tetraquark_correlator
    from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators

    compiled = compile_correlators(
        {"tetraquark": _tetraquark_correlator()},
        RuntimeCompilationConfig(mode_size=45, shared_orthonormal_laph_basis=True),
    )
    online = _allocate_online_buffers(compiled.schedule, memory_space="default", reuse_strategy="best_fit")
    online_total = sum(slot.capacity_bytes for slot in online[0].values())
    budget = compiled.physical_plan.statistics.total_buffer_capacity_bytes
    assert budget < online_total
    admitted = physical.build_physical_execution_plan(compiled.schedule, memory_limit_bytes=budget)
    admitted.validate()
    assert admitted.source_schedule is compiled.schedule
    assert admitted.statistics.total_buffer_capacity_bytes == budget
    assert admitted.statistics.peak_active_capacity_bytes <= _allocation_peak_capacity(compiled.schedule, *online)
    with pytest.raises(MemoryError, match="exceeds memory_limit_bytes"):
        physical.build_physical_execution_plan(compiled.schedule, memory_limit_bytes=budget - 1)


def test_higher_peak_candidate_is_only_admitted_to_satisfy_a_hard_pool_budget(monkeypatch):
    requests = (
        _request("large", 64, 0, 0),
        _request("small_first", 16, 1, 3),
        _request("small_second", 16, 2, 3),
        _request("consumer", 16, 3, 3),
    )
    schedule = _schedule(requests)
    online = (
        {f"buffer_{number}": PhysicalBufferSlot(f"buffer_{number}", capacity, "complex128")
         for number, capacity in enumerate((64, 16, 16, 16))},
        {request.value_id: f"buffer_{number}" for number, request in enumerate(requests)},
    )
    lifetime = (
        {f"buffer_{number}": PhysicalBufferSlot(f"buffer_{number}", capacity, "complex128")
         for number, capacity in enumerate((64, 16, 16))},
        {"large": "buffer_0", "small_first": "buffer_0", "small_second": "buffer_1", "consumer": "buffer_2"},
    )
    assert _metrics(requests, online) == (112, 64)
    assert _metrics(requests, lifetime) == (96, 96)
    monkeypatch.setattr(physical, "_allocate_online_buffers", lambda *arguments, **keywords: online)
    monkeypatch.setattr(physical, "_allocate_lifetime_requests", lambda *arguments, **keywords: lifetime)
    selected = _select_buffer_allocation(schedule, memory_space="default", reuse_strategy="best_fit", memory_limit_bytes=None)
    assert selected == online
    selected = _select_buffer_allocation(schedule, memory_space="default", reuse_strategy="best_fit", memory_limit_bytes=96)
    assert selected == lifetime
