"""The limit for separately mapped blocks around kernel building."""

from __future__ import annotations

import pytest

import concord_lqcd.backend.runtime.heap as runtime_heap


@pytest.fixture
def mallopt_calls(monkeypatch):
    """Replace the C library call, so that this process keeps its own setting."""

    calls: list[tuple[int, int]] = []

    def mallopt(option: int, value: int) -> int:
        calls.append((option, value))
        return 1

    monkeypatch.setattr(runtime_heap, "libc_function", lambda name: mallopt)
    monkeypatch.setattr(runtime_heap, "_limit_fixed", False)
    return calls


def test_kernel_building_changes_nothing_in_a_process_that_did_not_ask(mallopt_calls):
    with runtime_heap.kernel_building():
        pass
    assert mallopt_calls == []


def test_kernel_building_lowers_the_fixed_limit_and_sets_it_back(mallopt_calls):
    high = (-3, runtime_heap.BLOCK_LIMIT_BYTES)
    low = (-3, runtime_heap.KERNEL_BUILDING_BLOCK_LIMIT_BYTES)

    assert runtime_heap.fix_block_limit()
    # With the limit, the free space kept at the top of the heap is fixed.
    fixed = [high, (-1, runtime_heap.HEAP_TOP_KEPT_BYTES)]
    assert mallopt_calls == fixed
    with runtime_heap.kernel_building():
        assert mallopt_calls == fixed + [low]
    assert mallopt_calls == fixed + [low, high]

    # Also when building the kernel fails.
    with pytest.raises(ValueError):
        with runtime_heap.kernel_building():
            raise ValueError("no kernel")
    assert mallopt_calls == fixed + [low, high, low, high]


def test_the_limit_is_not_fixed_without_the_c_library_call(monkeypatch):
    monkeypatch.setattr(runtime_heap, "libc_function", lambda name: None)
    monkeypatch.setattr(runtime_heap, "_limit_fixed", False)

    assert not runtime_heap.fix_block_limit()
    with runtime_heap.kernel_building():
        pass
