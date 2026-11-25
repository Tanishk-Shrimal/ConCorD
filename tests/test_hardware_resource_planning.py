"""Regression tests for hardware discovery and safe single-device planning."""

from __future__ import annotations

import pytest

from tests.plan_costs import _meson_correlator
from concord_lqcd.backend.runtime import (
    BatchPlanningError,
    HardwareBudgetPolicy,
    HardwareDeviceProfile,
    HardwareProfile,
    InsufficientDeviceMemoryError,
    MemorySafetyMargin,
    ResourcePlanningError,
    RuntimeCompilationConfig,
    compile_correlators,
    detect_hardware_profile,
    estimate_batched_execution_memory,
    estimate_execution_memory,
    plan_time_batch_candidates,
    plan_tiled_execution,
    plan_execution_resources,
    revalidate_execution_resources,
)
from concord_lqcd.backend.runtime import hardware as hardware_module
from concord_lqcd.backend.runtime.hardware import (
    _proc_cpu_model,
    _self_cgroup_memory_bytes,
    _usable_cpu_count,
)


@pytest.fixture(scope="module")
def meson_plan():
    compilation = compile_correlators(
        {"meson": _meson_correlator()},
        RuntimeCompilationConfig(mode_size=4),
    )
    return compilation.physical_plan


@pytest.fixture
def exact_policy():
    no_margin = MemorySafetyMargin(fraction=0.0, minimum_bytes=0)
    return HardwareBudgetPolicy(
        cpu_margin=no_margin,
        cuda_margin=no_margin,
    )


@pytest.fixture
def resident_only_policy():
    no_margin = MemorySafetyMargin(fraction=0.0, minimum_bytes=0)
    return HardwareBudgetPolicy(
        cpu_margin=no_margin,
        cuda_margin=no_margin,
        allow_host_tiling=False,
    )


def _cpu(*, total: int, available: int | None) -> HardwareDeviceProfile:
    return HardwareDeviceProfile(
        device="cpu",
        kind="cpu",
        name="Synthetic CPU",
        total_memory_bytes=total,
        available_memory_bytes=available,
        logical_processors=8,
    )


def _cuda(index: int, *, total: int, available: int) -> HardwareDeviceProfile:
    return HardwareDeviceProfile(
        device=f"cuda:{index}",
        kind="cuda",
        name=f"Synthetic GPU {index}",
        total_memory_bytes=total,
        available_memory_bytes=available,
        compute_capability=(9, 0),
        multiprocessor_count=100 + index,
    )


def _profile(
    host: HardwareDeviceProfile,
    *accelerators: HardwareDeviceProfile,
) -> HardwareProfile:
    return HardwareProfile(
        host=host,
        accelerators=tuple(accelerators),
        system="TestOS",
        machine="test-machine",
        torch_num_threads=4,
        torch_num_interop_threads=2,
    )


def test_linux_cpu_model_prefers_model_name(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor : 0\nmodel name : Example Research CPU 9000\n",
        encoding="utf-8",
    )

    assert _proc_cpu_model(cpuinfo) == "Example Research CPU 9000"


def test_execution_memory_estimate_includes_pool_and_largest_transient(meson_plan):
    estimate = estimate_execution_memory(meson_plan)

    assert estimate.buffer_pool_bytes == (
        meson_plan.statistics.total_buffer_capacity_bytes
    )
    assert estimate.largest_transient_value_bytes == max(
        binding.size_bytes for binding in meson_plan.value_bindings.values()
    )
    assert estimate.known_working_set_bytes == (
        estimate.buffer_pool_bytes + estimate.largest_transient_value_bytes
    )


def test_resident_time_batch_candidates_are_bounded_by_safe_memory(
    meson_plan,
    exact_policy,
):
    batch_four = estimate_batched_execution_memory(meson_plan, 4)
    profile = _profile(
        _cpu(
            total=2 * batch_four.known_peak_bytes,
            available=batch_four.known_peak_bytes,
        )
    )
    deployment = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="cpu",
        policy=exact_policy,
    )

    feasible = plan_time_batch_candidates(
        meson_plan,
        deployment,
        work_units=5,
    )
    explicit = plan_time_batch_candidates(
        meson_plan,
        deployment,
        work_units=5,
        requested_batch_size=3,
    )

    assert feasible.batch_sizes == (1, 2, 4)
    assert explicit.batch_sizes == (1, 3)
    assert batch_four.buffer_pool_bytes == (
        4 * estimate_execution_memory(meson_plan).buffer_pool_bytes
    )


def test_host_staged_time_batching_keeps_safe_sequential_fallback(
    meson_plan,
    exact_policy,
):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    profile = _profile(
        _cpu(total=4 * required, available=2 * required),
        _cuda(0, total=required, available=required - 1),
    )
    deployment = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="auto",
        policy=exact_policy,
    )
    assert deployment.execution_mode == "host_tiled"

    automatic = plan_time_batch_candidates(
        meson_plan,
        deployment,
        work_units=4,
    )
    assert automatic.batch_sizes == (1,)
    with pytest.raises(BatchPlanningError, match="unavailable"):
        plan_time_batch_candidates(
            meson_plan,
            deployment,
            work_units=4,
            requested_batch_size=2,
        )


def test_cpu_only_profile_selects_cpu_and_derives_buffer_budget(
    meson_plan,
    exact_policy,
):
    estimate = estimate_execution_memory(meson_plan)
    available = estimate.known_working_set_bytes + 4096
    profile = _profile(_cpu(total=2 * available, available=available))

    deployment = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="auto",
        policy=exact_policy,
    )

    assert deployment.selected_device == "cpu"
    assert deployment.selected_budget.fits is True
    assert deployment.selected_budget.recommended_buffer_limit_bytes == (
        available - estimate.largest_transient_value_bytes
    )
    assert deployment.selected_budget.headroom_after_known_plan_bytes == 4096


def test_auto_prefers_feasible_cuda_and_chooses_most_headroom(
    meson_plan,
    exact_policy,
):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    profile = _profile(
        _cpu(total=20 * required, available=10 * required),
        _cuda(0, total=4 * required, available=2 * required),
        _cuda(1, total=6 * required, available=3 * required),
    )

    deployment = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="auto",
        policy=exact_policy,
    )

    assert deployment.selected_device == "cuda:1"


def test_auto_uses_host_staged_cuda_when_pool_exceeds_vram(
    meson_plan,
    exact_policy,
):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    profile = _profile(
        _cpu(total=4 * required, available=2 * required),
        _cuda(0, total=required, available=required - 1),
    )

    deployment = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="auto",
        policy=exact_policy,
    )

    assert deployment.selected_device == "cuda:0"
    assert deployment.storage_device == "cpu"
    assert deployment.execution_mode == "host_tiled"
    assert deployment.tiled_plan is not None
    assert deployment.tiled_plan.peak_device_bytes <= required - 1
    assert deployment.memory_verified is True


def test_auto_derives_index_tiles_when_one_full_operation_exceeds_vram(
    meson_plan,
    exact_policy,
):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    full_operations = plan_tiled_execution(
        meson_plan,
        device_memory_limit_bytes=10**18,
    )
    device_limit = full_operations.full_peak_device_bytes - 1
    assert device_limit < required
    profile = _profile(
        _cpu(total=4 * required, available=2 * required),
        _cuda(0, total=2 * device_limit, available=device_limit),
    )

    deployment = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="auto",
        policy=exact_policy,
    )

    assert deployment.execution_mode == "host_tiled"
    assert deployment.tiled_plan is not None
    assert deployment.tiled_plan.uses_index_tiling
    assert deployment.tiled_plan.peak_device_bytes <= device_limit


def test_auto_falls_back_to_cpu_when_host_tiling_is_disabled(
    meson_plan,
    resident_only_policy,
):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    profile = _profile(
        _cpu(total=4 * required, available=2 * required),
        _cuda(0, total=required, available=required - 1),
    )

    deployment = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="auto",
        policy=resident_only_policy,
    )

    assert deployment.selected_device == "cpu"
    assert deployment.storage_device == "cpu"
    assert deployment.execution_mode == "resident"
    assert deployment.tiled_plan is None


def test_explicit_device_never_silently_falls_back(
    meson_plan,
    resident_only_policy,
):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    profile = _profile(
        _cpu(total=4 * required, available=2 * required),
        _cuda(0, total=required, available=required - 1),
    )

    with pytest.raises(
        InsufficientDeviceMemoryError,
        match="Explicit device requests do not fall back",
    ):
        plan_execution_resources(
            meson_plan,
            profile,
            requested_device="cuda:0",
            policy=resident_only_policy,
        )


def test_explicit_cuda_can_use_verified_host_staging(meson_plan, exact_policy):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    profile = _profile(
        _cpu(total=4 * required, available=2 * required),
        _cuda(0, total=required, available=required - 1),
    )

    deployment = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="cuda:0",
        policy=exact_policy,
    )

    assert deployment.selected_device == "cuda:0"
    assert deployment.storage_device == "cpu"
    assert deployment.execution_mode == "host_tiled"
    assert deployment.tiled_plan is not None


def test_explicit_unknown_memory_is_preserved_but_auto_requires_measurement(
    meson_plan,
    exact_policy,
):
    profile = _profile(_cpu(total=16 * 1024**3, available=None))

    explicit = plan_execution_resources(
        meson_plan,
        profile,
        requested_device="cpu",
        policy=exact_policy,
    )
    assert explicit.selected_device == "cpu"
    assert explicit.selected_budget.memory_verified is False

    with pytest.raises(ResourcePlanningError, match="measurable available-memory"):
        plan_execution_resources(
            meson_plan,
            profile,
            requested_device="auto",
            policy=exact_policy,
        )


def test_auto_reports_no_feasible_device_with_required_and_available_bytes(
    meson_plan,
    exact_policy,
):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    profile = _profile(
        _cpu(total=required, available=required - 2),
        _cuda(0, total=required, available=required - 1),
    )

    with pytest.raises(
        InsufficientDeviceMemoryError,
        match="known working set.*largest safe device limit",
    ):
        plan_execution_resources(
            meson_plan,
            profile,
            requested_device="auto",
            policy=exact_policy,
        )


def test_missing_explicit_device_and_unsupported_device_type_fail_clearly(
    meson_plan,
    exact_policy,
):
    profile = _profile(_cpu(total=8 * 1024**3, available=6 * 1024**3))

    with pytest.raises(ResourcePlanningError, match="was not detected"):
        plan_execution_resources(
            meson_plan,
            profile,
            requested_device="cuda:0",
            policy=exact_policy,
        )
    with pytest.raises(ResourcePlanningError, match="does not support device type"):
        plan_execution_resources(
            meson_plan,
            profile,
            requested_device="meta",
            policy=exact_policy,
        )


def test_safety_margin_is_retained_before_plan_feasibility(meson_plan):
    estimate = estimate_execution_memory(meson_plan)
    reserve = 1024
    available = estimate.known_working_set_bytes + reserve - 1
    margin = MemorySafetyMargin(fraction=0.0, minimum_bytes=reserve)
    policy = HardwareBudgetPolicy(cpu_margin=margin, cuda_margin=margin)
    profile = _profile(_cpu(total=2 * available, available=available))

    with pytest.raises(InsufficientDeviceMemoryError):
        plan_execution_resources(
            meson_plan,
            profile,
            requested_device="cpu",
            policy=policy,
        )


def test_kernel_batch_staging_policy_is_backend_specific_and_replaceable():
    cpu = _cpu(total=64 * 1024**3, available=48 * 1024**3)
    cuda = _cuda(0, total=80 * 1024**3, available=72 * 1024**3)
    policy = HardwareBudgetPolicy()

    assert policy.kernel_batch_staging_limit_for(cpu) == 16 * 1024**3
    assert policy.kernel_batch_staging_limit_for(cuda) == 32 * 1024**3

    unbounded = HardwareBudgetPolicy(
        cpu_kernel_batch_staging_limit_bytes=None,
        cuda_kernel_batch_staging_limit_bytes=None,
    )
    assert unbounded.kernel_batch_staging_limit_for(cpu) is None
    assert unbounded.kernel_batch_staging_limit_for(cuda) is None

    disabled = HardwareBudgetPolicy(
        cpu_kernel_batch_staging_limit_bytes=0,
        cuda_kernel_batch_staging_limit_bytes=0,
    )
    assert disabled.kernel_batch_staging_limit_for(cpu) == 0
    assert disabled.kernel_batch_staging_limit_for(cuda) == 0

    with pytest.raises(ValueError, match="cpu_kernel_batch"):
        HardwareBudgetPolicy(cpu_kernel_batch_staging_limit_bytes=-1)


def test_revalidation_rejects_memory_that_became_unavailable(
    meson_plan,
    exact_policy,
):
    required = estimate_execution_memory(meson_plan).known_working_set_bytes
    initial = _profile(_cpu(total=4 * required, available=2 * required))
    deployment = plan_execution_resources(
        meson_plan,
        initial,
        requested_device="auto",
        policy=exact_policy,
    )
    reduced = _profile(_cpu(total=4 * required, available=required - 1))

    with pytest.raises(InsufficientDeviceMemoryError, match="do not fall back"):
        revalidate_execution_resources(
            meson_plan,
            deployment,
            hardware=reduced,
        )


def test_cgroup_v2_limit_caps_host_memory_snapshot(tmp_path):
    membership = tmp_path / "cgroup"
    membership.write_text("0::/job/step\n", encoding="utf-8")
    directory = tmp_path / "mount" / "job" / "step"
    directory.mkdir(parents=True)
    (directory / "memory.max").write_text("10000\n", encoding="ascii")
    (directory / "memory.current").write_text("2500\n", encoding="ascii")

    assert _self_cgroup_memory_bytes(membership, tmp_path / "mount") == (
        10000,
        7500,
    )


def test_live_detector_returns_a_valid_cpu_snapshot_without_allocating_tensors():
    profile = detect_hardware_profile()

    assert profile.host.device == "cpu"
    assert profile.host.kind == "cpu"
    assert profile.host.logical_processors is not None
    assert profile.host.logical_processors > 0
    assert profile.torch_num_threads is not None
    assert tuple(device.device for device in profile.devices)[0] == "cpu"


def test_cpu_count_prefers_process_affinity_over_smaller_process_cpu_count(
    monkeypatch,
):
    monkeypatch.setattr(
        hardware_module.os,
        "sched_getaffinity",
        lambda process: set(range(64)),
        raising=False,
    )
    monkeypatch.setattr(
        hardware_module.os,
        "process_cpu_count",
        lambda: 1,
        raising=False,
    )

    assert _usable_cpu_count() == 64


def test_cpu_count_falls_back_when_affinity_is_unavailable(monkeypatch):
    def unavailable(process):
        raise OSError("affinity unavailable")

    monkeypatch.setattr(
        hardware_module.os,
        "sched_getaffinity",
        unavailable,
        raising=False,
    )
    monkeypatch.setattr(
        hardware_module.os,
        "process_cpu_count",
        lambda: 12,
        raising=False,
    )

    assert _usable_cpu_count() == 12
