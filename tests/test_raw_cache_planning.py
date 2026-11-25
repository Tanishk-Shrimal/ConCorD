"""Raw-field cache admission respects time batches and shared memory limits."""

from dataclasses import replace
from math import prod

import h5py
import numpy as np
import pytest

import concord_lqcd.job.runner as job_runner
from concord_lqcd.backend.runtime.hardware import HardwareDeviceProfile, HardwareProfile
from concord_lqcd.backend.runtime.resource_planning import (
    HardwareBudgetPolicy,
    MemorySafetyMargin,
    resolve_runtime_device,
)
from concord_lqcd.backend.runtime.batching import estimate_batched_execution_memory
from concord_lqcd.job import JobSpec, compile_job_plan, run_job
from concord_lqcd.job._raw_cache_planning import plan_raw_eigenvector_cache
from concord_lqcd.job.data import (
    HDF5DistillationKernelProvider,
    _build_job_configuration_runtime_resources,
)
from tests.test_job_runner import _spec, _write_data


def _matrix_spec(tmp_path, *, time_batch_size=2):
    spec = _spec(tmp_path, time_batch_size=time_batch_size, momentum=(1, 0, 0))
    spec.operators.file.write_text(
        "\n".join((
            "from concord_lqcd.core import Correlator, Operator, c, cbar, u, ubar, gamma5, shift",
            "def build():",
            "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * u('y','a','s2'), momentum=(1,0,0))",
            "    moved = Operator(cbar('y','a','s1') * gamma5('s1','s2') * shift(u('y','a','s2'), '+x'), momentum=(1,0,0))",
            "    source = Operator(ubar('x','b','s3') * gamma5('s3','s4') * c('x','b','s4'), momentum=(-1,0,0))",
            "    return {'local': Correlator(sink, source), 'shifted': Correlator(moved, source)}",
        )),
        encoding="utf-8",
    )
    return spec


def _hardware(available=8 * 1024**3):
    return HardwareProfile(host=HardwareDeviceProfile(
        device="cpu", kind="cpu", name="Cache budget test CPU",
        total_memory_bytes=8 * 1024**3,
        available_memory_bytes=available,
        logical_processors=2,
    ))


@pytest.fixture
def cache_inputs(tmp_path):
    spec = _matrix_spec(tmp_path)
    plan = compile_job_plan(spec)
    deployment = resolve_runtime_device(
        plan.physical_plan, "cpu", hardware=_hardware(),
    )
    resources = _build_job_configuration_runtime_resources(
        spec, plan, configuration=0, time_contexts=spec.time_contexts,
        execution_plan=deployment,
    )
    field_bytes = prod(spec.lattice.spatial_extent) * 3 * plan.config.mode_size * plan.config.element_bytes
    return spec, plan, deployment, resources, field_bytes


def _admission(cache_inputs, *, fields=2, automatic=True, resources=None, deployment=None):
    _, plan, original_deployment, original_resources, field_bytes = cache_inputs
    if deployment is None:
        deployment = replace(
            original_deployment,
            policy=replace(
                original_deployment.policy,
                raw_eigenvector_cache_limit_bytes=fields * field_bytes,
            ),
        )
    return plan_raw_eigenvector_cache(
        original_resources if resources is None else resources,
        plan.physical_plan, deployment,
        batch_size=2, automatic_batch=automatic,
        element_bytes=plan.config.element_bytes,
    )


def test_automatic_batch_fits_complete_reused_working_set(cache_inputs):
    result = _admission(cache_inputs)
    assert result.batch_size == 1
    assert result.reserved_bytes == 2 * cache_inputs[-1]
    assert "reduced from 2 to 1" in result.warning


def test_larger_cache_preserves_requested_batch(cache_inputs):
    result = _admission(cache_inputs, fields=3)
    assert result.batch_size == 2
    assert result.reserved_bytes == 3 * cache_inputs[-1]
    assert result.warning is None


def test_explicit_batch_is_not_silently_reduced(cache_inputs):
    result = _admission(cache_inputs, automatic=False)
    assert result.batch_size == 2
    assert result.reserved_bytes == 2 * cache_inputs[-1]
    assert "keeping the explicit batch unchanged" in result.warning


def test_repeated_times_do_not_reserve_duplicate_fields(cache_inputs):
    resources = (cache_inputs[3][0], cache_inputs[3][0])
    result = _admission(cache_inputs, resources=resources)
    assert result.batch_size == 2
    assert result.reserved_bytes == 2 * cache_inputs[-1]
    assert result.warning is None


def test_partial_resume_uses_only_pending_time_contexts(cache_inputs):
    _, plan, deployment, resources, field_bytes = cache_inputs
    result = plan_raw_eigenvector_cache(
        resources[1:], plan.physical_plan,
        replace(deployment, policy=replace(
            deployment.policy, raw_eigenvector_cache_limit_bytes=2 * field_bytes,
        )),
        batch_size=1, automatic_batch=True,
        element_bytes=plan.config.element_bytes,
    )
    assert result.batch_size == 1
    assert result.reserved_bytes == 2 * field_bytes


@pytest.mark.parametrize("maximum_bytes", [0, 1])
def test_disabled_or_subfield_cache_does_not_limit_batch(cache_inputs, maximum_bytes):
    deployment = cache_inputs[2]
    deployment = replace(deployment, policy=replace(
        deployment.policy, raw_eigenvector_cache_limit_bytes=maximum_bytes,
    ))
    result = _admission(cache_inputs, deployment=deployment)
    assert result.batch_size == 2
    assert result.reserved_bytes == 0
    assert result.warning is None


def test_reservation_rounds_down_to_complete_fields(cache_inputs):
    field_bytes = cache_inputs[-1]
    deployment = cache_inputs[2]
    deployment = replace(deployment, policy=replace(
        deployment.policy,
        raw_eigenvector_cache_limit_bytes=3 * field_bytes - 1,
    ))
    result = _admission(cache_inputs, deployment=deployment)
    assert result.batch_size == 1
    assert result.reserved_bytes == 2 * field_bytes


def test_safe_memory_headroom_also_bounds_raw_storage(cache_inputs):
    _, plan, _, _, field_bytes = cache_inputs
    peak = estimate_batched_execution_memory(plan.physical_plan, 2).known_peak_bytes
    policy = HardwareBudgetPolicy(
        cpu_margin=MemorySafetyMargin(0.0, 0),
        raw_eigenvector_cache_limit_bytes=3 * field_bytes,
    )
    deployment = resolve_runtime_device(
        plan.physical_plan, "cpu", hardware=_hardware(peak + 2 * field_bytes),
        policy=policy,
    )
    result = _admission(cache_inputs, deployment=deployment)
    final_peak = estimate_batched_execution_memory(
        plan.physical_plan, result.batch_size,
    ).known_peak_bytes
    assert result.batch_size == 1
    assert final_peak + result.reserved_bytes <= peak + 2 * field_bytes
    assert result.reserved_bytes <= policy.raw_eigenvector_cache_limit_bytes


def test_unknown_headroom_disables_raw_admission(cache_inputs):
    plan = cache_inputs[1]
    deployment = resolve_runtime_device(
        plan.physical_plan, "cpu", hardware=_hardware(None),
    )
    result = _admission(cache_inputs, deployment=deployment)
    assert result.batch_size == 2
    assert result.reserved_bytes == 0


def test_single_kernel_per_time_does_not_impose_cache_batch_limit(tmp_path):
    spec = _spec(tmp_path, momentum=(1, 0, 0))
    plan = compile_job_plan(spec)
    deployment = resolve_runtime_device(
        plan.physical_plan, "cpu", hardware=_hardware(),
        policy=HardwareBudgetPolicy(raw_eigenvector_cache_limit_bytes=384),
    )
    resources = _build_job_configuration_runtime_resources(
        spec, plan, configuration=0, time_contexts=spec.time_contexts,
        execution_plan=deployment,
    )
    result = plan_raw_eigenvector_cache(
        resources, plan.physical_plan, deployment,
        batch_size=2, automatic_batch=True, element_bytes=16,
    )
    assert result.batch_size == 2
    assert result.warning is None


def test_custom_provider_subclasses_keep_existing_path(cache_inputs):
    class CustomProvider(HDF5DistillationKernelProvider):
        pass

    resources = cache_inputs[3]
    original = resources[0].kernel_providers["distillation"]
    provider = CustomProvider(
        original.spec, configuration=0, spatial_extent=original.spatial_extent,
        temporal_extent=original.temporal_extent, mode_size=original.mode_size,
        device="cpu",
    )
    custom_resources = tuple(replace(
        resource, kernel_providers={"distillation": provider},
    ) for resource in resources)
    result = _admission(cache_inputs, resources=custom_resources)
    assert result.batch_size == 2
    assert result.provider_limits == ()


@pytest.mark.parametrize("scalar_type, cache_mib, expected_batch, retained_mib", [
    ("complex128", 256, 4, 243),
    ("complex128", 512, 8, 324),
    ("complex64", 256, 8, 162),
])
def test_nv64_eight_sink_working_set_regression(
    tmp_path, scalar_type, cache_mib, expected_batch, retained_mib,
):
    spec = _matrix_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["lattice"] = {"spatial_extent": [24, 24, 24], "temporal_extent": 8}
    payload["compile"]["mode_size"] = 64
    payload["compile"]["scalar_type"] = scalar_type
    payload["run"]["time_batch_size"] = "auto"
    payload["run"]["times"]["generator"]["source"] = 0
    payload["run"]["times"]["generator"]["sinks"] = {"values": list(range(8))}
    spec = JobSpec.model_validate(payload)
    plan = compile_job_plan(spec)
    deployment = resolve_runtime_device(
        plan.physical_plan, "cpu", hardware=_hardware(),
        policy=HardwareBudgetPolicy(raw_eigenvector_cache_limit_bytes=cache_mib * 1024**2),
    )
    resources = _build_job_configuration_runtime_resources(
        spec, plan, configuration=0, time_contexts=spec.time_contexts,
        execution_plan=deployment,
    )
    result = plan_raw_eigenvector_cache(
        resources, plan.physical_plan, deployment,
        batch_size=8, automatic_batch=True, element_bytes=plan.config.element_bytes,
    )
    assert result.batch_size == expected_batch
    assert result.reserved_bytes == retained_mib * 1024**2


def test_final_short_batch_is_included(cache_inputs):
    resources = cache_inputs[3]
    result = _admission(
        cache_inputs, fields=3, resources=(resources[0], resources[1], resources[0]),
    )
    assert result.batch_size == 2
    assert result.reserved_bytes == 3 * cache_inputs[-1]


def test_distinct_providers_share_one_total_cap(cache_inputs):
    resources = cache_inputs[3]
    original = resources[0].kernel_providers["distillation"]
    other = HDF5DistillationKernelProvider(
        original.spec, configuration=0, spatial_extent=original.spatial_extent,
        temporal_extent=original.temporal_extent, mode_size=original.mode_size,
        device="cpu",
    )
    split_resources = (
        replace(resources[0], kernel_providers={"distillation": original, "alias": original}),
        replace(resources[1], kernel_providers={"distillation": other}),
    )
    result = _admission(cache_inputs, fields=4, resources=split_resources)
    assert result.batch_size == 2
    assert len(result.provider_limits) == 2
    assert result.reserved_bytes == 4 * cache_inputs[-1]
    assert all(limit == 2 * cache_inputs[-1] for _, limit in result.provider_limits)


def test_cuda_metadata_uses_device_headroom_not_free_host_ram(cache_inputs):
    _, plan, _, _, field_bytes = cache_inputs
    peak = estimate_batched_execution_memory(plan.physical_plan, 2).known_peak_bytes
    available = peak + 2 * field_bytes
    profile = HardwareProfile(
        host=_hardware().host,
        accelerators=(HardwareDeviceProfile(
            device="cuda:0", kind="cuda", name="Synthetic CUDA memory domain",
            total_memory_bytes=8 * 1024**3, available_memory_bytes=available,
        ),),
    )
    deployment = resolve_runtime_device(
        plan.physical_plan, "cuda:0", hardware=profile,
        policy=HardwareBudgetPolicy(
            cuda_margin=MemorySafetyMargin(0.0, 0),
            raw_eigenvector_cache_limit_bytes=3 * field_bytes,
        ),
    )
    result = _admission(cache_inputs, deployment=deployment)
    assert result.batch_size == 1
    final_peak = estimate_batched_execution_memory(plan.physical_plan, 1).known_peak_bytes
    assert result.reserved_bytes + final_peak <= available


@pytest.mark.parametrize("automatic, sink_count, cache_fields, expected_batch", [
    (False, 2, 2, 2),
    (True, 2, 2, 1),
    (False, 6, 4, 6),
    (True, 6, 4, 4),
])
def test_runner_outputs_and_shared_memory_accounting(
    tmp_path, monkeypatch, automatic, sink_count, cache_fields, expected_batch,
):
    spec = _matrix_spec(tmp_path, time_batch_size="auto" if automatic else sink_count)
    if sink_count > 2:
        payload = spec.model_dump(mode="python")
        payload["run"]["times"]["generator"]["sinks"] = {"values": list(range(sink_count))}
        spec = JobSpec.model_validate(payload)
    _write_data(tmp_path)
    random = np.random.default_rng(20261020)
    for path in sorted(tmp_path.glob("*.h5")):
        with h5py.File(path, "r+") as handle:
            if path.name.startswith("peram_"):
                shape = handle["tau_1"].shape
                for time in range(spec.lattice.temporal_extent):
                    if f"tau_{time}" not in handle:
                        handle.create_dataset(f"tau_{time}", shape=shape, dtype=np.complex128)
            for dataset in handle.values():
                values = random.normal(size=dataset.shape) + 1j * random.normal(size=dataset.shape)
                if dataset.name == "/evec":
                    for time in range(values.shape[0]):
                        orthonormal, _ = np.linalg.qr(values[time].reshape(-1, values.shape[-1]))
                        values[time] = orthonormal.reshape(values[time].shape)
                dataset[...] = values
    plan = compile_job_plan(spec)
    field_bytes = prod(spec.lattice.spatial_extent) * 3 * plan.config.mode_size * plan.config.element_bytes
    select = job_runner.select_time_batch_size
    reads = []
    read = h5py.Dataset.__getitem__

    def record_read(dataset, selection):
        result = read(dataset, selection)
        if dataset.name == "/evec":
            reads.append(np.asarray(result).nbytes)
        return result

    def select_batch(current, *arguments, **keywords):
        configured = current.model_copy(update={
            "run": current.run.model_copy(update={"time_batch_size": sink_count}),
        })
        return select(configured, *arguments, **keywords)

    monkeypatch.setattr(job_runner, "select_time_batch_size", select_batch)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_read)
    baseline_payload = spec.model_dump(mode="python")
    baseline_payload["output"]["file"] = tmp_path / "baseline.h5"
    baseline = JobSpec.model_validate(baseline_payload)
    run_job(
        baseline, plan, hardware=_hardware(),
        budget_policy=HardwareBudgetPolicy(raw_eigenvector_cache_limit_bytes=0),
    )
    baseline_read_bytes = sum(reads)
    reads.clear()

    statistics = []
    clear = HDF5DistillationKernelProvider.clear_raw_eigenvector_cache

    def record_clear(provider):
        statistics.append(provider.raw_eigenvector_cache_statistics)
        clear(provider)

    monkeypatch.setattr(HDF5DistillationKernelProvider, "clear_raw_eigenvector_cache", record_clear)
    policy = HardwareBudgetPolicy(raw_eigenvector_cache_limit_bytes=cache_fields * field_bytes)
    message = "automatic time batch reduced" if automatic else f"explicit time batch size {sink_count}"
    with pytest.warns(UserWarning, match=message):
        summary = run_job(spec, plan, hardware=_hardware(), budget_policy=policy)
    assert summary.all_complete
    assert summary.executed_work_units == sink_count * len(spec.configurations)
    assert summary.time_batch_size == expected_batch
    assert summary.time_batch_selection_source == ("cache_bounded" if automatic else "configured")
    if automatic:
        assert sum(reads) < baseline_read_bytes
        unique_times = {1, *(range(sink_count) if sink_count > 2 else (3, 4))}
        assert all(
            item.evictions <= len(unique_times) - cache_fields
            for item in statistics
        )
    for decision in summary.configuration_decisions:
        peak = estimate_batched_execution_memory(
            plan.physical_plan, decision.batch_size,
        ).known_peak_bytes
        reserved = (
            peak + decision.raw_eigenvector_cache_limit_bytes
            + decision.primitive_cache_plan.planned_bytes
            + decision.propagation_prefetch_plan.device_staging_bytes
        )
        assert decision.raw_eigenvector_cache_limit_bytes == cache_fields * field_bytes
        assert reserved + decision.kernel_batch_staging_limit_bytes <= 6 * 1024**3
    with h5py.File(baseline.output.file, "r") as expected, h5py.File(spec.output.file, "r") as actual:
        for name in expected["correlators"]:
            expected_values = expected[f"correlators/{name}/values"][...]
            actual_values = actual[f"correlators/{name}/values"][...]
            scale = np.max(np.abs(expected_values))
            assert np.max(np.abs(actual_values - expected_values)) / scale < 1e-12
