"""Focused tests for the production configuration/time job runner."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import concord_lqcd.backend.runtime.heap as runtime_heap
import concord_lqcd.job._workers as job_workers
import concord_lqcd.job.runner as job_runner
from concord_lqcd.backend.runtime import (
    HardwareBudgetPolicy,
    HardwareDeviceProfile,
    HardwareProfile,
)
from concord_lqcd.backend.runtime.numerical_executor import (
    GenericNumericalExecutor,
    RuntimeBufferPool,
)
from concord_lqcd.job import (
    AdaptiveCalibrationPolicy,
    JobDataError,
    JobRunError,
    JobRunRecord,
    JobSpec,
    TimeBatchCalibrationPolicy,
    build_job_runtime_resources,
    iter_job_work_units,
    load_concord_plan,
    run_job,
    write_job_plan,
)


T = 6
LX = 2
LY = 1
LZ = 1
NV = 2
CONFIGURATIONS = (0, 4)
SINKS = (3, 4)
SOURCE = 1


def _write_operator(path: Path, *, momentum=(0, 0, 0)) -> None:
    path.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, c, cbar, u, ubar, gamma5",
                "def build():",
                f"    p = {tuple(momentum)!r}",
                "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * u('y','a','s2'), momentum=p)",
                "    source = Operator(ubar('x','b','s3') * gamma5('s3','s4') * c('x','b','s4'), momentum=tuple(-v for v in p))",
                "    return {'D': Correlator(sink, source)}",
            ]
        ),
        encoding="utf-8",
    )


def _write_data(tmp_path: Path) -> None:
    volume_shape = (T, LZ, LY, LX, 3, NV)
    base_grid = np.arange(np.prod(volume_shape), dtype=np.float64).reshape(volume_shape)

    for configuration in CONFIGURATIONS:
        evec = (base_grid + 10 * configuration) / 31.0
        evec = evec + 1j * ((base_grid + configuration) % 7) / 17.0
        with h5py.File(tmp_path / f"evec_{configuration}.h5", "w") as handle:
            handle.create_dataset("evec", data=evec.astype(np.complex128))

        for flavor_index, flavor in enumerate(("u", "c")):
            with h5py.File(
                tmp_path / f"peram_{flavor}_{configuration}.h5", "w"
            ) as handle:
                # Both Wick orientations are possible, so every concrete source
                # time used by the two-point work grid is present.
                for source_time in (SOURCE,) + SINKS:
                    raw = np.arange(4 * 4 * NV * NV, dtype=np.float64).reshape(
                        4, 4, NV, NV
                    )
                    data = np.empty((T, 4, 4, NV, NV), dtype=np.complex128)
                    for sink_time in range(T):
                        offset = (
                            1000 * configuration
                            + 100 * (flavor_index + 1)
                            + 10 * source_time
                            + sink_time
                        )
                        data[sink_time] = (
                            offset + raw / 23.0 + 1j * (raw % 5) / 29.0
                        )
                    handle.create_dataset(f"tau_{source_time}", data=data)


def _spec(
    tmp_path: Path,
    *,
    on_existing: str = "resume",
    contraction_strategy: str = "greedy",
    time_batch_size: str | int = 1,
    momentum=(0, 0, 0),
    workers: int | str = 1,
    threads_per_worker: int | str = "auto",
) -> JobSpec:
    operator_file = tmp_path / "operators.py"
    _write_operator(operator_file, momentum=momentum)
    return JobSpec.model_validate(
        {
            "schema_version": 1,
            "job": {"name": "runner_test"},
            "operators": {"file": str(operator_file), "factory": "build"},
            "plan": {"file": str(tmp_path / "runner.concord")},
            "lattice": {
                "spatial_extent": [LX, LY, LZ],
                "temporal_extent": T,
            },
            "compile": {
                "mode_size": NV,
                "scalar_type": "complex128",
                "contraction": {"strategy": contraction_strategy},
            },
            "run": {
                "device": "cpu",
                "time_batch_size": time_batch_size,
                "workers": workers,
                "threads_per_worker": threads_per_worker,
                "configurations": {"values": list(CONFIGURATIONS)},
                "times": {
                    "generator": {
                        "type": "two_point",
                        "source": SOURCE,
                        "sinks": {"values": list(SINKS)},
                    }
                },
            },
            "data": {
                "eigenvectors": {
                    "provider": "hdf5_distillation",
                    "path": str(tmp_path / "evec_{configuration}.h5"),
                    "dataset": "/evec",
                    "axes": ["time", "z", "y", "x", "color", "mode"],
                },
                "propagation": {
                    "provider": "hdf5_perambulator",
                    "ordinary": {
                        "path": str(
                            tmp_path / "peram_{flavor}_{configuration}.h5"
                        ),
                        "dataset": "/tau_{source_time}",
                        "axes": [
                            "sink_time",
                            "sink_spin",
                            "source_spin",
                            "sink_mode",
                            "source_mode",
                        ],
                    },
                },
            },
            "output": {
                "file": str(tmp_path / "runner_result.h5"),
                "on_existing": on_existing,
            },
        }
    )


def _stored_output(path: Path, cfg_index: int, time_index: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        group = handle["correlators/0"]
        assert group.attrs["correlator_id"] == "D"
        return np.asarray(group["values"][cfg_index, time_index])


def test_work_unit_order_is_configuration_outer_time_inner(tmp_path):
    spec = _spec(tmp_path)
    units = tuple(iter_job_work_units(spec))
    assert [unit.key for unit in units] == [
        (0, "2pt_src_1_snk_3"),
        (0, "2pt_src_1_snk_4"),
        (4, "2pt_src_1_snk_3"),
        (4, "2pt_src_1_snk_4"),
    ]
    assert [(u.configuration_index, u.time_index) for u in units] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_run_job_loads_saved_plan_executes_grid_and_writes_incrementally(tmp_path):
    spec = _spec(tmp_path)
    _write_data(tmp_path)
    write_job_plan(spec)

    # No plan object is passed: the runner must use the saved .concord artifact.
    summary = run_job(spec)
    assert summary.total_work_units == 4
    assert summary.executed_work_units == 4
    assert summary.skipped_work_units == 0
    assert summary.completed_work_units == 4
    assert summary.all_complete
    assert summary.correlator_ids == ("D",)
    assert summary.output_file == spec.output.file
    assert summary.time_batch_size == 1
    assert summary.time_batch_selection_source == "configured"
    assert summary.executed_batches == 4
    assert summary.resident_pool_allocations == len(CONFIGURATIONS)
    assert summary.resident_pool_allocations < summary.executed_batches
    assert summary.resident_pool_peak_bytes > 0
    assert summary.primitive_cache_hits > 0
    assert summary.primitive_cache_misses > 0
    assert summary.primitive_cache_allocation_failures == 0
    assert summary.primitive_cache_peak_bytes > 0
    assert summary.hdf5_file_opens > 0
    assert summary.hdf5_file_reuses > 0
    assert summary.propagation_prefetch_configurations == len(CONFIGURATIONS)
    assert summary.propagation_prefetch_batches == summary.executed_batches
    assert summary.propagation_prefetch_tensors > 0
    assert summary.propagation_prefetch_bytes > 0
    assert summary.propagation_prefetch_wait_seconds >= 0
    assert summary.propagation_prefetch_allocation_fallbacks == 0
    assert summary.timing.execution_wall_seconds > 0.0
    assert sum(seconds for _, seconds in summary.timing.stages) == pytest.approx(
        summary.timing.execution_wall_seconds
    )
    assert summary.timing.numerical_execution_seconds > 0.0
    assert summary.timing.output_writing_seconds > 0.0
    assert tuple(
        decision.configuration for decision in summary.configuration_decisions
    ) == CONFIGURATIONS
    assert all(
        decision.primitive_cache_plan is not None
        and decision.propagation_prefetch_plan is not None
        and decision.propagation_prefetch_plan.enabled
        and decision.kernel_batch_staging_limit_bytes is not None
        and decision.kernel_batch_staging_limit_bytes >= 0
        for decision in summary.configuration_decisions
    )
    assert all(
        decision.kernel_batch_staging_limit_bytes <= 16 * 1024**3
        for decision in summary.configuration_decisions
    )

    loaded_plan = load_concord_plan(spec.plan.file)
    with h5py.File(spec.output.file, "r") as handle:
        assert handle.attrs["plan_sha256"] == loaded_plan.manifest.plan_sha256
        assert tuple(handle["work/configurations"][...]) == CONFIGURATIONS
        assert tuple(handle["completed"][...].reshape(-1)) == (True, True, True, True)
        assert set(handle["correlators"].keys()) == {"0"}
        assert handle["correlators/0"].attrs["correlator_id"] == "D"

    # Compare every persisted work unit against direct execution of the same
    # frozen plan/runtime resources.
    for cfg_index, configuration in enumerate(CONFIGURATIONS):
        for time_index, context in enumerate(spec.time_contexts):
            resources = build_job_runtime_resources(
                spec,
                loaded_plan,
                configuration=configuration,
                time_context=context,
            )
            direct = GenericNumericalExecutor(
                resources.materializer(loaded_plan),
                device=resources.storage_device,
                compute_device=resources.device,
                tiled_plan=resources.execution_plan.tiled_plan,
            ).execute().output("D")
            stored = _stored_output(spec.output.file, cfg_index, time_index)
            assert np.allclose(stored, direct.detach().cpu().numpy())


def test_runner_reports_persistent_kernel_cache_hits_and_builds(tmp_path):
    base_spec = _spec(tmp_path, momentum=(1, 0, 0))
    _write_data(tmp_path)
    payload = base_spec.model_dump(mode="python")
    payload["data"]["kernel_cache"] = {
        "directory": str(tmp_path / "kernel-cache")
    }
    cold_spec = JobSpec.model_validate(payload)
    plan = write_job_plan(cold_spec)

    cold = run_job(cold_spec, plan)

    assert cold.kernel_cache_enabled
    assert cold.kernel_cache_misses > 0
    assert cold.kernel_cache_writes > 0
    assert cold.kernel_cache_invalid_entries == 0
    assert cold.kernel_cache_write_failures == 0

    warm_payload = cold_spec.model_dump(mode="python")
    warm_payload["output"]["file"] = tmp_path / "warm_result.h5"
    warm_spec = JobSpec.model_validate(warm_payload)
    warm = run_job(warm_spec, plan)

    assert warm.kernel_cache_enabled
    assert warm.kernel_cache_hits > 0
    assert warm.kernel_cache_misses == 0
    assert warm.kernel_cache_writes == 0


def test_explicit_time_batch_matches_independent_sequential_execution(tmp_path):
    spec = _spec(tmp_path, time_batch_size=2)
    _write_data(tmp_path)
    plan = write_job_plan(spec)

    summary = run_job(spec, plan)

    assert summary.all_complete
    assert summary.time_batch_size == 2
    assert summary.time_batch_selection_source == "configured"
    assert summary.executed_work_units == 4
    assert summary.executed_batches == 2
    assert summary.propagation_prefetch_configurations == 0
    assert summary.propagation_prefetch_batches == 0
    assert len(summary.configuration_decisions) == len(CONFIGURATIONS)
    assert all(
        decision.propagation_prefetch_plan is not None
        and not decision.propagation_prefetch_plan.enabled
        and decision.propagation_prefetch_plan.reason
        and decision.kernel_batch_staging_limit_bytes is not None
        and decision.kernel_batch_staging_limit_bytes >= 0
        for decision in summary.configuration_decisions
    )
    for cfg_index, configuration in enumerate(CONFIGURATIONS):
        for time_index, context in enumerate(spec.time_contexts):
            resources = build_job_runtime_resources(
                spec,
                plan,
                configuration=configuration,
                time_context=context,
            )
            direct = GenericNumericalExecutor(
                resources.materializer(plan),
                device=resources.storage_device,
                compute_device=resources.device,
                tiled_plan=resources.execution_plan.tiled_plan,
            ).execute().output("D")
            torch.testing.assert_close(
                torch.from_numpy(_stored_output(spec.output.file, cfg_index, time_index)),
                direct.detach().cpu(),
                # A leading batch axis can select a different BLAS/einsum
                # reduction order from independent execution.  Use the same
                # complex128 agreement policy as runtime batch calibration.
                rtol=1e-10,
                atol=1e-11,
            )


def test_runner_applies_replaceable_kernel_batch_staging_cap(tmp_path):
    spec = _spec(tmp_path, time_batch_size=2)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    staging_cap = 64 * 1024**2
    policy = HardwareBudgetPolicy(
        cpu_kernel_batch_staging_limit_bytes=staging_cap,
    )

    summary = run_job(spec, plan, budget_policy=policy)

    assert summary.all_complete
    assert summary.configuration_decisions
    assert all(
        decision.kernel_batch_staging_limit_bytes == staging_cap
        for decision in summary.configuration_decisions
    )


def test_resident_runner_prepares_resources_once_per_configuration(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path, time_batch_size=2)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    original = job_runner._build_job_configuration_runtime_resources
    prepared_configurations = []

    def counted(*args, **kwargs):
        prepared_configurations.append(kwargs["configuration"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        job_runner,
        "_build_job_configuration_runtime_resources",
        counted,
    )

    summary = run_job(spec, plan)

    assert summary.all_complete
    assert prepared_configurations == list(CONFIGURATIONS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_resident_prefetch_matches_independent_cpu_execution(tmp_path):
    spec = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(spec)

    summary = run_job(spec, plan, device="cuda:0")
    torch.cuda.synchronize()

    assert summary.all_complete
    assert summary.selected_device == "cuda:0"
    assert summary.storage_device == "cuda:0"
    assert summary.execution_mode == "resident"
    assert summary.propagation_prefetch_configurations == len(CONFIGURATIONS)
    assert summary.propagation_prefetch_batches == summary.executed_batches
    assert summary.propagation_prefetch_tensors > 0
    assert summary.propagation_prefetch_allocation_fallbacks == 0
    assert all(
        decision.propagation_prefetch_plan is not None
        and decision.propagation_prefetch_plan.enabled
        for decision in summary.configuration_decisions
    )

    for configuration_index, configuration in enumerate(CONFIGURATIONS):
        for time_index, context in enumerate(spec.time_contexts):
            resources = build_job_runtime_resources(
                spec,
                plan,
                configuration=configuration,
                time_context=context,
                device="cpu",
            )
            reference = GenericNumericalExecutor(
                resources.materializer(plan),
                device=resources.storage_device,
                compute_device=resources.device,
                tiled_plan=resources.execution_plan.tiled_plan,
            ).execute().output("D")
            torch.testing.assert_close(
                torch.from_numpy(
                    _stored_output(
                        spec.output.file,
                        configuration_index,
                        time_index,
                    )
                ),
                reference.detach().cpu(),
                rtol=1e-10,
                atol=1e-11,
            )


def test_automatic_time_batch_calibrates_complete_feasible_set(tmp_path):
    spec = _spec(tmp_path, time_batch_size="auto")
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    profile = HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Time Batch Test CPU",
            total_memory_bytes=64 * 1024**3,
            available_memory_bytes=48 * 1024**3,
            logical_processors=8,
        ),
        system="TestOS",
        machine="time-batch-test-machine",
        torch_num_threads=4,
        torch_num_interop_threads=2,
    )
    cache = tmp_path / "time-batch-calibration.json"

    summary = run_job(
        spec,
        plan,
        device="cpu",
        hardware=profile,
        time_batch_calibration_policy=TimeBatchCalibrationPolicy(
            warmups=0,
            repeats=1,
        ),
        calibration_cache_path=cache,
    )

    assert summary.all_complete
    assert summary.time_batch_size in (1, 2)
    assert summary.time_batch_selection_source == "calibrated"
    assert summary.executed_batches in (2, 4)
    assert cache.is_file()


def test_automatic_time_batch_respects_estimated_calibration_budget(tmp_path):
    spec = _spec(tmp_path, time_batch_size="auto")
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    profile = HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Budget Test CPU",
            total_memory_bytes=64 * 1024**3,
            available_memory_bytes=48 * 1024**3,
            logical_processors=8,
        ),
        system="TestOS",
        machine="time-batch-budget-machine",
    )

    with pytest.warns(UserWarning, match="estimated wall-time budget"):
        summary = run_job(
            spec,
            plan,
            hardware=profile,
            time_batch_calibration_policy=TimeBatchCalibrationPolicy(
                max_estimated_seconds=1e-12,
            ),
            calibration_cache_path=tmp_path / "budgeted-calibration.json",
        )

    assert summary.all_complete
    assert summary.time_batch_size == 1
    assert summary.time_batch_selection_source == "calibrated"
    assert summary.executed_batches == 4


def test_automatic_time_batch_guard_preserves_sequential_fallback(tmp_path):
    spec = _spec(tmp_path, time_batch_size="auto")
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    profile = HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Guard Test CPU",
            total_memory_bytes=64 * 1024**3,
            available_memory_bytes=48 * 1024**3,
            logical_processors=8,
        ),
        system="TestOS",
        machine="time-batch-guard-machine",
    )

    with pytest.warns(UserWarning, match="allocation guard"):
        summary = run_job(
            spec,
            plan,
            hardware=profile,
            time_batch_calibration_policy=TimeBatchCalibrationPolicy(
                warmups=0,
                repeats=1,
                max_memory_bytes=1,
            ),
            calibration_cache_path=tmp_path / "guarded-calibration.json",
        )

    assert summary.all_complete
    assert summary.time_batch_size == 1
    assert summary.time_batch_selection_source == "sequential_fallback"
    assert summary.executed_batches == 4


def test_resume_skips_complete_work_units_and_reexecutes_only_incomplete_slot(tmp_path):
    spec = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    first = run_job(spec, plan)
    assert first.executed_work_units == 4

    second = run_job(spec, plan)
    assert second.executed_work_units == 0
    assert second.skipped_work_units == 4
    assert second.completed_work_units == 4

    # Simulate an interrupted slot: numerical bytes may be partial/stale, but
    # completed=False is authoritative and must force exactly that work unit to run.
    with h5py.File(spec.output.file, "r+") as handle:
        handle["completed"][0, 1] = False
        handle["correlators/0/values"][0, 1] = np.nan + 1j * np.nan

    resumed = run_job(spec, plan)
    assert resumed.executed_work_units == 1
    assert resumed.skipped_work_units == 3
    assert resumed.completed_work_units == 4
    assert resumed.all_complete
    assert np.isfinite(_stored_output(spec.output.file, 0, 1)).all()


def test_existing_output_plan_or_runtime_mismatch_is_rejected_before_resume(tmp_path):
    spec = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    run_job(spec, plan)

    with h5py.File(spec.output.file, "r+") as handle:
        original = handle.attrs["plan_sha256"]
        handle.attrs["plan_sha256"] = "0" * 64
    with pytest.raises(JobRunError, match="different compiled plan"):
        run_job(spec, plan)

    with h5py.File(spec.output.file, "r+") as handle:
        handle.attrs["plan_sha256"] = original
        handle.attrs["run_fingerprint"] = "0" * 64
    with pytest.raises(JobRunError, match="runtime specification differs"):
        run_job(spec, plan)


def test_existing_output_policy_error_and_overwrite(tmp_path):
    spec = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    run_job(spec, plan)

    error_spec = spec.model_copy(
        update={
            "output": spec.output.model_copy(update={"on_existing": "error"}),
        }
    )
    with pytest.raises(FileExistsError):
        run_job(error_spec, plan)

    overwrite_spec = spec.model_copy(
        update={
            "output": spec.output.model_copy(update={"on_existing": "overwrite"}),
        }
    )
    overwritten = run_job(overwrite_spec, plan)
    assert overwritten.executed_work_units == 4
    assert overwritten.skipped_work_units == 0
    assert overwritten.completed_work_units == 4


def test_device_override_does_not_change_resume_identity(tmp_path):
    # CPU is used for both invocations in this test environment; the important
    # contract is that the explicit runtime device is not part of result identity.
    spec = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    first = run_job(spec, plan, device="cpu")
    second = run_job(spec, plan, device=torch.device("cpu"))
    assert first.executed_work_units == 4
    assert second.executed_work_units == 0
    assert second.skipped_work_units == 4


def test_auto_device_executes_on_feasible_synthetic_cpu_profile(tmp_path):
    spec = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    profile = HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Test CPU",
            total_memory_bytes=64 * 1024**3,
            available_memory_bytes=48 * 1024**3,
            logical_processors=8,
        ),
        system="TestOS",
        machine="test-machine",
        torch_num_threads=4,
        torch_num_interop_threads=2,
    )

    summary = run_job(spec, plan, device="auto", hardware=profile)

    assert summary.all_complete
    assert summary.selected_device == "cpu"
    assert summary.storage_device == "cpu"
    assert summary.execution_mode == "resident"
    assert summary.hardware_profile is profile
    assert summary.memory_verified is True
    assert summary.known_working_set_bytes > 0
    assert summary.peak_tile_bytes is None
    assert summary.tiled_operations == 0


def test_adaptive_job_calibrates_and_reuses_device_local_selection(tmp_path):
    spec = _spec(tmp_path, contraction_strategy="adaptive")
    _write_data(tmp_path)
    portable_plan = write_job_plan(spec)
    assert portable_plan.contraction_order.strategy == "adaptive"
    profile = HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Test CPU",
            total_memory_bytes=64 * 1024**3,
            available_memory_bytes=48 * 1024**3,
            logical_processors=8,
        ),
        system="TestOS",
        machine="test-machine",
        torch_num_threads=4,
        torch_num_interop_threads=2,
    )
    cache = tmp_path / "adaptive-calibration.json"
    policy = AdaptiveCalibrationPolicy(warmups=0, repeats=1)

    first = run_job(
        spec,
        device="cpu",
        hardware=profile,
        adaptive_calibration_policy=policy,
        calibration_cache_path=cache,
    )
    second = run_job(
        spec,
        device="cpu",
        hardware=profile,
        adaptive_calibration_policy=policy,
        calibration_cache_path=cache,
    )

    assert first.all_complete
    assert first.contraction_path in ("greedy", "reuse-aware")
    assert first.contraction_selection_source == "calibrated"
    assert cache.is_file()
    assert second.executed_work_units == 0
    assert second.contraction_path == first.contraction_path
    assert second.contraction_selection_source == "calibrated"


def test_completed_resume_does_not_require_execution_hardware(tmp_path):
    spec = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    first = run_job(spec, plan, device="cpu")
    assert first.all_complete
    cpu_only = HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Test CPU",
            total_memory_bytes=8 * 1024**3,
            available_memory_bytes=4 * 1024**3,
            logical_processors=4,
        ),
        system="TestOS",
        machine="test-machine",
    )

    resumed = run_job(
        spec,
        plan,
        device="cuda:99",
        hardware=cpu_only,
    )

    assert resumed.executed_work_units == 0
    assert resumed.skipped_work_units == resumed.total_work_units
    assert resumed.selected_device == "not-required"
    assert resumed.storage_device == "not-required"
    assert resumed.execution_mode == "not-required"
    assert resumed.memory_verified is False
    assert resumed.time_batch_selection_source == "not-required"
    assert resumed.executed_batches == 0
    assert resumed.resident_pool_allocations == 0
    assert resumed.resident_pool_peak_bytes == 0
    assert resumed.primitive_cache_hits == 0
    assert resumed.primitive_cache_misses == 0
    assert resumed.primitive_cache_allocation_failures == 0
    assert resumed.primitive_cache_peak_bytes == 0
    assert resumed.hdf5_file_opens == 0
    assert resumed.propagation_prefetch_configurations == 0
    assert resumed.propagation_prefetch_batches == 0
    assert resumed.propagation_prefetch_tensors == 0
    assert resumed.propagation_prefetch_bytes == 0
    assert resumed.propagation_prefetch_wait_seconds == 0
    assert resumed.propagation_prefetch_allocation_fallbacks == 0
    assert resumed.hdf5_file_reuses == 0
    assert resumed.configuration_decisions == ()


# Worker tests use one thread per worker: earlier tests in this process may
# already have started OpenMP threads, which a forked child cannot use.


def _all_outputs(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        assert handle["completed"][...].all()
        return np.asarray(handle["correlators/0/values"][...])


def test_split_contiguous_gives_nearly_equal_ordered_shares():
    assert job_workers.split_contiguous(tuple(range(7)), 3) == ((0, 1, 2), (3, 4), (5, 6))
    assert job_workers.split_contiguous(tuple(range(2)), 5) == ((0,), (1,))
    with pytest.raises(ValueError):
        job_workers.split_contiguous((1,), 0)


def test_worker_cpu_blocks_are_spread_over_the_node(monkeypatch):
    # 64 cores in four memory domains of 16 consecutive CPUs, no hyperthreads.
    monkeypatch.setattr(job_workers, "usable_cpus", lambda: tuple(range(64)))
    monkeypatch.setattr(
        job_workers, "cpu_topology", lambda: {cpu: (cpu // 16, (cpu // 32, cpu)) for cpu in range(64)}
    )
    assert [cpus[0] for cpus in job_workers.worker_cpu_sets(12, 1)] == [
        0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 58
    ]
    assert job_workers.worker_cpu_sets(16, 4)[1] == (4, 5, 6, 7)
    assert job_workers.worker_cpu_sets(4, 16)[3] == tuple(range(48, 64))
    assert job_workers.worker_cpu_sets(3, 30) == (None, None, None)


def test_worker_cpu_blocks_use_whole_cores_before_hyperthreads(monkeypatch):
    # 24 cores with two hyperthreads each: CPUs n and n + 24 share a core; even
    # CPUs are in domain 0, odd CPUs in domain 1.
    monkeypatch.setattr(job_workers, "usable_cpus", lambda: tuple(range(48)))
    monkeypatch.setattr(
        job_workers, "cpu_topology", lambda: {cpu: (cpu % 2, (cpu % 2, cpu % 24)) for cpu in range(48)}
    )
    assert job_workers.worker_cpu_sets(2, 1) == ((0,), (1,))
    assert job_workers.worker_cpu_sets(2, 4) == ((0, 2, 4, 6), (1, 3, 5, 7))
    blocks = job_workers.worker_cpu_sets(48, 1)
    assert sorted(cpu for (cpu,) in blocks) == list(range(48))


def test_node_shape_counts_physical_cores_and_the_smallest_domain(monkeypatch):
    monkeypatch.setattr(job_workers, "usable_cpus", lambda: tuple(range(64)))
    monkeypatch.setattr(
        job_workers, "cpu_topology", lambda: {cpu: (cpu // 16, (cpu // 32, cpu)) for cpu in range(64)}
    )
    assert job_workers.node_shape() == (64, 16)
    # Cores 0-20 (even ones in domain 0, odd ones in domain 1) and CPUs 24 and
    # 25, the hyperthreads of cores 0 and 1.
    cpus = (*range(21), 24, 25)
    monkeypatch.setattr(job_workers, "usable_cpus", lambda: cpus)
    monkeypatch.setattr(
        job_workers, "cpu_topology", lambda: {cpu: (cpu % 2, (cpu % 2, cpu % 24)) for cpu in cpus}
    )
    assert job_workers.node_shape() == (21, 10)


def test_workers_write_the_same_results_as_one_process(tmp_path):
    single = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(single)
    run_job(single, plan)
    expected = _all_outputs(single.output.file)

    spec = _spec(tmp_path, workers=2, threads_per_worker=1).model_copy(
        update={
            "output": single.output.model_copy(
                update={"file": tmp_path / "workers_result.h5", "on_existing": "error"}
            )
        }
    )
    threads_before = torch.get_num_threads()
    summary = run_job(spec, plan)
    assert torch.get_num_threads() == threads_before
    np.testing.assert_allclose(_all_outputs(spec.output.file), expected, rtol=1e-13, atol=0)

    assert summary.workers == 2
    assert summary.threads_per_worker == 1
    assert summary.executed_work_units == summary.completed_work_units == 4
    assert summary.executed_batches == 4
    assert [report.work_units for report in summary.worker_reports] == [2, 2]
    assert all(report.threads == 1 for report in summary.worker_reports)
    assert all(report.timing.numerical_execution_seconds > 0 for report in summary.worker_reports)
    assert summary.selected_device == "cpu"
    assert len(summary.configuration_decisions) == 2
    assert sum(seconds for _, seconds in summary.timing.stages) == pytest.approx(
        summary.timing.execution_wall_seconds
    )

    record = JobRunRecord.from_summary(
        summary, job_name="runner_test", plan_action="reused", plan_preparation_seconds=0.0
    )
    assert JobRunRecord.from_dict(record.to_dict()).workers == summary.worker_reports


def test_workers_resume_only_incomplete_slots_and_skip_spare_workers(tmp_path):
    spec = _spec(tmp_path, workers=8, threads_per_worker=1)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    first = run_job(spec, plan)
    assert first.workers == 4  # one work unit each; no empty workers
    with h5py.File(spec.output.file, "r+") as handle:
        handle["completed"][1, 0] = False
        handle["correlators/0/values"][1, 0] = np.nan + 1j * np.nan

    resumed = run_job(spec, plan)
    assert resumed.executed_work_units == 1
    assert resumed.skipped_work_units == 3
    assert resumed.workers == 1
    assert resumed.all_complete
    assert np.isfinite(_stored_output(spec.output.file, 1, 0)).all()

    finished = run_job(spec, plan)
    assert finished.executed_work_units == 0
    assert finished.worker_reports == ()


def test_worker_failure_stops_the_run_with_the_original_error(tmp_path):
    spec = _spec(tmp_path, workers=2, threads_per_worker=1)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    (tmp_path / f"peram_u_{CONFIGURATIONS[1]}.h5").unlink()

    with pytest.raises((JobDataError, JobRunError)) as failure:
        run_job(spec, plan)
    assert any("traceback of worker 1" in note for note in failure.value.__notes__)
    with h5py.File(spec.output.file, "r") as handle:
        # The healthy worker's results are kept for resume.
        assert tuple(handle["completed"][0]) == (True, True)
        assert not handle["completed"][1].any()


def test_workers_that_cannot_start_threads_fail_instead_of_hanging(tmp_path, monkeypatch):
    spec = _spec(tmp_path, workers=2, threads_per_worker=2)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    torch.set_num_threads(2)
    torch.ones(1 << 20).sum()  # the parent now owns an OpenMP thread pool
    monkeypatch.setattr(job_workers, "READY_TIMEOUT_SECONDS", 5.0)

    with pytest.raises(JobRunError, match="did not start"):
        run_job(spec, plan)


def test_auto_workers_fall_back_to_one_thread_when_threaded_workers_cannot_start(
    tmp_path, monkeypatch
):
    spec = _spec(tmp_path, workers="auto")
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    torch.set_num_threads(2)
    torch.ones(1 << 20).sum()  # the parent now owns an OpenMP thread pool
    monkeypatch.setattr(job_workers, "READY_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(
        job_runner,
        "auto_worker_split",
        lambda **_: job_runner.WorkerSplit(2, 2, True, "auto: two workers"),
    )

    with pytest.warns(RuntimeWarning, match="one thread per worker"):
        summary = run_job(spec, plan)
    assert summary.all_complete
    assert (summary.workers, summary.threads_per_worker) == (2, 1)
    assert summary.worker_selection.endswith("threaded workers did not start")


def test_auto_workers_split_the_job_by_plan_cost_and_node_shape(tmp_path, monkeypatch):
    single = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(single)
    run_job(single, plan)
    expected = _all_outputs(single.output.file)
    spec = _spec(tmp_path, workers="auto").model_copy(
        update={
            "output": single.output.model_copy(
                update={"file": tmp_path / "auto_result.h5", "on_existing": "error"}
            )
        }
    )

    # A plan this small runs in one process, on one memory domain's cores.
    monkeypatch.setattr(job_runner, "node_shape", lambda: (8, 2))
    threads_before = torch.get_num_threads()
    small = run_job(spec, plan)
    assert torch.get_num_threads() == threads_before
    assert (small.workers, small.threads_per_worker) == (1, 2)
    assert small.worker_selection.startswith("auto: one process (4 pending work units")

    # Treated as a large job, it gets one worker per work unit, two cores each.
    monkeypatch.setattr("concord_lqcd.job._worker_split.MINIMUM_PARALLEL_SECONDS", 0.0)
    monkeypatch.setattr(job_runner, "node_shape", lambda: (8, 1))
    large_spec = spec.model_copy(
        update={"output": spec.output.model_copy(update={"file": tmp_path / "auto_large.h5"})}
    )
    large = run_job(large_spec, plan)
    assert (large.workers, large.threads_per_worker) == (4, 1)
    assert large.worker_selection.startswith("auto: 1 thread per worker")
    np.testing.assert_allclose(_all_outputs(large_spec.output.file), expected, rtol=1e-13, atol=0)
    record = JobRunRecord.from_summary(
        large, job_name="runner_test", plan_action="reused", plan_preparation_seconds=0.0
    )
    assert JobRunRecord.from_dict(record.to_dict()).worker_selection == large.worker_selection


def _small_node(available_gib: int) -> HardwareProfile:
    return HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Test CPU",
            total_memory_bytes=available_gib * 1024**3,
            available_memory_bytes=available_gib * 1024**3,
            logical_processors=8,
        ),
        system="TestOS",
        machine="test-machine",
    )


def test_auto_workers_measure_their_memory_first_when_it_may_not_fit(tmp_path, monkeypatch):
    single = _spec(tmp_path)
    _write_data(tmp_path)
    plan = write_job_plan(single)
    run_job(single, plan)
    expected = _all_outputs(single.output.file)
    spec = _spec(tmp_path, workers="auto").model_copy(
        update={
            "output": single.output.model_copy(
                update={"file": tmp_path / "measured.h5", "on_existing": "error"}
            )
        }
    )
    monkeypatch.setattr("concord_lqcd.job._worker_split.MINIMUM_PARALLEL_SECONDS", 0.0)
    monkeypatch.setattr(job_runner, "node_shape", lambda: (8, 1))
    pool_writes = tmp_path / "pool_writes"
    write_all = RuntimeBufferPool.write_all

    def recorded_write_all(pool: RuntimeBufferPool) -> None:
        # Runs in the forked workers, so it reports through a file.
        with pool_writes.open("a") as out:
            out.write(f"{os.getpid()}\n")
        write_all(pool)

    monkeypatch.setattr(RuntimeBufferPool, "write_all", recorded_write_all)
    limit_fixes = tmp_path / "limit_fixes"
    fix_block_limit = job_runner.fix_block_limit

    def recorded_fix_block_limit() -> bool:
        with limit_fixes.open("a") as out:
            out.write(f"{os.getpid()}\n")
        return fix_block_limit()

    monkeypatch.setattr(job_runner, "fix_block_limit", recorded_fix_block_limit)

    # 6 GiB free: two workers keep the 2 GiB margin each, but with the rough
    # 3 GiB per worker beyond the plan's buffers only one fits, so one worker
    # runs one work unit first, with its buffers written before it, and the
    # measurement decides for the other three.
    summary = run_job(spec, plan, hardware=_small_node(6))

    np.testing.assert_allclose(_all_outputs(spec.output.file), expected, rtol=1e-13, atol=0)
    assert summary.all_complete and summary.executed_work_units == 4
    assert [report.work_units for report in summary.worker_reports] == [1, 2, 1]
    assert [report.index for report in summary.worker_reports] == [0, 1, 2]
    assert (summary.workers, summary.threads_per_worker) == (2, 1)
    assert "by measured memory" in summary.worker_selection
    assert summary.worker_selection.endswith("measured first by 1 worker, one work unit each")
    # Only the measuring worker wrote its buffer pool first.
    assert len(pool_writes.read_text().split()) == 1
    # Each of the three workers, and not this process, fixed its limit for
    # separately mapped blocks (which kernel building then lowers).
    workers_with_fixed_limit = set(limit_fixes.read_text().split())
    assert len(workers_with_fixed_limit) == 3
    assert str(os.getpid()) not in workers_with_fixed_limit
    assert not runtime_heap._limit_fixed
    peaks = [report.peak_private_bytes for report in summary.worker_reports]
    assert all(peak is not None and peak > 0 for peak in peaks)
    assert all(
        report.peak_private_bytes >= report.private_bytes for report in summary.worker_reports
    )
    record = JobRunRecord.from_summary(
        summary, job_name="runner_test", plan_action="reused", plan_preparation_seconds=0.0
    )
    assert JobRunRecord.from_dict(record.to_dict()).workers == summary.worker_reports


def test_peak_private_memory_counts_copied_pages_but_not_swapped_ones(monkeypatch):
    G = 1024**3
    # Start: 10 GiB resident, 1 GiB of it private (the rest shared with the
    # parent). The resident high-water mark rose by 1 GiB of new pages.
    monkeypatch.setattr(job_runner, "_max_rss_bytes", lambda: 11 * G)
    # 1 GiB of shared pages was copied (now private), 7 GiB went to swap.
    monkeypatch.setattr(job_runner, "_memory_usage", lambda: (3 * G, 2 * G, 7 * G))
    assert job_runner._peak_private_bytes((10 * G, 1 * G, 0)) == 3 * G
    # Without swap the same drop of shared pages is all copies.
    monkeypatch.setattr(job_runner, "_memory_usage", lambda: (10 * G, 9 * G, 0))
    assert job_runner._peak_private_bytes((10 * G, 1 * G, 0)) == 10 * G
    job_runner._return_freed_memory()


def test_auto_workers_keep_the_rough_count_when_memory_cannot_be_measured(tmp_path, monkeypatch):
    spec = _spec(tmp_path, workers="auto")
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    monkeypatch.setattr("concord_lqcd.job._worker_split.MINIMUM_PARALLEL_SECONDS", 0.0)
    monkeypatch.setattr(job_runner, "node_shape", lambda: (8, 1))
    monkeypatch.setattr(job_runner, "_memory_usage", lambda: None)

    summary = run_job(spec, plan, hardware=_small_node(6))

    assert summary.all_complete
    assert [report.work_units for report in summary.worker_reports] == [1, 3]
    assert [report.index for report in summary.worker_reports] == [0, 1]
    assert summary.workers == 1
    assert summary.worker_selection.endswith("memory not measurable, 1 worker")
    assert all(report.peak_private_bytes is None for report in summary.worker_reports)


def test_threads_per_worker_sets_one_process_threads_for_the_run_only(tmp_path):
    spec = _spec(tmp_path, threads_per_worker=1)
    _write_data(tmp_path)
    plan = write_job_plan(spec)
    torch.set_num_threads(2)
    summary = run_job(spec, plan)
    assert torch.get_num_threads() == 2
    assert summary.workers == 1
    assert summary.threads_per_worker == 1
    assert summary.worker_reports == ()


def test_workers_require_cpu_execution(tmp_path):
    payload = _spec(tmp_path).model_dump(mode="json")
    payload["run"].update(device="cuda", workers=2)
    with pytest.raises(ValueError, match="runs on the CPU"):
        JobSpec.model_validate(payload)
