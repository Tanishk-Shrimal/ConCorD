"""Versioned run-record serialization and timing invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from concord_lqcd.job.run_record import (
    JobRunHardware,
    JobRunRecord,
    JobRunTiming,
    RunRecordError,
    load_job_run_record,
    write_job_run_record,
)
from concord_lqcd.visualization import _runtime_html


def _record(tmp_path: Path) -> JobRunRecord:
    timing = JobRunTiming.measured(
        execution_wall_seconds=10.0,
        result_management_seconds=0.5,
        runtime_planning_seconds=1.0,
        data_preparation_seconds=1.5,
        input_wait_seconds=0.5,
        numerical_execution_seconds=5.0,
        output_writing_seconds=1.0,
    )
    return JobRunRecord(
        job_name="report-test",
        output_file=tmp_path / "result.h5",
        plan_sha256="a" * 64,
        plan_action="loaded",
        plan_preparation_seconds=0.25,
        timing=timing,
        total_work_units=64,
        executed_work_units=64,
        skipped_work_units=0,
        completed_work_units=64,
        correlator_ids=("pion", "nucleon"),
        contraction_path="reuse-aware",
        contraction_selection_source="configured",
        time_batch_size=8,
        time_batch_selection_source="configured",
        executed_batches=8,
        selected_device="cuda:0",
        storage_device="cuda:0",
        execution_mode="resident",
        memory_verified=True,
        known_working_set_bytes=1024,
        peak_tile_bytes=None,
        tiled_operations=0,
        resident_pool_peak_bytes=512,
        primitive_cache_hits=10,
        primitive_cache_misses=2,
        hdf5_file_opens=2,
        hdf5_file_reuses=20,
        propagation_prefetch_bytes=4096,
        propagation_prefetch_wait_seconds=0.1,
        kernel_cache_enabled=True,
        kernel_cache_hits=7,
        kernel_cache_misses=3,
        kernel_cache_writes=3,
        kernel_cache_invalid_entries=1,
        kernel_cache_write_failures=0,
        hardware=JobRunHardware(
            host_name="Test CPU",
            logical_cpus=32,
            host_memory_bytes=128 * 1024**3,
            compute_device="cuda:0",
            compute_name="Test GPU",
            compute_memory_bytes=80 * 1024**3,
            torch_num_threads=32,
            torch_num_interop_threads=1,
        ),
    )


def test_run_record_round_trip_is_versioned_and_lossless(tmp_path):
    record = _record(tmp_path)
    path = write_job_run_record(record, tmp_path / "run.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 3
    assert payload["timing"]["execution_wall_seconds"] == 10.0
    assert payload["hardware"]["compute_name"] == "Test GPU"
    assert load_job_run_record(path) == record


def test_runtime_report_summarizes_persistent_kernel_reuse(tmp_path):
    content = _runtime_html(_record(tmp_path))

    assert "Reusable kernels" in content
    assert "Reused from earlier runs" in content
    assert "Computed afresh" in content
    assert "Cache writes" in content
    assert "Invalid saved entries ignored" in content


def test_run_record_loads_v1_without_hardware_metadata(tmp_path):
    payload = _record(tmp_path).to_dict()
    payload["format_version"] = 1
    payload.pop("hardware")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_job_run_record(path)

    assert loaded.hardware is None


def test_run_record_loads_v2_without_kernel_cache_statistics(tmp_path):
    payload = _record(tmp_path).to_dict()
    payload["format_version"] = 2
    payload.pop("hardware")
    for name in (
        "kernel_cache_enabled",
        "kernel_cache_hits",
        "kernel_cache_misses",
        "kernel_cache_writes",
        "kernel_cache_invalid_entries",
        "kernel_cache_write_failures",
    ):
        payload["resources"].pop(name)
    path = tmp_path / "legacy-v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_job_run_record(path)

    assert loaded.kernel_cache_enabled is False
    assert loaded.kernel_cache_hits == 0
    assert loaded.kernel_cache_misses == 0


def test_run_timing_rejects_overlapping_top_level_categories():
    with pytest.raises(RunRecordError, match="exceed"):
        JobRunTiming.measured(
            execution_wall_seconds=1.0,
            result_management_seconds=0.3,
            runtime_planning_seconds=0.3,
            data_preparation_seconds=0.3,
            input_wait_seconds=0.3,
            numerical_execution_seconds=0.3,
            output_writing_seconds=0.3,
        )
