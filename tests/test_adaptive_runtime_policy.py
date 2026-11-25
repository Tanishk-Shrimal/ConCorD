"""Regression tests for device-local adaptive execution policy."""

from __future__ import annotations

import json

from tests.plan_costs import _baryon_correlator, _meson_correlator
from concord_lqcd.backend.runtime import (
    CalibrationCache,
    CalibrationCandidate,
    CalibrationKey,
    CalibrationRecord,
    HardwareDeviceProfile,
    HardwareProfile,
    RuntimeCompilationConfig,
    compile_correlators,
    performance_environment_fingerprint,
    physical_plan_fingerprint,
    select_calibration_candidate,
)


def _key(name: str) -> CalibrationKey:
    return CalibrationKey(
        workload_fingerprint="workload",
        candidate_fingerprint=f"candidate:{name}",
        environment_fingerprint="environment",
        deployment_fingerprint="deployment",
        scalar_type="complex128",
    )


def _profile(*, available: int, threads: int = 4) -> HardwareProfile:
    return HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Synthetic CPU",
            total_memory_bytes=16 * 1024**3,
            available_memory_bytes=available,
            logical_processors=8,
        ),
        system="TestOS",
        machine="test-machine",
        torch_num_threads=threads,
        torch_num_interop_threads=2,
    )


def test_adaptive_compilation_freezes_portable_meta_strategy():
    compilation = compile_correlators(
        {"meson": _meson_correlator()},
        RuntimeCompilationConfig(
            mode_size=2,
            contraction_strategy="adaptive",
        ),
    )

    assert compilation.config.contraction_strategy == "adaptive"
    assert compilation.contraction_order.strategy == "adaptive"
    assert "greedy" in compilation.contraction_order.selected_strategies
    compilation.validate()


def test_adaptive_candidate_construction_is_graph_based_not_hadron_labeled():
    compilation = compile_correlators(
        {"arbitrary_name": _baryon_correlator()},
        RuntimeCompilationConfig(
            mode_size=4,
            contraction_strategy="adaptive",
        ),
    )

    assert tuple(compilation.correlator_ids) == ("arbitrary_name",)
    assert compilation.contraction_order.strategy == "adaptive"
    assert compilation.cost_report.scheduled_contraction_flops > 0


def test_environment_fingerprint_excludes_volatile_available_memory():
    first = performance_environment_fingerprint(
        _profile(available=12 * 1024**3),
        "cpu",
    )
    second = performance_environment_fingerprint(
        _profile(available=4 * 1024**3),
        "cpu",
    )
    changed_threads = performance_environment_fingerprint(
        _profile(available=12 * 1024**3, threads=2),
        "cpu",
    )

    assert first == second
    assert changed_threads != first


def test_cuda_environment_fingerprint_includes_host_characteristics():
    def profile(host_name: str) -> HardwareProfile:
        return HardwareProfile(
            host=HardwareDeviceProfile(
                device="cpu",
                kind="cpu",
                name=host_name,
                total_memory_bytes=32 * 1024**3,
                available_memory_bytes=24 * 1024**3,
                logical_processors=16,
            ),
            accelerators=(
                HardwareDeviceProfile(
                    device="cuda:0",
                    kind="cuda",
                    name="Synthetic GPU",
                    total_memory_bytes=16 * 1024**3,
                    available_memory_bytes=12 * 1024**3,
                    compute_capability=(9, 0),
                    multiprocessor_count=32,
                ),
            ),
            system="TestOS",
            machine="test-machine",
            torch_num_threads=8,
            torch_num_interop_threads=2,
        )

    first = performance_environment_fingerprint(profile("Host A"), "cuda:0")
    second = performance_environment_fingerprint(profile("Host B"), "cuda:0")

    assert first != second


def test_physical_plan_fingerprint_tracks_candidate_execution_graph():
    greedy = compile_correlators(
        {"baryon": _baryon_correlator()},
        RuntimeCompilationConfig(mode_size=4, contraction_strategy="greedy"),
    )
    reuse = compile_correlators(
        {"baryon": _baryon_correlator()},
        RuntimeCompilationConfig(mode_size=4, contraction_strategy="reuse-aware"),
    )

    assert physical_plan_fingerprint(greedy.physical_plan) != (
        physical_plan_fingerprint(reuse.physical_plan)
    )


def test_cache_round_trip_is_versioned_and_atomic(tmp_path):
    path = tmp_path / "calibration.json"
    record = CalibrationRecord.measured(
        key=_key("greedy"),
        candidate_name="greedy",
        samples_seconds_per_work_unit=(0.3, 0.2, 0.25),
    )
    cache = CalibrationCache(path)
    cache.put(record)
    cache.save()

    loaded = CalibrationCache.load(path)
    restored = loaded.get(record.key)

    assert restored == record
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["records"]) == 1


def test_invalid_or_stale_cache_is_ignored_with_warning(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text('{"schema_version": 999, "records": []}', encoding="utf-8")

    cache = CalibrationCache.load(path)

    assert cache.records == ()
    assert cache.warnings
    assert "ignored invalid calibration cache" in cache.warnings[0]


def test_partial_calibration_never_biases_candidate_selection(tmp_path):
    baseline = CalibrationCandidate(
        name="greedy",
        key=_key("greedy"),
        fallback_rank=(0,),
    )
    alternate = CalibrationCandidate(
        name="reuse-aware",
        key=_key("reuse-aware"),
        fallback_rank=(1,),
    )
    cache = CalibrationCache(tmp_path / "calibration.json")
    cache.put(
        CalibrationRecord.measured(
            key=alternate.key,
            candidate_name=alternate.name,
            samples_seconds_per_work_unit=(0.01,),
        )
    )

    selection = select_calibration_candidate((baseline, alternate), cache)

    assert selection.source == "analytical_fallback"
    assert selection.candidate.name == "greedy"
    assert selection.missing_candidates == ("greedy",)


def test_complete_calibration_selects_measured_winner(tmp_path):
    greedy = CalibrationCandidate(
        name="greedy",
        key=_key("greedy"),
        fallback_rank=(0,),
    )
    reuse = CalibrationCandidate(
        name="reuse-aware",
        key=_key("reuse-aware"),
        fallback_rank=(1,),
    )
    cache = CalibrationCache(tmp_path / "calibration.json")
    cache.put(
        CalibrationRecord.measured(
            key=greedy.key,
            candidate_name=greedy.name,
            samples_seconds_per_work_unit=(0.12, 0.11, 0.13),
        )
    )
    cache.put(
        CalibrationRecord.measured(
            key=reuse.key,
            candidate_name=reuse.name,
            samples_seconds_per_work_unit=(0.02, 0.01, 0.015),
        )
    )

    selection = select_calibration_candidate((greedy, reuse), cache)

    assert selection.source == "calibrated"
    assert selection.candidate.name == "reuse-aware"
    assert selection.median_seconds_per_work_unit == 0.015
