"""Device-local selection of exact time-context batch sizes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import time
from typing import Mapping
import warnings

import torch

from concord_lqcd.backend.planner.physical import MaterializePrimitiveOperation
from concord_lqcd.backend.runtime.batched_executor import BatchedNumericalExecutor
from concord_lqcd.backend.runtime.batching import (
    BatchPlanningError,
    TimeBatchFeasibilityPlan,
    plan_time_batch_candidates,
)
from concord_lqcd.backend.runtime.calibration import (
    CalibrationCache,
    CalibrationCacheError,
    CalibrationCandidate,
    CalibrationKey,
    CalibrationRecord,
    default_calibration_cache_path,
    deployment_fingerprint,
    performance_environment_fingerprint,
    physical_plan_fingerprint,
    select_calibration_candidate,
)
from concord_lqcd.backend.runtime.hardware import HardwareProfile, detect_hardware_profile
from concord_lqcd.backend.runtime.numerical_executor import GenericNumericalExecutor
from concord_lqcd.backend.runtime.resource_planning import DeviceExecutionPlan

from .plan import ConcordPlan
from .schema import JobSpec
from .synthetic import (
    DeterministicSyntheticTensorFactory,
    build_synthetic_materializer,
    prime_synthetic_materializer,
)


class TimeBatchPlanningError(ValueError):
    """Raised when batch calibration cannot preserve exact execution semantics."""


_GIB = 1024**3


@dataclass(frozen=True)
class TimeBatchCalibrationPolicy:
    """Bounded empirical policy for memory-feasible batch candidates."""

    warmups: int = 0
    repeats: int = 1
    seed: int = 20260824
    max_memory_bytes: int = 16 * _GIB
    max_scheduled_flops_per_sample: int = 10_000_000_000_000
    max_estimated_seconds: float = 180.0

    def __post_init__(self) -> None:
        for field_name, allow_zero in (
            ("warmups", True),
            ("repeats", False),
            ("seed", True),
            ("max_memory_bytes", False),
            ("max_scheduled_flops_per_sample", False),
        ):
            value = getattr(self, field_name)
            minimum = 0 if allow_zero else 1
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{field_name} must be a {qualifier} integer")
        if (
            not isinstance(self.max_estimated_seconds, (int, float))
            or isinstance(self.max_estimated_seconds, bool)
            or not isfinite(self.max_estimated_seconds)
            or self.max_estimated_seconds <= 0
        ):
            raise ValueError("max_estimated_seconds must be a positive finite number")


@dataclass(frozen=True)
class TimeBatchSelection:
    """Selected batch size plus its memory and measurement provenance."""

    batch_size: int
    source: str
    feasibility: TimeBatchFeasibilityPlan
    median_seconds_per_work_unit: float | None
    calibration_cache: Path
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.batch_size not in self.feasibility.batch_sizes:
            raise ValueError("selected time batch is absent from feasible candidates")
        if self.source not in (
            "calibrated",
            "configured",
            "sequential_fallback",
        ):
            raise ValueError("unsupported time-batch selection source")
        if self.source == "calibrated" and self.median_seconds_per_work_unit is None:
            raise ValueError("calibrated time batch requires a measured timing")
        if self.source != "calibrated" and self.median_seconds_per_work_unit is not None:
            raise ValueError("unmeasured time-batch selection cannot claim a timing")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _execute_candidate(
    plan: ConcordPlan,
    deployment: DeviceExecutionPlan,
    materializers,
) -> dict[str, torch.Tensor]:
    values = tuple(materializers)
    if len(values) == 1:
        result = GenericNumericalExecutor(
            values[0],
            device=deployment.storage_torch_device,
            compute_device=deployment.torch_device,
            tiled_plan=deployment.tiled_plan,
        ).execute()
        return {
            correlator_id: tensor.detach().cpu().clone().unsqueeze(0)
            for correlator_id, tensor in result.outputs.items()
        }
    if deployment.execution_mode != "resident":
        raise TimeBatchPlanningError(
            "true time batching currently requires resident execution"
        )
    result = BatchedNumericalExecutor(
        values,
        device=deployment.storage_torch_device,
    ).execute()
    return {
        correlator_id: tensor.detach().cpu().clone()
        for correlator_id, tensor in result.outputs.items()
    }


def _assert_outputs_agree(
    reference: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    scalar_type: str,
) -> None:
    if set(reference) != set(actual):
        raise TimeBatchPlanningError("time-batch candidates produced different outputs")
    rtol, atol = (1e-4, 1e-5) if scalar_type == "complex64" else (1e-10, 1e-11)
    for correlator_id in sorted(reference):
        expected = reference[correlator_id]
        value = actual[correlator_id]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise TimeBatchPlanningError(
                f"time-batch output {correlator_id!r} has incompatible metadata"
            )
        try:
            torch.testing.assert_close(value, expected, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise TimeBatchPlanningError(
                f"time-batch output {correlator_id!r} disagrees: {exc}"
            ) from exc


def _candidate_name(batch_size: int) -> str:
    return f"time-batch:{batch_size}"


def _primitive_input_bytes(plan: ConcordPlan) -> int:
    return sum(
        plan.physical_plan.binding(step.operation.value_id).size_bytes
        for step in plan.physical_plan.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
    )


def _bounded_calibration_feasibility(
    plan: ConcordPlan,
    feasibility: TimeBatchFeasibilityPlan,
    policy: TimeBatchCalibrationPolicy,
) -> tuple[TimeBatchFeasibilityPlan, tuple[str, ...]]:
    """Apply an independent synthetic-allocation guard to safe candidates."""

    input_bytes = _primitive_input_bytes(plan)
    memory_limit_bytes = min(
        policy.max_memory_bytes,
        feasibility.safe_memory_limit_bytes
        if feasibility.safe_memory_limit_bytes is not None
        else policy.max_memory_bytes,
    )
    def calibration_bytes(candidate) -> int:
        return (
            candidate.memory.known_peak_bytes
            + candidate.batch_size * input_bytes
        )

    def calibration_flops(candidate) -> int:
        return plan.cost_report.scheduled_contraction_flops * candidate.batch_size

    retained = tuple(
        candidate
        for candidate in feasibility.candidates
        if candidate.batch_size == 1
        or (
            calibration_bytes(candidate) <= memory_limit_bytes
            and calibration_flops(candidate)
            <= policy.max_scheduled_flops_per_sample
        )
    )
    # Batch size one is a non-calibrated fallback and must survive even if the
    # synthetic benchmark guard is smaller than the real sequential plan.
    sequential = feasibility.candidates[0]
    assert retained and retained[0] == sequential
    if retained == feasibility.candidates:
        return feasibility, ()
    removed = tuple(
        candidate
        for candidate in feasibility.candidates
        if candidate not in retained
    )
    messages: list[str] = []
    if any(
        calibration_bytes(candidate) > memory_limit_bytes
        for candidate in removed
    ):
        messages.append(
            "time-batch calibration candidates were capped by the synthetic "
            f"allocation guard ({memory_limit_bytes} bytes)"
        )
    if any(
        calibration_flops(candidate) > policy.max_scheduled_flops_per_sample
        for candidate in removed
    ):
        messages.append(
            "time-batch calibration candidates were capped by the scheduled-FLOP "
            f"guard ({policy.max_scheduled_flops_per_sample} FLOPs per sample)"
        )
    warning_values = tuple(messages)
    return (
        TimeBatchFeasibilityPlan(
            requested_batch_size=feasibility.requested_batch_size,
            candidates=retained,
            selected_device=feasibility.selected_device,
            execution_mode=feasibility.execution_mode,
            safe_memory_limit_bytes=feasibility.safe_memory_limit_bytes,
            reason=f"{feasibility.reason}; {'; '.join(warning_values)}",
        ),
        warning_values,
    )


def _budgeted_calibration_feasibility(
    feasibility: TimeBatchFeasibilityPlan,
    policy: TimeBatchCalibrationPolicy,
    baseline_seconds_per_work_unit: float,
) -> tuple[TimeBatchFeasibilityPlan, tuple[str, ...]]:
    retained = []
    estimated_seconds = 0.0
    for candidate in feasibility.candidates:
        projected_seconds = estimated_seconds + (
            1.5
            * baseline_seconds_per_work_unit
            * candidate.batch_size
            * (policy.warmups + policy.repeats)
        )
        if candidate.batch_size != 1 and projected_seconds > policy.max_estimated_seconds:
            break
        retained.append(candidate)
        estimated_seconds = projected_seconds
    if len(retained) == len(feasibility.candidates):
        return feasibility, ()
    message = (
        "time-batch calibration candidates were capped by the estimated "
        f"wall-time budget ({policy.max_estimated_seconds:g} seconds)"
    )
    return (
        TimeBatchFeasibilityPlan(
            requested_batch_size=feasibility.requested_batch_size,
            candidates=tuple(retained),
            selected_device=feasibility.selected_device,
            execution_mode=feasibility.execution_mode,
            safe_memory_limit_bytes=feasibility.safe_memory_limit_bytes,
            reason=f"{feasibility.reason}; {message}",
        ),
        (message,),
    )


def _calibration_candidates(
    plan: ConcordPlan,
    deployment: DeviceExecutionPlan,
    hardware: HardwareProfile,
    feasibility: TimeBatchFeasibilityPlan,
    policy: TimeBatchCalibrationPolicy,
) -> tuple[CalibrationCandidate, ...]:
    environment = performance_environment_fingerprint(
        hardware,
        deployment.selected_device,
    )
    physical = physical_plan_fingerprint(plan.physical_plan)
    deployed = deployment_fingerprint(deployment)
    result = []
    for candidate in feasibility.candidates:
        batch_size = candidate.batch_size
        key = CalibrationKey(
            workload_fingerprint=plan.manifest.job_fingerprint,
            candidate_fingerprint=(
                f"time-batch-v2:{physical}:{batch_size}:"
                f"{candidate.memory.known_peak_bytes}:"
                f"{policy.seed}:{policy.warmups}:{policy.repeats}"
            ),
            environment_fingerprint=environment,
            deployment_fingerprint=deployed,
            scalar_type=plan.config.scalar_type,
            work_units_per_sample=batch_size,
        )
        result.append(
            CalibrationCandidate(
                name=_candidate_name(batch_size),
                key=key,
                # Incomplete or unavailable measurements must preserve the
                # established sequential execution path.
                fallback_rank=(0 if batch_size == 1 else 1, batch_size),
            )
        )
    return tuple(result)


def _measure_candidates(
    spec: JobSpec,
    plan: ConcordPlan,
    deployment: DeviceExecutionPlan,
    feasibility: TimeBatchFeasibilityPlan,
    candidates: tuple[CalibrationCandidate, ...],
    cache: CalibrationCache,
    policy: TimeBatchCalibrationPolicy,
) -> None:
    reference_first_context: Mapping[str, torch.Tensor] | None = None
    pending: list[CalibrationRecord] = []
    for feasible, candidate in zip(feasibility.candidates, candidates):
        batch_size = feasible.batch_size
        sample_flops = plan.cost_report.scheduled_contraction_flops * batch_size
        if sample_flops > policy.max_scheduled_flops_per_sample:
            raise TimeBatchPlanningError(
                f"time-batch calibration FLOP guard refused batch {batch_size}: "
                f"{sample_flops} exceeds {policy.max_scheduled_flops_per_sample}"
            )
        factory = DeterministicSyntheticTensorFactory(policy.seed)
        materializers = tuple(
            build_synthetic_materializer(plan, factory, context)
            for context in spec.time_contexts[:batch_size]
        )
        for materializer in materializers:
            prime_synthetic_materializer(materializer)

        device = deployment.torch_device
        for _ in range(policy.warmups):
            _synchronize(device)
            _execute_candidate(plan, deployment, materializers)
            _synchronize(device)

        samples: list[float] = []
        first_snapshot: Mapping[str, torch.Tensor] | None = None
        last_snapshot: Mapping[str, torch.Tensor] | None = None
        for _ in range(policy.repeats):
            _synchronize(device)
            started = time.perf_counter_ns()
            last_snapshot = _execute_candidate(plan, deployment, materializers)
            _synchronize(device)
            elapsed = (time.perf_counter_ns() - started) / 1e9
            seconds_per_work_unit = elapsed / batch_size
            if (
                not isfinite(seconds_per_work_unit)
                or seconds_per_work_unit <= 0.0
            ):
                raise TimeBatchPlanningError(
                    "time-batch calibration produced an invalid timing"
                )
            samples.append(seconds_per_work_unit)
            if first_snapshot is None:
                first_snapshot = last_snapshot
        assert first_snapshot is not None
        assert last_snapshot is not None
        if any(
            not bool(torch.isfinite(tensor).all())
            for tensor in last_snapshot.values()
        ):
            raise TimeBatchPlanningError(
                f"time-batch candidate {batch_size} produced non-finite output"
            )
        _assert_outputs_agree(
            first_snapshot,
            last_snapshot,
            scalar_type=plan.config.scalar_type,
        )
        first_context = {
            correlator_id: tensor[0]
            for correlator_id, tensor in last_snapshot.items()
        }
        if reference_first_context is None:
            reference_first_context = first_context
        else:
            _assert_outputs_agree(
                reference_first_context,
                first_context,
                scalar_type=plan.config.scalar_type,
            )
        if cache.get(candidate.key) is None:
            pending.append(
                CalibrationRecord.measured(
                    key=candidate.key,
                    candidate_name=candidate.name,
                    samples_seconds_per_work_unit=samples,
                )
            )
    for record in pending:
        cache.put(record)


def select_time_batch_size(
    spec: JobSpec,
    plan: ConcordPlan,
    deployment: DeviceExecutionPlan,
    *,
    hardware: HardwareProfile | None = None,
    calibration_policy: TimeBatchCalibrationPolicy | None = None,
    cache_path: str | Path | None = None,
) -> TimeBatchSelection:
    """Choose one safe batch size without changing the portable plan."""

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be a JobSpec")
    if not isinstance(plan, ConcordPlan):
        raise TypeError("plan must be a ConcordPlan")
    if not isinstance(deployment, DeviceExecutionPlan):
        raise TypeError("deployment must be a DeviceExecutionPlan")
    plan.validate()
    target_cache = (
        default_calibration_cache_path() if cache_path is None else Path(cache_path)
    )
    try:
        feasibility = plan_time_batch_candidates(
            plan.physical_plan,
            deployment,
            work_units=len(spec.time_contexts),
            requested_batch_size=spec.run.time_batch_size,
        )
    except BatchPlanningError as exc:
        raise TimeBatchPlanningError(str(exc)) from exc

    if isinstance(spec.run.time_batch_size, int):
        selected_size = min(spec.run.time_batch_size, len(spec.time_contexts))
        return TimeBatchSelection(
            batch_size=selected_size,
            source="configured",
            feasibility=feasibility,
            median_seconds_per_work_unit=None,
            calibration_cache=target_cache,
        )
    if len(feasibility.candidates) == 1:
        return TimeBatchSelection(
            batch_size=1,
            source="sequential_fallback",
            feasibility=feasibility,
            median_seconds_per_work_unit=None,
            calibration_cache=target_cache,
        )

    policy = calibration_policy or TimeBatchCalibrationPolicy()
    feasibility, guard_warnings = _bounded_calibration_feasibility(
        plan,
        feasibility,
        policy,
    )
    if len(feasibility.candidates) == 1:
        for message in guard_warnings:
            warnings.warn(message, UserWarning, stacklevel=2)
        return TimeBatchSelection(
            batch_size=1,
            source="sequential_fallback",
            feasibility=feasibility,
            median_seconds_per_work_unit=None,
            calibration_cache=target_cache,
            warnings=guard_warnings,
        )

    profile = detect_hardware_profile() if hardware is None else hardware
    candidates = _calibration_candidates(
        plan,
        deployment,
        profile,
        feasibility,
        policy,
    )
    cache = CalibrationCache.load(target_cache)
    warning_messages = [*guard_warnings, *cache.warnings]
    selection = select_calibration_candidate(candidates, cache)
    if selection.source != "calibrated":
        try:
            baseline = cache.get(candidates[0].key)
            if baseline is None:
                _measure_candidates(
                    spec,
                    plan,
                    deployment,
                    TimeBatchFeasibilityPlan(
                        requested_batch_size=feasibility.requested_batch_size,
                        candidates=(feasibility.candidates[0],),
                        selected_device=feasibility.selected_device,
                        execution_mode=feasibility.execution_mode,
                        safe_memory_limit_bytes=feasibility.safe_memory_limit_bytes,
                        reason=feasibility.reason,
                    ),
                    candidates[:1],
                    cache,
                    policy,
                )
                baseline = cache.get(candidates[0].key)
            assert baseline is not None
            feasibility, budget_warnings = _budgeted_calibration_feasibility(
                feasibility,
                policy,
                baseline.median_seconds_per_work_unit,
            )
            warning_messages.extend(budget_warnings)
            candidates = candidates[:len(feasibility.candidates)]
            selection = select_calibration_candidate(candidates, cache)
            if selection.source != "calibrated":
                _measure_candidates(
                    spec,
                    plan,
                    deployment,
                    feasibility,
                    candidates,
                    cache,
                    policy,
                )
                selection = select_calibration_candidate(candidates, cache)
            try:
                cache.save()
            except CalibrationCacheError as exc:
                warning_messages.append(str(exc))
        except (MemoryError, RuntimeError, TimeBatchPlanningError) as exc:
            warning_messages.append(
                "time-batch numerical calibration unavailable; used sequential "
                f"fallback: {exc}"
            )
            for message in warning_messages:
                warnings.warn(message, UserWarning, stacklevel=2)
            return TimeBatchSelection(
                batch_size=1,
                source="sequential_fallback",
                feasibility=feasibility,
                median_seconds_per_work_unit=None,
                calibration_cache=cache.path,
                warnings=tuple(warning_messages),
            )

    selected_size = int(selection.candidate.name.rsplit(":", 1)[1])
    source = (
        "calibrated"
        if selection.source == "calibrated"
        else "sequential_fallback"
    )
    for message in warning_messages:
        warnings.warn(message, UserWarning, stacklevel=2)
    return TimeBatchSelection(
        batch_size=selected_size,
        source=source,
        feasibility=feasibility,
        median_seconds_per_work_unit=selection.median_seconds_per_work_unit,
        calibration_cache=cache.path,
        warnings=tuple(warning_messages),
    )


__all__ = [
    "TimeBatchCalibrationPolicy",
    "TimeBatchPlanningError",
    "TimeBatchSelection",
    "select_time_batch_size",
]
