"""Hardware-adaptive specialization of a portable ``ConcordPlan``.

The saved plan retains a hardware-independent global mathematical DAG and an
``adaptive`` compatibility baseline.  Immediately before a job run, this
module derives a bounded pair of equivalent complete execution candidates,
checks current device feasibility, and either reuses exact local calibration
records or times deterministic synthetic tensors.  Only the transient
execution plan changes; symbolic, Wick, PhysicsIR, resource, and correlator
identities remain frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import time
from typing import Mapping
import warnings

import torch

from concord_lqcd.backend.planner.physical import MaterializePrimitiveOperation
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
from concord_lqcd.backend.runtime.pipeline import (
    RuntimeExecutionCandidate,
    build_adaptive_execution_candidates,
)
from concord_lqcd.backend.runtime.resource_planning import (
    DeviceExecutionPlan,
    HardwareBudgetPolicy,
    ResourcePlanningError,
    estimate_execution_memory,
    resolve_runtime_device,
)
from concord_lqcd.compiler import ContractionSizeModel

from .plan import ConcordPlan, specialize_concord_plan
from .schema import ExpandedTimeContext, JobSpec
from .synthetic import (
    DeterministicSyntheticTensorFactory as _SyntheticTensorFactory,
    build_synthetic_materializer as _materializer,
    prime_synthetic_materializer,
)


_GIB = 1024**3


class AdaptivePlanningError(ValueError):
    """Raised when adaptive specialization cannot preserve its invariants."""


@dataclass(frozen=True)
class AdaptiveCalibrationPolicy:
    """Bounded first-run calibration policy for one exact job graph."""

    warmups: int = 1
    repeats: int = 3
    seed: int = 20260824
    max_memory_bytes: int = 4 * _GIB
    max_scheduled_flops_per_sample: int = 10_000_000_000

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


@dataclass(frozen=True)
class AdaptivePlanSelection:
    """Transient plan plus inspectable selection provenance."""

    plan: ConcordPlan
    candidate_strategy: str
    source: str
    selected_device: str
    median_seconds_per_work_unit: float | None
    calibration_cache: Path
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if not isinstance(self.plan, ConcordPlan):
            raise TypeError("adaptive selection requires a ConcordPlan")
        if not self.candidate_strategy or not self.selected_device:
            raise ValueError("adaptive selection names must be non-empty")
        if self.source not in ("calibrated", "analytical_fallback", "not_requested"):
            raise ValueError("unsupported adaptive selection source")


def _primitive_input_bytes(plan: ConcordPlan) -> int:
    return sum(
        plan.physical_plan.binding(step.operation.value_id).size_bytes
        for step in plan.physical_plan.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
        and step.operation.kind in ("kernel", "propagation")
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _execute_once(
    plan: ConcordPlan,
    deployment: DeviceExecutionPlan,
    factory: _SyntheticTensorFactory,
    time_context: ExpandedTimeContext,
) -> dict[str, torch.Tensor]:
    result = GenericNumericalExecutor(
        _materializer(plan, factory, time_context),
        device=deployment.storage_torch_device,
        compute_device=deployment.torch_device,
        tiled_plan=deployment.tiled_plan,
    ).execute()
    return {
        correlator_id: tensor.detach().cpu().clone()
        for correlator_id, tensor in result.outputs.items()
    }


def _prime_inputs(
    plan: ConcordPlan,
    factory: _SyntheticTensorFactory,
    time_context: ExpandedTimeContext,
) -> None:
    prime_synthetic_materializer(_materializer(plan, factory, time_context))


def _assert_outputs_agree(
    reference: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    scalar_type: str,
) -> None:
    if set(reference) != set(actual):
        raise AdaptivePlanningError("adaptive candidates produced different output sets")
    rtol, atol = (1e-4, 1e-5) if scalar_type == "complex64" else (1e-10, 1e-11)
    for correlator_id in sorted(reference):
        expected = reference[correlator_id]
        value = actual[correlator_id]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise AdaptivePlanningError(
                f"adaptive candidate output {correlator_id!r} has incompatible metadata"
            )
        try:
            torch.testing.assert_close(value, expected, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AdaptivePlanningError(
                f"adaptive candidate output {correlator_id!r} disagrees: {exc}"
            ) from exc


@dataclass(frozen=True)
class _PreparedCandidate:
    compiler_candidate: RuntimeExecutionCandidate
    plan: ConcordPlan
    deployment: DeviceExecutionPlan
    calibration: CalibrationCandidate


def _prepare_candidates(
    base: ConcordPlan,
    hardware: HardwareProfile,
    requested_device: str | torch.device,
    budget_policy: HardwareBudgetPolicy | None,
) -> tuple[_PreparedCandidate, ...]:
    size_model = ContractionSizeModel(
        mode_size=base.config.mode_size,
        spin_size=base.config.spin_size,
    )
    compiler_candidates = build_adaptive_execution_candidates(
        base.global_cse,
        size_model,
        base.config,
    )
    specialized = tuple(
        (
            candidate,
            specialize_concord_plan(
                base,
                candidate.stages,
                diagnostics={"adaptive_candidate": candidate.strategy},
            ),
        )
        for candidate in compiler_candidates
    )

    # Keep path calibration on one compute device.  Choosing between CPU and
    # GPU is a separate policy problem with different transfer, I/O, and
    # batching tradeoffs.  Use the first candidate that can establish the
    # requested/automatic device, then resolve every candidate explicitly on
    # that same device (resident or host-tiled as required).
    shared_device: str | None = None
    initial_failures: list[str] = []
    for candidate, plan in specialized:
        try:
            initial = resolve_runtime_device(
                plan.physical_plan,
                requested_device,
                hardware=hardware,
                policy=budget_policy,
            )
        except ResourcePlanningError as exc:
            initial_failures.append(f"{candidate.strategy}: {exc}")
            continue
        shared_device = initial.selected_device
        break
    if shared_device is None:
        raise AdaptivePlanningError(
            "no adaptive execution candidate can establish a feasible device: "
            + "; ".join(initial_failures)
        )

    result: list[_PreparedCandidate] = []
    failures: list[str] = []
    for candidate, plan in specialized:
        try:
            deployment = resolve_runtime_device(
                plan.physical_plan,
                shared_device,
                hardware=hardware,
                policy=budget_policy,
            )
        except ResourcePlanningError as exc:
            failures.append(f"{candidate.strategy}: {exc}")
            continue
        environment = performance_environment_fingerprint(
            hardware,
            deployment.selected_device,
        )
        key = CalibrationKey(
            workload_fingerprint=base.manifest.job_fingerprint,
            candidate_fingerprint=physical_plan_fingerprint(plan.physical_plan),
            environment_fingerprint=environment,
            deployment_fingerprint=deployment_fingerprint(deployment),
            scalar_type=base.config.scalar_type,
        )
        estimate = estimate_execution_memory(plan.physical_plan)
        fallback_rank = (
            plan.cost_report.scheduled_contraction_flops,
            estimate.known_working_set_bytes,
            plan.cost_report.critical_path_nodes,
        )
        result.append(
            _PreparedCandidate(
                compiler_candidate=candidate,
                plan=plan,
                deployment=deployment,
                calibration=CalibrationCandidate(
                    name=candidate.strategy,
                    key=key,
                    fallback_rank=fallback_rank,
                ),
            )
        )
    if not result:
        detail = "; ".join(failures) if failures else "no candidates were produced"
        raise AdaptivePlanningError(
            f"no adaptive execution candidate is feasible: {detail}"
        )
    return tuple(result)


def _measure_candidates(
    candidates: tuple[_PreparedCandidate, ...],
    cache: CalibrationCache,
    time_context: ExpandedTimeContext,
    policy: AdaptiveCalibrationPolicy,
) -> None:
    for item in candidates:
        estimate = estimate_execution_memory(item.plan.physical_plan)
        conservative = estimate.known_working_set_bytes + _primitive_input_bytes(item.plan)
        if conservative > policy.max_memory_bytes:
            raise AdaptivePlanningError(
                "adaptive calibration memory guard refused candidate "
                f"{item.compiler_candidate.strategy!r}: {conservative} bytes exceeds "
                f"{policy.max_memory_bytes} bytes"
            )
        if (
            item.plan.cost_report.scheduled_contraction_flops
            > policy.max_scheduled_flops_per_sample
        ):
            raise AdaptivePlanningError(
                "adaptive calibration FLOP guard refused candidate "
                f"{item.compiler_candidate.strategy!r}"
            )

    outputs: dict[str, Mapping[str, torch.Tensor]] = {}
    pending_records: list[CalibrationRecord] = []
    for item in candidates:
        factory = _SyntheticTensorFactory(policy.seed)
        _prime_inputs(item.plan, factory, time_context)
        device = item.deployment.torch_device
        for _ in range(policy.warmups):
            _synchronize(device)
            _execute_once(item.plan, item.deployment, factory, time_context)
            _synchronize(device)

        samples: list[float] = []
        first_snapshot: Mapping[str, torch.Tensor] | None = None
        snapshot: Mapping[str, torch.Tensor] | None = None
        for _ in range(policy.repeats):
            _synchronize(device)
            started = time.perf_counter_ns()
            snapshot = _execute_once(item.plan, item.deployment, factory, time_context)
            _synchronize(device)
            elapsed = (time.perf_counter_ns() - started) / 1e9
            if not isfinite(elapsed) or elapsed <= 0.0:
                raise AdaptivePlanningError("adaptive calibration produced invalid timing")
            samples.append(elapsed)
            if first_snapshot is None:
                first_snapshot = snapshot
        assert snapshot is not None
        assert first_snapshot is not None
        if any(
            not bool(torch.isfinite(tensor).all())
            for tensor in snapshot.values()
        ):
            raise AdaptivePlanningError(
                f"adaptive candidate {item.compiler_candidate.strategy!r} "
                "produced non-finite output"
            )
        _assert_outputs_agree(
            first_snapshot,
            snapshot,
            scalar_type=item.plan.config.scalar_type,
        )
        outputs[item.compiler_candidate.strategy] = snapshot
        pending_records.append(
            CalibrationRecord.measured(
                key=item.calibration.key,
                candidate_name=item.compiler_candidate.strategy,
                samples_seconds_per_work_unit=samples,
            )
        )

    reference_name = candidates[0].compiler_candidate.strategy
    reference = outputs[reference_name]
    for item in candidates[1:]:
        _assert_outputs_agree(
            reference,
            outputs[item.compiler_candidate.strategy],
            scalar_type=item.plan.config.scalar_type,
        )
    for record in pending_records:
        cache.put(record)


def specialize_plan_for_hardware(
    spec: JobSpec,
    plan: ConcordPlan,
    *,
    device: str | torch.device | None = None,
    hardware: HardwareProfile | None = None,
    budget_policy: HardwareBudgetPolicy | None = None,
    calibration_policy: AdaptiveCalibrationPolicy | None = None,
    cache_path: str | Path | None = None,
) -> AdaptivePlanSelection:
    """Return the execution plan selected for the current environment.

    Non-adaptive plans pass through unchanged.  Adaptive plans use a complete
    timing set when available.  On a cache miss, bounded synthetic calibration
    is attempted; if calibration is unsafe or unavailable, the deterministic
    analytical rank is used and the reason is surfaced as a warning.
    """

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be a JobSpec")
    if not isinstance(plan, ConcordPlan):
        raise TypeError("plan must be a ConcordPlan")
    plan.validate()
    requested_device = spec.run.device if device is None else device
    target_cache = (
        default_calibration_cache_path() if cache_path is None else Path(cache_path)
    )
    if plan.config.contraction_strategy != "adaptive":
        deployment = resolve_runtime_device(
            plan.physical_plan,
            requested_device,
            hardware=hardware,
            policy=budget_policy,
        )
        return AdaptivePlanSelection(
            plan=plan,
            candidate_strategy=plan.contraction_order.strategy,
            source="not_requested",
            selected_device=deployment.selected_device,
            median_seconds_per_work_unit=None,
            calibration_cache=target_cache,
        )

    profile = detect_hardware_profile() if hardware is None else hardware
    chosen_policy = calibration_policy or AdaptiveCalibrationPolicy()
    candidates = _prepare_candidates(
        plan,
        profile,
        requested_device,
        budget_policy,
    )
    cache = CalibrationCache.load(target_cache)
    warning_messages = list(cache.warnings)
    selection = select_calibration_candidate(
        tuple(item.calibration for item in candidates),
        cache,
    )
    if selection.source != "calibrated":
        try:
            _measure_candidates(
                candidates,
                cache,
                spec.time_contexts[0],
                chosen_policy,
            )
            selection = select_calibration_candidate(
                tuple(item.calibration for item in candidates),
                cache,
            )
            try:
                cache.save()
            except CalibrationCacheError as exc:
                warning_messages.append(str(exc))
        except (AdaptivePlanningError, MemoryError, RuntimeError) as exc:
            warning_messages.append(
                f"adaptive numerical calibration unavailable; used deterministic "
                f"analytical fallback: {exc}"
            )

    selected = next(
        item
        for item in candidates
        if item.compiler_candidate.strategy == selection.candidate.name
    )
    final_plan = specialize_concord_plan(
        plan,
        selected.compiler_candidate.stages,
        diagnostics={
            "adaptive_selection": {
                "candidate_strategy": selected.compiler_candidate.strategy,
                "source": selection.source,
                "selected_device": selected.deployment.selected_device,
                "median_seconds_per_work_unit": selection.median_seconds_per_work_unit,
                "calibration_key": selected.calibration.key.digest,
            }
        },
    )
    for message in warning_messages:
        warnings.warn(message, UserWarning, stacklevel=2)
    return AdaptivePlanSelection(
        plan=final_plan,
        candidate_strategy=selected.compiler_candidate.strategy,
        source=selection.source,
        selected_device=selected.deployment.selected_device,
        median_seconds_per_work_unit=selection.median_seconds_per_work_unit,
        calibration_cache=cache.path,
        warnings=tuple(warning_messages),
    )


__all__ = [
    "AdaptiveCalibrationPolicy",
    "AdaptivePlanSelection",
    "AdaptivePlanningError",
    "specialize_plan_for_hardware",
]
