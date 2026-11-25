"""Conservative memory feasibility for runtime-only work-unit batching."""

from __future__ import annotations

from dataclasses import dataclass

from concord_lqcd.backend.planner.physical import (
    MaterializePrimitiveOperation,
    PhysicalExecutionPlan,
)

from .resource_planning import DeviceExecutionPlan, estimate_execution_memory


class BatchPlanningError(ValueError):
    """Raised when an explicit or automatic batch request is unsafe."""


@dataclass(frozen=True)
class BatchedExecutionMemoryEstimate:
    """Conservative resident-memory estimate for one leading-axis batch."""

    batch_size: int
    buffer_pool_bytes: int
    operation_transient_bytes: int
    primitive_staging_bytes: int
    known_peak_bytes: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch memory estimate requires a positive batch size")
        for field_name in (
            "buffer_pool_bytes",
            "operation_transient_bytes",
            "primitive_staging_bytes",
            "known_peak_bytes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        expected = self.buffer_pool_bytes + max(
            self.operation_transient_bytes,
            self.primitive_staging_bytes,
        )
        if self.known_peak_bytes != expected:
            raise ValueError("batched memory peak disagrees with its components")


@dataclass(frozen=True)
class TimeBatchCandidate:
    batch_size: int
    memory: BatchedExecutionMemoryEstimate

    def __post_init__(self) -> None:
        if self.batch_size != self.memory.batch_size:
            raise ValueError("time-batch candidate size disagrees with its estimate")


@dataclass(frozen=True)
class TimeBatchFeasibilityPlan:
    """Memory-feasible candidate sizes before empirical selection."""

    requested_batch_size: str | int
    candidates: tuple[TimeBatchCandidate, ...]
    selected_device: str
    execution_mode: str
    safe_memory_limit_bytes: int | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates or self.candidates[0].batch_size != 1:
            raise ValueError("time-batch feasibility must retain batch size one")
        sizes = tuple(candidate.batch_size for candidate in self.candidates)
        if sizes != tuple(sorted(set(sizes))):
            raise ValueError("time-batch candidate sizes must be unique and sorted")
        if not self.selected_device or not self.reason:
            raise ValueError("time-batch feasibility metadata must be non-empty")
        if self.execution_mode not in ("resident", "host_tiled"):
            raise ValueError("time-batch feasibility has an unknown execution mode")
        if (
            self.safe_memory_limit_bytes is not None
            and self.safe_memory_limit_bytes < 0
        ):
            raise ValueError("time-batch safe memory limit cannot be negative")

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(candidate.batch_size for candidate in self.candidates)


def estimate_batched_execution_memory(
    plan: PhysicalExecutionPlan,
    batch_size: int,
) -> BatchedExecutionMemoryEstimate:
    """Estimate pool, contraction temporary, and primitive stack staging.

    Batch size one is executed by the established sequential executor and uses
    its existing estimate.  For a true batch, every physical buffer and
    operation result carries the leading axis.  Primitive stacking can briefly
    retain both the per-context tensors and their dense stacked tensor, so two
    copies of the largest primitive are included conservatively.
    """

    if not isinstance(plan, PhysicalExecutionPlan):
        raise TypeError("plan must be a PhysicalExecutionPlan")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    plan.validate()
    base = estimate_execution_memory(plan)
    if batch_size == 1:
        return BatchedExecutionMemoryEstimate(
            batch_size=1,
            buffer_pool_bytes=base.buffer_pool_bytes,
            operation_transient_bytes=base.largest_transient_value_bytes,
            primitive_staging_bytes=0,
            known_peak_bytes=base.known_working_set_bytes,
        )

    largest_primitive = max(
        (
            plan.binding(step.operation.value_id).size_bytes
            for step in plan.steps
            if isinstance(step.operation, MaterializePrimitiveOperation)
        ),
        default=0,
    )
    pool_bytes = batch_size * base.buffer_pool_bytes
    operation_bytes = batch_size * base.largest_transient_value_bytes
    primitive_bytes = 2 * batch_size * largest_primitive
    return BatchedExecutionMemoryEstimate(
        batch_size=batch_size,
        buffer_pool_bytes=pool_bytes,
        operation_transient_bytes=operation_bytes,
        primitive_staging_bytes=primitive_bytes,
        known_peak_bytes=pool_bytes + max(operation_bytes, primitive_bytes),
    )


def _candidate_sizes(maximum: int) -> tuple[int, ...]:
    values = [1]
    value = 2
    while value <= maximum:
        values.append(value)
        value *= 2
    if maximum > 1 and values[-1] != maximum:
        values.append(maximum)
    return tuple(sorted(set(values)))


def plan_time_batch_candidates(
    plan: PhysicalExecutionPlan,
    deployment: DeviceExecutionPlan,
    *,
    work_units: int,
    requested_batch_size: str | int = "auto",
) -> TimeBatchFeasibilityPlan:
    """Return safe candidate sizes for one configuration's time contexts.

    Host-staged execution and unknown memory measurements retain the proven
    sequential path.  An explicit value greater than one fails instead of
    silently changing the request under either condition.
    """

    if not isinstance(plan, PhysicalExecutionPlan):
        raise TypeError("plan must be a PhysicalExecutionPlan")
    if not isinstance(deployment, DeviceExecutionPlan):
        raise TypeError("deployment must be a DeviceExecutionPlan")
    if deployment.estimate != estimate_execution_memory(plan):
        raise BatchPlanningError("deployment belongs to a different physical plan")
    if (
        not isinstance(work_units, int)
        or isinstance(work_units, bool)
        or work_units <= 0
    ):
        raise ValueError("work_units must be a positive integer")
    if requested_batch_size != "auto" and (
        not isinstance(requested_batch_size, int)
        or isinstance(requested_batch_size, bool)
        or requested_batch_size <= 0
    ):
        raise BatchPlanningError("time batch size must be 'auto' or a positive integer")

    sequential = TimeBatchCandidate(1, estimate_batched_execution_memory(plan, 1))
    safe_limit = deployment.selected_budget.safe_working_set_limit_bytes
    if work_units == 1:
        reason = "one time context is available"
        candidates = (sequential,)
    elif deployment.execution_mode != "resident":
        reason = "host-staged execution is not yet batch-aware"
        candidates = (sequential,)
    elif safe_limit is None:
        reason = "available device memory is not measurable"
        candidates = (sequential,)
    else:
        feasible = []
        for size in _candidate_sizes(work_units):
            estimate = estimate_batched_execution_memory(plan, size)
            if size == 1 or estimate.known_peak_bytes <= safe_limit:
                feasible.append(TimeBatchCandidate(size, estimate))
        candidates = tuple(feasible)
        reason = "candidate sizes are bounded by the current safe resident-memory limit"

    if isinstance(requested_batch_size, int):
        selected_size = min(requested_batch_size, work_units)
        estimate = estimate_batched_execution_memory(plan, selected_size)
        supported = (
            selected_size == 1
            or (
                deployment.execution_mode == "resident"
                and safe_limit is not None
                and estimate.known_peak_bytes <= safe_limit
            )
        )
        if not supported:
            if deployment.execution_mode != "resident" or safe_limit is None:
                detail = reason
            else:
                detail = (
                    f"estimated peak {estimate.known_peak_bytes} bytes exceeds "
                    f"safe limit {safe_limit} bytes"
                )
            raise BatchPlanningError(
                f"explicit time batch size {requested_batch_size} is unavailable: {detail}"
            )
        selected = TimeBatchCandidate(selected_size, estimate)
        candidates = (sequential,) if selected_size == 1 else (sequential, selected)
        reason = "explicit batch size was validated against the current deployment"

    return TimeBatchFeasibilityPlan(
        requested_batch_size=requested_batch_size,
        candidates=candidates,
        selected_device=deployment.selected_device,
        execution_mode=deployment.execution_mode,
        safe_memory_limit_bytes=safe_limit,
        reason=reason,
    )


__all__ = [
    "BatchPlanningError",
    "BatchedExecutionMemoryEstimate",
    "TimeBatchCandidate",
    "TimeBatchFeasibilityPlan",
    "estimate_batched_execution_memory",
    "plan_time_batch_candidates",
]
