"""Feasibility-based single-device planning from a hardware snapshot.

The planner uses only exact physical-plan byte counts plus an explicit safety
policy.  It does not benchmark devices or claim that FLOP counts predict
wall-clock speed.  Profile-guided selection with seeded synthetic tensors is a
future layer above this deterministic feasibility boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Literal

import torch

from concord_lqcd.backend.planner.physical import PhysicalExecutionPlan

from .hardware import (
    HardwareDeviceProfile,
    HardwareProfile,
    detect_hardware_profile,
)
from .tiling import TiledExecutionPlan, TilingPlanningError, plan_tiled_execution


_GIB = 1024**3
ExecutionMode = Literal["resident", "host_tiled"]


class ResourcePlanningError(ValueError):
    """Raised when no safe deployment can be derived from supplied resources."""


class InsufficientDeviceMemoryError(MemoryError, ResourcePlanningError):
    """Raised when a requested or automatic device cannot fit the known plan."""


@dataclass(frozen=True)
class MemorySafetyMargin:
    """Memory retained for the OS, allocator, providers, and library workspace."""

    fraction: float
    minimum_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fraction, (int, float))
            or isinstance(self.fraction, bool)
            or not 0.0 <= float(self.fraction) < 1.0
        ):
            raise ValueError("memory safety fraction must be in [0, 1)")
        if (
            not isinstance(self.minimum_bytes, int)
            or isinstance(self.minimum_bytes, bool)
            or self.minimum_bytes < 0
        ):
            raise ValueError("minimum memory safety reserve must be non-negative")

    def reserve(self, available_bytes: int) -> int:
        if available_bytes < 0:
            raise ValueError("available_bytes must be non-negative")
        proportional = ceil(available_bytes * float(self.fraction))
        return min(available_bytes, max(self.minimum_bytes, proportional))


@dataclass(frozen=True)
class HardwareBudgetPolicy:
    """Conservative defaults for one-process, one-device execution.

    These defaults are deliberately policy rather than physics. Applications
    may supply a different immutable policy after measuring their provider and
    contraction workspace requirements.
    """

    cpu_margin: MemorySafetyMargin = field(
        default_factory=lambda: MemorySafetyMargin(0.20, 2 * _GIB)
    )
    cuda_margin: MemorySafetyMargin = field(
        default_factory=lambda: MemorySafetyMargin(0.10, 1 * _GIB)
    )
    cpu_kernel_batch_staging_limit_bytes: int | None = 16 * _GIB
    cuda_kernel_batch_staging_limit_bytes: int | None = 32 * _GIB
    prefer_cuda: bool = True
    allow_host_tiling: bool = True
    raw_eigenvector_cache_limit_bytes: int = 256 * 1024**2

    def __post_init__(self) -> None:
        if not isinstance(self.cpu_margin, MemorySafetyMargin):
            raise TypeError("cpu_margin must be a MemorySafetyMargin")
        if not isinstance(self.cuda_margin, MemorySafetyMargin):
            raise TypeError("cuda_margin must be a MemorySafetyMargin")
        for field_name in (
            "cpu_kernel_batch_staging_limit_bytes",
            "cuda_kernel_batch_staging_limit_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer or None"
                )
        if not isinstance(self.prefer_cuda, bool):
            raise TypeError("prefer_cuda must be boolean")
        if (
            not isinstance(self.raw_eigenvector_cache_limit_bytes, int)
            or isinstance(self.raw_eigenvector_cache_limit_bytes, bool)
            or self.raw_eigenvector_cache_limit_bytes < 0
        ):
            raise ValueError("raw_eigenvector_cache_limit_bytes must be a non-negative integer")
        if not isinstance(self.allow_host_tiling, bool):
            raise TypeError("allow_host_tiling must be boolean")

    def margin_for(self, device: HardwareDeviceProfile) -> MemorySafetyMargin:
        if device.kind == "cpu":
            return self.cpu_margin
        if device.kind == "cuda":
            return self.cuda_margin
        raise ResourcePlanningError(f"unsupported device kind {device.kind!r}")

    def kernel_batch_staging_limit_for(
        self,
        device: HardwareDeviceProfile,
    ) -> int | None:
        """Return the optional provider-staging cap for one memory domain.

        ``None`` delegates the complete measured headroom to a provider. Zero
        disables provider-owned kernel batching without changing execution
        semantics.
        """

        if device.kind == "cpu":
            return self.cpu_kernel_batch_staging_limit_bytes
        if device.kind == "cuda":
            return self.cuda_kernel_batch_staging_limit_bytes
        raise ResourcePlanningError(f"unsupported device kind {device.kind!r}")


@dataclass(frozen=True)
class ExecutionMemoryEstimate:
    """Known resident-memory requirement of the numerical executor.

    Resident execution allocates every physical buffer up front. Each operation
    then produces a tensor before copying it into the assigned buffer, so the
    largest scheduled value is one exact additional transient. Host-tiled
    execution uses this as a conservative host-RAM prerequisite. Provider-
    internal and backend-library workspace remains covered only by the policy
    reserve.
    """

    buffer_pool_bytes: int
    largest_transient_value_bytes: int
    known_working_set_bytes: int

    def __post_init__(self) -> None:
        for field_name in (
            "buffer_pool_bytes",
            "largest_transient_value_bytes",
            "known_working_set_bytes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.known_working_set_bytes != (
            self.buffer_pool_bytes + self.largest_transient_value_bytes
        ):
            raise ValueError("known working-set bytes disagree with their components")


@dataclass(frozen=True)
class DeviceMemoryBudget:
    """Plan feasibility and safe capacity for one detected memory domain."""

    device: HardwareDeviceProfile
    estimate: ExecutionMemoryEstimate
    safety_reserve_bytes: int | None
    safe_working_set_limit_bytes: int | None
    recommended_buffer_limit_bytes: int | None
    headroom_after_known_plan_bytes: int | None
    fits: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.device, HardwareDeviceProfile):
            raise TypeError("device budget requires a HardwareDeviceProfile")
        if not isinstance(self.estimate, ExecutionMemoryEstimate):
            raise TypeError("device budget requires an ExecutionMemoryEstimate")
        optional_bytes = (
            self.safety_reserve_bytes,
            self.safe_working_set_limit_bytes,
            self.recommended_buffer_limit_bytes,
            self.headroom_after_known_plan_bytes,
        )
        if self.fits is None:
            if any(value is not None for value in optional_bytes):
                raise ValueError("unverified device budget cannot contain derived bytes")
            return
        if not isinstance(self.fits, bool):
            raise TypeError("device budget fits state must be boolean or None")
        if any(value is None for value in optional_bytes):
            raise ValueError("verified device budget is missing derived bytes")
        assert self.safety_reserve_bytes is not None
        assert self.safe_working_set_limit_bytes is not None
        assert self.recommended_buffer_limit_bytes is not None
        assert self.headroom_after_known_plan_bytes is not None
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in optional_bytes
        ):
            raise TypeError("device budget byte fields must be integers")
        if min(
            self.safety_reserve_bytes,
            self.safe_working_set_limit_bytes,
            self.recommended_buffer_limit_bytes,
        ) < 0:
            raise ValueError("device budget reserve and limits must be non-negative")
        if self.fits != (self.headroom_after_known_plan_bytes >= 0):
            raise ValueError("device budget feasibility disagrees with its headroom")

    @property
    def memory_verified(self) -> bool:
        return self.fits is not None


@dataclass(frozen=True)
class DeviceExecutionPlan:
    """Selected deployment for one frozen physical plan.

    ``resident`` keeps the physical buffer pool and compute on the selected
    device. ``host_tiled`` keeps the complete pool on CPU and executes bounded
    contraction tiles on the selected CUDA device.
    """

    requested_device: str
    selected_device: str
    estimate: ExecutionMemoryEstimate
    selected_budget: DeviceMemoryBudget
    candidate_budgets: tuple[DeviceMemoryBudget, ...]
    policy: HardwareBudgetPolicy
    execution_mode: ExecutionMode
    storage_device: str
    tiled_plan: TiledExecutionPlan | None = None
    hardware_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_budgets", tuple(self.candidate_budgets))
        object.__setattr__(self, "hardware_warnings", tuple(self.hardware_warnings))
        if not self.requested_device or not self.selected_device:
            raise ValueError("requested and selected devices must be non-empty")
        if self.selected_budget.device.device != self.selected_device:
            raise ValueError("selected device disagrees with selected memory budget")
        if self.selected_budget.estimate != self.estimate:
            raise ValueError("selected memory budget uses a different estimate")
        if self.selected_budget not in self.candidate_budgets:
            raise ValueError("selected memory budget is absent from candidate budgets")
        if any(budget.estimate != self.estimate for budget in self.candidate_budgets):
            raise ValueError("candidate device budgets use inconsistent estimates")
        if not isinstance(self.policy, HardwareBudgetPolicy):
            raise TypeError("execution plan policy must be a HardwareBudgetPolicy")
        if self.execution_mode not in ("resident", "host_tiled"):
            raise ValueError("execution mode must be resident or host_tiled")
        if not self.storage_device:
            raise ValueError("execution storage device must be non-empty")
        if self.execution_mode == "resident":
            if self.storage_device != self.selected_device:
                raise ValueError("resident execution must compute and store on one device")
            if self.tiled_plan is not None:
                raise ValueError("resident execution cannot contain a tiled plan")
            if self.selected_budget.fits is False:
                raise ValueError("resident execution exceeds its verified device budget")
        else:
            if self.selected_budget.device.kind != "cuda":
                raise ValueError("host-tiled execution requires a CUDA compute device")
            if self.selected_budget.fits is not False:
                raise ValueError(
                    "host-tiled execution requires a verified non-resident CUDA plan"
                )
            if self.storage_device != "cpu":
                raise ValueError("host-tiled execution requires CPU-resident storage")
            if not isinstance(self.tiled_plan, TiledExecutionPlan):
                raise TypeError("host-tiled execution requires a TiledExecutionPlan")
            host_budget = next(
                (
                    budget
                    for budget in self.candidate_budgets
                    if budget.device.device == "cpu"
                ),
                None,
            )
            if host_budget is None or host_budget.fits is not True:
                raise ValueError("host-tiled execution requires a verified fitting host")
            if self.selected_budget.safe_working_set_limit_bytes is None:
                raise ValueError("host-tiled execution requires measured CUDA memory")
            if self.tiled_plan.device_memory_limit_bytes != (
                self.selected_budget.safe_working_set_limit_bytes
            ):
                raise ValueError("tiled plan uses a different CUDA memory limit")
            if self.tiled_plan.peak_device_bytes > (
                self.selected_budget.safe_working_set_limit_bytes
            ):
                raise ValueError("tiled execution exceeds the selected CUDA budget")
        if any(
            not isinstance(item, str) or not item
            for item in self.hardware_warnings
        ):
            raise ValueError("hardware warnings must be non-empty strings")

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.selected_device)

    @property
    def storage_torch_device(self) -> torch.device:
        return torch.device(self.storage_device)

    @property
    def memory_verified(self) -> bool:
        if self.execution_mode == "resident":
            return self.selected_budget.memory_verified
        return True

    @property
    def uses_index_tiling(self) -> bool:
        return self.tiled_plan is not None and self.tiled_plan.uses_index_tiling


def estimate_execution_memory(plan: PhysicalExecutionPlan) -> ExecutionMemoryEstimate:
    """Return the exact currently modeled pool-plus-result working set."""

    if not isinstance(plan, PhysicalExecutionPlan):
        raise TypeError("plan must be a PhysicalExecutionPlan")
    plan.validate()
    largest_value = max(
        (binding.size_bytes for binding in plan.value_bindings.values()),
        default=0,
    )
    pool = plan.statistics.total_buffer_capacity_bytes
    return ExecutionMemoryEstimate(
        buffer_pool_bytes=pool,
        largest_transient_value_bytes=largest_value,
        known_working_set_bytes=pool + largest_value,
    )


def _device_budget(
    device: HardwareDeviceProfile,
    estimate: ExecutionMemoryEstimate,
    policy: HardwareBudgetPolicy,
) -> DeviceMemoryBudget:
    available = device.available_memory_bytes
    if not device.memory_is_known:
        return DeviceMemoryBudget(
            device=device,
            estimate=estimate,
            safety_reserve_bytes=None,
            safe_working_set_limit_bytes=None,
            recommended_buffer_limit_bytes=None,
            headroom_after_known_plan_bytes=None,
            fits=None,
        )
    assert available is not None
    reserve = policy.margin_for(device).reserve(available)
    safe_limit = available - reserve
    buffer_limit = max(0, safe_limit - estimate.largest_transient_value_bytes)
    headroom = safe_limit - estimate.known_working_set_bytes
    return DeviceMemoryBudget(
        device=device,
        estimate=estimate,
        safety_reserve_bytes=reserve,
        safe_working_set_limit_bytes=safe_limit,
        recommended_buffer_limit_bytes=buffer_limit,
        headroom_after_known_plan_bytes=headroom,
        fits=headroom >= 0,
    )


def _canonical_requested_device(value: str | torch.device) -> str:
    if isinstance(value, str) and value == "auto":
        return "auto"
    try:
        device = value if isinstance(value, torch.device) else torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ResourcePlanningError(f"invalid runtime device {value!r}") from exc
    if device.type == "cpu":
        if device.index not in (None, 0):
            raise ResourcePlanningError("CPU device indices other than zero are unsupported")
        return "cpu"
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else device.index}"
    raise ResourcePlanningError(
        f"automatic resource planning does not support device type {device.type!r}"
    )


def _automatic_rank(budget: DeviceMemoryBudget, policy: HardwareBudgetPolicy) -> tuple:
    kind_rank = 0 if policy.prefer_cuda and budget.device.kind == "cuda" else 1
    if not policy.prefer_cuda and budget.device.kind == "cpu":
        kind_rank = 0
    headroom = budget.headroom_after_known_plan_bytes
    assert headroom is not None
    return kind_rank, -headroom, budget.device.device


def _host_tiled_candidate(
    plan: PhysicalExecutionPlan,
    accelerator: DeviceMemoryBudget,
    budgets: tuple[DeviceMemoryBudget, ...],
    policy: HardwareBudgetPolicy,
) -> TiledExecutionPlan | None:
    if not policy.allow_host_tiling or accelerator.device.kind != "cuda":
        return None
    safe_limit = accelerator.safe_working_set_limit_bytes
    if safe_limit is None or safe_limit <= 0:
        return None
    host = next(
        (budget for budget in budgets if budget.device.device == "cpu"),
        None,
    )
    # The full resident CPU estimate is conservative for host-staged execution:
    # tiled contraction outputs are written directly into the pool, but using
    # this stronger condition avoids overstating host feasibility.
    if host is None or host.fits is not True:
        return None
    try:
        return plan_tiled_execution(
            plan,
            device_memory_limit_bytes=safe_limit,
        )
    except TilingPlanningError:
        return None


def plan_execution_resources(
    plan: PhysicalExecutionPlan,
    hardware: HardwareProfile,
    *,
    requested_device: str | torch.device = "auto",
    policy: HardwareBudgetPolicy | None = None,
) -> DeviceExecutionPlan:
    """Select and validate a resident or host-tiled deployment.

    Automatic selection considers only devices with measured available memory.
    An explicit device with unknown memory is preserved but marked unverified;
    an explicit CUDA device that cannot hold the resident plan may retain
    complete values in verified host RAM and execute bounded contraction tiles.
    Explicit requests never switch to a different compute device.
    """

    if not isinstance(hardware, HardwareProfile):
        raise TypeError("hardware must be a HardwareProfile")
    chosen_policy = policy or HardwareBudgetPolicy()
    if not isinstance(chosen_policy, HardwareBudgetPolicy):
        raise TypeError("policy must be a HardwareBudgetPolicy")
    estimate = estimate_execution_memory(plan)
    budgets = tuple(
        _device_budget(device, estimate, chosen_policy)
        for device in hardware.devices
    )
    canonical = _canonical_requested_device(requested_device)
    execution_mode: ExecutionMode = "resident"
    storage_device: str
    tiled_plan: TiledExecutionPlan | None = None

    if canonical == "auto":
        resident_cuda = [
            budget
            for budget in budgets
            if budget.device.kind == "cuda" and budget.fits is True
        ]
        resident_cpu = next(
            (
                budget
                for budget in budgets
                if budget.device.kind == "cpu" and budget.fits is True
            ),
            None,
        )
        tiled_cuda = [
            (budget, candidate)
            for budget in budgets
            if budget.device.kind == "cuda" and budget.fits is False
            for candidate in [_host_tiled_candidate(plan, budget, budgets, chosen_policy)]
            if candidate is not None
        ]

        if chosen_policy.prefer_cuda and resident_cuda:
            selected = min(
                resident_cuda,
                key=lambda item: _automatic_rank(item, chosen_policy),
            )
        elif chosen_policy.prefer_cuda and tiled_cuda:
            selected, tiled_plan = min(
                tiled_cuda,
                key=lambda item: (
                    item[1].total_contraction_calls,
                    -item[1].device_memory_limit_bytes,
                    item[0].device.device,
                ),
            )
            execution_mode = "host_tiled"
        elif resident_cpu is not None:
            selected = resident_cpu
        elif resident_cuda:
            selected = min(
                resident_cuda,
                key=lambda item: _automatic_rank(item, chosen_policy),
            )
        elif tiled_cuda:
            selected, tiled_plan = min(
                tiled_cuda,
                key=lambda item: (
                    item[1].total_contraction_calls,
                    -item[1].device_memory_limit_bytes,
                    item[0].device.device,
                ),
            )
            execution_mode = "host_tiled"
        else:
            known = [budget for budget in budgets if budget.memory_verified]
            if known:
                best = max(
                    known,
                    key=lambda item: item.safe_working_set_limit_bytes or 0,
                )
                raise InsufficientDeviceMemoryError(
                    "No detected device safely fits the current execution plan: "
                    f"known working set {estimate.known_working_set_bytes} bytes; "
                    f"largest safe device limit {best.safe_working_set_limit_bytes} "
                    f"bytes on {best.device.device}."
                )
            raise ResourcePlanningError(
                "Automatic device selection requires a measurable available-memory "
                "snapshot, but none of the detected devices supplied one."
            )
    else:
        selected = next(
            (budget for budget in budgets if budget.device.device == canonical),
            None,
        )
        if selected is None:
            detected = ", ".join(device.device for device in hardware.devices)
            raise ResourcePlanningError(
                f"Requested device {canonical!r} was not detected; available devices: "
                f"{detected or 'none'}."
            )
        if selected.fits is False:
            tiled_plan = _host_tiled_candidate(
                plan,
                selected,
                budgets,
                chosen_policy,
            )
            if tiled_plan is None:
                staging_detail = (
                    ", and no legal host-tiled deployment fits"
                    if selected.device.kind == "cuda"
                    else ""
                )
                raise InsufficientDeviceMemoryError(
                    f"Requested device {canonical!r} cannot safely fit the current plan: "
                    f"known working set {estimate.known_working_set_bytes} bytes, safe "
                    f"limit {selected.safe_working_set_limit_bytes} bytes"
                    f"{staging_detail}. Explicit device requests do not "
                    "fall back to a different compute device."
                )
            execution_mode = "host_tiled"

    storage_device = "cpu" if execution_mode == "host_tiled" else selected.device.device

    return DeviceExecutionPlan(
        requested_device=canonical,
        selected_device=selected.device.device,
        estimate=estimate,
        selected_budget=selected,
        candidate_budgets=budgets,
        policy=chosen_policy,
        execution_mode=execution_mode,
        storage_device=storage_device,
        tiled_plan=tiled_plan,
        hardware_warnings=hardware.warnings,
    )


def resolve_runtime_device(
    plan: PhysicalExecutionPlan,
    requested_device: str | torch.device,
    *,
    hardware: HardwareProfile | None = None,
    policy: HardwareBudgetPolicy | None = None,
) -> DeviceExecutionPlan:
    """Detect hardware when necessary and return a validated deployment plan."""

    profile = detect_hardware_profile() if hardware is None else hardware
    return plan_execution_resources(
        plan,
        profile,
        requested_device=requested_device,
        policy=policy,
    )


def revalidate_execution_resources(
    plan: PhysicalExecutionPlan,
    deployment: DeviceExecutionPlan,
    *,
    hardware: HardwareProfile | None = None,
) -> DeviceExecutionPlan:
    """Refresh availability for the already-selected device without fallback."""

    if not isinstance(deployment, DeviceExecutionPlan):
        raise TypeError("deployment must be a DeviceExecutionPlan")
    current_estimate = estimate_execution_memory(plan)
    if current_estimate != deployment.estimate:
        raise ResourcePlanningError(
            "execution resource plan belongs to a different physical plan"
        )
    return resolve_runtime_device(
        plan,
        deployment.selected_device,
        hardware=hardware,
        policy=deployment.policy,
    )


__all__ = [
    "ExecutionMode",
    "ResourcePlanningError",
    "InsufficientDeviceMemoryError",
    "MemorySafetyMargin",
    "HardwareBudgetPolicy",
    "ExecutionMemoryEstimate",
    "DeviceMemoryBudget",
    "DeviceExecutionPlan",
    "estimate_execution_memory",
    "plan_execution_resources",
    "resolve_runtime_device",
    "revalidate_execution_resources",
]
