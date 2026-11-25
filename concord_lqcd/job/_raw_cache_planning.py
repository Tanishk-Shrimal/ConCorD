"""Coordinate resident time batches with configuration-local raw-field storage."""

from dataclasses import dataclass
from math import prod
from typing import Sequence

from concord_lqcd.backend.planner.physical import PhysicalExecutionPlan
from concord_lqcd.backend.runtime.batching import estimate_batched_execution_memory
from concord_lqcd.backend.runtime.resource_binding import KernelMaterializationRequest
from concord_lqcd.backend.runtime.resource_planning import DeviceExecutionPlan

from .data import HDF5DistillationKernelProvider, JobRuntimeResources


@dataclass(frozen=True)
class RawEigenvectorCacheAdmission:
    batch_size: int
    provider_limits: tuple[tuple[HDF5DistillationKernelProvider, int], ...]
    warning: str | None = None

    @property
    def reserved_bytes(self) -> int:
        return sum(limit for _, limit in self.provider_limits)


def plan_raw_eigenvector_cache(
    resources: Sequence[JobRuntimeResources],
    physical_plan: PhysicalExecutionPlan,
    deployment: DeviceExecutionPlan,
    *,
    batch_size: int,
    automatic_batch: bool,
    element_bytes: int,
) -> RawEigenvectorCacheAdmission:
    """Reserve complete raw-field working sets before other optional caches.

    Only exact built-in providers participate. Automatic batches may shrink;
    explicit batches retain their requested size even when only partial raw
    retention fits. No tensor or scientific dataset is loaded by this planner.
    """

    values = tuple(resources)
    safe_limit = deployment.selected_budget.safe_working_set_limit_bytes
    policy_limit = deployment.policy.raw_eigenvector_cache_limit_bytes
    if (
        not values or not policy_limit or safe_limit is None
        or deployment.execution_mode != "resident"
    ):
        return RawEigenvectorCacheAdmission(batch_size, ())

    providers: dict[int, HDF5DistillationKernelProvider] = {}
    provider_times: dict[int, list[set[int]]] = {}
    reused_providers: set[int] = set()
    for context_index, resource in enumerate(values):
        binding = resource.binding_plan
        if binding is None:
            raise ValueError("raw cache planning requires prebound resources")
        for request in binding.materializations.values():
            if not isinstance(request, KernelMaterializationRequest):
                continue
            provider = resource.kernel_providers[request.provider_id]
            if type(provider) is not HDF5DistillationKernelProvider:
                continue
            identity = id(provider)
            if identity not in providers:
                providers[identity] = provider
                provider_times[identity] = [set() for _ in values]
            times = set(request.operator_times or (0,))
            context_times = provider_times[identity][context_index]
            if times.intersection(context_times):
                reused_providers.add(identity)
            context_times.update(times)
    if not providers:
        return RawEigenvectorCacheAdmission(batch_size, ())

    identities = tuple(providers)
    field_bytes = tuple(
        prod(providers[identity].spatial_extent)
        * 3 * providers[identity].mode_size * element_bytes
        for identity in identities
    )
    context_times = tuple(provider_times[identity] for identity in identities)
    total_bytes = tuple(
        len(set().union(*times)) * size
        for times, size in zip(context_times, field_bytes, strict=True)
    )

    def available_bytes(size: int) -> int:
        peak = estimate_batched_execution_memory(
            physical_plan, size
        ).known_peak_bytes
        return min(policy_limit, max(0, safe_limit - peak))

    def required_bytes(size: int) -> tuple[int, ...]:
        return tuple(
            max(
                len(set().union(*times[start : start + size]))
                for start in range(0, len(values), size)
            ) * field_size if identity in reused_providers else 0
            for identity, times, field_size in zip(
                identities, context_times, field_bytes, strict=True
            )
        )

    requested_size = batch_size
    capacity = available_bytes(batch_size)
    required = required_bytes(batch_size)
    if available_bytes(1) < min(field_bytes):
        return RawEigenvectorCacheAdmission(batch_size, ())
    while automatic_batch and batch_size > 1 and sum(required) > capacity:
        batch_size = 1 << ((batch_size - 1).bit_length() - 1)
        capacity = available_bytes(batch_size)
        required = required_bytes(batch_size)

    if sum(required) <= capacity:
        limits = list(required)
    else:
        share = capacity // len(identities)
        limits = [
            min(total, share // size * size)
            for total, size in zip(total_bytes, field_bytes, strict=True)
        ]
    remaining = capacity - sum(limits)
    for index, (total, size) in enumerate(zip(total_bytes, field_bytes, strict=True)):
        extra = min(total - limits[index], remaining // size * size)
        limits[index] += extra
        remaining -= extra

    warning = None
    if batch_size != requested_size:
        warning = (
            f"automatic time batch reduced from {requested_size} to {batch_size} "
            "to respect the raw eigenvector cache working-set budget"
        )
        if sum(required) > capacity:
            warning += "; even batch size one requires partial raw retention"
    elif batch_size > 1 and sum(required) > capacity:
        warning = (
            f"explicit time batch size {batch_size} needs {sum(required)} bytes "
            f"for its raw eigenvector working set, but only {capacity} bytes "
            "are available within the cache and safe-memory limits; keeping "
            "the explicit batch unchanged with partial raw retention. Use "
            "time_batch_size: auto or increase the raw cache policy limit "
            "within the available memory budget to avoid repeated reads"
        )
    return RawEigenvectorCacheAdmission(
        batch_size,
        tuple(
            (providers[identity], limit)
            for identity, limit in zip(identities, limits, strict=True)
        ),
        warning,
    )
