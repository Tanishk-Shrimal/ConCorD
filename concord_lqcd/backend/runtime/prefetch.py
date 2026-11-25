"""Bounded asynchronous propagation prefetch for resident execution.

The physical plan remains unchanged.  This module stages only ordinary
propagation tensors served by ConCorD's built-in HDF5 handler, and only one time
batch ahead of execution. Exact primitive-cache residency has priority;
prefetch planning uses the measured memory headroom left after those reusable
tensors. Generalized propagation remains behind its separate validation gate.

CPU execution overlaps one background HDF5 reader with current contractions.
CUDA execution additionally uses pinned host memory and a dedicated transfer
stream, then synchronizes the transfer event before installing the tensors in
their request-scoped materializers.  Unsupported/custom handlers retain the
proven synchronous path.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Sequence

import torch

from .materialization import PrimitiveMaterializer, materialize_primitive_batch
from .resource_planning import DeviceExecutionPlan


@dataclass(frozen=True)
class PropagationPrefetchPlan:
    """Exact depth-one staging decision for one configuration."""

    enabled: bool
    reason: str
    batch_size: int
    batch_count: int
    eligible_requests: int
    maximum_batch_requests: int
    staged_tensor_bytes: int
    maximum_tensor_bytes: int
    host_staging_bytes: int
    device_staging_bytes: int
    available_device_headroom_bytes: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("prefetch enabled state must be boolean")
        if not self.reason:
            raise ValueError("prefetch plan reason must be non-empty")
        for field_name in (
            "batch_size",
            "batch_count",
            "eligible_requests",
            "maximum_batch_requests",
            "staged_tensor_bytes",
            "maximum_tensor_bytes",
            "host_staging_bytes",
            "device_staging_bytes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.batch_size <= 0:
            raise ValueError("prefetch batch size must be positive")
        if self.batch_count <= 0:
            raise ValueError("prefetch batch count must be positive")
        headroom = self.available_device_headroom_bytes
        if headroom is not None and (
            not isinstance(headroom, int)
            or isinstance(headroom, bool)
            or headroom < 0
        ):
            raise ValueError(
                "available_device_headroom_bytes must be non-negative or None"
            )
        if self.maximum_batch_requests > self.eligible_requests:
            raise ValueError("maximum prefetch batch requests exceed total requests")
        if self.maximum_tensor_bytes > self.staged_tensor_bytes:
            raise ValueError("maximum tensor exceeds the complete staging allocation")
        if self.enabled:
            if self.batch_count < 2:
                raise ValueError("enabled prefetch requires at least two batches")
            if self.maximum_batch_requests == 0 or self.staged_tensor_bytes == 0:
                raise ValueError("enabled prefetch requires eligible tensors")
            if headroom is None or self.device_staging_bytes > headroom:
                raise ValueError("enabled prefetch exceeds device staging headroom")


def _disabled_plan(
    *,
    reason: str,
    batch_size: int,
    batch_count: int,
    eligible_requests: int = 0,
    maximum_batch_requests: int = 0,
    staged_tensor_bytes: int = 0,
    maximum_tensor_bytes: int = 0,
    host_staging_bytes: int = 0,
    device_staging_bytes: int = 0,
    available_device_headroom_bytes: int | None = None,
) -> PropagationPrefetchPlan:
    return PropagationPrefetchPlan(
        enabled=False,
        reason=reason,
        batch_size=batch_size,
        batch_count=batch_count,
        eligible_requests=eligible_requests,
        maximum_batch_requests=maximum_batch_requests,
        staged_tensor_bytes=staged_tensor_bytes,
        maximum_tensor_bytes=maximum_tensor_bytes,
        host_staging_bytes=host_staging_bytes,
        device_staging_bytes=device_staging_bytes,
        available_device_headroom_bytes=available_device_headroom_bytes,
    )


def plan_propagation_prefetch(
    materializers: Sequence[PrimitiveMaterializer],
    *,
    batch_size: int,
    execution_plan: DeviceExecutionPlan,
    available_device_headroom_bytes: int | None,
) -> PropagationPrefetchPlan:
    """Plan one next-batch HDF5 staging allocation from measured headroom.

    ``available_device_headroom_bytes`` is the selected device capacity left
    after the active resident batch and the exact primitive-cache admission
    plan.  For CUDA the host pinned-memory requirement is checked independently
    against the detected CPU safe working-set limit.
    """

    values = tuple(materializers)
    if not values:
        raise ValueError("propagation prefetch planning requires materializers")
    if any(not isinstance(item, PrimitiveMaterializer) for item in values):
        raise TypeError(
            "propagation prefetch planning requires PrimitiveMaterializer entries"
        )
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("prefetch batch_size must be a positive integer")
    if not isinstance(execution_plan, DeviceExecutionPlan):
        raise TypeError("prefetch planning requires a DeviceExecutionPlan")
    if available_device_headroom_bytes is not None and (
        not isinstance(available_device_headroom_bytes, int)
        or isinstance(available_device_headroom_bytes, bool)
        or available_device_headroom_bytes < 0
    ):
        raise ValueError(
            "available_device_headroom_bytes must be non-negative or None"
        )

    source_plan = values[0].binding_plan.source_plan
    if any(item.binding_plan.source_plan is not source_plan for item in values[1:]):
        raise ValueError("prefetch materializers must bind the same physical plan")
    if execution_plan.execution_mode != "resident":
        raise ValueError("propagation prefetch planning requires resident execution")

    batch_count = ceil(len(values) / batch_size)
    if batch_count < 2:
        return _disabled_plan(
            reason="one resident batch is sufficient",
            batch_size=batch_size,
            batch_count=batch_count,
            available_device_headroom_bytes=available_device_headroom_bytes,
        )

    eligible_requests = 0
    maximum_batch_requests = 0
    staged_tensor_bytes = 0
    maximum_tensor_bytes = 0
    batch_tensor_bytes: list[int] = []
    batch_maximum_tensor_bytes: list[int] = []
    for cursor in range(0, len(values), batch_size):
        batch_requests = 0
        batch_bytes = 0
        batch_maximum = 0
        for materializer in values[cursor : cursor + batch_size]:
            for value_id in materializer.prefetchable_hdf5_propagation_values():
                size_bytes = source_plan.binding(value_id).size_bytes
                batch_requests += 1
                batch_bytes += size_bytes
                batch_maximum = max(batch_maximum, size_bytes)
                maximum_tensor_bytes = max(maximum_tensor_bytes, size_bytes)
        eligible_requests += batch_requests
        maximum_batch_requests = max(maximum_batch_requests, batch_requests)
        staged_tensor_bytes = max(staged_tensor_bytes, batch_bytes)
        batch_tensor_bytes.append(batch_bytes)
        batch_maximum_tensor_bytes.append(batch_maximum)

    if maximum_batch_requests == 0:
        return _disabled_plan(
            reason="no uncached ordinary HDF5 propagation tensors are eligible",
            batch_size=batch_size,
            batch_count=batch_count,
            available_device_headroom_bytes=available_device_headroom_bytes,
        )

    selected_kind = execution_plan.selected_budget.device.kind
    # Batch N's installed values may remain live while the single worker
    # finishes batch N+1.  Bound the actual double buffer by adjacent batches,
    # including a zero-request batch, instead of assuming two maximum batches.
    device_staging_bytes = max(
        (
            batch_tensor_bytes[index] + batch_tensor_bytes[index + 1]
            for index in range(len(batch_tensor_bytes) - 1)
        ),
        default=staged_tensor_bytes,
    )
    if selected_kind == "cuda":
        # Pinning proceeds one tensor at a time, so host peak for one transfer is
        # all pinned tensors plus its largest current pageable HDF5 tensor.  The
        # installed current batch is already device-resident, not host-resident.
        host_staging_bytes = max(
            batch_bytes + batch_maximum
            for batch_bytes, batch_maximum in zip(
                batch_tensor_bytes,
                batch_maximum_tensor_bytes,
            )
        )
    elif selected_kind == "cpu":
        host_staging_bytes = device_staging_bytes
    else:
        return _disabled_plan(
            reason=f"selected device kind {selected_kind!r} has no prefetch backend",
            batch_size=batch_size,
            batch_count=batch_count,
            eligible_requests=eligible_requests,
            maximum_batch_requests=maximum_batch_requests,
            staged_tensor_bytes=staged_tensor_bytes,
            maximum_tensor_bytes=maximum_tensor_bytes,
            host_staging_bytes=0,
            device_staging_bytes=0,
            available_device_headroom_bytes=available_device_headroom_bytes,
        )

    if available_device_headroom_bytes is None:
        return _disabled_plan(
            reason="selected-device memory headroom is not measurable",
            batch_size=batch_size,
            batch_count=batch_count,
            eligible_requests=eligible_requests,
            maximum_batch_requests=maximum_batch_requests,
            staged_tensor_bytes=staged_tensor_bytes,
            maximum_tensor_bytes=maximum_tensor_bytes,
            host_staging_bytes=host_staging_bytes,
            device_staging_bytes=device_staging_bytes,
            available_device_headroom_bytes=None,
        )
    if device_staging_bytes > available_device_headroom_bytes:
        return _disabled_plan(
            reason="next-batch tensors exceed remaining selected-device headroom",
            batch_size=batch_size,
            batch_count=batch_count,
            eligible_requests=eligible_requests,
            maximum_batch_requests=maximum_batch_requests,
            staged_tensor_bytes=staged_tensor_bytes,
            maximum_tensor_bytes=maximum_tensor_bytes,
            host_staging_bytes=host_staging_bytes,
            device_staging_bytes=device_staging_bytes,
            available_device_headroom_bytes=available_device_headroom_bytes,
        )

    if selected_kind == "cuda":
        host_budget = next(
            (
                budget
                for budget in execution_plan.candidate_budgets
                if budget.device.kind == "cpu"
            ),
            None,
        )
        host_limit = (
            host_budget.safe_working_set_limit_bytes
            if host_budget is not None
            else None
        )
        if host_limit is None:
            return _disabled_plan(
                reason="pinned-host staging capacity is not measurable",
                batch_size=batch_size,
                batch_count=batch_count,
                eligible_requests=eligible_requests,
                maximum_batch_requests=maximum_batch_requests,
                staged_tensor_bytes=staged_tensor_bytes,
                maximum_tensor_bytes=maximum_tensor_bytes,
                host_staging_bytes=host_staging_bytes,
                device_staging_bytes=device_staging_bytes,
                available_device_headroom_bytes=available_device_headroom_bytes,
            )
        if host_staging_bytes > host_limit:
            return _disabled_plan(
                reason="next-batch pinned tensors exceed safe host capacity",
                batch_size=batch_size,
                batch_count=batch_count,
                eligible_requests=eligible_requests,
                maximum_batch_requests=maximum_batch_requests,
                staged_tensor_bytes=staged_tensor_bytes,
                maximum_tensor_bytes=maximum_tensor_bytes,
                host_staging_bytes=host_staging_bytes,
                device_staging_bytes=device_staging_bytes,
                available_device_headroom_bytes=available_device_headroom_bytes,
            )

    return PropagationPrefetchPlan(
        enabled=True,
        reason="enabled by exact device and host staging budgets",
        batch_size=batch_size,
        batch_count=batch_count,
        eligible_requests=eligible_requests,
        maximum_batch_requests=maximum_batch_requests,
        staged_tensor_bytes=staged_tensor_bytes,
        maximum_tensor_bytes=maximum_tensor_bytes,
        host_staging_bytes=host_staging_bytes,
        device_staging_bytes=device_staging_bytes,
        available_device_headroom_bytes=available_device_headroom_bytes,
    )


@dataclass(frozen=True)
class PropagationPrefetchStatistics:
    submitted_batches: int
    completed_batches: int
    staged_tensors: int
    staged_bytes: int
    wait_seconds: float
    allocation_fallbacks: int


@dataclass(frozen=True)
class _StagedTensor:
    materializer: PrimitiveMaterializer
    value_id: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class _StagedBatch:
    values: tuple[_StagedTensor, ...]
    staged_bytes: int
    event: torch.cuda.Event | None = None
    pinned_sources: tuple[torch.Tensor, ...] = ()


class PropagationPrefetchSession:
    """Own a single background reader and at most one staged batch."""

    def __init__(
        self,
        plan: PropagationPrefetchPlan,
        *,
        device: str | torch.device,
    ) -> None:
        if not isinstance(plan, PropagationPrefetchPlan):
            raise TypeError("prefetch session requires a PropagationPrefetchPlan")
        self.plan = plan
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self._future: Future[_StagedBatch] | None = None
        self._closed = False
        self._runtime_enabled = plan.enabled
        self._submitted_batches = 0
        self._completed_batches = 0
        self._staged_tensors = 0
        self._staged_bytes = 0
        self._wait_seconds = 0.0
        self._allocation_fallbacks = 0
        self._transfer_stream: torch.cuda.Stream | None = None
        if self._runtime_enabled and self.device.type == "cuda":
            try:
                self._transfer_stream = torch.cuda.Stream(device=self.device)
            except (MemoryError, RuntimeError) as exc:
                if not self._is_optional_allocation_failure(exc):
                    raise
                self._runtime_enabled = False
                self._allocation_fallbacks += 1
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="concord-hdf5-prefetch")
            if self._runtime_enabled
            else None
        )

    @staticmethod
    def _is_optional_allocation_failure(exc: BaseException) -> bool:
        if isinstance(exc, MemoryError):
            return True
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "out of memory",
                "pin_memory",
                "pin memory",
                "cuda driver version is insufficient",
                "no cuda gpus are available",
                "cannot access accelerator device",
                "cuda initialization error",
                "can't start new thread",
            )
        )

    def _load_batch(
        self,
        entries: tuple[tuple[PrimitiveMaterializer, str], ...],
    ) -> _StagedBatch:
        grouped: dict[str, list[PrimitiveMaterializer]] = {}
        for materializer, value_id in entries:
            grouped.setdefault(value_id, []).append(materializer)

        staged: list[_StagedTensor] = []
        pinned_sources: list[torch.Tensor] = []
        staged_bytes = 0
        if self.device.type == "cuda":
            assert self._transfer_stream is not None
            with torch.cuda.device(self.device), torch.cuda.stream(
                self._transfer_stream
            ):
                for value_id, group in grouped.items():
                    batch = materialize_primitive_batch(group, value_id)
                    if batch is not None:
                        pinned = batch.tensor.pin_memory()
                        transferred = pinned.to(
                            device=self.device,
                            non_blocking=True,
                        )
                        pinned_sources.append(pinned)
                        for materializer, tensor in zip(
                            group,
                            transferred.unbind(0),
                            strict=True,
                        ):
                            staged.append(_StagedTensor(materializer, value_id, tensor))
                        staged_bytes += (
                            transferred.numel() * transferred.element_size()
                        )
                        continue

                    for materializer in group:
                        host = materializer.materialize_hdf5_propagation_for_prefetch(
                            value_id
                        )
                        pinned = host.pin_memory()
                        transferred = pinned.to(
                            device=self.device,
                            non_blocking=True,
                        )
                        pinned_sources.append(pinned)
                        staged.append(_StagedTensor(materializer, value_id, transferred))
                        staged_bytes += (
                            transferred.numel() * transferred.element_size()
                        )
                event = torch.cuda.Event(blocking=False)
                event.record(self._transfer_stream)
            return _StagedBatch(
                values=tuple(staged),
                staged_bytes=staged_bytes,
                event=event,
                pinned_sources=tuple(pinned_sources),
            )

        for value_id, group in grouped.items():
            batch = materialize_primitive_batch(group, value_id)
            if batch is not None:
                for materializer, tensor in zip(
                    group,
                    batch.tensor.unbind(0),
                    strict=True,
                ):
                    staged.append(_StagedTensor(materializer, value_id, tensor))
                staged_bytes += batch.tensor.numel() * batch.tensor.element_size()
                continue

            for materializer in group:
                tensor = materializer.materialize_hdf5_propagation_for_prefetch(
                    value_id
                )
                staged.append(_StagedTensor(materializer, value_id, tensor))
                staged_bytes += tensor.numel() * tensor.element_size()
        return _StagedBatch(values=tuple(staged), staged_bytes=staged_bytes)

    def submit(self, materializers: Sequence[PrimitiveMaterializer]) -> bool:
        """Submit one batch if enabled; return whether background work started."""

        if self._closed:
            raise RuntimeError("prefetch session is closed")
        if self._future is not None:
            raise RuntimeError("prefetch depth is limited to one in-flight batch")
        if not self._runtime_enabled:
            return False
        values = tuple(materializers)
        if not values:
            raise ValueError("prefetch submission requires materializers")
        if len(values) > self.plan.batch_size:
            raise ValueError("prefetch batch exceeds its planned batch size")
        if any(not isinstance(item, PrimitiveMaterializer) for item in values):
            raise TypeError("prefetch submission requires PrimitiveMaterializer entries")

        entries = tuple(
            (materializer, value_id)
            for materializer in values
            for value_id in materializer.prefetchable_hdf5_propagation_values()
        )
        if not entries:
            return False
        assert self._executor is not None
        try:
            self._future = self._executor.submit(self._load_batch, entries)
        except (MemoryError, RuntimeError) as exc:
            if not self._is_optional_allocation_failure(exc):
                raise
            self._runtime_enabled = False
            self._allocation_fallbacks += 1
            return False
        self._submitted_batches += 1
        return True

    def resolve(self) -> bool:
        """Install the pending batch, or preserve synchronous fallback on OOM."""

        if self._closed:
            raise RuntimeError("prefetch session is closed")
        future = self._future
        if future is None:
            return False
        self._future = None
        started = perf_counter()
        try:
            batch = future.result()
            if batch.event is not None:
                batch.event.synchronize()
        except (MemoryError, RuntimeError) as exc:
            self._wait_seconds += perf_counter() - started
            if not self._is_optional_allocation_failure(exc):
                raise
            self._runtime_enabled = False
            self._allocation_fallbacks += 1
            return False
        self._wait_seconds += perf_counter() - started

        if batch.staged_bytes > self.plan.staged_tensor_bytes:
            raise RuntimeError("prefetch exceeded its exact staging byte plan")
        for value in batch.values:
            value.materializer.install_prefetched_hdf5_propagation(
                value.value_id,
                value.tensor,
            )
        self._completed_batches += 1
        self._staged_tensors += len(batch.values)
        self._staged_bytes += batch.staged_bytes
        return True

    @property
    def statistics(self) -> PropagationPrefetchStatistics:
        return PropagationPrefetchStatistics(
            submitted_batches=self._submitted_batches,
            completed_batches=self._completed_batches,
            staged_tensors=self._staged_tensors,
            staged_bytes=self._staged_bytes,
            wait_seconds=self._wait_seconds,
            allocation_fallbacks=self._allocation_fallbacks,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def __enter__(self) -> "PropagationPrefetchSession":
        if self._closed:
            raise RuntimeError("prefetch session cannot be reopened")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "PropagationPrefetchPlan",
    "PropagationPrefetchStatistics",
    "PropagationPrefetchSession",
    "plan_propagation_prefetch",
]
