"""Configuration-scoped persistent resident numerical execution.

The physical plan remains a single-work-unit description.  This runtime-only
session allocates one maximum-batch buffer pool and reuses its storage for every
batch in one gauge configuration, including a shorter final batch.  Result
tensors are views into that storage and must be consumed before ``release()``.

Work units run through a :class:`PreparedResidentProgram` per batch size, which
decides every step once and gives bitwise the same results as the step-by-step
executors.  ``CONCORD_CHECKED_EXECUTION=1`` selects the step-by-step executors,
which repeat every check for every work unit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch

from concord_lqcd.backend.planner.physical import PhysicalExecutionPlan

from .batched_executor import (
    BatchedNumericalExecutionResult,
    BatchedNumericalExecutor,
)
from .materialization import PrimitiveMaterializer
from .numerical_executor import (
    GenericNumericalExecutor,
    NumericalExecutionResult,
    RuntimeBufferPool,
)
from .prepared_executor import PreparedResidentProgram, prepare_plan_steps


ResidentExecutionResult = NumericalExecutionResult | BatchedNumericalExecutionResult


def checked_execution_requested() -> bool:
    """Whether ``CONCORD_CHECKED_EXECUTION`` asks for step-by-step execution."""

    return os.environ.get("CONCORD_CHECKED_EXECUTION", "").strip() not in ("", "0")


@dataclass(frozen=True)
class ResidentExecutionSessionStatistics:
    pool_allocations: int
    pool_allocation_bytes: int
    maximum_batch_size: int
    executions: int
    released_executions: int
    failed_executions: int


class ResidentExecutionSession:
    """Reuse one resident physical buffer pool across numerical batches.

    A session is intentionally scoped by the job runner to one configuration.
    ``execute`` leaves output values active so callers can synchronously persist
    them.  ``release`` invalidates those views and makes the same allocation
    available for the next batch.
    """

    def __init__(
        self,
        plan: PhysicalExecutionPlan,
        *,
        device: str | torch.device,
        maximum_batch_size: int,
        prepared: bool | None = None,
    ) -> None:
        if not isinstance(plan, PhysicalExecutionPlan):
            raise TypeError("resident execution session requires a PhysicalExecutionPlan")
        if (
            not isinstance(maximum_batch_size, int)
            or isinstance(maximum_batch_size, bool)
            or maximum_batch_size <= 0
        ):
            raise ValueError("maximum_batch_size must be a positive integer")
        plan.validate()
        self.plan = plan
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.maximum_batch_size = maximum_batch_size
        self.pool = RuntimeBufferPool(
            plan,
            device=self.device,
            batch_size=maximum_batch_size,
        )
        self._pool_allocation_bytes = self.pool.allocation_bytes
        self.prepared = not checked_execution_requested() if prepared is None else bool(prepared)
        self._programs: dict[int, PreparedResidentProgram] = {}
        self._result_live = False
        self._closed = False
        self._executions = 0
        self._released_executions = 0
        self._failed_executions = 0

    @property
    def result_live(self) -> bool:
        return self._result_live

    def execute(
        self,
        materializers: Sequence[PrimitiveMaterializer],
    ) -> ResidentExecutionResult:
        if self._closed:
            raise RuntimeError("resident execution session is closed")
        if self._result_live or not self.pool.is_idle:
            raise RuntimeError(
                "previous resident outputs must be consumed and released before reuse"
            )
        values = tuple(materializers)
        if not values:
            raise ValueError("resident execution requires at least one materializer")
        if len(values) > self.maximum_batch_size:
            raise ValueError(
                f"resident batch size {len(values)} exceeds session capacity "
                f"{self.maximum_batch_size}"
            )
        if any(not isinstance(item, PrimitiveMaterializer) for item in values):
            raise TypeError("resident execution requires PrimitiveMaterializer entries")
        if any(
            item.binding_plan.source_plan is not self.plan
            for item in values
        ):
            raise ValueError("resident materializers bind a different physical plan")

        self.pool.activate_batch_size(len(values))
        try:
            if self.prepared:
                program = self._programs.get(len(values))
                if program is None:
                    program = PreparedResidentProgram(
                        self.plan,
                        self.pool,
                        prepare_plan_steps(self.plan, len(values), self.device.type),
                    )
                    self._programs[len(values)] = program
                result: ResidentExecutionResult = program.execute(values)
            elif len(values) == 1:
                result = GenericNumericalExecutor(
                    values[0],
                    device=self.device,
                    buffer_pool=self.pool,
                ).execute()
            else:
                result = BatchedNumericalExecutor(
                    values,
                    device=self.device,
                    buffer_pool=self.pool,
                ).execute()
        except Exception:
            self.pool.reset()
            self._failed_executions += 1
            raise

        self._executions += 1
        self._result_live = True
        return result

    def release(self) -> None:
        """Invalidate the current result views without freeing pool storage."""

        if self._closed:
            raise RuntimeError("resident execution session is closed")
        if not self._result_live:
            raise RuntimeError("resident execution session has no live result")
        self.pool.reset()
        self._result_live = False
        self._released_executions += 1

    @property
    def statistics(self) -> ResidentExecutionSessionStatistics:
        return ResidentExecutionSessionStatistics(
            pool_allocations=1,
            pool_allocation_bytes=self._pool_allocation_bytes,
            maximum_batch_size=self.maximum_batch_size,
            executions=self._executions,
            released_executions=self._released_executions,
            failed_executions=self._failed_executions,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._result_live:
            raise RuntimeError("cannot close a resident session with live outputs")
        self.pool.close()
        self._closed = True

    def __enter__(self) -> "ResidentExecutionSession":
        if self._closed:
            raise RuntimeError("resident execution session cannot be reopened")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._result_live:
            self.pool.reset()
            self._result_live = False
        self.close()


__all__ = [
    "checked_execution_requested",
    "ResidentExecutionResult",
    "ResidentExecutionSessionStatistics",
    "ResidentExecutionSession",
]
