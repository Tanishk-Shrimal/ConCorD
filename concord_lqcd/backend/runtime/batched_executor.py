"""Exact leading-axis batching for independent physical-plan work units.

The frozen compiler plan describes one configuration/time-context evaluation.
This module executes several resource bindings for that same plan together by
prepending one private batch index to every primitive, intermediate, and output.
The batch index is never reduced, so no contraction can couple distinct work
units.  Batch size one remains the original :class:`GenericNumericalExecutor`
path and host-staged index tiling is deliberately excluded until its memory
model represents this additional leading axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from concord_lqcd.backend.planner.physical import (
    ContractOperation,
    CorrelatorSumOperation,
    MaterializePrimitiveOperation,
    PhysicalExecutionPlan,
    PhysicalInputUse,
    WickTermProductOperation,
)

from .contraction import (
    contract_labeled_tensors,
    execute_direct_binary_gemm_into,
    prepare_direct_binary_gemm,
)
from .materialization import (
    PrimitiveMaterializer,
    materialize_primitive_batch,
    shared_materialization_cache_key,
)
from .numerical_executor import (
    RuntimeBufferPool,
    _torch_dtype,
)


def _private_batch_index(plan: PhysicalExecutionPlan) -> str:
    used: set[str] = set()
    for step in plan.steps:
        operation = step.operation
        for use in getattr(operation, "inputs", ()):
            used.update(binding.index for binding in use.bindings)
        used.update(getattr(operation, "output_indices", ()))
    value = "__concord_time_batch__"
    while value in used:
        value += "_"
    return value


def _batched_input_tensor_and_indices(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    use: PhysicalInputUse,
    batch_index: str,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    binding = plan.binding(use.value_id)
    ports = tuple(item.port for item in use.bindings)
    if ports != binding.axis_labels:
        raise ValueError(
            f"Batched physical input use of {use.value_id!r} binds ports "
            f"{ports!r}, but its runtime value axes are {binding.axis_labels!r}."
        )
    return (
        pool.read(use.value_id),
        (batch_index, *(item.index for item in use.bindings)),
    )


def _batched_contract_inputs(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    inputs: tuple[PhysicalInputUse, ...],
    output_indices: tuple[str, ...],
    batch_index: str,
) -> torch.Tensor:
    tensors = tuple(
        _batched_input_tensor_and_indices(plan, pool, use, batch_index)
        for use in inputs
    )
    return contract_labeled_tensors(
        tensors,
        (batch_index, *output_indices),
    )


def _execute_batched_contraction(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: ContractOperation,
    batch_index: str,
) -> torch.Tensor:
    return _batched_contract_inputs(
        plan,
        pool,
        operation.inputs,
        operation.output_indices,
        batch_index,
    )


def _execute_batched_contraction_into_if_possible(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: ContractOperation,
    batch_index: str,
) -> torch.Tensor | None:
    inputs = tuple(
        _batched_input_tensor_and_indices(plan, pool, use, batch_index)
        for use in operation.inputs
    )
    contraction = prepare_direct_binary_gemm(
        inputs,
        (batch_index, *operation.output_indices),
        allow_layout_copies=pool.device.type == "cpu",
    )
    if contraction is None:
        return None
    target = pool.reserve(operation.node_id)
    return execute_direct_binary_gemm_into(contraction, target)


def _execute_batched_wick_term(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: WickTermProductOperation,
    batch_index: str,
) -> torch.Tensor:
    output_binding = plan.binding(operation.node_id)
    dtype = _torch_dtype(output_binding.scalar_type)
    if operation.inputs:
        value = _batched_contract_inputs(
            plan,
            pool,
            operation.inputs,
            operation.output_indices,
            batch_index,
        )
        coefficient = torch.tensor(
            operation.coefficient,
            dtype=dtype,
            device=value.device,
        )
        return value * coefficient
    if output_binding.shape:
        raise ValueError("A zero-input batched Wick term must have scalar work-unit output.")
    return torch.full(
        (pool.batch_size,),
        operation.coefficient,
        dtype=dtype,
        device=pool.device,
    )


def _execute_batched_correlator_sum(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: CorrelatorSumOperation,
) -> torch.Tensor:
    output_binding = plan.binding(operation.node_id)
    dtype = _torch_dtype(output_binding.scalar_type)
    expected_shape = (pool.batch_size, *output_binding.shape)
    if not operation.inputs:
        return torch.zeros(expected_shape, dtype=dtype, device=pool.device)

    tensors = [pool.read(use.value_id) for use in operation.inputs]
    for tensor in tensors:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Batched correlator sum {operation.correlator_id!r} received "
                f"shape {tuple(tensor.shape)}, expected {expected_shape}."
            )
        if tensor.dtype != dtype:
            raise TypeError(
                f"Batched correlator sum {operation.correlator_id!r} received "
                f"dtype {tensor.dtype}, expected {dtype}."
            )

    result = torch.zeros(expected_shape, dtype=dtype, device=pool.device)
    for tensor in tensors:
        result.add_(tensor)
    return result


@dataclass(frozen=True)
class BatchedNumericalExecutionResult:
    """Outputs from several independent time contexts in one leading axis."""

    source_plan: PhysicalExecutionPlan
    device: torch.device
    outputs: Mapping[str, torch.Tensor]
    batch_size: int
    executed_steps: int

    def output(self, correlator_id: str) -> torch.Tensor:
        return self.outputs[correlator_id]

    def outputs_at(self, batch_index: int) -> dict[str, torch.Tensor]:
        if (
            not isinstance(batch_index, int)
            or isinstance(batch_index, bool)
            or batch_index < 0
            or batch_index >= self.batch_size
        ):
            raise IndexError("batched numerical output index is out of range")
        return {
            correlator_id: tensor[batch_index]
            for correlator_id, tensor in self.outputs.items()
        }

    def validate(self) -> None:
        self.source_plan.validate()
        if self.batch_size <= 1:
            raise ValueError("Batched numerical results require batch_size greater than one.")
        if self.executed_steps != len(self.source_plan.steps):
            raise ValueError("Batched execution did not execute every physical-plan step.")
        if set(self.outputs) != set(self.source_plan.outputs):
            raise ValueError("Batched result does not contain exactly the requested outputs.")
        for correlator_id, output in self.source_plan.outputs.items():
            tensor = self.outputs[correlator_id]
            binding = self.source_plan.binding(output.value_id)
            expected_shape = (self.batch_size, *binding.shape)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Batched output {correlator_id!r} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}."
                )
            if tensor.dtype != _torch_dtype(binding.scalar_type):
                raise TypeError(
                    f"Batched output {correlator_id!r} has dtype {tensor.dtype}, "
                    f"expected {_torch_dtype(binding.scalar_type)}."
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"Batched output {correlator_id!r} is on {tensor.device}, "
                    f"expected {self.device}."
                )


class BatchedNumericalExecutor:
    """Execute equivalent resource bindings together on one resident device."""

    def __init__(
        self,
        materializers: Sequence[PrimitiveMaterializer],
        *,
        device: str | torch.device,
        buffer_pool: RuntimeBufferPool | None = None,
    ) -> None:
        values = tuple(materializers)
        if len(values) <= 1:
            raise ValueError(
                "BatchedNumericalExecutor requires at least two materializers; "
                "use GenericNumericalExecutor for batch size one."
            )
        if any(not isinstance(item, PrimitiveMaterializer) for item in values):
            raise TypeError("batched execution requires PrimitiveMaterializer entries")
        plan = values[0].binding_plan.source_plan
        if any(item.binding_plan.source_plan is not plan for item in values[1:]):
            raise ValueError("all batched materializers must bind the same physical plan")
        for item in values:
            item.binding_plan.validate()
        self.materializers = values
        self.plan = plan
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.batch_size = len(values)
        self.batch_index = _private_batch_index(plan)
        self.invariant_materialization_value_ids = frozenset(
            step.operation.value_id
            for step in plan.steps
            if isinstance(step.operation, MaterializePrimitiveOperation)
            and shared_materialization_cache_key(
                values,
                step.operation.value_id,
            )
            is not None
        )
        if buffer_pool is not None:
            if not isinstance(buffer_pool, RuntimeBufferPool):
                raise TypeError("buffer_pool must be a RuntimeBufferPool or None")
            if buffer_pool.plan is not self.plan:
                raise ValueError("runtime buffer pool belongs to a different physical plan")
            if buffer_pool.device != self.device:
                raise ValueError("runtime buffer pool belongs to a different device")
            if buffer_pool.batch_size != self.batch_size:
                raise ValueError(
                    "batched execution size disagrees with active buffer-pool size"
                )
            if not buffer_pool.is_idle:
                raise RuntimeError("batched execution requires an idle runtime buffer pool")
        self.buffer_pool = buffer_pool

    def _materialize(self, operation: MaterializePrimitiveOperation) -> torch.Tensor:
        if operation.value_id in self.invariant_materialization_value_ids:
            value = self.materializers[0].materialize(operation.value_id)
            if (
                value.value_id != operation.value_id
                or value.resource_id != operation.resource_id
                or value.buffer_id != operation.output_buffer_id
            ):
                raise ValueError(
                    "Batched primitive materializer returned metadata inconsistent "
                    "with the physical operation."
                )
            tensor = (
                value.tensor
                if value.tensor.device == self.device
                else value.tensor.to(device=self.device)
            )
            return tensor.unsqueeze(0).expand(self.batch_size, *tensor.shape)

        batch = materialize_primitive_batch(
            self.materializers,
            operation.value_id,
        )
        if batch is not None:
            if (
                batch.value_id != operation.value_id
                or batch.resource_id != operation.resource_id
                or batch.buffer_id != operation.output_buffer_id
            ):
                raise ValueError(
                    "Batched primitive materializer returned metadata inconsistent "
                    "with the physical operation."
                )
            return (
                batch.tensor
                if batch.tensor.device == self.device
                else batch.tensor.to(device=self.device)
            )

        tensors: list[torch.Tensor] = []
        for materializer in self.materializers:
            value = materializer.materialize(operation.value_id)
            if (
                value.value_id != operation.value_id
                or value.resource_id != operation.resource_id
                or value.buffer_id != operation.output_buffer_id
            ):
                raise ValueError(
                    "Batched primitive materializer returned metadata inconsistent "
                    "with the physical operation."
                )
            tensors.append(
                value.tensor
                if value.tensor.device == self.device
                else value.tensor.to(device=self.device)
            )
        return torch.stack(tensors, dim=0)

    def execute(self) -> BatchedNumericalExecutionResult:
        self.plan.validate()
        pool = self.buffer_pool or RuntimeBufferPool(
            self.plan,
            device=self.device,
            batch_size=self.batch_size,
        )

        for plan_step in self.plan.steps:
            operation = plan_step.operation
            if pool.active_capacity_bytes != (
                self.batch_size * plan_step.active_capacity_before
            ):
                raise RuntimeError(
                    f"Batched active capacity before step {plan_step.step_index} "
                    "disagrees with the physical plan."
                )

            wrote_directly = False
            if isinstance(operation, MaterializePrimitiveOperation):
                value = self._materialize(operation)
            elif isinstance(operation, ContractOperation):
                value = _execute_batched_contraction_into_if_possible(
                    self.plan,
                    pool,
                    operation,
                    self.batch_index,
                )
                if value is None:
                    value = _execute_batched_contraction(
                        self.plan,
                        pool,
                        operation,
                        self.batch_index,
                    )
                else:
                    wrote_directly = True
            elif isinstance(operation, WickTermProductOperation):
                value = _execute_batched_wick_term(
                    self.plan,
                    pool,
                    operation,
                    self.batch_index,
                )
            elif isinstance(operation, CorrelatorSumOperation):
                value = _execute_batched_correlator_sum(self.plan, pool, operation)
            else:  # pragma: no cover - physical plan validates the closed union
                raise TypeError(f"Unsupported physical operation {type(operation)!r}.")

            if not wrote_directly:
                pool.write(operation.node_id, value)
            if pool.active_capacity_bytes != (
                self.batch_size * plan_step.peak_active_capacity_during_step
            ):
                raise RuntimeError(
                    f"Batched active capacity during step {plan_step.step_index} "
                    "disagrees with the physical plan."
                )

            for released_value in plan_step.released_values:
                pool.release(released_value)
            if pool.active_capacity_bytes != (
                self.batch_size * plan_step.active_capacity_after
            ):
                raise RuntimeError(
                    f"Batched active capacity after step {plan_step.step_index} "
                    "disagrees with the physical plan."
                )

        expected_live = set(self.plan.source_schedule.output_nodes)
        if set(pool.active_values) != expected_live:
            raise RuntimeError(
                "Batched runtime final live-value set does not match retained outputs."
            )

        outputs: dict[str, torch.Tensor] = {}
        for correlator_id, output in self.plan.outputs.items():
            value = pool.read(output.value_id)
            if value.data_ptr() != pool.storage[output.buffer_id].data_ptr():
                raise RuntimeError("Batched output is not backed by its planned buffer.")
            outputs[correlator_id] = value

        result = BatchedNumericalExecutionResult(
            source_plan=self.plan,
            device=self.device,
            outputs=dict(outputs),
            batch_size=self.batch_size,
            executed_steps=len(self.plan.steps),
        )
        result.validate()
        return result


__all__ = [
    "BatchedNumericalExecutionResult",
    "BatchedNumericalExecutor",
]
