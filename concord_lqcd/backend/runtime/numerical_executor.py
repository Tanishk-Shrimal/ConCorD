"""Generic numerical execution for the new ConCorD physical pipeline.

Consumes a frozen PhysicalExecutionPlan plus PrimitiveMaterializer, executes the
plan in order, stores live values in the assigned reusable buffers, and releases
values exactly according to the schedule. An optional runtime-only tile plan
keeps those buffers on a storage device while executing bounded contractions on
a distinct compute device. Primitive distinctions end at materialization:
contractions operate only on tensors and explicit index bindings.

The physical buffer pool models persistent logical-value storage. PyTorch and
opt_einsum may still allocate temporary workspace while evaluating one
operation; planner capacity is therefore not a bound on total allocator usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as cartesian_product
from math import prod
from typing import Mapping

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
from .materialization import PrimitiveMaterializer
from .tiling import OperationTilePlan, TiledExecutionPlan


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _torch_dtype(scalar_type: str) -> torch.dtype:
    try:
        return _TORCH_DTYPES[scalar_type]
    except KeyError as exc:
        raise ValueError(
            f"Generic numerical execution does not know scalar type {scalar_type!r}. "
            f"Supported types are {tuple(_TORCH_DTYPES)}."
        ) from exc


def _numel(shape: tuple[int, ...]) -> int:
    return prod(shape) if shape else 1


class RuntimeBufferPool:
    """Concrete single-device storage implementing the abstract buffer plan.

    ``batch_size`` is a runtime-only leading multiplicity.  The frozen physical
    plan continues to describe one work unit; batched execution allocates the
    same reusable pool layout for several independent work units and prepends a
    leading axis to every logical value.  The default remains exactly the
    original single-work-unit behavior.
    """

    def __init__(
        self,
        plan: PhysicalExecutionPlan,
        *,
        device: str | torch.device = "cpu",
        batch_size: int = 1,
    ) -> None:
        plan.validate()
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("runtime buffer-pool batch_size must be positive")
        self.plan = plan
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.capacity_batch_size = batch_size
        self.batch_size = batch_size
        self.storage: dict[str, torch.Tensor] = {}
        self.active_owner: dict[str, str] = {}
        self.active_values: set[str] = set()
        self._active_capacity_bytes = 0

        layouts: dict[str, tuple[torch.dtype, int]] = {}
        for buffer_id, slot in plan.buffers.items():
            dtype = _torch_dtype(slot.scalar_type)
            element_size = torch.empty((), dtype=dtype).element_size()
            if slot.capacity_bytes % element_size != 0:
                raise ValueError(
                    f"Physical buffer {buffer_id!r} has capacity {slot.capacity_bytes} bytes, "
                    f"which is not divisible by runtime element size {element_size} for "
                    f"{slot.scalar_type!r}."
                )
            layouts[buffer_id] = (dtype, self.batch_size * slot.capacity_bytes // element_size)

        # Check the descriptive compiler byte model against the concrete torch
        # representation for every scheduled value, not just primitives.
        for value_id, binding in plan.value_bindings.items():
            dtype = _torch_dtype(binding.scalar_type)
            element_size = torch.empty((), dtype=dtype).element_size()
            runtime_size = _numel(binding.shape) * element_size
            if runtime_size != binding.size_bytes:
                raise ValueError(
                    f"Physical value {value_id!r} is planned as {binding.size_bytes} bytes, "
                    f"but runtime dtype {binding.scalar_type!r} and shape {binding.shape} "
                    f"require {runtime_size} bytes."
                )
            if runtime_size > plan.buffer(binding.buffer_id).capacity_bytes:
                raise ValueError(
                    f"Physical value {value_id!r} does not fit its planned runtime buffer."
                )

        for buffer_id, (dtype, elements) in layouts.items():
            self.storage[buffer_id] = torch.empty(elements, dtype=dtype, device=self.device)

    @property
    def active_capacity_bytes(self) -> int:
        return self._active_capacity_bytes

    @property
    def allocation_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.storage.values())

    @property
    def is_idle(self) -> bool:
        return not self.active_values and not self.active_owner

    def write_all(self) -> None:
        """Write every buffer once, so that all of the pool is in memory now.

        The storage is allocated empty, and the system backs each page with
        memory only when it is first written.
        """

        for tensor in self.storage.values():
            tensor.zero_()

    def activate_batch_size(self, batch_size: int) -> None:
        """Select an active prefix without reallocating maximum-capacity storage."""

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("active runtime batch_size must be positive")
        if batch_size > self.capacity_batch_size:
            raise ValueError(
                f"active batch size {batch_size} exceeds pool capacity "
                f"{self.capacity_batch_size}"
            )
        if not self.is_idle:
            raise RuntimeError("cannot change batch size while runtime values are active")
        self.batch_size = batch_size

    def reset(self, *, batch_size: int | None = None) -> None:
        """Release logical ownership while retaining all physical allocations."""

        self.active_values.clear()
        self.active_owner.clear()
        self._active_capacity_bytes = 0
        if batch_size is not None:
            self.activate_batch_size(batch_size)

    def close(self) -> None:
        if not self.is_idle:
            raise RuntimeError("cannot close a runtime buffer pool with active values")
        self.storage.clear()

    @property
    def active_value_ids(self) -> tuple[str, ...]:
        return tuple(
            value_id
            for value_id in self.plan.value_bindings
            if value_id in self.active_values
        )

    def _view(self, value_id: str) -> torch.Tensor:
        binding = self.plan.binding(value_id)
        flat = self.storage[binding.buffer_id]
        shape = (
            binding.shape
            if self.batch_size == 1
            else (self.batch_size, *binding.shape)
        )
        prefix = flat[: self.batch_size * _numel(binding.shape)]
        return prefix.reshape(shape) if shape else prefix.reshape(())

    def planned_view(self, value_id: str) -> torch.Tensor:
        """The view a value occupies at the active batch size, without ownership checks."""

        return self._view(value_id)

    def read(self, value_id: str) -> torch.Tensor:
        if value_id not in self.active_values:
            raise RuntimeError(f"Runtime attempted to read inactive value {value_id!r}.")
        binding = self.plan.binding(value_id)
        if self.active_owner.get(binding.buffer_id) != value_id:
            raise RuntimeError(
                f"Runtime buffer {binding.buffer_id!r} is not owned by value {value_id!r}."
            )
        return self._view(value_id)

    def write(self, value_id: str, tensor: torch.Tensor) -> torch.Tensor:
        binding = self.plan.binding(value_id)
        expected_dtype = _torch_dtype(binding.scalar_type)
        expected_shape = (
            binding.shape
            if self.batch_size == 1
            else (self.batch_size, *binding.shape)
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Runtime result {value_id!r} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}."
            )
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"Runtime result {value_id!r} has dtype {tensor.dtype}, expected "
                f"{expected_dtype}."
            )
        runtime_bytes = tensor.numel() * tensor.element_size()
        expected_bytes = self.batch_size * binding.size_bytes
        if runtime_bytes != expected_bytes:
            raise ValueError(
                f"Runtime result {value_id!r} occupies {runtime_bytes} bytes, "
                f"expected {expected_bytes}."
            )

        target = self.reserve(value_id)
        source = tensor if tensor.device == self.device else tensor.to(device=self.device)
        target.copy_(source)
        return target

    def reserve(self, value_id: str) -> torch.Tensor:
        """Claim a value's planned buffer before writing it incrementally.

        Resident execution normally produces a complete temporary and then uses
        :meth:`write`.  Host-staged tiled execution instead claims the complete
        host output buffer first and fills disjoint slices directly, avoiding a
        complete accelerator-resident result.
        """

        binding = self.plan.binding(value_id)
        buffer_id = binding.buffer_id
        owner = self.active_owner.get(buffer_id)
        if owner is not None:
            raise RuntimeError(
                f"Runtime attempted to overwrite active buffer {buffer_id!r}, currently "
                f"owned by {owner!r}, with value {value_id!r}."
            )
        if value_id in self.active_values:
            raise RuntimeError(f"Runtime attempted to produce value {value_id!r} twice.")
        self.active_owner[buffer_id] = value_id
        self.active_values.add(value_id)
        self._active_capacity_bytes += self.batch_size * self.plan.buffer(buffer_id).capacity_bytes
        return self._view(value_id)

    def release(self, value_id: str) -> None:
        if value_id not in self.active_values:
            raise RuntimeError(f"Runtime attempted to release inactive value {value_id!r}.")
        binding = self.plan.binding(value_id)
        owner = self.active_owner.get(binding.buffer_id)
        if owner != value_id:
            raise RuntimeError(
                f"Runtime release for {value_id!r} found buffer {binding.buffer_id!r} "
                f"owned by {owner!r}."
            )
        self.active_values.remove(value_id)
        del self.active_owner[binding.buffer_id]
        self._active_capacity_bytes -= (
            self.batch_size * self.plan.buffer(binding.buffer_id).capacity_bytes
        )

    def retain_outputs(self, value_ids: tuple[str, ...]) -> None:
        """Leave an idle pool as a completed step-by-step execution leaves it.

        Prepared execution writes through views fixed in advance and skips the
        per-step ownership bookkeeping, which it checks once per plan instead.
        Afterwards exactly the retained output values are active.
        """

        if not self.is_idle:
            raise RuntimeError("retained outputs can only be restored on an idle pool")
        for value_id in value_ids:
            self.reserve(value_id)


def _input_tensor_and_indices(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    use: PhysicalInputUse,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    binding = plan.binding(use.value_id)
    ports = tuple(item.port for item in use.bindings)
    if ports != binding.axis_labels:
        raise ValueError(
            f"Physical input use of {use.value_id!r} binds ports {ports!r}, but its "
            f"runtime value axes are {binding.axis_labels!r}."
        )
    return pool.read(use.value_id), tuple(item.index for item in use.bindings)


def _contract_labeled_inputs(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    inputs: tuple[PhysicalInputUse, ...],
    output_indices: tuple[str, ...],
) -> torch.Tensor:
    tensors = tuple(
        _input_tensor_and_indices(plan, pool, use)
        for use in inputs
    )
    return contract_labeled_tensors(tensors, output_indices)


def _execute_contraction(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: ContractOperation,
) -> torch.Tensor:
    return _contract_labeled_inputs(plan, pool, operation.inputs, operation.output_indices)


def _execute_contraction_into_if_possible(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: ContractOperation,
) -> torch.Tensor | None:
    inputs = tuple(
        _input_tensor_and_indices(plan, pool, use)
        for use in operation.inputs
    )
    contraction = prepare_direct_binary_gemm(
        inputs,
        operation.output_indices,
        allow_layout_copies=pool.device.type == "cpu",
    )
    if contraction is None:
        return None
    target = pool.reserve(operation.node_id)
    return execute_direct_binary_gemm_into(contraction, target)


def _execute_wick_term(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: WickTermProductOperation,
) -> torch.Tensor:
    output_binding = plan.binding(operation.node_id)
    dtype = _torch_dtype(output_binding.scalar_type)
    if operation.inputs:
        product_value = _contract_labeled_inputs(
            plan, pool, operation.inputs, operation.output_indices
        )
        coefficient = torch.tensor(
            operation.coefficient,
            dtype=dtype,
            device=product_value.device,
        )
        return product_value * coefficient
    if output_binding.shape:
        raise ValueError("A zero-input Wick term must be scalar.")
    return torch.tensor(operation.coefficient, dtype=dtype, device=pool.device)


def _execute_correlator_sum(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: CorrelatorSumOperation,
) -> torch.Tensor:
    output_binding = plan.binding(operation.node_id)
    dtype = _torch_dtype(output_binding.scalar_type)
    if not operation.inputs:
        if output_binding.shape:
            raise ValueError("An empty correlator sum must be scalar.")
        return torch.zeros((), dtype=dtype, device=pool.device)

    tensors = [pool.read(use.value_id) for use in operation.inputs]
    for tensor in tensors:
        if tuple(tensor.shape) != output_binding.shape:
            raise ValueError(
                f"Correlator sum {operation.correlator_id!r} received an input shape "
                f"{tuple(tensor.shape)} incompatible with output {output_binding.shape}."
            )
        if tensor.dtype != dtype:
            raise TypeError(
                f"Correlator sum {operation.correlator_id!r} received dtype "
                f"{tensor.dtype}, expected {dtype}."
            )

    result = torch.zeros(output_binding.shape, dtype=dtype, device=pool.device)
    for tensor in tensors:
        result.add_(tensor)
    return result


def _tile_slices(size: int, tile_size: int) -> tuple[slice, ...]:
    return tuple(
        slice(start, min(size, start + tile_size))
        for start in range(0, size, tile_size)
    )


def _tile_assignments(
    indices: tuple[str, ...],
    operation_plan: OperationTilePlan,
):
    sizes = operation_plan.size_by_index
    tiles = operation_plan.tile_by_index
    tiled = tuple(index for index in indices if tiles[index] < sizes[index])
    if not tiled:
        yield {}
        return
    choices = tuple(_tile_slices(sizes[index], tiles[index]) for index in tiled)
    for slices in cartesian_product(*choices):
        yield dict(zip(tiled, slices))


def _execute_tiled_operation_into(
    plan: PhysicalExecutionPlan,
    pool: RuntimeBufferPool,
    operation: ContractOperation | WickTermProductOperation,
    operation_plan: OperationTilePlan,
    target: torch.Tensor,
    *,
    compute_device: torch.device,
) -> torch.Tensor:
    """Execute one physical operation in exact host-staged index tiles."""

    if operation.node_id != operation_plan.node_id:
        raise ValueError("tiled operation and operation plan ids disagree")
    output_assignments = tuple(
        _tile_assignments(operation.output_indices, operation_plan)
    )
    reduction_assignments = tuple(
        _tile_assignments(operation_plan.contracted_indices, operation_plan)
    )
    if len(output_assignments) != operation_plan.output_tile_count:
        raise RuntimeError("runtime output-tile count disagrees with its tile plan")
    if len(reduction_assignments) != operation_plan.reduction_tile_count:
        raise RuntimeError("runtime reduction-tile count disagrees with its tile plan")

    calls = 0
    for output_slices in output_assignments:
        accumulator: torch.Tensor | None = None
        for reduction_slices in reduction_assignments:
            slices_by_index = dict(output_slices)
            slices_by_index.update(reduction_slices)
            tiled_inputs: list[tuple[torch.Tensor, tuple[str, ...]]] = []
            for use in operation.inputs:
                source, indices = _input_tensor_and_indices(plan, pool, use)
                source_slices = tuple(
                    slices_by_index.get(index, slice(None))
                    for index in indices
                )
                source_view = source[source_slices] if source_slices else source
                # ``copy=True`` is intentional even in CPU-only numerical tests:
                # an einsum backend may return an operand for a unary/no-op
                # contraction, and coefficient accumulation must never mutate a
                # host-resident physical input buffer.
                device_tile = source_view.to(
                    device=compute_device,
                    copy=True,
                    # The tile-byte model counts logical elements, so force a
                    # dense accelerator allocation even when the host slice has
                    # inherited non-compact strides.
                    memory_format=torch.contiguous_format,
                )
                tiled_inputs.append((device_tile, indices))

            partial = contract_labeled_tensors(
                tuple(tiled_inputs),
                operation.output_indices,
            )
            if accumulator is None:
                accumulator = partial
            else:
                accumulator.add_(partial)
            # Do not retain an additional Python reference while the next tile
            # inputs/result are allocated. ``accumulator`` owns the first
            # partial or contains the accumulated value thereafter.
            del partial
            calls += 1

        if accumulator is None:  # pragma: no cover - tile plan requires inputs
            raise RuntimeError("tiled contraction produced no partial result")
        if isinstance(operation, WickTermProductOperation):
            accumulator.mul_(operation.coefficient)

        target_slices = tuple(
            output_slices.get(index, slice(None))
            for index in operation.output_indices
        )
        target_view = target[target_slices] if target_slices else target
        host_tile = (
            accumulator
            if accumulator.device == target.device
            else accumulator.to(device=target.device)
        )
        if tuple(host_tile.shape) != tuple(target_view.shape):
            raise RuntimeError(
                f"tiled result for {operation.node_id!r} has shape "
                f"{tuple(host_tile.shape)}, expected {tuple(target_view.shape)}"
            )
        target_view.copy_(host_tile)
        # Keep the modeled device lifetime local to this output tile. Without
        # explicit deletion, loop variables can retain the previous partial
        # while PyTorch allocates the next one.
        del host_tile
        del accumulator
        del tiled_inputs

    if calls != operation_plan.contraction_calls:
        raise RuntimeError("runtime contraction-call count disagrees with tile plan")
    return target


@dataclass(frozen=True)
class NumericalExecutionResult:
    """Requested correlator tensors produced by one physical-plan execution."""

    source_plan: PhysicalExecutionPlan
    device: torch.device
    compute_device: torch.device
    outputs: Mapping[str, torch.Tensor]
    executed_steps: int

    def output(self, correlator_id: str) -> torch.Tensor:
        return self.outputs[correlator_id]

    def validate(self) -> None:
        self.source_plan.validate()
        if self.executed_steps != len(self.source_plan.steps):
            raise ValueError("Numerical execution did not execute every physical-plan step.")
        if set(self.outputs) != set(self.source_plan.outputs):
            raise ValueError("Numerical result does not contain exactly the requested outputs.")
        for correlator_id, output_binding in self.source_plan.outputs.items():
            tensor = self.outputs[correlator_id]
            binding = self.source_plan.binding(output_binding.value_id)
            if tuple(tensor.shape) != binding.shape:
                raise ValueError(
                    f"Numerical output {correlator_id!r} has shape {tuple(tensor.shape)}, "
                    f"expected {binding.shape}."
                )
            if tensor.dtype != _torch_dtype(binding.scalar_type):
                raise TypeError(
                    f"Numerical output {correlator_id!r} has dtype {tensor.dtype}, "
                    f"expected {_torch_dtype(binding.scalar_type)}."
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"Numerical output {correlator_id!r} is on {tensor.device}, "
                    f"expected {self.device}."
                )


class GenericNumericalExecutor:
    """Execute one frozen resource-bound physical plan resident or host-staged."""

    def __init__(
        self,
        materializer: PrimitiveMaterializer,
        *,
        device: str | torch.device = "cpu",
        compute_device: str | torch.device | None = None,
        tiled_plan: TiledExecutionPlan | None = None,
        buffer_pool: RuntimeBufferPool | None = None,
    ) -> None:
        if not isinstance(materializer, PrimitiveMaterializer):
            raise TypeError("GenericNumericalExecutor requires a PrimitiveMaterializer.")
        materializer.binding_plan.validate()
        self.materializer = materializer
        self.binding_plan = materializer.binding_plan
        self.plan = self.binding_plan.source_plan
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.compute_device = (
            self.device
            if compute_device is None
            else (
                compute_device
                if isinstance(compute_device, torch.device)
                else torch.device(compute_device)
            )
        )
        self.tiled_plan = tiled_plan
        if buffer_pool is not None:
            if not isinstance(buffer_pool, RuntimeBufferPool):
                raise TypeError("buffer_pool must be a RuntimeBufferPool or None")
            if buffer_pool.plan is not self.plan:
                raise ValueError("runtime buffer pool belongs to a different physical plan")
            if buffer_pool.device != self.device:
                raise ValueError("runtime buffer pool belongs to a different device")
            if buffer_pool.batch_size != 1:
                raise ValueError("generic execution requires active buffer-pool batch size one")
            if not buffer_pool.is_idle:
                raise RuntimeError("generic execution requires an idle runtime buffer pool")
        self.buffer_pool = buffer_pool
        if tiled_plan is None:
            if self.compute_device != self.device:
                raise ValueError(
                    "a distinct compute device requires a tiled execution plan"
                )
        else:
            if not isinstance(tiled_plan, TiledExecutionPlan):
                raise TypeError("tiled_plan must be a TiledExecutionPlan")
            if tiled_plan.source_plan is not self.plan:
                raise ValueError("tiled execution plan belongs to a different physical plan")

    def execute(self) -> NumericalExecutionResult:
        self.plan.validate()
        self.binding_plan.validate()
        pool = self.buffer_pool or RuntimeBufferPool(self.plan, device=self.device)

        for plan_step in self.plan.steps:
            operation = plan_step.operation
            if pool.active_capacity_bytes != plan_step.active_capacity_before:
                raise RuntimeError(
                    f"Runtime active capacity before step {plan_step.step_index} disagrees "
                    "with the physical plan."
                )

            operation_tile = (
                self.tiled_plan.operations.get(operation.node_id)
                if self.tiled_plan is not None
                else None
            )
            wrote_directly = False
            if operation_tile is not None:
                if not isinstance(
                    operation,
                    (ContractOperation, WickTermProductOperation),
                ):
                    raise TypeError("tile plan attached to a non-contraction operation")
                target = pool.reserve(operation.node_id)
                value = _execute_tiled_operation_into(
                    self.plan,
                    pool,
                    operation,
                    operation_tile,
                    target,
                    compute_device=self.compute_device,
                )
                wrote_directly = True
            elif isinstance(operation, MaterializePrimitiveOperation):
                materialized = self.materializer.materialize(operation.value_id)
                if (
                    materialized.value_id != operation.value_id
                    or materialized.resource_id != operation.resource_id
                    or materialized.buffer_id != operation.output_buffer_id
                ):
                    raise ValueError(
                        "Primitive materializer returned metadata inconsistent with the "
                        "physical primitive operation."
                    )
                value = materialized.tensor
            elif isinstance(operation, ContractOperation):
                value = _execute_contraction_into_if_possible(
                    self.plan,
                    pool,
                    operation,
                )
                if value is None:
                    value = _execute_contraction(self.plan, pool, operation)
                else:
                    wrote_directly = True
            elif isinstance(operation, WickTermProductOperation):
                value = _execute_wick_term(self.plan, pool, operation)
            elif isinstance(operation, CorrelatorSumOperation):
                value = _execute_correlator_sum(self.plan, pool, operation)
            else:  # pragma: no cover
                raise TypeError(f"Unsupported physical operation {type(operation)!r}.")

            if not wrote_directly:
                pool.write(operation.node_id, value)
            if pool.active_capacity_bytes != plan_step.peak_active_capacity_during_step:
                raise RuntimeError(
                    f"Runtime active capacity during step {plan_step.step_index} disagrees "
                    "with the physical plan."
                )

            for released_value in plan_step.released_values:
                pool.release(released_value)

            if pool.active_capacity_bytes != plan_step.active_capacity_after:
                raise RuntimeError(
                    f"Runtime active capacity after step {plan_step.step_index} disagrees "
                    "with the physical plan."
                )

        expected_live = set(self.plan.source_schedule.output_nodes)
        if set(pool.active_values) != expected_live:
            raise RuntimeError(
                "Runtime final live-value set does not match retained logical outputs."
            )

        outputs: dict[str, torch.Tensor] = {}
        for correlator_id, output_binding in self.plan.outputs.items():
            value = pool.read(output_binding.value_id)
            if value.data_ptr() != pool.storage[output_binding.buffer_id].data_ptr():
                raise RuntimeError("Requested output is not backed by its planned buffer.")
            outputs[correlator_id] = value

        result = NumericalExecutionResult(
            source_plan=self.plan,
            device=self.device,
            compute_device=self.compute_device,
            outputs=dict(outputs),
            executed_steps=len(self.plan.steps),
        )
        result.validate()
        return result


__all__ = [
    "RuntimeBufferPool",
    "NumericalExecutionResult",
    "GenericNumericalExecutor",
]
