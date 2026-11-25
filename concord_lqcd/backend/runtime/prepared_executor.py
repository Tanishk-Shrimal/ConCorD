"""Resident execution with every physical-plan step prepared once.

``ResidentExecutionSession`` runs the same frozen plan for every work unit of
one configuration.  Which buffer view each value uses, whether a binary
contraction is a direct matrix product (its orientation, axis order and layout
copies), the opt_einsum expression otherwise, the Wick-term coefficients and
the plan's capacity bookkeeping depend only on the plan, the device type and
the active batch size, never on tensor values.  They are decided once per
process on storage-free ``meta`` tensors, bound once per session to the pool
storage, and reused for every work unit.

Each work unit performs exactly the PyTorch operations of
``GenericNumericalExecutor`` and ``BatchedNumericalExecutor`` on the same
buffers, so its results are bitwise identical.  Host-staged tiled execution
keeps the step-by-step executor.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Sequence

import torch

from concord_lqcd.backend.planner.physical import (
    ContractOperation,
    CorrelatorSumOperation,
    MaterializePrimitiveOperation,
    PhysicalExecutionPlan,
    PhysicalInputUse,
    WickTermProductOperation,
)

from .batched_executor import (
    BatchedNumericalExecutionResult,
    BatchedNumericalExecutor,
    _private_batch_index,
)
from .contraction import labeled_contraction_expression, prepare_direct_binary_gemm
from .materialization import PrimitiveMaterializer
from .numerical_executor import (
    NumericalExecutionResult,
    RuntimeBufferPool,
    _numel,
    _torch_dtype,
)


_DIRECT = 0
_EINSUM = 1
_MATERIALIZE = 2
_WICK = 3
_CONSTANT = 4
_SUM = 5


@dataclass(frozen=True)
class PreparedPlanSteps:
    """Storage-free decisions for every step of one plan at one batch size.

    It holds no reference to the plan object itself, so a process-wide cache
    of these records does not keep plans alive.
    """

    batch_size: int
    device_type: str
    steps: tuple[tuple, ...]
    retained_values: tuple[str, ...]


def check_capacity_bookkeeping(plan: PhysicalExecutionPlan) -> None:
    """Replay the executors' per-step ownership and capacity checks once.

    The step-by-step executors repeat these checks for every work unit, scaled
    by the batch size.  They depend only on the plan, so one replay without
    tensors proves them for every work unit and batch size.
    """

    owner: dict[str, str] = {}
    active: set[str] = set()
    capacity = 0
    for plan_step in plan.steps:
        node_id = plan_step.operation.node_id
        if capacity != plan_step.active_capacity_before:
            raise RuntimeError(
                f"Runtime active capacity before step {plan_step.step_index} disagrees "
                "with the physical plan."
            )
        buffer_id = plan.binding(node_id).buffer_id
        if buffer_id in owner:
            raise RuntimeError(
                f"Runtime attempted to overwrite active buffer {buffer_id!r}, currently "
                f"owned by {owner[buffer_id]!r}, with value {node_id!r}."
            )
        if node_id in active:
            raise RuntimeError(f"Runtime attempted to produce value {node_id!r} twice.")
        owner[buffer_id] = node_id
        active.add(node_id)
        capacity += plan.buffer(buffer_id).capacity_bytes
        if capacity != plan_step.peak_active_capacity_during_step:
            raise RuntimeError(
                f"Runtime active capacity during step {plan_step.step_index} disagrees "
                "with the physical plan."
            )
        for value_id in plan_step.released_values:
            if value_id not in active:
                raise RuntimeError(
                    f"Runtime attempted to release inactive value {value_id!r}."
                )
            released_buffer = plan.binding(value_id).buffer_id
            if owner.get(released_buffer) != value_id:
                raise RuntimeError(
                    f"Runtime release for {value_id!r} found buffer {released_buffer!r} "
                    f"owned by {owner.get(released_buffer)!r}."
                )
            active.remove(value_id)
            del owner[released_buffer]
            capacity -= plan.buffer(released_buffer).capacity_bytes
        if capacity != plan_step.active_capacity_after:
            raise RuntimeError(
                f"Runtime active capacity after step {plan_step.step_index} disagrees "
                "with the physical plan."
            )
    if active != set(plan.source_schedule.output_nodes):
        raise RuntimeError(
            "Runtime final live-value set does not match retained logical outputs."
        )


def _value_shape(
    plan: PhysicalExecutionPlan, value_id: str, batch_size: int
) -> tuple[int, ...]:
    shape = plan.binding(value_id).shape
    return shape if batch_size == 1 else (batch_size, *shape)


def _meta_view(
    plan: PhysicalExecutionPlan, value_id: str, batch_size: int
) -> torch.Tensor:
    """A storage-free tensor with the shape and strides of the pool's view."""

    binding = plan.binding(value_id)
    flat = torch.empty(
        batch_size * _numel(binding.shape),
        dtype=_torch_dtype(binding.scalar_type),
        device="meta",
    )
    shape = _value_shape(plan, value_id, batch_size)
    return flat.reshape(shape) if shape else flat.reshape(())


def _labeled_input(
    plan: PhysicalExecutionPlan,
    use: PhysicalInputUse,
    batch_index: str | None,
) -> tuple[str, tuple[str, ...]]:
    binding = plan.binding(use.value_id)
    ports = tuple(item.port for item in use.bindings)
    if ports != binding.axis_labels:
        raise ValueError(
            f"Physical input use of {use.value_id!r} binds ports {ports!r}, but its "
            f"runtime value axes are {binding.axis_labels!r}."
        )
    indices = tuple(item.index for item in use.bindings)
    return use.value_id, indices if batch_index is None else (batch_index, *indices)


def _check_output_shape(
    inputs: tuple[tuple[torch.Tensor, tuple[str, ...]], ...],
    output_indices: tuple[str, ...],
    expected: tuple[int, ...],
    node_id: str,
) -> None:
    sizes: dict[str, int] = {}
    for tensor, indices in inputs:
        sizes.update(zip(indices, tensor.shape))
    shape = tuple(sizes[index] for index in output_indices)
    if shape != expected:
        raise ValueError(
            f"Runtime result {node_id!r} has shape {shape}, expected {expected}."
        )


def _prepare_steps(
    plan: PhysicalExecutionPlan, batch_size: int, device_type: str
) -> PreparedPlanSteps:
    plan.validate()
    check_capacity_bookkeeping(plan)
    batch_index = None if batch_size == 1 else _private_batch_index(plan)
    meta: dict[str, torch.Tensor] = {}

    def meta_view(value_id: str) -> torch.Tensor:
        view = meta.get(value_id)
        if view is None:
            view = meta[value_id] = _meta_view(plan, value_id, batch_size)
        return view

    def labeled(
        operation: ContractOperation | WickTermProductOperation,
    ) -> tuple[tuple[str, ...], tuple[tuple[torch.Tensor, tuple[str, ...]], ...], tuple[str, ...]]:
        uses = tuple(_labeled_input(plan, use, batch_index) for use in operation.inputs)
        output = tuple(operation.output_indices)
        if batch_index is not None:
            output = (batch_index, *output)
        return (
            tuple(value_id for value_id, _ in uses),
            tuple((meta_view(value_id), indices) for value_id, indices in uses),
            output,
        )

    steps: list[tuple] = []
    for plan_step in plan.steps:
        operation = plan_step.operation
        node_id = operation.node_id
        binding = plan.binding(node_id)
        dtype = _torch_dtype(binding.scalar_type)
        target_shape = _value_shape(plan, node_id, batch_size)
        if isinstance(operation, ContractOperation):
            value_ids, inputs, output = labeled(operation)
            direct = prepare_direct_binary_gemm(
                inputs,
                output,
                allow_layout_copies=device_type == "cpu",
            )
            if direct is not None:
                if direct.output_shape != target_shape:
                    raise ValueError(
                        f"Direct contraction target has shape {target_shape}, expected "
                        f"{direct.output_shape}."
                    )
                steps.append((_DIRECT, node_id, value_ids, direct))
            else:
                _check_output_shape(inputs, output, target_shape, node_id)
                expression = labeled_contraction_expression(inputs, output)
                steps.append((_EINSUM, node_id, value_ids, expression))
        elif isinstance(operation, MaterializePrimitiveOperation):
            steps.append((
                _MATERIALIZE,
                node_id,
                operation,
                target_shape,
                dtype,
                batch_size * binding.size_bytes,
            ))
        elif isinstance(operation, WickTermProductOperation):
            if operation.inputs:
                value_ids, inputs, output = labeled(operation)
                _check_output_shape(inputs, output, target_shape, node_id)
                expression = labeled_contraction_expression(inputs, output)
                steps.append((
                    _WICK, node_id, value_ids, expression, operation.coefficient, dtype,
                ))
            else:
                if binding.shape:
                    raise ValueError("A zero-input Wick term must be scalar.")
                steps.append((_CONSTANT, node_id, operation.coefficient, dtype))
        elif isinstance(operation, CorrelatorSumOperation):
            value_ids = tuple(use.value_id for use in operation.inputs)
            if not value_ids and binding.shape and batch_size == 1:
                raise ValueError("An empty correlator sum must be scalar.")
            for value_id in value_ids:
                if _value_shape(plan, value_id, batch_size) != target_shape:
                    raise ValueError(
                        f"Correlator sum {operation.correlator_id!r} received an input "
                        f"shape {_value_shape(plan, value_id, batch_size)} incompatible "
                        f"with output {target_shape}."
                    )
                if _torch_dtype(plan.binding(value_id).scalar_type) != dtype:
                    raise TypeError(
                        f"Correlator sum {operation.correlator_id!r} received dtype "
                        f"{plan.binding(value_id).scalar_type}, expected {dtype}."
                    )
            steps.append((_SUM, node_id, value_ids, target_shape, dtype))
        else:  # pragma: no cover - physical plan validates the closed union
            raise TypeError(f"Unsupported physical operation {type(operation)!r}.")

    return PreparedPlanSteps(
        batch_size=batch_size,
        device_type=device_type,
        steps=tuple(steps),
        retained_values=tuple(plan.source_schedule.output_nodes),
    )


_PREPARED_STEPS: dict[tuple[int, int, str], tuple[weakref.ref, PreparedPlanSteps]] = {}


def prepare_plan_steps(
    plan: PhysicalExecutionPlan, batch_size: int, device_type: str
) -> PreparedPlanSteps:
    """Return the storage-free step decisions, prepared once per process."""

    key = (id(plan), batch_size, device_type)
    entry = _PREPARED_STEPS.get(key)
    if entry is not None and entry[0]() is plan:
        return entry[1]
    prepared = _prepare_steps(plan, batch_size, device_type)
    try:
        reference = weakref.ref(plan, lambda _, key=key: _PREPARED_STEPS.pop(key, None))
    except TypeError:  # pragma: no cover - physical plans are weak-referenceable
        return prepared
    _PREPARED_STEPS[key] = (reference, prepared)
    return prepared


class PreparedResidentProgram:
    """Prepared steps bound to the storage of one resident pool.

    Binding creates every buffer view and fixed operand once. The pool must
    keep its storage and active batch size for the program's lifetime, which a
    configuration-scoped resident session guarantees.
    """

    def __init__(
        self,
        plan: PhysicalExecutionPlan,
        pool: RuntimeBufferPool,
        prepared: PreparedPlanSteps,
    ) -> None:
        if pool.plan is not plan:
            raise ValueError("runtime buffer pool belongs to a different physical plan")
        if pool.batch_size != prepared.batch_size:
            raise ValueError("prepared steps disagree with the active pool batch size")
        if pool.device.type != prepared.device_type:
            raise ValueError("prepared steps were made for a different device type")
        if not pool.is_idle:
            raise RuntimeError("binding prepared steps requires an idle runtime buffer pool")
        self.plan = plan
        self.pool = pool
        self.batch_size = prepared.batch_size
        device = pool.device
        views: dict[str, torch.Tensor] = {}

        def view(value_id: str) -> torch.Tensor:
            tensor = views.get(value_id)
            if tensor is None:
                tensor = views[value_id] = pool.planned_view(value_id)
            return tensor

        bound: list[tuple] = []
        for step in prepared.steps:
            kind = step[0]
            if kind == _DIRECT:
                _, node_id, value_ids, direct = step
                left = view(value_ids[direct.left_input]).permute(direct.left_permutation)
                right = view(value_ids[1 - direct.left_input]).permute(
                    direct.right_permutation
                )
                if (
                    left.shape != direct.left.shape
                    or left.stride() != direct.left.stride()
                    or right.shape != direct.right.shape
                    or right.stride() != direct.right.stride()
                ):
                    raise RuntimeError(
                        f"Prepared direct contraction {node_id!r} disagrees with the "
                        "layout of its pool views."
                    )
                bound.append((
                    kind,
                    torch.bmm if direct.batched else torch.mm,
                    left,
                    None if direct.copy_left_layout else left.view(direct.left_matrix_shape),
                    direct.left_matrix_shape,
                    right,
                    None if direct.copy_right_layout else right.view(direct.right_matrix_shape),
                    direct.right_matrix_shape,
                    view(node_id).view(direct.matrix_output_shape),
                ))
            elif kind == _EINSUM:
                _, node_id, value_ids, expression = step
                bound.append((
                    kind, expression, tuple(view(value_id) for value_id in value_ids),
                    view(node_id),
                ))
            elif kind == _MATERIALIZE:
                _, node_id, operation, shape, dtype, size_bytes = step
                bound.append((kind, operation, view(node_id), shape, dtype, size_bytes))
            elif kind == _WICK:
                _, node_id, value_ids, expression, coefficient, dtype = step
                bound.append((
                    kind,
                    expression,
                    tuple(view(value_id) for value_id in value_ids),
                    torch.tensor(coefficient, dtype=dtype, device=device),
                    view(node_id),
                ))
            elif kind == _CONSTANT:
                _, node_id, coefficient, dtype = step
                value = (
                    torch.tensor(coefficient, dtype=dtype, device=device)
                    if self.batch_size == 1
                    else torch.full(
                        (self.batch_size,), coefficient, dtype=dtype, device=device
                    )
                )
                bound.append((kind, value, view(node_id)))
            else:
                _, node_id, value_ids, shape, dtype = step
                bound.append((
                    kind, tuple(view(value_id) for value_id in value_ids), shape, dtype,
                    view(node_id),
                ))
        self._steps = tuple(bound)
        self._retained = prepared.retained_values

        outputs: dict[str, torch.Tensor] = {}
        for correlator_id, output in plan.outputs.items():
            tensor = view(output.value_id)
            if tensor.data_ptr() != pool.storage[output.buffer_id].data_ptr():
                raise RuntimeError("Requested output is not backed by its planned buffer.")
            outputs[correlator_id] = tensor
        self._outputs = outputs

    def execute(
        self, materializers: Sequence[PrimitiveMaterializer]
    ) -> NumericalExecutionResult | BatchedNumericalExecutionResult:
        pool = self.pool
        if pool.batch_size != self.batch_size:
            raise RuntimeError("prepared program disagrees with the active pool batch size")
        if not pool.is_idle:
            raise RuntimeError("prepared execution requires an idle runtime buffer pool")
        values = tuple(materializers)
        if len(values) != self.batch_size:
            raise ValueError(
                f"prepared program expects {self.batch_size} materializers, got {len(values)}"
            )
        if self.batch_size == 1:
            materializer = values[0]
            materializer.binding_plan.validate()

            def materialize(operation: MaterializePrimitiveOperation) -> torch.Tensor:
                value = materializer.materialize(operation.value_id)
                if (
                    value.value_id != operation.value_id
                    or value.resource_id != operation.resource_id
                    or value.buffer_id != operation.output_buffer_id
                ):
                    raise ValueError(
                        "Primitive materializer returned metadata inconsistent with the "
                        "physical primitive operation."
                    )
                return value.tensor
        else:
            materialize = BatchedNumericalExecutor(
                values, device=pool.device, buffer_pool=pool
            )._materialize

        device = pool.device
        for step in self._steps:
            kind = step[0]
            if kind == _DIRECT:
                _, gemm, left, left_matrix, left_shape, right, right_matrix, right_shape, target = step
                gemm(
                    left_matrix if left_matrix is not None
                    else left.contiguous().view(left_shape),
                    right_matrix if right_matrix is not None
                    else right.contiguous().view(right_shape),
                    out=target,
                )
            elif kind == _EINSUM:
                _, expression, inputs, target = step
                target.copy_(expression(*inputs, backend="torch"))
            elif kind == _MATERIALIZE:
                _, operation, target, shape, dtype, size_bytes = step
                tensor = materialize(operation)
                if tuple(tensor.shape) != shape:
                    raise ValueError(
                        f"Runtime result {operation.node_id!r} has shape "
                        f"{tuple(tensor.shape)}, expected {shape}."
                    )
                if tensor.dtype != dtype:
                    raise TypeError(
                        f"Runtime result {operation.node_id!r} has dtype {tensor.dtype}, "
                        f"expected {dtype}."
                    )
                if tensor.numel() * tensor.element_size() != size_bytes:
                    raise ValueError(
                        f"Runtime result {operation.node_id!r} occupies "
                        f"{tensor.numel() * tensor.element_size()} bytes, expected "
                        f"{size_bytes}."
                    )
                target.copy_(tensor if tensor.device == device else tensor.to(device=device))
            elif kind == _WICK:
                _, expression, inputs, coefficient, target = step
                target.copy_(expression(*inputs, backend="torch") * coefficient)
            elif kind == _CONSTANT:
                _, value, target = step
                target.copy_(value)
            else:
                _, inputs, shape, dtype, target = step
                result = torch.zeros(shape, dtype=dtype, device=device)
                for tensor in inputs:
                    result.add_(tensor)
                target.copy_(result)

        pool.retain_outputs(self._retained)
        if self.batch_size == 1:
            return NumericalExecutionResult(
                source_plan=self.plan,
                device=device,
                compute_device=device,
                outputs=dict(self._outputs),
                executed_steps=len(self._steps),
            )
        return BatchedNumericalExecutionResult(
            source_plan=self.plan,
            device=device,
            outputs=dict(self._outputs),
            batch_size=self.batch_size,
            executed_steps=len(self._steps),
        )


__all__ = [
    "PreparedPlanSteps",
    "PreparedResidentProgram",
    "check_capacity_bookkeeping",
    "prepare_plan_steps",
]
