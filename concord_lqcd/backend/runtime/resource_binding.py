"""Exact resource binding for the new ConCorD physical execution pipeline.

This layer sits between abstract physical planning and numerical runtime
execution. It resolves the remaining runtime information needed by primitive
materialization without changing algebra, scheduling, or buffer allocation.

The three identities remain strictly separate::

    compiler resource id != logical value id != physical buffer id

A resource binding is created once per *used* compiler resource. A
materialization request is then created for every scheduled primitive value by
combining that resource binding with the value's exact operator-slot placement
and a concrete operator-time context.

Important: ``GlobalTensorResource.payload`` is an exemplar describing *what*
resource to construct. Its original operator ids/position labels are not an
occurrence placement. The authoritative occurrence placement is the scoped
placement carried by the primitive value/physical operation. Resource binding
therefore never infers execution times from exemplar operator ids.

No tensor is loaded here. In particular, propagation sources are bound by exact
``resource_id``; there is deliberately no flavor fallback such as
``P_data[flavor]``. Ordinary and generalized perambulators differ only in their
``PropagationRecipe``/external source binding, not in downstream contraction
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeAlias

from concord_lqcd.backend.planner.physical import (
    MaterializePrimitiveOperation,
    PhysicalExecutionPlan,
)
from concord_lqcd.compiler.distillation_lowering import KernelRecipe, identify_kernel_resource
from concord_lqcd.compiler.physics_ir import SpinTensorRecord
from concord_lqcd.compiler.propagation_lowering import PropagationRecipe
from concord_lqcd.core.dirac_basis import get_spin_matrices


_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


@dataclass(frozen=True)
class OperatorTimeContext:
    """Concrete Euclidean times for frontend operator slots, scoped by arity.

    ``times_by_arity[n][i]`` is the time assigned to frontend operator slot
    ``i`` in an ``n``-operator correlator. Arity scoping is required because
    partial CSE deliberately keeps, for example, ``op1`` in a two-point graph
    distinct from ``op1`` in a three-point graph.
    """

    times_by_arity: Mapping[int, tuple[int, ...]]

    def __post_init__(self) -> None:
        normalized: dict[int, tuple[int, ...]] = {}
        for arity, times in self.times_by_arity.items():
            if not isinstance(arity, int) or isinstance(arity, bool) or arity < 0:
                raise ValueError("Operator arity keys must be non-negative integers.")
            values = tuple(times)
            if len(values) != arity:
                raise ValueError(
                    f"Operator-time context for arity {arity} contains {len(values)} "
                    f"times; expected {arity}."
                )
            for value in values:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError("Operator times must be integer lattice timeslices.")
            normalized[arity] = values
        object.__setattr__(self, "times_by_arity", normalized)

    def time_for(self, arity: int, operator_id: int) -> int:
        try:
            times = self.times_by_arity[arity]
        except KeyError as exc:
            raise ValueError(
                f"No operator-time context was supplied for correlator arity {arity}."
            ) from exc
        if operator_id < 0 or operator_id >= len(times):
            raise ValueError(
                f"Operator id {operator_id} is outside the arity-{arity} placement."
            )
        return times[operator_id]


@dataclass(frozen=True)
class ExternalTensorSource:
    """Concrete external tensor source selected for one propagation resource.

    Examples include an HDF5 file/group, an in-memory tensor registry entry, or
    a custom dataset service. ``kind`` tells the later loader how to interpret
    ``location``; ``dataset`` and ``metadata`` carry source-specific details
    without entering compiler IR.
    """

    source_id: str
    kind: str
    location: str
    dataset: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("External tensor source_id must be non-empty.")
        if not self.kind:
            raise ValueError("External tensor source kind must be non-empty.")
        if not self.location:
            raise ValueError("External tensor source location must be non-empty.")
        if self.dataset == "":
            raise ValueError("External tensor dataset must be non-empty when provided.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class KernelResourceBinding:
    """Bind one spatial-kernel resource to a runtime tensor provider.

    ``recipe`` is the global resource exemplar. Its operator ids are not used as
    occurrence placement; each request carries the authoritative scoped
    placement separately.
    """

    resource_id: str
    provider_id: str
    recipe: KernelRecipe


@dataclass(frozen=True)
class SpinResourceBinding:
    """Self-contained numerical binding for one reusable spin resource."""

    resource_id: str
    tensor: SpinTensorRecord
    matrix: tuple[tuple[complex, ...], ...]


@dataclass(frozen=True)
class PropagationResourceBinding:
    """Bind one Stage-5 propagation resource to an exact external source."""

    resource_id: str
    recipe: PropagationRecipe
    source: ExternalTensorSource


ResourceBinding: TypeAlias = (
    KernelResourceBinding | SpinResourceBinding | PropagationResourceBinding
)


@dataclass(frozen=True)
class PrimitiveMaterializationRequest:
    """Common exact request metadata for one scheduled primitive value."""

    step_index: int
    node_id: str
    value_id: str
    resource_id: str
    placement: tuple[object, ...]
    output_buffer_id: str
    expected_shape: tuple[int, ...]
    scalar_type: str


@dataclass(frozen=True)
class KernelMaterializationRequest(PrimitiveMaterializationRequest):
    provider_id: str
    recipe: KernelRecipe
    operator_ids: tuple[int, ...]
    operator_times: tuple[int, ...]


@dataclass(frozen=True)
class SpinMaterializationRequest(PrimitiveMaterializationRequest):
    tensor_kind: str
    matrix: tuple[tuple[complex, ...], ...]
    operator_id: int | None
    operator_time: int | None


@dataclass(frozen=True)
class PropagationMaterializationRequest(PrimitiveMaterializationRequest):
    source: ExternalTensorSource
    recipe: PropagationRecipe
    source_operator_id: int
    source_time: int
    insertion_operator_ids: tuple[int, ...]
    insertion_times: tuple[int, ...]
    sink_operator_id: int
    sink_time: int


MaterializationRequest: TypeAlias = (
    KernelMaterializationRequest
    | SpinMaterializationRequest
    | PropagationMaterializationRequest
)


@dataclass(frozen=True)
class ResourceBindingStatistics:
    used_resources: int
    kernel_resources: int
    spin_resources: int
    propagation_resources: int
    primitive_materializations: int


@dataclass(frozen=True)
class ResourceBindingPlan:
    """Exact resource/provider/dataset bindings for one physical plan instance."""

    source_plan: PhysicalExecutionPlan
    time_context: OperatorTimeContext
    bindings: Mapping[str, ResourceBinding]
    materializations: Mapping[str, MaterializationRequest]
    statistics: ResourceBindingStatistics
    spin_basis: str = "DR"

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, _MAPPING_PROXY_TYPE):
            object.__setattr__(
                self,
                "bindings",
                MappingProxyType(dict(self.bindings)),
            )
        if not isinstance(self.materializations, _MAPPING_PROXY_TYPE):
            object.__setattr__(
                self,
                "materializations",
                MappingProxyType(dict(self.materializations)),
            )

    def binding(self, resource_id: str) -> ResourceBinding:
        return self.bindings[resource_id]

    def materialization(self, value_id: str) -> MaterializationRequest:
        return self.materializations[value_id]

    def validate(self) -> None:
        self.source_plan.validate()
        if self.spin_basis not in ("DR", "DP"):
            raise ValueError("spin_basis must be 'DR' or 'DP'.")
        primitive_operations = _used_primitive_operations(self.source_plan)
        used_resource_ids = tuple(dict.fromkeys(op.resource_id for op in primitive_operations))
        if set(self.bindings) != set(used_resource_ids):
            raise ValueError(
                "Resource-binding map does not match resources used by scheduled primitives."
            )
        if set(self.materializations) != {op.value_id for op in primitive_operations}:
            raise ValueError(
                "Materialization-request map does not match scheduled primitive values."
            )

        for operation in primitive_operations:
            request = self.materializations[operation.value_id]
            binding = self.bindings[operation.resource_id]
            physical_binding = self.source_plan.binding(operation.value_id)
            if (
                request.step_index != operation.step_index
                or request.node_id != operation.node_id
                or request.value_id != operation.value_id
                or request.resource_id != operation.resource_id
                or request.placement != operation.placement
                or request.output_buffer_id != operation.output_buffer_id
                or request.expected_shape != physical_binding.shape
                or request.scalar_type != physical_binding.scalar_type
            ):
                raise ValueError(
                    f"Materialization request for {operation.value_id!r} disagrees with "
                    "the physical primitive operation."
                )

            arity, raw = _scoped_placement(operation.placement)
            if operation.kind == "kernel":
                if not isinstance(binding, KernelResourceBinding) or not isinstance(
                    request, KernelMaterializationRequest
                ):
                    raise ValueError("Kernel primitive has incompatible resource binding/request.")
                expected_ids = _kernel_operator_ids(raw, arity)
                expected_times = tuple(
                    self.time_context.time_for(arity, operator_id)
                    for operator_id in expected_ids
                )
                if (
                    request.provider_id != binding.provider_id
                    or request.recipe != binding.recipe
                    or request.operator_ids != expected_ids
                    or request.operator_times != expected_times
                ):
                    raise ValueError("Kernel materialization does not match its binding/placement.")
            elif operation.kind == "spin":
                if not isinstance(binding, SpinResourceBinding) or not isinstance(
                    request, SpinMaterializationRequest
                ):
                    raise ValueError("Spin primitive has incompatible resource binding/request.")
                expected_id = _spin_operator_id(raw, arity)
                expected_time = (
                    self.time_context.time_for(arity, expected_id)
                    if expected_id is not None
                    else None
                )
                if (
                    request.tensor_kind != binding.tensor.kind
                    or request.matrix != binding.matrix
                    or request.operator_id != expected_id
                    or request.operator_time != expected_time
                ):
                    raise ValueError("Spin materialization does not match its binding/placement.")
            elif operation.kind == "propagation":
                if not isinstance(binding, PropagationResourceBinding) or not isinstance(
                    request, PropagationMaterializationRequest
                ):
                    raise ValueError(
                        "Propagation primitive has incompatible resource binding/request."
                    )
                source_id, insertion_ids, sink_id = _path_operator_ids(raw, arity)
                if len(insertion_ids) != len(binding.recipe.insertion_specs):
                    raise ValueError(
                        "Propagation placement insertion count disagrees with Stage-5 recipe."
                    )
                if (
                    request.source != binding.source
                    or request.recipe != binding.recipe
                    or request.source_operator_id != source_id
                    or request.source_time != self.time_context.time_for(arity, source_id)
                    or request.insertion_operator_ids != insertion_ids
                    or request.insertion_times
                    != tuple(
                        self.time_context.time_for(arity, operator_id)
                        for operator_id in insertion_ids
                    )
                    or request.sink_operator_id != sink_id
                    or request.sink_time != self.time_context.time_for(arity, sink_id)
                ):
                    raise ValueError(
                        "Propagation materialization does not match its binding/placement."
                    )
            else:
                raise ValueError(f"Unsupported primitive kind {operation.kind!r}.")

        expected_statistics = ResourceBindingStatistics(
            used_resources=len(self.bindings),
            kernel_resources=sum(
                isinstance(item, KernelResourceBinding) for item in self.bindings.values()
            ),
            spin_resources=sum(
                isinstance(item, SpinResourceBinding) for item in self.bindings.values()
            ),
            propagation_resources=sum(
                isinstance(item, PropagationResourceBinding) for item in self.bindings.values()
            ),
            primitive_materializations=len(self.materializations),
        )
        if self.statistics != expected_statistics:
            raise ValueError("Resource-binding statistics are inconsistent.")


@dataclass(frozen=True)
class ResourceBindingTemplate:
    """Configuration-local bindings shared by operator-time contexts."""

    source_plan: PhysicalExecutionPlan
    bindings: Mapping[str, ResourceBinding]
    primitive_operations: tuple[MaterializePrimitiveOperation, ...]
    statistics: ResourceBindingStatistics
    spin_basis: str = "DR"

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, _MAPPING_PROXY_TYPE):
            object.__setattr__(
                self,
                "bindings",
                MappingProxyType(dict(self.bindings)),
            )
        object.__setattr__(
            self,
            "primitive_operations",
            tuple(self.primitive_operations),
        )
        if self.spin_basis not in ("DR", "DP"):
            raise ValueError("spin_basis must be 'DR' or 'DP'.")

    def bind(
        self,
        operator_times: OperatorTimeContext | Mapping[int, Sequence[int]],
    ) -> ResourceBindingPlan:
        """Specialize only time-dependent materialization requests."""

        time_context = _normalize_time_context(operator_times)
        plan = ResourceBindingPlan(
            source_plan=self.source_plan,
            time_context=time_context,
            bindings=self.bindings,
            materializations=_bind_materialization_requests(
                self.source_plan,
                self.primitive_operations,
                self.bindings,
                time_context,
            ),
            statistics=self.statistics,
            spin_basis=self.spin_basis,
        )
        plan.validate()
        return plan


def _normalize_time_context(
    operator_times: OperatorTimeContext | Mapping[int, Sequence[int]],
) -> OperatorTimeContext:
    if isinstance(operator_times, OperatorTimeContext):
        return operator_times
    return OperatorTimeContext({arity: tuple(times) for arity, times in operator_times.items()})


def _global_resources(plan: PhysicalExecutionPlan):
    logical_dag = plan.source_schedule.source_analysis.source_dag
    return logical_dag.source_dag.source_plan.source_dag.resources


def _used_primitive_operations(
    plan: PhysicalExecutionPlan,
) -> tuple[MaterializePrimitiveOperation, ...]:
    return tuple(
        step.operation
        for step in plan.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
    )


def _freeze_spin_matrix(
    kind: str,
    *,
    spin_basis: str = "DR",
) -> tuple[tuple[complex, ...], ...]:
    matrices = get_spin_matrices(spin_basis)
    try:
        matrix = matrices[kind]
    except KeyError as exc:
        raise ValueError(f"No numerical spin matrix is registered for {kind!r}.") from exc
    if tuple(matrix.shape) != (4, 4):
        raise ValueError(f"Spin matrix {kind!r} does not have shape (4, 4).")
    return tuple(tuple(complex(value) for value in row) for row in matrix)


def _scoped_placement(placement: tuple[object, ...]) -> tuple[int, tuple[object, ...]]:
    if len(placement) != 3 or placement[0] != "operator_arity":
        raise ValueError(
            "Primitive placement must carry ('operator_arity', arity, raw_placement)."
        )
    arity = placement[1]
    raw = placement[2]
    if not isinstance(arity, int) or isinstance(arity, bool) or arity < 0:
        raise ValueError("Primitive placement carries an invalid operator arity.")
    if not isinstance(raw, tuple) or not raw:
        raise ValueError("Primitive placement carries an invalid raw placement tuple.")
    return arity, raw


def _validate_operator_id(operator_id: object, arity: int, *, role: str) -> int:
    if not isinstance(operator_id, int) or isinstance(operator_id, bool):
        raise ValueError(f"{role} operator placement is not an integer operator id.")
    if operator_id < 0 or operator_id >= arity:
        raise ValueError(
            f"{role} operator id {operator_id} lies outside correlator arity {arity}."
        )
    return operator_id


def _kernel_operator_ids(raw: tuple[object, ...], arity: int) -> tuple[int, ...]:
    if len(raw) != 2 or raw[0] != "kernel_ops" or not isinstance(raw[1], tuple):
        raise ValueError("Kernel primitive does not carry a kernel_ops placement.")
    ids = tuple(
        _validate_operator_id(operator_id, arity, role="Kernel")
        for operator_id in raw[1]
    )
    if len(ids) != len(set(ids)):
        raise ValueError("Kernel placement contains duplicate operator ids.")
    return ids


def _spin_operator_id(raw: tuple[object, ...], arity: int) -> int | None:
    if len(raw) != 2 or raw[0] != "spin_op":
        raise ValueError("Spin primitive does not carry a spin_op placement.")
    operator_id = raw[1]
    if operator_id is None:
        return None
    return _validate_operator_id(operator_id, arity, role="Spin")


def _path_operator_ids(
    raw: tuple[object, ...],
    arity: int,
) -> tuple[int, tuple[int, ...], int]:
    if len(raw) != 4 or raw[0] != "path_ops" or not isinstance(raw[2], tuple):
        raise ValueError("Propagation primitive does not carry a path_ops placement.")
    source = _validate_operator_id(raw[1], arity, role="Propagation source")
    insertions = tuple(
        _validate_operator_id(operator_id, arity, role="Propagation insertion")
        for operator_id in raw[2]
    )
    sink = _validate_operator_id(raw[3], arity, role="Propagation sink")
    return source, insertions, sink


def _bind_resource(
    resource,
    *,
    propagation_sources: Mapping[str, ExternalTensorSource],
    kernel_provider_id: str,
    kernel_provider_ids: Mapping[str, str],
    spin_basis: str,
) -> ResourceBinding:
    resource_id = resource.resource_id
    if resource.kind == "kernel":
        recipe = resource.payload
        if not isinstance(recipe, KernelRecipe):
            raise TypeError("Kernel global resource does not carry a KernelRecipe payload.")
        if identify_kernel_resource(recipe) != resource_id:
            raise ValueError("Kernel global resource id disagrees with its recipe identity.")
        provider_id = kernel_provider_ids.get(resource_id, kernel_provider_id)
        if not provider_id:
            raise ValueError(f"Kernel resource {resource_id!r} has an empty provider id.")
        return KernelResourceBinding(resource_id, provider_id, recipe)

    if resource.kind == "spin":
        tensor = resource.payload
        if not isinstance(tensor, SpinTensorRecord):
            raise TypeError("Spin global resource does not carry a SpinTensorRecord payload.")
        return SpinResourceBinding(
            resource_id=resource_id,
            tensor=tensor,
            matrix=_freeze_spin_matrix(tensor.kind, spin_basis=spin_basis),
        )

    if resource.kind == "propagation":
        recipe = resource.payload
        if not isinstance(recipe, PropagationRecipe):
            raise TypeError(
                "Propagation global resource does not carry a Stage-5 PropagationRecipe payload."
            )
        if recipe.resource_id != resource_id:
            raise ValueError("Propagation global resource id disagrees with its recipe identity.")
        try:
            source = propagation_sources[resource_id]
        except KeyError as exc:
            raise ValueError(
                f"No exact external tensor source was supplied for propagation resource "
                f"{resource_id!r}."
            ) from exc
        if not isinstance(source, ExternalTensorSource):
            raise TypeError(
                f"Propagation source for {resource_id!r} must be ExternalTensorSource."
            )
        return PropagationResourceBinding(resource_id, recipe, source)

    raise ValueError(f"Unsupported global tensor resource kind {resource.kind!r}.")


def _common_request_kwargs(
    plan: PhysicalExecutionPlan,
    operation: MaterializePrimitiveOperation,
) -> dict[str, object]:
    binding = plan.binding(operation.value_id)
    return {
        "step_index": operation.step_index,
        "node_id": operation.node_id,
        "value_id": operation.value_id,
        "resource_id": operation.resource_id,
        "placement": operation.placement,
        "output_buffer_id": operation.output_buffer_id,
        "expected_shape": binding.shape,
        "scalar_type": binding.scalar_type,
    }


def _kernel_request(
    plan: PhysicalExecutionPlan,
    operation: MaterializePrimitiveOperation,
    binding: KernelResourceBinding,
    time_context: OperatorTimeContext,
) -> KernelMaterializationRequest:
    arity, raw = _scoped_placement(operation.placement)
    operator_ids = _kernel_operator_ids(raw, arity)
    return KernelMaterializationRequest(
        **_common_request_kwargs(plan, operation),
        provider_id=binding.provider_id,
        recipe=binding.recipe,
        operator_ids=operator_ids,
        operator_times=tuple(time_context.time_for(arity, op) for op in operator_ids),
    )


def _spin_request(
    plan: PhysicalExecutionPlan,
    operation: MaterializePrimitiveOperation,
    binding: SpinResourceBinding,
    time_context: OperatorTimeContext,
) -> SpinMaterializationRequest:
    arity, raw = _scoped_placement(operation.placement)
    operator_id = _spin_operator_id(raw, arity)
    return SpinMaterializationRequest(
        **_common_request_kwargs(plan, operation),
        tensor_kind=binding.tensor.kind,
        matrix=binding.matrix,
        operator_id=operator_id,
        operator_time=(
            time_context.time_for(arity, operator_id) if operator_id is not None else None
        ),
    )


def _propagation_request(
    plan: PhysicalExecutionPlan,
    operation: MaterializePrimitiveOperation,
    binding: PropagationResourceBinding,
    time_context: OperatorTimeContext,
) -> PropagationMaterializationRequest:
    arity, raw = _scoped_placement(operation.placement)
    source_op, insertion_ops, sink_op = _path_operator_ids(raw, arity)
    if len(insertion_ops) != len(binding.recipe.insertion_specs):
        raise ValueError(
            "Propagation placement insertion count disagrees with generalized recipe."
        )
    return PropagationMaterializationRequest(
        **_common_request_kwargs(plan, operation),
        source=binding.source,
        recipe=binding.recipe,
        source_operator_id=source_op,
        source_time=time_context.time_for(arity, source_op),
        insertion_operator_ids=insertion_ops,
        insertion_times=tuple(time_context.time_for(arity, op) for op in insertion_ops),
        sink_operator_id=sink_op,
        sink_time=time_context.time_for(arity, sink_op),
    )


def _bind_materialization_requests(
    physical_plan: PhysicalExecutionPlan,
    primitive_operations: Sequence[MaterializePrimitiveOperation],
    bindings: Mapping[str, ResourceBinding],
    time_context: OperatorTimeContext,
) -> dict[str, MaterializationRequest]:
    materializations: dict[str, MaterializationRequest] = {}
    for operation in primitive_operations:
        binding = bindings[operation.resource_id]
        if operation.kind == "kernel":
            if not isinstance(binding, KernelResourceBinding):
                raise TypeError("Kernel primitive resolved to a non-kernel binding.")
            request = _kernel_request(
                physical_plan,
                operation,
                binding,
                time_context,
            )
        elif operation.kind == "spin":
            if not isinstance(binding, SpinResourceBinding):
                raise TypeError("Spin primitive resolved to a non-spin binding.")
            request = _spin_request(
                physical_plan,
                operation,
                binding,
                time_context,
            )
        elif operation.kind == "propagation":
            if not isinstance(binding, PropagationResourceBinding):
                raise TypeError(
                    "Propagation primitive resolved to a non-propagation binding."
                )
            request = _propagation_request(
                physical_plan,
                operation,
                binding,
                time_context,
            )
        else:
            raise ValueError(f"Unsupported primitive kind {operation.kind!r}.")
        if operation.value_id in materializations:
            raise ValueError(f"Duplicate primitive value id {operation.value_id!r}.")
        materializations[operation.value_id] = request
    return materializations


def prepare_resource_binding_template(
    physical_plan: PhysicalExecutionPlan,
    *,
    propagation_sources: Mapping[str, ExternalTensorSource],
    kernel_provider_id: str = "distillation",
    kernel_provider_ids: Mapping[str, str] | None = None,
    spin_basis: str = "DR",
) -> ResourceBindingTemplate:
    """Resolve configuration-invariant resource bindings exactly once."""

    physical_plan.validate()
    if not kernel_provider_id:
        raise ValueError("kernel_provider_id must be non-empty.")
    if spin_basis not in ("DR", "DP"):
        raise ValueError("spin_basis must be 'DR' or 'DP'.")
    provider_overrides = dict(kernel_provider_ids or {})
    resources = _global_resources(physical_plan)
    primitive_operations = _used_primitive_operations(physical_plan)
    used_resource_ids = tuple(
        dict.fromkeys(operation.resource_id for operation in primitive_operations)
    )

    bindings: dict[str, ResourceBinding] = {}
    for resource_id in used_resource_ids:
        try:
            resource = resources[resource_id]
        except KeyError as exc:
            raise ValueError(
                f"Physical primitive references missing global resource {resource_id!r}."
            ) from exc
        bindings[resource_id] = _bind_resource(
            resource,
            propagation_sources=propagation_sources,
            kernel_provider_id=kernel_provider_id,
            kernel_provider_ids=provider_overrides,
            spin_basis=spin_basis,
        )

    return ResourceBindingTemplate(
        source_plan=physical_plan,
        bindings=bindings,
        primitive_operations=primitive_operations,
        statistics=ResourceBindingStatistics(
            used_resources=len(bindings),
            kernel_resources=sum(
                isinstance(item, KernelResourceBinding) for item in bindings.values()
            ),
            spin_resources=sum(
                isinstance(item, SpinResourceBinding) for item in bindings.values()
            ),
            propagation_resources=sum(
                isinstance(item, PropagationResourceBinding) for item in bindings.values()
            ),
            primitive_materializations=len(primitive_operations),
        ),
        spin_basis=spin_basis,
    )


def build_resource_binding_plan(
    physical_plan: PhysicalExecutionPlan,
    *,
    operator_times: OperatorTimeContext | Mapping[int, Sequence[int]],
    propagation_sources: Mapping[str, ExternalTensorSource],
    kernel_provider_id: str = "distillation",
    kernel_provider_ids: Mapping[str, str] | None = None,
    spin_basis: str = "DR",
) -> ResourceBindingPlan:
    """Bind every scheduled primitive resource to an exact runtime source."""

    template = prepare_resource_binding_template(
        physical_plan,
        propagation_sources=propagation_sources,
        kernel_provider_id=kernel_provider_id,
        kernel_provider_ids=kernel_provider_ids,
        spin_basis=spin_basis,
    )
    return template.bind(operator_times)
