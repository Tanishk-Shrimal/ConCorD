"""Static analysis for the backend-independent logical execution DAG.

This layer derives exact per-value tensor metadata and dependency information
from ``LogicalExecutionDAG`` without making any execution-order or hardware
choices. Shapes describe one logical value at its exact operator placement;
time batching, device residency, storage layout, and resource-file dimensions
belong to later physical planning/runtime layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral
from typing import Literal, Mapping

from concord_lqcd.compiler._validation import _has_static_validation
from concord_lqcd.compiler.contraction_order import ContractionSizeModel, IndexSpace
from concord_lqcd.compiler.logical_execution import (
    LogicalContraction,
    LogicalCorrelatorSum,
    LogicalExecutionDAG,
    LogicalPrimitiveEvaluation,
    LogicalValueUse,
    LogicalWickTermProduct,
)


LogicalValueKind = Literal["primitive", "contraction", "wick_term", "correlator_sum"]


@dataclass(frozen=True)
class StaticNumericalModel:
    """Backend-neutral scalar storage model used for static byte estimates.

    ``mode_size`` and ``spin_size`` are inherited from the contraction-order
    size model so path optimization and static shape analysis use exactly the
    same index dimensions. ``scalar_type`` is descriptive only; no NumPy or
    PyTorch dtype enters compiler IR.
    """

    mode_size: int
    spin_size: int = 4
    scalar_type: str = "complex128"
    element_bytes: int = 16

    def __post_init__(self) -> None:
        dimensions = ContractionSizeModel(self.mode_size, self.spin_size)
        object.__setattr__(self, "mode_size", dimensions.mode_size)
        object.__setattr__(self, "spin_size", dimensions.spin_size)
        if not isinstance(self.scalar_type, str):
            raise TypeError("scalar_type must be a non-empty string.")
        if not self.scalar_type:
            raise ValueError("scalar_type must be non-empty.")
        if isinstance(self.element_bytes, bool) or not isinstance(self.element_bytes, Integral):
            raise TypeError("element_bytes must be an integer, not a boolean or rounded value.")
        if self.element_bytes <= 0:
            raise ValueError("element_bytes must be positive.")
        object.__setattr__(self, "element_bytes", int(self.element_bytes))

    def index_size(self, space: IndexSpace) -> int:
        if space == "mode":
            return self.mode_size
        if space == "spin":
            return self.spin_size
        raise ValueError(f"Unsupported index space {space!r}.")


@dataclass(frozen=True)
class StaticValueDescriptor:
    """Exact logical tensor metadata for one DAG-produced value.

    ``axis_labels`` are the public interface ports/indices of the logical value,
    in numerical axis order. ``axis_spaces`` records the semantic index space of
    each axis independently of its numerical dimension. This distinction is
    essential when two spaces happen to have the same size (for example
    ``mode_size == spin_size == 4``). Scalars expose no axes and contain one
    numerical element.
    """

    value_id: str
    kind: LogicalValueKind
    axis_labels: tuple[str, ...]
    axis_spaces: tuple[IndexSpace, ...]
    shape: tuple[int, ...]
    scalar_type: str
    element_bytes: int
    element_count: int
    size_bytes: int

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_scalar(self) -> bool:
        return not self.shape

    def size_for_axis(self, label: str) -> int:
        matches = [
            size for axis, size in zip(self.axis_labels, self.shape)
            if axis == label
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Value {self.value_id!r} does not expose exactly one axis {label!r}."
            )
        return matches[0]

    def space_for_axis(self, label: str) -> IndexSpace:
        matches = [
            space for axis, space in zip(self.axis_labels, self.axis_spaces)
            if axis == label
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Value {self.value_id!r} does not expose exactly one axis {label!r}."
            )
        return matches[0]


@dataclass(frozen=True)
class StaticDependencyRecord:
    """Static producer/consumer information for one logical DAG node.

    Dependency and consumer tuples retain repeated occurrences. This preserves
    the distinction between one consumer using a value once and using the same
    value multiple times (for example a Wick term containing ``L * L``).
    """

    node_id: str
    dependencies: tuple[str, ...]
    consumers: tuple[str, ...]
    depth_from_sources: int
    distance_to_output: int | None

    @property
    def dependency_occurrences(self) -> int:
        return len(self.dependencies)

    @property
    def consumer_occurrences(self) -> int:
        return len(self.consumers)

    @property
    def unique_consumers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.consumers))

    @property
    def fanout(self) -> int:
        return len(self.unique_consumers)

    @property
    def is_live(self) -> bool:
        return self.distance_to_output is not None


@dataclass(frozen=True)
class StaticExecutionStatistics:
    values: int
    scalar_values: int
    total_materialized_bytes: int
    largest_value_bytes: int
    max_rank: int
    dependency_occurrences: int
    unique_dependency_edges: int
    max_fanout: int
    critical_path_nodes: int
    dead_nodes: int


@dataclass(frozen=True)
class StaticExecutionAnalysis:
    """Backend-independent static analysis result for a logical execution DAG."""

    source_dag: LogicalExecutionDAG
    numerical_model: StaticNumericalModel
    values: Mapping[str, StaticValueDescriptor]
    dependencies: Mapping[str, StaticDependencyRecord]
    topological_order: tuple[str, ...]
    roots: tuple[str, ...]
    output_nodes: tuple[str, ...]
    dead_nodes: tuple[str, ...]
    statistics: StaticExecutionStatistics

    def value(self, value_id: str) -> StaticValueDescriptor:
        return self.values[value_id]

    def dependency(self, node_id: str) -> StaticDependencyRecord:
        return self.dependencies[node_id]

    def validate(self) -> None:
        """Check derived metadata against the attached logical source, without data I/O."""

        if _has_static_validation(self):
            return
        expected = analyze_static_execution(
            self.source_dag,
            scalar_type=self.numerical_model.scalar_type,
            element_bytes=self.numerical_model.element_bytes,
        )
        for descriptor in self.values.values():
            dimensions = descriptor.shape + (
                descriptor.element_bytes, descriptor.element_count, descriptor.size_bytes,
            )
            if any(isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
                   for size in dimensions):
                raise ValueError("Static tensor dimensions and byte counts must be positive integers.")
        for name in ("numerical_model", "values", "dependencies", "topological_order",
                     "roots", "output_nodes", "dead_nodes", "statistics"):
            if getattr(self, name) != getattr(expected, name):
                raise ValueError(f"Static analysis {name} disagrees with its logical source.")


def _primitive_port_space(kind: str, port: str) -> IndexSpace:
    if kind == "kernel":
        return "mode"
    if kind == "spin":
        return "spin"
    if kind == "propagation":
        if port in ("sink_mode", "source_mode"):
            return "mode"
        if port in ("sink_spin", "source_spin"):
            return "spin"
        raise ValueError(f"Unknown propagation tensor port {port!r}.")
    raise ValueError(f"Unsupported primitive tensor kind {kind!r}.")


def _descriptor(
    *,
    value_id: str,
    kind: LogicalValueKind,
    axis_labels: tuple[str, ...],
    axis_spaces: tuple[IndexSpace, ...],
    shape: tuple[int, ...],
    model: StaticNumericalModel,
) -> StaticValueDescriptor:
    if len(axis_labels) != len(axis_spaces) or len(axis_labels) != len(shape):
        raise ValueError(
            f"Value {value_id!r} has inconsistent axis-label, index-space, and shape ranks."
        )
    if len(axis_labels) != len(set(axis_labels)):
        raise ValueError(f"Value {value_id!r} exposes duplicate axis labels.")
    if any(size <= 0 for size in shape):
        raise ValueError(f"Value {value_id!r} has a non-positive tensor dimension.")
    for space, size in zip(axis_spaces, shape):
        expected_size = model.index_size(space)
        if size != expected_size:
            raise ValueError(
                f"Value {value_id!r} axis in {space!r} space has dimension {size}, "
                f"expected {expected_size}."
            )
    elements = prod(shape) if shape else 1
    return StaticValueDescriptor(
        value_id=value_id,
        kind=kind,
        axis_labels=axis_labels,
        axis_spaces=axis_spaces,
        shape=shape,
        scalar_type=model.scalar_type,
        element_bytes=model.element_bytes,
        element_count=elements,
        size_bytes=elements * model.element_bytes,
    )


def _primitive_descriptor(
    node: LogicalPrimitiveEvaluation,
    model: StaticNumericalModel,
) -> StaticValueDescriptor:
    axis_spaces = tuple(
        _primitive_port_space(node.kind, port)
        for port in node.output_ports
    )
    shape = tuple(model.index_size(space) for space in axis_spaces)
    return _descriptor(
        value_id=node.node_id,
        kind="primitive",
        axis_labels=node.output_ports,
        axis_spaces=axis_spaces,
        shape=shape,
        model=model,
    )


def _index_metadata_from_uses(
    uses: tuple[LogicalValueUse, ...],
    values: Mapping[str, StaticValueDescriptor],
) -> dict[str, tuple[IndexSpace, int]]:
    metadata: dict[str, tuple[IndexSpace, int]] = {}
    for use in uses:
        try:
            descriptor = values[use.value_id]
        except KeyError as exc:
            raise ValueError(
                f"Static analysis references unavailable value {use.value_id!r}."
            ) from exc

        expected_ports = descriptor.axis_labels
        actual_ports = tuple(binding.port for binding in use.bindings)
        if actual_ports != expected_ports:
            raise ValueError(
                f"Static use of value {use.value_id!r} has ports {actual_ports!r}, "
                f"expected {expected_ports!r}."
            )

        for binding, axis_space, axis_size in zip(
            use.bindings,
            descriptor.axis_spaces,
            descriptor.shape,
        ):
            previous = metadata.get(binding.index)
            if previous is not None:
                previous_space, previous_size = previous
                if previous_space != axis_space:
                    raise ValueError(
                        f"Logical index {binding.index!r} joins incompatible index spaces "
                        f"{previous_space!r} and {axis_space!r}."
                    )
                if previous_size != axis_size:
                    raise ValueError(
                        f"Logical index {binding.index!r} joins incompatible dimensions "
                        f"{previous_size} and {axis_size}."
                    )
            metadata[binding.index] = (axis_space, axis_size)
    return metadata


def _contraction_descriptor(
    node: LogicalContraction,
    values: Mapping[str, StaticValueDescriptor],
    model: StaticNumericalModel,
) -> StaticValueDescriptor:
    index_metadata = _index_metadata_from_uses(node.inputs, values)
    missing = set(node.output_indices) - set(index_metadata)
    if missing:
        raise ValueError(
            f"Contraction {node.node_id!r} outputs unknown indices {sorted(missing)}."
        )
    axis_spaces = tuple(index_metadata[index][0] for index in node.output_indices)
    shape = tuple(index_metadata[index][1] for index in node.output_indices)
    descriptor = _descriptor(
        value_id=node.node_id,
        kind="contraction",
        axis_labels=node.output_indices,
        axis_spaces=axis_spaces,
        shape=shape,
        model=model,
    )
    if descriptor.element_count != node.output_elements:
        raise ValueError(
            f"Contraction {node.node_id!r} static output size "
            f"{descriptor.element_count} disagrees with optimized-path metadata "
            f"{node.output_elements}."
        )
    return descriptor


def _wick_term_descriptor(
    node: LogicalWickTermProduct,
    values: Mapping[str, StaticValueDescriptor],
    model: StaticNumericalModel,
) -> StaticValueDescriptor:
    index_metadata = _index_metadata_from_uses(node.inputs, values)
    if set(index_metadata) != set(node.output_indices):
        raise ValueError(
            f"Wick term {node.node_id!r} static input interface does not match its "
            "declared outputs."
        )
    axis_spaces = tuple(index_metadata[index][0] for index in node.output_indices)
    shape = tuple(index_metadata[index][1] for index in node.output_indices)
    return _descriptor(
        value_id=node.node_id,
        kind="wick_term",
        axis_labels=node.output_indices,
        axis_spaces=axis_spaces,
        shape=shape,
        model=model,
    )


def _correlator_descriptor(
    node: LogicalCorrelatorSum,
    values: Mapping[str, StaticValueDescriptor],
    model: StaticNumericalModel,
) -> StaticValueDescriptor:
    if not node.term_ids:
        if node.output_indices:
            raise ValueError(
                f"Empty correlator sum {node.node_id!r} cannot infer non-scalar output "
                "dimensions without contributing terms."
            )
        return _descriptor(
            value_id=node.node_id,
            kind="correlator_sum",
            axis_labels=(),
            axis_spaces=(),
            shape=(),
            model=model,
        )

    first = values[node.term_ids[0]]
    if first.axis_labels != node.output_indices:
        raise ValueError(
            f"Correlator {node.node_id!r} output interface disagrees with its first term."
        )
    for term_id in node.term_ids[1:]:
        term = values[term_id]
        if (
            term.axis_labels != first.axis_labels
            or term.axis_spaces != first.axis_spaces
            or term.shape != first.shape
        ):
            raise ValueError(
                f"Correlator {node.node_id!r} sums terms with incompatible shapes/index spaces."
            )

    return _descriptor(
        value_id=node.node_id,
        kind="correlator_sum",
        axis_labels=node.output_indices,
        axis_spaces=first.axis_spaces,
        shape=first.shape,
        model=model,
    )


def _infer_value_descriptors(
    dag: LogicalExecutionDAG,
    model: StaticNumericalModel,
    topological_order: tuple[str, ...],
) -> dict[str, StaticValueDescriptor]:
    values: dict[str, StaticValueDescriptor] = {}
    for node_id in topological_order:
        node = dag.node(node_id)
        if isinstance(node, LogicalPrimitiveEvaluation):
            values[node_id] = _primitive_descriptor(node, model)
        elif isinstance(node, LogicalContraction):
            values[node_id] = _contraction_descriptor(node, values, model)
        elif isinstance(node, LogicalWickTermProduct):
            values[node_id] = _wick_term_descriptor(node, values, model)
        elif isinstance(node, LogicalCorrelatorSum):
            values[node_id] = _correlator_descriptor(node, values, model)
        else:
            raise TypeError(f"Unsupported logical execution node {type(node).__name__}.")
    return values


def _dependency_records(
    dag: LogicalExecutionDAG,
    topological_order: tuple[str, ...],
) -> tuple[dict[str, StaticDependencyRecord], tuple[str, ...], tuple[str, ...]]:
    dependencies = {
        node_id: tuple(dag.dependencies(node_id))
        for node_id in topological_order
    }
    consumers: dict[str, list[str]] = {node_id: [] for node_id in topological_order}
    for node_id in topological_order:
        for dependency in dependencies[node_id]:
            if dependency not in consumers:
                raise ValueError(
                    f"Logical node {node_id!r} references unknown dependency {dependency!r}."
                )
            consumers[dependency].append(node_id)

    depth: dict[str, int] = {}
    for node_id in topological_order:
        deps = dependencies[node_id]
        depth[node_id] = 0 if not deps else 1 + max(depth[dep] for dep in deps)

    output_nodes = tuple(dict.fromkeys(dag.outputs.values()))
    output_set = set(output_nodes)
    distance: dict[str, int | None] = {node_id: None for node_id in topological_order}
    for node_id in reversed(topological_order):
        if node_id in output_set:
            distance[node_id] = 0
            continue
        live_consumer_distances = [
            distance[consumer]
            for consumer in dict.fromkeys(consumers[node_id])
            if distance[consumer] is not None
        ]
        if live_consumer_distances:
            distance[node_id] = 1 + max(live_consumer_distances)

    records = {
        node_id: StaticDependencyRecord(
            node_id=node_id,
            dependencies=dependencies[node_id],
            consumers=tuple(consumers[node_id]),
            depth_from_sources=depth[node_id],
            distance_to_output=distance[node_id],
        )
        for node_id in topological_order
    }
    roots = tuple(node_id for node_id in topological_order if not dependencies[node_id])
    dead = tuple(node_id for node_id in topological_order if distance[node_id] is None)
    return records, roots, dead


def analyze_static_execution(
    dag: LogicalExecutionDAG,
    *,
    scalar_type: str = "complex128",
    element_bytes: int = 16,
) -> StaticExecutionAnalysis:
    """Derive exact logical tensor sizes, spaces, and dependency metadata.

    The mode/spin dimensions are inherited from the already-selected contraction
    order. ``scalar_type``/``element_bytes`` remain explicit analysis inputs so
    compiler analysis does not depend on a PyTorch/NumPy dtype object.
    """

    dag.validate()
    size_model = dag.source_dag.source_plan.size_model
    model = StaticNumericalModel(
        mode_size=size_model.mode_size,
        spin_size=size_model.spin_size,
        scalar_type=scalar_type,
        element_bytes=element_bytes,
    )
    order = dag.topological_order()
    values = _infer_value_descriptors(dag, model, order)
    dependencies, roots, dead = _dependency_records(dag, order)
    output_nodes = tuple(dict.fromkeys(dag.outputs.values()))

    unique_edges = {
        (dependency, node_id)
        for node_id in order
        for dependency in dag.dependencies(node_id)
    }
    critical_path_nodes = (
        max((dependencies[node_id].depth_from_sources for node_id in output_nodes), default=-1)
        + 1
    )

    analysis = StaticExecutionAnalysis(
        source_dag=dag,
        numerical_model=model,
        values=dict(values),
        dependencies=dict(dependencies),
        topological_order=order,
        roots=roots,
        output_nodes=output_nodes,
        dead_nodes=dead,
        statistics=StaticExecutionStatistics(
            values=len(values),
            scalar_values=sum(value.is_scalar for value in values.values()),
            total_materialized_bytes=sum(value.size_bytes for value in values.values()),
            largest_value_bytes=max((value.size_bytes for value in values.values()), default=0),
            max_rank=max((value.rank for value in values.values()), default=0),
            dependency_occurrences=sum(len(dag.dependencies(node_id)) for node_id in order),
            unique_dependency_edges=len(unique_edges),
            max_fanout=max((record.fanout for record in dependencies.values()), default=0),
            critical_path_nodes=critical_path_nodes,
            dead_nodes=len(dead),
        ),
    )
    return analysis
