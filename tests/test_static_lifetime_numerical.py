"""Independent shape, dependency, and dense-value lifetime checks."""

from collections import Counter
from dataclasses import replace
from functools import cache

import numpy as np
import pytest

from concord_lqcd.backend.runtime import RuntimeCompilationConfig
from concord_lqcd.compiler import (
    ComponentComputation, ComponentTensorUse, ComponentUse, ContractionSizeModel,
    GlobalCSEStatistics, GlobalComputationDAG, GlobalCorrelatorUse,
    GlobalTensorResource, GlobalWickTermUse, IntermediateComputation,
    IntermediateInputUse, LogicalContraction, LogicalCorrelatorSum,
    LogicalPrimitiveEvaluation, LogicalWickTermProduct, StaticNumericalModel,
    TensorLegBinding, analyze_static_execution, build_logical_execution_dag,
    build_logical_schedule, build_partial_intermediate_cse,
    optimize_global_contraction_order,
)


def _bindings(ports, indices):
    return tuple(TensorLegBinding(port, index) for port, index in zip(ports, indices, strict=True))


def _source(mode_size=2):
    outputs = ("mode_left", "mode_right", "spin_left", "spin_right")
    mixed = ComponentComputation("mixed", "manual_mixed", (
        ComponentTensorUse("kernel", "kernel", "K", ("kernel_ops", (0,)),
                           _bindings(("left", "right"), ("mode_left", "mode_join"))),
        ComponentTensorUse("propagation", "propagation", "P", ("prop_path_ops", (0, 1)),
                           _bindings(("sink_mode", "sink_spin", "source_mode", "source_spin"),
                                     ("mode_join", "spin_join", "mode_right", "spin_left"))),
        ComponentTensorUse("spin", "spin", "S", ("spin_op", 0),
                           _bindings(("left", "right"), ("spin_join", "spin_right"))),
    ), outputs)
    loop = ComponentComputation("loop", "manual_spin_loop", (
        ComponentTensorUse("trace", "spin", "S", ("spin_op", 0),
                           _bindings(("left", "right"), ("trace_index", "trace_index"))),
    ), ())
    mixed_use = ComponentUse("mixed", _bindings(outputs, outputs))
    loop_use = ComponentUse("loop")
    global_dag = GlobalComputationDAG(
        resources={resource: GlobalTensorResource(resource, kind, object())
                   for resource, kind in (("K", "kernel"), ("P", "propagation"), ("S", "spin"))},
        components={"mixed": mixed, "loop": loop},
        correlators=(
            GlobalCorrelatorUse("matrix", (GlobalWickTermUse(0, 1 + 0.5j,
                                                           (mixed_use, loop_use, loop_use)),)),
            GlobalCorrelatorUse("second", (GlobalWickTermUse(0, -0.3j, (mixed_use, loop_use)),)),
            GlobalCorrelatorUse("empty", ()),
            GlobalCorrelatorUse("constant", (GlobalWickTermUse(0, 2 - 3j, ()),)),
        ),
        statistics=GlobalCSEStatistics(9, 3, 5, 2),
    )
    order = optimize_global_contraction_order(global_dag, ContractionSizeModel(mode_size), strategy="greedy")
    partial = build_partial_intermediate_cse(order)
    exemplar = next(value for value in partial.primitive_values.values() if value.kind == "kernel")
    dead = replace(exemplar, value_id="dead_kernel")
    dead_trace = IntermediateComputation(
        "dead_trace", "manual_dead_trace",
        (IntermediateInputUse(dead.value_id, _bindings(dead.ports, ("dead_index", "dead_index"))),),
        ("dead_index",), (), mode_size, 1,
    )
    partial = replace(partial,
                      primitive_values={**partial.primitive_values, dead.value_id: dead},
                      intermediates={**partial.intermediates, dead_trace.intermediate_id: dead_trace})
    return build_logical_execution_dag(partial)


def _arrays(mode_size, dtype):
    generator = np.random.default_rng(4329)
    return {resource: (generator.normal(size=shape) + 1j * generator.normal(size=shape)).astype(dtype)
            for resource, shape in (("K", (mode_size, mode_size)),
                                    ("P", (mode_size, 4, mode_size, 4)), ("S", (4, 4)))}


def _direct(arrays):
    mixed = np.einsum("ab,bscd,se->acde", arrays["K"], arrays["P"], arrays["S"], optimize=False)
    loop = np.trace(arrays["S"])
    return {"matrix": (1 + 0.5j) * mixed * loop**2,
            "second": -0.3j * mixed * loop, "empty": np.asarray(0j), "constant": np.asarray(2 - 3j)}


def _contract(uses, values, ports, outputs):
    labels = sorted({binding.index for use in uses for binding in use.bindings})
    numbers = {label: number for number, label in enumerate(labels)}
    arguments = []
    for use in uses:
        bindings = {binding.port: binding.index for binding in use.bindings}
        arguments.extend((values[use.value_id], [numbers[bindings[port]] for port in ports[use.value_id]]))
    arguments.append([numbers[label] for label in outputs])
    return np.einsum(*arguments, optimize=False) if uses else np.asarray(1 + 0j)


def _replay(logical, arrays, dtype):
    values, ports, spaces = {}, {}, {}
    for node_id in logical.topological_order():
        node = logical.node(node_id)
        if isinstance(node, LogicalPrimitiveEvaluation):
            value = arrays[node.resource_id]
            ports[node_id] = node.output_ports
            spaces[node_id] = tuple("mode" if node.kind == "kernel" or port.endswith("_mode") else "spin"
                                    for port in node.output_ports)
        elif isinstance(node, (LogicalContraction, LogicalWickTermProduct)):
            value = _contract(node.inputs, values, ports, node.output_indices)
            if isinstance(node, LogicalWickTermProduct):
                value = node.coefficient * value
            metadata = {binding.index: space for use in node.inputs
                        for binding, space in zip(use.bindings, spaces[use.value_id], strict=True)}
            ports[node_id] = node.output_indices
            spaces[node_id] = tuple(metadata[index] for index in node.output_indices)
        elif isinstance(node, LogicalCorrelatorSum):
            value = sum((values[term] for term in node.term_ids), np.asarray(0j, dtype=dtype))
            ports[node_id] = node.output_indices
            spaces[node_id] = spaces[node.term_ids[0]] if node.term_ids else ()
        else:
            raise AssertionError(type(node))
        values[node_id] = np.asarray(value, dtype=dtype)
    return values, ports, spaces


def _reachable(logical):
    live = set()
    pending = list(logical.outputs.values())
    while pending:
        node_id = pending.pop()
        if node_id not in live:
            live.add(node_id)
            pending.extend(logical.node(node_id).dependencies)
    return live


@pytest.mark.parametrize("mode_size", (2, 4))
@pytest.mark.parametrize("dtype", ("complex64", "complex128"))
@pytest.mark.parametrize("strategy", ("topological", "critical_path", "memory_aware"))
@pytest.mark.parametrize("include_dead", (False, True))
def test_numpy_shapes_and_independent_live_tensor_ledger(mode_size, dtype, strategy, include_dead):
    arrays = _arrays(mode_size, dtype)
    reference = _direct(arrays)
    logical = _source(mode_size)
    values, ports, spaces = _replay(logical, arrays, dtype)
    analysis = analyze_static_execution(logical, scalar_type=dtype, element_bytes=np.dtype(dtype).itemsize)
    schedule = build_logical_schedule(analysis, strategy=strategy, include_dead=include_dead)

    for name, output_id in logical.outputs.items():
        np.testing.assert_allclose(values[output_id], reference[name],
                                   rtol=1e-5 if dtype == "complex64" else 1e-13, atol=1e-6)
    assert np.max(np.abs(reference["matrix"])) > 0.1
    depths = {}
    consumer_map = {identity: tuple(consumer for consumer in logical.topological_order()
                                   for dependency in logical.node(consumer).dependencies if dependency == identity)
                    for identity in logical.node_ids}

    @cache
    def distance_to_output(identity):
        if identity in logical.outputs.values():
            return 0
        child_distances = [distance_to_output(consumer) for consumer in consumer_map[identity]]
        distances = [distance for distance in child_distances if distance is not None]
        return 1 + max(distances) if distances else None

    for node_id in logical.topological_order():
        descriptor = analysis.value(node_id)
        node = logical.node(node_id)
        assert descriptor.axis_labels == ports[node_id]
        assert descriptor.axis_spaces == spaces[node_id]
        assert descriptor.shape == values[node_id].shape
        assert descriptor.element_count == values[node_id].size
        assert descriptor.size_bytes == values[node_id].nbytes
        dependencies = node.dependencies
        consumers = consumer_map[node_id]
        depths[node_id] = 1 + max((depths[dependency] for dependency in dependencies), default=-1)
        assert analysis.dependency(node_id).dependencies == dependencies
        assert analysis.dependency(node_id).consumers == consumers
        assert analysis.dependency(node_id).depth_from_sources == depths[node_id]
        assert analysis.dependency(node_id).distance_to_output == distance_to_output(node_id)
        assert analysis.dependency(node_id).fanout == len(set(consumers))
    live = _reachable(logical)
    assert set(analysis.dead_nodes) == set(values) - live == {"dead_kernel", "dead_trace"}
    assert set(schedule.order) == (set(values) if include_dead else live)
    assert analysis.statistics.total_materialized_bytes == sum(value.nbytes for value in values.values())
    assert analysis.statistics.values == len(values)
    assert analysis.statistics.scalar_values == sum(value.ndim == 0 for value in values.values())
    assert analysis.statistics.largest_value_bytes == max(value.nbytes for value in values.values())
    assert analysis.statistics.max_rank == max(value.ndim for value in values.values())
    edges = tuple((dependency, identity) for identity in logical.node_ids
                  for dependency in logical.node(identity).dependencies)
    assert analysis.statistics.dependency_occurrences == len(edges)
    assert analysis.statistics.unique_dependency_edges == len(set(edges))
    assert analysis.statistics.max_fanout == max(len(set(consumers)) for consumers in consumer_map.values())
    assert analysis.statistics.dead_nodes == 2
    assert analysis.statistics.critical_path_nodes == 1 + max(depths[output] for output in logical.outputs.values())

    resident, produced = set(), set()
    requested = set(logical.outputs.values())
    peak = 0
    for number, step in enumerate(schedule.steps):
        node = logical.node(step.node_id)
        ready = {candidate for candidate in schedule.order if candidate not in produced
                 and set(logical.node(candidate).dependencies) <= produced}
        assert step.ready_width == len(ready) and step.node_id in ready
        assert set(node.dependencies) <= resident
        before = sum(values[identity].nbytes for identity in resident)
        assert step.live_bytes_before == before
        assert step.output_bytes == values[step.node_id].nbytes
        peak = max(peak, before + values[step.node_id].nbytes)
        assert step.peak_live_bytes_during_step == before + values[step.node_id].nbytes
        future = Counter(dependency for future_id in schedule.order[number + 1:]
                         for dependency in logical.node(future_id).dependencies)
        released_inputs = tuple(dependency for dependency in dict.fromkeys(node.dependencies)
                                if dependency not in requested and not future[dependency])
        released = released_inputs + ((step.node_id,) if step.node_id not in requested and not future[step.node_id] else ())
        assert step.released_values == released
        assert step.releasable_input_bytes == sum(values[identity].nbytes for identity in released_inputs)
        resident.add(step.node_id)
        resident.difference_update(released)
        produced.add(step.node_id)
        assert step.live_bytes_after == sum(values[identity].nbytes for identity in resident)
        lifetime = schedule.lifetime(step.node_id)
        consumer_steps = tuple(position for position, consumer_id in enumerate(schedule.order)
                               for dependency in logical.node(consumer_id).dependencies if dependency == step.node_id)
        assert lifetime.consumer_steps == consumer_steps
        assert lifetime.produced_at == number
        assert lifetime.release_after_step == (None if step.node_id in requested else max(consumer_steps, default=number))
    assert resident == requested
    assert schedule.statistics.peak_live_bytes == peak
    assert schedule.statistics.final_live_bytes == sum(values[identity].nbytes for identity in requested)
    assert schedule.statistics.scheduled_nodes == len(schedule.order)
    assert schedule.statistics.skipped_dead_nodes == (0 if include_dead else 2)
    assert schedule.statistics.max_ready_width == max(step.ready_width for step in schedule.steps)
    assert schedule.statistics.released_values == len(schedule.order) - len(requested)
    assert schedule.statistics.total_estimated_contraction_flops == sum(
        logical.node(identity).estimated_flops for identity in schedule.order
        if isinstance(logical.node(identity), LogicalContraction)
    )
    repeated = next(term for term in logical.terms if term.correlator_id == "matrix")
    assert repeated.inputs[1].value_id == repeated.inputs[2].value_id
    assert len(schedule.lifetime(repeated.inputs[1].value_id).consumer_steps) >= 3


@pytest.mark.parametrize("model_type", (StaticNumericalModel, RuntimeCompilationConfig))
@pytest.mark.parametrize("field", ("mode_size", "spin_size", "element_bytes"))
@pytest.mark.parametrize("value", (True, np.bool_(True), 3.0, 2.5, float("nan"), float("inf"), "3"))
def test_early_size_and_byte_models_reject_non_integer_inputs(model_type, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        model_type(**{"mode_size": 2, field: value})


@pytest.mark.parametrize("model_type", (StaticNumericalModel, RuntimeCompilationConfig))
def test_integer_scalar_dimensions_and_byte_sizes_are_normalized(model_type):
    model = model_type(mode_size=np.int64(2), spin_size=np.int32(4), element_bytes=np.int32(16))
    assert type(model.mode_size) is int and model.mode_size == 2
    assert type(model.spin_size) is int and model.spin_size == 4
    assert type(model.element_bytes) is int and model.element_bytes == 16


@pytest.mark.parametrize("field", ("shape", "axis_spaces", "size_bytes", "value_id", "float_shape"))
def test_scheduler_rejects_static_descriptors_altered_from_the_source(field):
    analysis = analyze_static_execution(_source(mode_size=4))
    identity = next(identity for identity, value in analysis.values.items() if value.rank > 0)
    descriptor = analysis.values[identity]
    changes = {"shape": tuple(size + 1 for size in descriptor.shape),
               "axis_spaces": tuple("mode" if space == "spin" else "spin" for space in descriptor.axis_spaces),
               "size_bytes": descriptor.size_bytes + 1, "value_id": "wrong_identity",
               "float_shape": tuple(float(size) for size in descriptor.shape)}
    changed_field = "shape" if field == "float_shape" else field
    values = {**analysis.values, identity: replace(descriptor, **{changed_field: changes[field]})}
    with pytest.raises(ValueError):
        build_logical_schedule(replace(analysis, values=values))


@pytest.mark.parametrize("field", ("roots", "output_nodes", "dead_nodes", "statistics", "dependencies"))
def test_scheduler_rejects_static_dependency_metadata_altered_from_the_source(field):
    analysis = analyze_static_execution(_source())
    if field == "statistics":
        changed = replace(analysis.statistics, largest_value_bytes=1)
    elif field == "dependencies":
        identity = next(iter(analysis.dependencies))
        changed = {**analysis.dependencies, identity: replace(analysis.dependencies[identity], depth_from_sources=999)}
    else:
        changed = ()
    with pytest.raises(ValueError):
        build_logical_schedule(replace(analysis, **{field: changed}))


@pytest.mark.parametrize("field", ("output_bytes", "releasable_input_bytes", "live_bytes_before",
                                   "live_bytes_after", "peak_live_bytes_during_step", "released_values",
                                   "ready_width", "criticality", "estimated_contraction_flops"))
def test_schedule_rejects_corrupted_step_accounting(field):
    schedule = build_logical_schedule(analyze_static_execution(_source()))
    position = next(number for number, step in enumerate(schedule.steps) if step.released_values)
    step = schedule.steps[position]
    changed = () if field == "released_values" else getattr(step, field) + 1
    steps = list(schedule.steps)
    steps[position] = replace(step, **{field: changed})
    with pytest.raises(ValueError):
        replace(schedule, steps=tuple(steps)).validate()


@pytest.mark.parametrize("field", ("scheduled_nodes", "skipped_dead_nodes", "total_estimated_contraction_flops",
                                   "max_ready_width", "released_values"))
def test_schedule_rejects_corrupted_summary_statistics(field):
    schedule = build_logical_schedule(analyze_static_execution(_source()))
    with pytest.raises(ValueError):
        replace(schedule, statistics=replace(schedule.statistics, **{field: getattr(schedule.statistics, field) + 1})).validate()


def test_schedule_cannot_replace_a_requested_correlator_by_its_intermediate_term():
    schedule = build_logical_schedule(analyze_static_execution(_source()))
    logical = schedule.source_analysis.source_dag
    output = logical.outputs["matrix"]
    term = next(term.node_id for term in logical.terms if term.correlator_id == "matrix")
    outputs = tuple(term if identity == output else identity for identity in schedule.output_nodes)
    lifetimes = {identity: replace(lifetime, retained_output=identity in outputs)
                 for identity, lifetime in schedule.lifetimes.items()}
    with pytest.raises(ValueError):
        replace(schedule, output_nodes=outputs, lifetimes=lifetimes).validate()


def test_schedule_cannot_silently_drop_one_requested_correlator():
    analysis = analyze_static_execution(_source())
    omitted = analysis.output_nodes[0]
    dependencies = {**analysis.dependencies, omitted: replace(analysis.dependencies[omitted], distance_to_output=None)}
    altered = replace(analysis, output_nodes=analysis.output_nodes[1:], dependencies=dependencies)
    with pytest.raises(ValueError):
        build_logical_schedule(altered)
