"""Focused tests for backend-independent logical scheduling."""

from dataclasses import replace
from itertools import count

import pytest

from concord_lqcd.backend.runtime.pipeline import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.compiler import (
    ContractionSizeModel,
    IntermediateComputation,
    IntermediateInputUse,
    TensorLegBinding,
    analyze_static_execution,
    build_logical_execution_dag,
    build_logical_schedule,
    build_partial_intermediate_cse,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.logical_scheduling import _priority_key
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import (
    CorrMatrix,
    Correlator,
    Operator,
    c,
    cbar,
    d,
    dbar,
    gamma5,
    gamma_identity,
    reset_registry,
    u,
    ubar,
)
from concord_lqcd.core.wick import wick_contract


def _logical(mode_size=5):
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * u("y", "a", "s2")
    )
    source = Operator(
        ubar("x", "b", "s3") * gamma5("s3", "s4") * c("x", "b", "s4")
    )
    correlator = Correlator(sink, source)
    physics_ir = lower_to_physics_ir(correlator, wick_contract(correlator))
    global_dag = lower_physics_ir_collection_to_global_cse({"C00": physics_ir})
    order = optimize_global_contraction_order(
        global_dag,
        ContractionSizeModel(mode_size=mode_size),
        strategy="greedy",
    )
    partial = build_partial_intermediate_cse(order)
    return build_logical_execution_dag(partial)


def test_frontend_schedule_is_dependency_valid_and_retains_only_outputs():
    logical = _logical(mode_size=6)
    analysis = analyze_static_execution(logical)
    schedule = build_logical_schedule(analysis)

    position = {node_id: idx for idx, node_id in enumerate(schedule.order)}
    assert schedule.statistics.scheduled_nodes == len(schedule.order)
    assert schedule.statistics.skipped_dead_nodes == 0
    assert set(schedule.output_nodes) == set(logical.outputs.values())

    for step in schedule.steps:
        for dependency in step.dependencies:
            assert position[dependency] < step.step_index

    expected_final = sum(
        analysis.value(node_id).size_bytes for node_id in schedule.output_nodes
    )
    assert schedule.statistics.final_live_bytes == expected_final
    assert schedule.statistics.peak_live_bytes >= expected_final
    assert schedule.steps[-1].live_bytes_after == expected_final


def test_schedule_lifetimes_match_consumers_and_release_after_final_use():
    analysis = analyze_static_execution(_logical(mode_size=5))
    schedule = build_logical_schedule(analysis, strategy="critical_path")

    for node_id in schedule.order:
        lifetime = schedule.lifetime(node_id)
        expected_consumer_steps = tuple(
            step.step_index
            for step in schedule.steps
            for dependency in step.dependencies
            if dependency == node_id
        )
        assert lifetime.consumer_steps == expected_consumer_steps
        if node_id in schedule.output_nodes:
            assert lifetime.retained_output
            assert lifetime.release_after_step is None
        else:
            assert not lifetime.retained_output
            assert lifetime.release_after_step == lifetime.last_used_at
            release_step = schedule.steps[lifetime.release_after_step]
            assert node_id in release_step.released_values


def test_topological_strategy_preserves_static_topological_order():
    analysis = analyze_static_execution(_logical(mode_size=5))
    schedule = build_logical_schedule(analysis, strategy="topological")

    expected = tuple(
        node_id
        for node_id in analysis.topological_order
        if analysis.dependency(node_id).is_live
    )
    assert schedule.order == expected


def test_default_schedule_skips_dead_nodes_but_can_include_them_explicitly():
    base = _logical(mode_size=5)
    exemplar = next(iter(base.primitives.values()))
    dead = replace(
        base.source_dag.primitive_values[exemplar.node_id], value_id="dead_primitive",
    )
    primitives = dict(base.source_dag.primitive_values)
    primitives[dead.value_id] = dead
    augmented = build_logical_execution_dag(replace(base.source_dag, primitive_values=primitives))

    analysis = analyze_static_execution(augmented)
    assert dead.value_id in analysis.dead_nodes

    normal = build_logical_schedule(analysis)
    assert dead.value_id not in normal.order
    assert normal.statistics.skipped_dead_nodes == 1

    complete = build_logical_schedule(analysis, include_dead=True)
    assert dead.value_id in complete.order
    assert complete.statistics.skipped_dead_nodes == 0
    assert complete.statistics.final_live_bytes == normal.statistics.final_live_bytes


def test_live_schedule_fanout_excludes_dead_consumers():
    base = _logical(mode_size=5)
    base_analysis = analyze_static_execution(base)
    primitive_id, primitive = next(iter(base.primitives.items()))
    descriptor = base_analysis.value(primitive_id)

    bindings = tuple(
        TensorLegBinding(port, f"dead_axis_{idx}")
        for idx, port in enumerate(primitive.output_ports)
    )
    dead_consumer = IntermediateComputation(
        intermediate_id="dead_consumer",
        signature="manual_dead_consumer",
        inputs=(IntermediateInputUse(primitive_id, bindings),),
        contracted_indices=(),
        output_indices=tuple(binding.index for binding in bindings),
        estimated_flops=0,
        output_elements=descriptor.element_count,
    )
    contractions = dict(base.source_dag.intermediates)
    contractions[dead_consumer.intermediate_id] = dead_consumer
    augmented = build_logical_execution_dag(replace(base.source_dag, intermediates=contractions))

    analysis = analyze_static_execution(augmented)
    assert dead_consumer.intermediate_id in analysis.dead_nodes
    full_fanout = analysis.dependency(primitive_id).fanout

    normal = build_logical_schedule(analysis)
    normal_step = next(step for step in normal.steps if step.node_id == primitive_id)
    live_consumers = {
        consumer
        for consumer in analysis.dependency(primitive_id).unique_consumers
        if analysis.dependency(consumer).is_live
    }
    assert normal_step.fanout == len(live_consumers)
    assert normal_step.fanout < full_fanout

    complete = build_logical_schedule(analysis, include_dead=True)
    complete_step = next(step for step in complete.steps if step.node_id == primitive_id)
    assert complete_step.fanout == full_fanout


def test_repeated_dependency_occurrences_are_preserved_in_lifetime_accounting():
    base = _logical(mode_size=5)
    original_term = base.terms[0]
    assert len(original_term.inputs) == 1
    global_dag = base.source_dag.source_plan.source_dag
    source_correlator = global_dag.correlators[0]
    source_term = source_correlator.terms[0]
    repeated_source_term = replace(
        source_term, components=source_term.components + source_term.components,
    )
    global_dag = replace(global_dag, correlators=(
        replace(source_correlator, terms=(repeated_source_term,)),
    ))
    partial = replace(
        base.source_dag,
        source_plan=replace(base.source_dag.source_plan, source_dag=global_dag),
    )
    dag = build_logical_execution_dag(partial)
    repeated_term = dag.terms[0]

    analysis = analyze_static_execution(dag)
    schedule = build_logical_schedule(analysis)

    repeated_value = repeated_term.inputs[0].value_id
    term_step = schedule.order.index(repeated_term.node_id)
    lifetime = schedule.lifetime(repeated_value)
    assert lifetime.consumer_steps[-2:] == (term_step, term_step)
    assert lifetime.last_used_at == term_step
    assert repeated_value in schedule.steps[term_step].released_values


def test_schedule_reports_exact_sum_of_selected_contraction_flops():
    logical = _logical(mode_size=7)
    analysis = analyze_static_execution(logical)
    schedule = build_logical_schedule(analysis, strategy="memory_aware")

    expected = sum(
        node.estimated_flops for node in logical.contractions.values()
    )
    assert schedule.statistics.total_estimated_contraction_flops == expected
    assert sum(step.estimated_contraction_flops for step in schedule.steps) == expected


def test_memory_aware_schedule_is_deterministic():
    analysis = analyze_static_execution(_logical(mode_size=5))
    first = build_logical_schedule(analysis, strategy="memory_aware")
    second = build_logical_schedule(analysis, strategy="memory_aware")

    assert first.order == second.order
    assert first.steps == second.steps
    assert first.lifetimes == second.lifetimes


def _scalar_channel_analysis(mode_size=3):
    """sigma and a two-pion operator, whose quark loops leave many small values ready at once."""
    reset_registry()
    labels = count()

    def meson(antiquark, quark, gamma, position):
        color, left, right = (f"m{next(labels)}" for _ in range(3))
        return Operator(
            antiquark(position, color, left) * gamma(left, right) * quark(position, color, right)
        )

    sigma = meson(ubar, u, gamma_identity, "x") + meson(dbar, d, gamma_identity, "x")
    pipi = (
        meson(dbar, u, gamma5, "x") * meson(ubar, d, gamma5, "y")
        + meson(ubar, d, gamma5, "x") * meson(dbar, u, gamma5, "y")
    )
    compilation = compile_correlators(
        CorrMatrix([sigma, pipi]), RuntimeCompilationConfig(mode_size=mode_size)
    )
    return compilation.static_analysis


def _full_scan_order(analysis, strategy, rescore=True):
    """Reference selection: score every ready node at every step.

    With ``rescore=False`` a node keeps the score it had when it became ready.
    """
    selected = [
        node_id
        for node_id in analysis.topological_order
        if analysis.dependency(node_id).is_live
    ]
    selected_set = set(selected)
    output_set = {node_id for node_id in analysis.output_nodes if node_id in selected_set}
    position = {node_id: idx for idx, node_id in enumerate(analysis.topological_order)}
    dependencies = {
        node_id: [
            dep for dep in analysis.dependency(node_id).dependencies if dep in selected_set
        ]
        for node_id in selected
    }
    consumers = {node_id: set() for node_id in selected}
    remaining = {node_id: 0 for node_id in selected}
    for node_id, deps in dependencies.items():
        for dep in deps:
            consumers[dep].add(node_id)
            remaining[dep] += 1
    fanout = {node_id: len(users) for node_id, users in consumers.items()}
    unsatisfied = {node_id: len(set(deps)) for node_id, deps in dependencies.items()}

    def score(node_id):
        return _priority_key(
            analysis,
            node_id,
            strategy=strategy,
            remaining_consumers=remaining,
            selected_fanout=fanout,
            scheduled_node_set=selected_set,
            output_set=output_set,
            topological_position=position,
        )

    ready = {node_id for node_id in selected if unsatisfied[node_id] == 0}
    frozen = {node_id: score(node_id) for node_id in ready}
    order = []
    while ready:
        node_id = max(ready, key=score if rescore else frozen.__getitem__)
        ready.remove(node_id)
        order.append(node_id)
        for dep in dependencies[node_id]:
            remaining[dep] -= 1
        for consumer in consumers[node_id]:
            unsatisfied[consumer] -= 1
            if unsatisfied[consumer] == 0:
                ready.add(consumer)
                frozen[consumer] = score(consumer)
    return tuple(order)


@pytest.mark.parametrize("strategy", ["memory_aware", "critical_path", "topological"])
def test_heap_selection_matches_a_full_scan_of_the_ready_set(strategy):
    analysis = _scalar_channel_analysis()
    schedule = build_logical_schedule(analysis, strategy=strategy)

    assert schedule.statistics.max_ready_width > 1
    assert schedule.order == _full_scan_order(analysis, strategy)


@pytest.mark.parametrize("strategy", ["memory_aware", "critical_path"])
def test_scalar_channel_exercises_rescoring_of_waiting_nodes(strategy):
    analysis = _scalar_channel_analysis()

    assert _full_scan_order(analysis, strategy) != _full_scan_order(
        analysis, strategy, rescore=False
    )
