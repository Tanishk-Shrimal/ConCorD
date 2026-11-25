"""Focused tests for backend-independent static execution analysis."""

from dataclasses import replace

import pytest

from concord_lqcd.compiler import (
    ComponentComputation,
    ComponentTensorUse,
    ComponentUse,
    ContractionSizeModel,
    GlobalCSEStatistics,
    GlobalComputationDAG,
    GlobalCorrelatorUse,
    GlobalTensorResource,
    GlobalWickTermUse,
    LogicalContraction,
    LogicalExecutionDAG,
    LogicalExecutionStatistics,
    LogicalPrimitiveEvaluation,
    LogicalValueUse,
    StaticNumericalModel,
    TensorLegBinding,
    analyze_static_execution,
    build_logical_execution_dag,
    build_partial_intermediate_cse,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import Correlator, Operator, c, cbar, gamma5, reset_registry, u, ubar
from concord_lqcd.core.wick import wick_contract


def _physics_ir(correlator):
    return lower_to_physics_ir(correlator, wick_contract(correlator))


def _meson_two_point():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * u("y", "a", "s2"),
    )
    source = Operator(
        ubar("x", "b", "s3")
        * gamma5("s3", "s4")
        * c("x", "b", "s4"),
    )
    return Correlator(sink, source)


def _logical(mode_size=5):
    global_dag = lower_physics_ir_collection_to_global_cse({
        "C00": _physics_ir(_meson_two_point())
    })
    order = optimize_global_contraction_order(
        global_dag,
        ContractionSizeModel(mode_size=mode_size),
        strategy="greedy",
    )
    return build_logical_execution_dag(build_partial_intermediate_cse(order))


def test_frontend_static_analysis_derives_exact_primitive_shapes_and_bytes():
    logical = _logical(mode_size=7)
    analysis = analyze_static_execution(logical)

    assert analysis.numerical_model == StaticNumericalModel(
        mode_size=7,
        spin_size=4,
        scalar_type="complex128",
        element_bytes=16,
    )
    assert analysis.statistics.values == len(logical.node_ids)
    assert analysis.statistics.dead_nodes == 0
    assert set(analysis.output_nodes) == set(logical.outputs.values())

    for primitive in logical.primitives.values():
        descriptor = analysis.value(primitive.node_id)
        assert descriptor.axis_labels == primitive.output_ports
        if primitive.kind == "spin":
            assert descriptor.shape == (4, 4)
        elif primitive.kind == "propagation":
            assert descriptor.shape == (7, 4, 7, 4)
        elif primitive.kind == "kernel":
            assert descriptor.shape == (7,) * len(primitive.output_ports)
        else:
            raise AssertionError(primitive.kind)
        assert descriptor.element_count == (
            1 if not descriptor.shape else __import__("math").prod(descriptor.shape)
        )
        assert descriptor.size_bytes == descriptor.element_count * 16


def test_static_analysis_uses_contraction_order_dimensions_but_dtype_is_explicit():
    logical = _logical(mode_size=3)
    analysis = analyze_static_execution(
        logical,
        scalar_type="complex64",
        element_bytes=8,
    )

    assert analysis.numerical_model.mode_size == 3
    assert analysis.numerical_model.spin_size == 4
    assert analysis.numerical_model.scalar_type == "complex64"
    assert all(value.element_bytes == 8 for value in analysis.values.values())

    propagation = next(
        analysis.value(node.node_id)
        for node in logical.primitives.values()
        if node.kind == "propagation"
    )
    assert propagation.shape == (3, 4, 3, 4)
    assert propagation.size_bytes == 3 * 4 * 3 * 4 * 8


def test_every_optimized_intermediate_shape_matches_recorded_output_elements():
    logical = _logical(mode_size=6)
    analysis = analyze_static_execution(logical)

    for contraction in logical.contractions.values():
        descriptor = analysis.value(contraction.node_id)
        assert descriptor.axis_labels == contraction.output_indices
        assert descriptor.element_count == contraction.output_elements


def test_dependency_analysis_preserves_repeated_use_occurrences_and_unique_fanout():
    tensor = ComponentTensorUse(
        slot="scalar",
        kind="kernel",
        resource_id="R",
        placement=("kernel_ops", (0,)),
        bindings=(),
    )
    component = ComponentComputation("scalar_component", "manual", (tensor,), ())
    global_dag = GlobalComputationDAG(
        resources={"R": GlobalTensorResource("R", "kernel", object())},
        components={component.computation_id: component},
        correlators=(GlobalCorrelatorUse("C", (GlobalWickTermUse(
            0, 1.0, (ComponentUse(component.computation_id), ComponentUse(component.computation_id)),
        ),)),),
        statistics=GlobalCSEStatistics(1, 1, 2, 1),
    )
    order = optimize_global_contraction_order(global_dag, ContractionSizeModel(5), strategy="greedy")
    dag = build_logical_execution_dag(build_partial_intermediate_cse(order))
    repeated_value = dag.terms[0].inputs[0].value_id
    term_id = dag.terms[0].node_id

    analysis = analyze_static_execution(dag)
    record = analysis.dependency(repeated_value)
    assert record.consumers == (term_id, term_id)
    assert record.consumer_occurrences == 2
    assert record.unique_consumers == (term_id,)
    assert record.fanout == 1
    assert analysis.statistics.dependency_occurrences == 4
    assert analysis.statistics.unique_dependency_edges == 3


def test_static_analysis_rejects_a_logical_layer_attached_to_an_unrelated_source():
    primitive = LogicalPrimitiveEvaluation(
        node_id="p",
        value_id="p",
        resource_id="R",
        kind="kernel",
        placement=("operator_arity", 1, ("kernel_ops", (0,))),
        output_ports=(),
    )
    term = __import__("concord_lqcd.compiler", fromlist=["LogicalWickTermProduct"]).LogicalWickTermProduct(
        node_id="term",
        correlator_id="C",
        term_index=0,
        inputs=(LogicalValueUse("p", ()), LogicalValueUse("p", ())),
        coefficient=1.0,
        output_indices=(),
    )
    corr = __import__("concord_lqcd.compiler", fromlist=["LogicalCorrelatorSum"]).LogicalCorrelatorSum(
        node_id="sum",
        correlator_id="C",
        term_ids=("term",),
        output_indices=(),
    )

    base = _logical(mode_size=5)
    known_resource = next(iter(base.source_dag.source_plan.source_dag.resources))
    primitive = replace(primitive, resource_id=known_resource)
    dag = LogicalExecutionDAG(
        source_dag=base.source_dag,
        primitives={"p": primitive},
        contractions={},
        terms=(term,),
        correlators=(corr,),
        outputs={"C": "sum"},
        statistics=LogicalExecutionStatistics(1, 0, 1, 1, 3),
    )

    with pytest.raises(ValueError, match="source"):
        analyze_static_execution(dag)


def test_critical_path_and_dependency_depth_are_derived_without_scheduling():
    logical = _logical(mode_size=5)
    analysis = analyze_static_execution(logical)

    for node_id in analysis.topological_order:
        record = analysis.dependency(node_id)
        for dep in record.dependencies:
            assert analysis.dependency(dep).depth_from_sources < record.depth_from_sources

    expected = max(
        analysis.dependency(node_id).depth_from_sources
        for node_id in analysis.output_nodes
    ) + 1
    assert analysis.statistics.critical_path_nodes == expected
    assert all(analysis.dependency(node_id).is_live for node_id in analysis.topological_order)


def test_static_analysis_rejects_contraction_output_size_inconsistent_with_path_metadata():
    logical = _logical(mode_size=5)
    contraction_id, contraction = next(iter(logical.contractions.items()))
    broken = replace(contraction, output_elements=contraction.output_elements + 1)
    broken_contractions = dict(logical.contractions)
    broken_contractions[contraction_id] = broken
    intermediates = dict(logical.source_dag.intermediates)
    intermediates[contraction_id] = replace(
        intermediates[contraction_id], output_elements=broken.output_elements,
    )
    broken_dag = replace(
        logical, contractions=broken_contractions,
        source_dag=replace(logical.source_dag, intermediates=intermediates),
    )

    with pytest.raises(ValueError, match="disagrees with optimized-path metadata"):
        analyze_static_execution(broken_dag)


def test_static_numerical_model_rejects_invalid_storage_parameters():
    with pytest.raises(ValueError, match="mode_size"):
        StaticNumericalModel(mode_size=0)
    with pytest.raises(ValueError, match="spin_size"):
        StaticNumericalModel(mode_size=5, spin_size=0)
    with pytest.raises(ValueError, match="scalar_type"):
        StaticNumericalModel(mode_size=5, scalar_type="")
    with pytest.raises(ValueError, match="element_bytes"):
        StaticNumericalModel(mode_size=5, element_bytes=0)
