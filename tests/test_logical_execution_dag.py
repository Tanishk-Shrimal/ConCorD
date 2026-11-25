"""Focused tests for backend-independent logical execution DAG construction."""

import pytest

from concord_lqcd.compiler import (
    ComponentComputation,
    ComponentContractionPlan,
    ComponentIntermediatePlan,
    ComponentTensorUse,
    ComponentUse,
    ContractionSizeModel,
    GlobalCSEStatistics,
    GlobalComputationDAG,
    GlobalContractionOrderPlan,
    GlobalCorrelatorUse,
    GlobalIntermediateDAG,
    GlobalTensorResource,
    GlobalWickTermUse,
    LogicalExecutionStatistics,
    PartialCSEStatistics,
    PrimitiveTensorValue,
    TensorLegBinding,
    build_logical_execution_dag,
    build_partial_intermediate_cse,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import (
    Correlator,
    Operator,
    c,
    cbar,
    gamma5,
    gammaX,
    reset_registry,
    s,
    sbar,
    u,
    ubar,
)
from concord_lqcd.core.wick import wick_contract


def _physics_ir(correlator):
    return lower_to_physics_ir(correlator, wick_contract(correlator))


def _to_logical(physics_by_id):
    global_dag = lower_physics_ir_collection_to_global_cse(physics_by_id)
    order = optimize_global_contraction_order(
        global_dag,
        ContractionSizeModel(mode_size=5),
        strategy="greedy",
    )
    partial = build_partial_intermediate_cse(order)
    return build_logical_execution_dag(partial), global_dag


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


def _disconnected_three_point():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * u("y", "a", "s2"),
    )
    current = Operator(
        sbar("z", "b", "s3")
        * gammaX("s3", "s4")
        * s("z", "b", "s4"),
        momentum=(1, 0, 0),
    )
    source = Operator(
        ubar("x", "c", "s5")
        * gamma5("s5", "s6")
        * c("x", "c", "s6"),
    )
    return Correlator(sink, current, source)


def test_frontend_scalar_correlator_becomes_complete_dependency_dag():
    logical, global_dag = _to_logical({"C00": _physics_ir(_meson_two_point())})

    assert logical.statistics.primitive_evaluations == len(logical.primitives)
    assert logical.statistics.contractions == len(logical.contractions)
    assert logical.statistics.wick_term_products == 1
    assert logical.statistics.correlator_sums == 1

    term = logical.terms[0]
    output = logical.correlators[0]
    assert term.coefficient == global_dag.correlators[0].terms[0].coefficient
    assert term.output_indices == ()
    assert output.output_indices == ()
    assert output.term_ids == (term.node_id,)
    assert logical.outputs["C00"] == output.node_id

    order = logical.topological_order()
    position = {node_id: idx for idx, node_id in enumerate(order)}
    for node_id in order:
        for dependency in logical.dependencies(node_id):
            assert position[dependency] < position[node_id]


def test_disconnected_wick_term_is_explicit_product_of_component_results():
    logical, global_dag = _to_logical({"C": _physics_ir(_disconnected_three_point())})

    assert len(global_dag.correlators[0].terms) == 1
    assert len(global_dag.correlators[0].terms[0].components) == 2

    term = logical.terms[0]
    assert len(term.inputs) == 2
    assert term.output_indices == ()
    assert term.coefficient == global_dag.correlators[0].terms[0].coefficient
    assert all(
        use.value_id in logical.primitives or use.value_id in logical.contractions
        for use in term.inputs
    )


def test_reused_component_value_feeds_distinct_correlator_assemblies():
    first = _physics_ir(_meson_two_point())
    second = _physics_ir(_meson_two_point())
    logical, global_dag = _to_logical({"C00": first, "C01": second})

    assert global_dag.statistics.unique_components == 1
    assert len(logical.terms) == 2
    assert len(logical.correlators) == 2

    first_input = logical.terms[0].inputs[0].value_id
    second_input = logical.terms[1].inputs[0].value_id
    assert first_input == second_input
    assert logical.terms[0].node_id != logical.terms[1].node_id
    assert logical.outputs["C00"] != logical.outputs["C01"]


def _manual_partial(
    *,
    component: ComponentComputation,
    component_plan: ComponentIntermediatePlan,
    primitive: PrimitiveTensorValue,
    correlator_terms: tuple[GlobalWickTermUse, ...],
):
    resource = GlobalTensorResource(
        resource_id=primitive.resource_id,
        kind=primitive.kind,
        payload=object(),
    )
    global_dag = GlobalComputationDAG(
        resources={primitive.resource_id: resource},
        components={component.computation_id: component},
        correlators=(GlobalCorrelatorUse("C", correlator_terms),),
        statistics=GlobalCSEStatistics(1, 1, len(correlator_terms), 1),
    )
    order = GlobalContractionOrderPlan(
        source_dag=global_dag,
        size_model=ContractionSizeModel(mode_size=5),
        strategy="fixed",
        memory_limit=None,
        component_plans={
            component.computation_id: ComponentContractionPlan(
                computation_id=component.computation_id,
                optimizer_backend="test",
                strategy="fixed",
                steps=(),
                result_ref=component.tensors[0].slot,
                output_indices=component.output_indices,
                estimated_flops=0,
                largest_intermediate_elements=1,
            )
        },
    )
    return GlobalIntermediateDAG(
        source_plan=order,
        primitive_values={primitive.value_id: primitive},
        intermediates={},
        component_plans={component.computation_id: component_plan},
        statistics=PartialCSEStatistics(1, 1, 0, 0, 0, 0),
    )


def test_component_result_binding_is_composed_into_term_external_interface():
    tensor = ComponentTensorUse(
        slot="t0",
        kind="spin",
        resource_id="R",
        placement=("operator_arity", 2, ("spin_op", 0)),
        bindings=(
            TensorLegBinding("left", "o0"),
            TensorLegBinding("right", "o1"),
        ),
    )
    component = ComponentComputation(
        computation_id="component",
        signature="manual",
        tensors=(tensor,),
        output_indices=("o0", "o1"),
    )
    primitive = PrimitiveTensorValue(
        value_id="pv",
        kind="spin",
        resource_id="R",
        placement=tensor.placement,
        ports=("right", "left"),
    )
    component_plan = ComponentIntermediatePlan(
        computation_id="component",
        step_uses=(),
        result_value_id="pv",
        result_bindings=(
            TensorLegBinding("right", "o1"),
            TensorLegBinding("left", "o0"),
        ),
    )
    term = GlobalWickTermUse(
        term_index=0,
        coefficient=2.0,
        components=(ComponentUse(
            computation_id="component",
            output_bindings=(
                TensorLegBinding("o0", "external_a"),
                TensorLegBinding("o1", "external_b"),
            ),
        ),),
    )

    logical = build_logical_execution_dag(_manual_partial(
        component=component,
        component_plan=component_plan,
        primitive=primitive,
        correlator_terms=(term,),
    ))

    use = logical.terms[0].inputs[0]
    assert use.bindings == (
        TensorLegBinding("right", "external_b"),
        TensorLegBinding("left", "external_a"),
    )
    assert logical.terms[0].output_indices == ("external_a", "external_b")
    assert logical.correlators[0].output_indices == ("external_a", "external_b")


def test_repeated_scalar_component_use_preserves_multiplicity():
    tensor = ComponentTensorUse(
        slot="t0",
        kind="kernel",
        resource_id="R",
        placement=("operator_arity", 2, ("kernel_ops", (0,))),
        bindings=(),
    )
    component = ComponentComputation(
        computation_id="scalar_component",
        signature="manual",
        tensors=(tensor,),
        output_indices=(),
    )
    primitive = PrimitiveTensorValue(
        value_id="scalar_value",
        kind="kernel",
        resource_id="R",
        placement=tensor.placement,
        ports=(),
    )
    component_plan = ComponentIntermediatePlan(
        computation_id="scalar_component",
        step_uses=(),
        result_value_id="scalar_value",
        result_bindings=(),
    )
    term = GlobalWickTermUse(
        term_index=0,
        coefficient=-3.0,
        components=(
            ComponentUse("scalar_component", ()),
            ComponentUse("scalar_component", ()),
        ),
    )

    logical = build_logical_execution_dag(_manual_partial(
        component=component,
        component_plan=component_plan,
        primitive=primitive,
        correlator_terms=(term,),
    ))

    assert len(logical.terms[0].inputs) == 2
    assert logical.terms[0].inputs[0].value_id == "scalar_value"
    assert logical.terms[0].inputs[1].value_id == "scalar_value"
    assert logical.terms[0].coefficient == -3.0


def test_correlator_terms_with_different_external_interfaces_are_rejected():
    tensor = ComponentTensorUse(
        slot="t0",
        kind="spin",
        resource_id="R",
        placement=("operator_arity", 2, ("spin_op", 0)),
        bindings=(TensorLegBinding("left", "o0"),),
    )
    component = ComponentComputation(
        computation_id="component",
        signature="manual",
        tensors=(tensor,),
        output_indices=("o0",),
    )
    primitive = PrimitiveTensorValue(
        value_id="pv",
        kind="spin",
        resource_id="R",
        placement=tensor.placement,
        ports=("left",),
    )
    component_plan = ComponentIntermediatePlan(
        computation_id="component",
        step_uses=(),
        result_value_id="pv",
        result_bindings=(TensorLegBinding("left", "o0"),),
    )
    terms = (
        GlobalWickTermUse(
            0,
            1.0,
            (ComponentUse("component", (TensorLegBinding("o0", "external_a"),)),),
        ),
        GlobalWickTermUse(
            1,
            1.0,
            (ComponentUse("component", (TensorLegBinding("o0", "external_b"),)),),
        ),
    )

    with pytest.raises(ValueError, match="do not share one external output interface"):
        build_logical_execution_dag(_manual_partial(
            component=component,
            component_plan=component_plan,
            primitive=primitive,
            correlator_terms=terms,
        ))
