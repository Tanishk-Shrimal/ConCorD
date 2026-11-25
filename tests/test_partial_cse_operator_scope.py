"""Regression tests for operator-slot scope in partial intermediate CSE."""

import json

from concord_lqcd.compiler import (
    ComponentComputation,
    ComponentContractionPlan,
    ComponentTensorUse,
    ContractionSizeModel,
    GlobalCSEStatistics,
    GlobalComputationDAG,
    GlobalContractionOrderPlan,
    TensorLegBinding,
    build_partial_intermediate_cse,
)


def _single_tensor_component(component_id: str, arity: int):
    tensor = ComponentTensorUse(
        slot="t0",
        kind="spin",
        resource_id="spin_tensor_shared",
        placement=("spin_op", 1),
        bindings=(
            TensorLegBinding("left", "o0"),
            TensorLegBinding("right", "o1"),
        ),
    )
    component = ComponentComputation(
        computation_id=component_id,
        signature=json.dumps({"arity": arity, "component": component_id}),
        tensors=(tensor,),
        output_indices=("o0", "o1"),
    )
    plan = ComponentContractionPlan(
        computation_id=component_id,
        optimizer_backend="test",
        strategy="fixed",
        steps=(),
        result_ref="t0",
        output_indices=("o0", "o1"),
        estimated_flops=0,
        largest_intermediate_elements=16,
    )
    return component, plan


def test_same_local_operator_number_in_different_npoint_arity_is_not_one_primitive_value():
    two_point, two_plan = _single_tensor_component("two_point", 2)
    three_point, three_plan = _single_tensor_component("three_point", 3)

    source_dag = GlobalComputationDAG(
        resources={},
        components={
            two_point.computation_id: two_point,
            three_point.computation_id: three_point,
        },
        correlators=(),
        statistics=GlobalCSEStatistics(0, 0, 0, 2),
    )
    order_plan = GlobalContractionOrderPlan(
        source_dag=source_dag,
        size_model=ContractionSizeModel(mode_size=5),
        strategy="fixed",
        memory_limit=None,
        component_plans={
            two_plan.computation_id: two_plan,
            three_plan.computation_id: three_plan,
        },
    )

    result = build_partial_intermediate_cse(order_plan)

    assert result.statistics.primitive_value_occurrences == 2
    assert result.statistics.unique_primitive_values == 2
    placements = {value.placement for value in result.primitive_values.values()}
    assert ("operator_arity", 2, ("spin_op", 1)) in placements
    assert ("operator_arity", 3, ("spin_op", 1)) in placements
