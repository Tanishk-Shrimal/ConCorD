"""Frontend-to-partial-CSE integration coverage."""

from concord_lqcd.compiler import (
    ContractionSizeModel,
    build_partial_intermediate_cse,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import Correlator, Operator, c, cbar, gamma5, reset_registry, u, ubar
from concord_lqcd.core.wick import wick_contract


def test_frontend_component_plans_lower_to_global_intermediate_dag():
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
    correlator = Correlator(sink, source)
    physics_ir = lower_to_physics_ir(correlator, wick_contract(correlator))

    global_dag = lower_physics_ir_collection_to_global_cse({"C00": physics_ir})
    order_plan = optimize_global_contraction_order(
        global_dag,
        ContractionSizeModel(mode_size=5),
        strategy="greedy",
    )
    partial = build_partial_intermediate_cse(order_plan)

    expected_steps = sum(len(plan.steps) for plan in order_plan.component_plans.values())
    assert partial.statistics.intermediate_occurrences == expected_steps
    assert set(partial.component_plans) == set(global_dag.components)

    for component_id, lowered in partial.component_plans.items():
        component = global_dag.components[component_id]
        assert lowered.computation_id == component_id
        assert {
            binding.index for binding in lowered.result_bindings
        } == set(component.output_indices)
        assert (
            partial.value_is_primitive(lowered.result_value_id)
            or partial.value_is_intermediate(lowered.result_value_id)
        )
