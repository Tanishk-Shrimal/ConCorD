"""Focused tests for backend-independent contraction-order optimization."""

import pytest

from concord_lqcd.compiler import (
    ComponentComputation,
    ComponentTensorUse,
    ContractionSizeModel,
    TensorLegBinding,
    infer_component_index_spaces,
    lower_physics_ir_collection_to_global_cse,
    optimize_component_contraction_order,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import Correlator, Operator, c, cbar, gamma5, reset_registry, u, ubar
from concord_lqcd.core.wick import wick_contract


def _meson_two_point():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * u("y", "a", "s2"),
        momentum=(0, 0, 0),
    )
    source = Operator(
        ubar("x", "b", "s3")
        * gamma5("s3", "s4")
        * c("x", "b", "s4"),
        momentum=(0, 0, 0),
    )
    return Correlator(sink, source)


def _physics_ir(correlator):
    return lower_to_physics_ir(correlator, wick_contract(correlator))


def _single_component_dag():
    physics = _physics_ir(_meson_two_point())
    dag = lower_physics_ir_collection_to_global_cse({"C00": physics})
    assert len(dag.components) == 1
    return dag


def test_mesonic_component_gets_valid_opt_einsum_contraction_plan():
    dag = _single_component_dag()
    size_model = ContractionSizeModel(mode_size=8)

    global_plan = optimize_global_contraction_order(
        dag,
        size_model,
        strategy="greedy",
    )

    assert len(global_plan.component_plans) == 1
    component_id, component = next(iter(dag.components.items()))
    plan = global_plan.component_plans[component_id]

    assert plan.optimizer_backend == "opt_einsum"
    assert plan.strategy == "greedy"
    assert plan.output_indices == ()
    assert plan.estimated_flops > 0
    assert plan.largest_intermediate_elements >= 1
    assert plan.steps
    assert plan.result_ref == plan.steps[-1].step_id
    plan.validate(component)


def test_mode_dimension_changes_cost_without_changing_component_identity():
    dag = _single_component_dag()
    component_id, component = next(iter(dag.components.items()))

    small = optimize_component_contraction_order(
        component,
        ContractionSizeModel(mode_size=4),
        strategy="greedy",
    )
    large = optimize_component_contraction_order(
        component,
        ContractionSizeModel(mode_size=16),
        strategy="greedy",
    )

    assert small.computation_id == large.computation_id == component_id
    assert large.estimated_flops > small.estimated_flops
    assert large.largest_intermediate_elements >= small.largest_intermediate_elements


def test_single_tensor_free_indices_are_preserved_by_unary_plan_step():
    component = ComponentComputation(
        computation_id="component_free_spin",
        signature="manual",
        tensors=(
            ComponentTensorUse(
                slot="t0",
                kind="spin",
                resource_id="spin_tensor_gamma5",
                placement=("spin_op", 0),
                bindings=(
                    TensorLegBinding("left", "o0"),
                    TensorLegBinding("right", "o1"),
                ),
            ),
        ),
        output_indices=("o0", "o1"),
    )

    plan = optimize_component_contraction_order(
        component,
        ContractionSizeModel(mode_size=8),
        strategy="greedy",
    )

    assert plan.output_indices == ("o0", "o1")
    assert len(plan.steps) == 1
    assert plan.steps[0].input_refs == ("t0",)
    assert plan.steps[0].output_indices == ("o0", "o1")
    plan.validate(component)


def test_index_space_validation_rejects_mode_spin_mismatch():
    component = ComponentComputation(
        computation_id="component_bad_space",
        signature="manual",
        tensors=(
            ComponentTensorUse(
                slot="t0",
                kind="kernel",
                resource_id="kernel_a",
                placement=("kernel_ops", (0,)),
                bindings=(TensorLegBinding("mode_0", "i0"),),
            ),
            ComponentTensorUse(
                slot="t1",
                kind="spin",
                resource_id="gamma",
                placement=("spin_op", 0),
                bindings=(
                    TensorLegBinding("left", "i0"),
                    TensorLegBinding("right", "o0"),
                ),
            ),
        ),
        output_indices=("o0",),
    )

    with pytest.raises(ValueError, match="incompatible spaces"):
        infer_component_index_spaces(component)


def test_global_order_optimizer_plans_each_unique_component_once():
    physics = _physics_ir(_meson_two_point())
    dag = lower_physics_ir_collection_to_global_cse({
        "C00": physics,
        "C11": physics,
    })

    assert dag.statistics.component_occurrences == 2
    assert dag.statistics.unique_components == 1

    plan = optimize_global_contraction_order(
        dag,
        ContractionSizeModel(mode_size=8),
        strategy="auto",
    )

    assert len(plan.component_plans) == dag.statistics.unique_components == 1
    assert set(plan.component_plans) == set(dag.components)
    assert plan.estimated_unique_flops > 0


def test_invalid_size_and_memory_limits_are_rejected():
    with pytest.raises(ValueError, match="mode_size"):
        ContractionSizeModel(mode_size=0)

    dag = _single_component_dag()
    component = next(iter(dag.components.values()))
    with pytest.raises(ValueError, match="memory_limit"):
        optimize_component_contraction_order(
            component,
            ContractionSizeModel(mode_size=8),
            memory_limit=0,
        )
