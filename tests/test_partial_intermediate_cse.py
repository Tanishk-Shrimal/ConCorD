"""Focused tests for recursive partial-intermediate CSE."""

from concord_lqcd.compiler import (
    ComponentComputation,
    ComponentContractionPlan,
    ComponentTensorUse,
    ContractionSizeModel,
    ContractionStep,
    GlobalCSEStatistics,
    GlobalComputationDAG,
    GlobalContractionOrderPlan,
    TensorLegBinding,
    build_partial_intermediate_cse,
)


def _tensor(slot, resource, left, right, *, placement=("kernel_ops", (0,))):
    return ComponentTensorUse(
        slot=slot,
        kind="kernel",
        resource_id=resource,
        placement=placement,
        bindings=(
            TensorLegBinding("p0", left),
            TensorLegBinding("p1", right),
        ),
    )


def _step(step_id, refs, indices, contracted, output, flops):
    return ContractionStep(
        step_id=step_id,
        input_refs=tuple(refs),
        input_indices=tuple(tuple(item) for item in indices),
        contracted_indices=tuple(contracted),
        output_indices=tuple(output),
        estimated_flops=flops,
        output_elements=25,
    )


def _component(component_id, tensors, outputs):
    return ComponentComputation(
        computation_id=component_id,
        signature=f"signature:{component_id}",
        tensors=tuple(tensors),
        output_indices=tuple(outputs),
    )


def _plan(component_id, steps, result_ref, outputs):
    return ComponentContractionPlan(
        computation_id=component_id,
        optimizer_backend="test",
        strategy="fixed",
        steps=tuple(steps),
        result_ref=result_ref,
        output_indices=tuple(outputs),
        estimated_flops=sum(step.estimated_flops for step in steps),
        largest_intermediate_elements=25,
    )


def _global_order(components, plans):
    dag = GlobalComputationDAG(
        resources={},
        components={component.computation_id: component for component in components},
        correlators=(),
        statistics=GlobalCSEStatistics(0, 0, 0, len(components)),
    )
    return GlobalContractionOrderPlan(
        source_dag=dag,
        size_model=ContractionSizeModel(mode_size=5),
        strategy="fixed",
        memory_limit=None,
        component_plans={plan.computation_id: plan for plan in plans},
    )


def test_shared_first_subtree_is_reused_across_different_components():
    comp_a = _component(
        "A",
        (
            _tensor("t0", "R_A", "a", "b"),
            _tensor("t1", "R_B", "b", "c"),
            _tensor("t2", "R_C", "c", "d"),
        ),
        ("a", "d"),
    )
    plan_a = _plan(
        "A",
        (
            _step("A:s0", ("t0", "t1"), (("a", "b"), ("b", "c")), ("b",), ("a", "c"), 100),
            _step("A:s1", ("A:s0", "t2"), (("a", "c"), ("c", "d")), ("c",), ("a", "d"), 200),
        ),
        "A:s1",
        ("a", "d"),
    )

    comp_b = _component(
        "B",
        (
            _tensor("u0", "R_A", "x", "y"),
            _tensor("u1", "R_B", "y", "z"),
            _tensor("u2", "R_D", "z", "w"),
        ),
        ("x", "w"),
    )
    plan_b = _plan(
        "B",
        (
            # Reverse both input order and intermediate output-axis order.  The
            # subtree is still the same numerical contraction after alpha-renaming.
            _step("B:s0", ("u1", "u0"), (("y", "z"), ("x", "y")), ("y",), ("z", "x"), 100),
            _step("B:s1", ("B:s0", "u2"), (("z", "x"), ("z", "w")), ("z",), ("x", "w"), 300),
        ),
        "B:s1",
        ("x", "w"),
    )

    result = build_partial_intermediate_cse(_global_order((comp_a, comp_b), (plan_a, plan_b)))

    assert result.statistics.intermediate_occurrences == 4
    assert result.statistics.unique_intermediates == 3
    assert result.statistics.intermediate_reuses == 1
    assert result.statistics.estimated_flops_saved == 100

    first_a = result.component_plans["A"].step_uses[0]
    first_b = result.component_plans["B"].step_uses[0]
    assert first_a.intermediate_id == first_b.intermediate_id
    assert first_a.output_bindings != first_b.output_bindings


def test_recursive_subtrees_are_reused_not_only_first_pair():
    comp_a = _component(
        "A",
        (
            _tensor("a0", "R_A", "a", "b"),
            _tensor("a1", "R_B", "b", "c"),
            _tensor("a2", "R_C", "c", "d"),
        ),
        ("a", "d"),
    )
    plan_a = _plan(
        "A",
        (
            _step("A:s0", ("a0", "a1"), (("a", "b"), ("b", "c")), ("b",), ("a", "c"), 10),
            _step("A:s1", ("A:s0", "a2"), (("a", "c"), ("c", "d")), ("c",), ("a", "d"), 20),
        ),
        "A:s1",
        ("a", "d"),
    )

    comp_b = _component(
        "B",
        (
            _tensor("b0", "R_A", "x", "y"),
            _tensor("b1", "R_B", "y", "z"),
            _tensor("b2", "R_C", "z", "w"),
            _tensor("b3", "R_D", "w", "v"),
        ),
        ("x", "v"),
    )
    plan_b = _plan(
        "B",
        (
            _step("B:s0", ("b0", "b1"), (("x", "y"), ("y", "z")), ("y",), ("x", "z"), 10),
            _step("B:s1", ("B:s0", "b2"), (("x", "z"), ("z", "w")), ("z",), ("x", "w"), 20),
            _step("B:s2", ("B:s1", "b3"), (("x", "w"), ("w", "v")), ("w",), ("x", "v"), 30),
        ),
        "B:s2",
        ("x", "v"),
    )

    result = build_partial_intermediate_cse(_global_order((comp_a, comp_b), (plan_a, plan_b)))

    assert result.statistics.intermediate_occurrences == 5
    assert result.statistics.unique_intermediates == 3
    assert result.statistics.intermediate_reuses == 2
    assert result.statistics.estimated_flops_saved == 30

    uses_a = result.component_plans["A"].step_uses
    uses_b = result.component_plans["B"].step_uses
    assert uses_a[0].intermediate_id == uses_b[0].intermediate_id
    assert uses_a[1].intermediate_id == uses_b[1].intermediate_id


def test_primitive_values_deduplicate_by_resource_and_exact_placement():
    comp_a = _component(
        "A",
        (_tensor("a0", "R_A", "a", "b"), _tensor("a1", "R_B", "b", "c")),
        ("a", "c"),
    )
    plan_a = _plan(
        "A",
        (_step("A:s0", ("a0", "a1"), (("a", "b"), ("b", "c")), ("b",), ("a", "c"), 10),),
        "A:s0",
        ("a", "c"),
    )

    comp_b = _component(
        "B",
        (_tensor("b0", "R_A", "x", "y"), _tensor("b1", "R_B", "y", "z")),
        ("x", "z"),
    )
    plan_b = _plan(
        "B",
        (_step("B:s0", ("b0", "b1"), (("x", "y"), ("y", "z")), ("y",), ("x", "z"), 10),),
        "B:s0",
        ("x", "z"),
    )

    result = build_partial_intermediate_cse(_global_order((comp_a, comp_b), (plan_a, plan_b)))

    assert result.statistics.primitive_value_occurrences == 4
    assert result.statistics.unique_primitive_values == 2
    assert result.statistics.primitive_value_reuses == 2
    assert result.statistics.unique_intermediates == 1


def test_different_placement_prevents_false_intermediate_reuse():
    comp_a = _component(
        "A",
        (_tensor("a0", "R_A", "a", "b"), _tensor("a1", "R_B", "b", "c")),
        ("a", "c"),
    )
    plan_a = _plan(
        "A",
        (_step("A:s0", ("a0", "a1"), (("a", "b"), ("b", "c")), ("b",), ("a", "c"), 10),),
        "A:s0",
        ("a", "c"),
    )

    comp_b = _component(
        "B",
        (
            _tensor("b0", "R_A", "x", "y", placement=("kernel_ops", (1,))),
            _tensor("b1", "R_B", "y", "z", placement=("kernel_ops", (1,))),
        ),
        ("x", "z"),
    )
    plan_b = _plan(
        "B",
        (_step("B:s0", ("b0", "b1"), (("x", "y"), ("y", "z")), ("y",), ("x", "z"), 10),),
        "B:s0",
        ("x", "z"),
    )

    result = build_partial_intermediate_cse(_global_order((comp_a, comp_b), (plan_a, plan_b)))

    assert result.statistics.unique_intermediates == 2
    assert result.statistics.intermediate_reuses == 0


def test_contracted_vs_exposed_index_structure_is_not_merged():
    comp_a = _component(
        "A",
        (_tensor("a0", "R_A", "a", "b"), _tensor("a1", "R_B", "b", "c")),
        ("a", "c"),
    )
    plan_a = _plan(
        "A",
        (_step("A:s0", ("a0", "a1"), (("a", "b"), ("b", "c")), ("b",), ("a", "c"), 10),),
        "A:s0",
        ("a", "c"),
    )

    # Same primitive values, but b is intentionally retained as an output index
    # rather than summed.  This is a different algebraic intermediate.
    comp_b = _component(
        "B",
        (_tensor("b0", "R_A", "x", "y"), _tensor("b1", "R_B", "y", "z")),
        ("x", "y", "z"),
    )
    plan_b = _plan(
        "B",
        (_step("B:s0", ("b0", "b1"), (("x", "y"), ("y", "z")), (), ("x", "y", "z"), 10),),
        "B:s0",
        ("x", "y", "z"),
    )

    result = build_partial_intermediate_cse(_global_order((comp_a, comp_b), (plan_a, plan_b)))

    assert result.statistics.unique_intermediates == 2


def test_missing_component_order_plan_is_rejected():
    component = _component(
        "A",
        (_tensor("a0", "R_A", "a", "b"),),
        ("a", "b"),
    )
    dag = GlobalComputationDAG(
        resources={},
        components={"A": component},
        correlators=(),
        statistics=GlobalCSEStatistics(0, 0, 0, 1),
    )
    order = GlobalContractionOrderPlan(
        source_dag=dag,
        size_model=ContractionSizeModel(mode_size=5),
        strategy="fixed",
        memory_limit=None,
        component_plans={},
    )

    import pytest

    with pytest.raises(ValueError, match="Missing contraction-order plan"):
        build_partial_intermediate_cse(order)
