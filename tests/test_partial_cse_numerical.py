"""Direct NumPy references for recursive intermediate reuse and its limits."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from concord_lqcd.compiler.contraction_graph import TensorLegBinding
from concord_lqcd.compiler.contraction_order import (
    ComponentContractionPlan,
    ContractionSizeModel,
    ContractionStep,
    GlobalContractionOrderPlan,
    optimize_component_contraction_order,
)
from concord_lqcd.compiler.global_cse import (
    ComponentComputation,
    ComponentTensorUse,
    GlobalComputationDAG,
    GlobalCSEStatistics,
)
from concord_lqcd.compiler.partial_cse import build_partial_intermediate_cse


def _tensor(slot, resource, indices, operator_id=0):
    return ComponentTensorUse(
        slot, "kernel", resource, ("kernel_ops", (operator_id,)),
        tuple(TensorLegBinding(f"mode_{axis}", index) for axis, index in enumerate(indices)),
    )


def _step(name, refs, inputs, outputs, dimension=3):
    involved = {index for indices in inputs for index in indices}
    removed = involved - set(outputs)
    factor = max(1, len(refs) - 1) + bool(removed)
    return ContractionStep(
        name, tuple(refs), tuple(inputs), tuple(sorted(removed)), tuple(outputs),
        factor * dimension ** len(involved), dimension ** len(outputs),
    )


def _plan(component, steps):
    return ComponentContractionPlan(
        component.computation_id, "independent_fixture", "fixed", tuple(steps),
        steps[-1].step_id if steps else component.tensors[0].slot,
        component.output_indices, sum(step.estimated_flops for step in steps),
        max((step.output_elements for step in steps), default=9),
    )


def _chain(name, resources, *, reverse=False, transpose=False, operator_id=0, arity=1):
    indices = tuple(f"{name}:index{number}" for number in range(len(resources) + 1))
    tensors = tuple(
        _tensor(f"{name}:tensor{number}", resource, indices[number:number + 2], operator_id)
        for number, resource in enumerate(resources)
    )
    component = ComponentComputation(
        name, json.dumps({"arity": arity}), tensors, (indices[0], indices[-1]),
    )
    steps = []
    current_ref = tensors[0].slot
    current_indices = indices[:2]
    for number, tensor in enumerate(tensors[1:]):
        refs = (current_ref, tensor.slot)
        inputs = (current_indices, tuple(binding.index for binding in tensor.bindings))
        if reverse:
            refs, inputs = refs[::-1], inputs[::-1]
        outputs = (indices[0], indices[number + 2])
        if transpose and number == 0 and len(resources) > 2:
            outputs = outputs[::-1]
        step = _step(f"{name}:step{number}", refs, inputs, outputs)
        steps.append(step)
        current_ref, current_indices = step.step_id, step.output_indices
    return component, _plan(component, steps)


def _order(pairs, mode_size=3, spin_size=4):
    components = {component.computation_id: component for component, _ in pairs}
    dag = GlobalComputationDAG({}, components, (), GlobalCSEStatistics(0, 0, 0, len(components)))
    return GlobalContractionOrderPlan(
        dag, ContractionSizeModel(mode_size, spin_size), "fixed", None,
        {plan.computation_id: plan for _, plan in pairs},
    )


def _single_step_chain(name, count, *, reverse=False, transpose=False):
    component, _ = _chain(name, tuple(f"resource{number}" for number in range(count)))
    tensors = component.tensors[::-1] if reverse else component.tensors
    outputs = component.output_indices[::-1] if transpose else component.output_indices
    component = replace(component, tensors=tensors, output_indices=outputs)
    step = _step(
        name + ":step", tuple(tensor.slot for tensor in tensors),
        tuple(tuple(binding.index for binding in tensor.bindings) for tensor in tensors),
        outputs,
    )
    return component, _plan(component, (step,))


def _contract(arrays, indices, outputs):
    labels = sorted({index for legs in indices for index in legs})
    label_numbers = {label: number for number, label in enumerate(labels)}
    arguments = []
    for array, legs in zip(arrays, indices, strict=True):
        arguments.extend((array, [label_numbers[index] for index in legs]))
    arguments.append([label_numbers[index] for index in outputs])
    return np.einsum(*arguments, optimize=False)


def _references(order):
    raw_values = {}
    expected = {}
    for component in order.source_dag.components.values():
        arity = json.loads(component.signature)["arity"]
        operands = []
        for tensor in component.tensors:
            ports = tuple(binding.port for binding in tensor.bindings)
            native_ports = tuple(sorted(ports))
            seed_payload = (tensor.kind, tensor.resource_id, arity, tensor.placement)
            digest = hashlib.sha256(repr(seed_payload).encode("utf-8")).digest()
            generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            shape = tuple(
                order.size_model.mode_size
                if tensor.kind == "kernel" or port.endswith("_mode")
                else order.size_model.spin_size
                for port in native_ports
            )
            native = generator.normal(size=shape) + 1j * generator.normal(size=shape)
            array = np.transpose(native, tuple(native_ports.index(port) for port in ports))
            placement = ("operator_arity", arity, tensor.placement)
            raw_values[(tensor.kind, tensor.resource_id, placement, ports)] = array
            operands.append(array)
        expected[component.computation_id] = _contract(
            operands,
            [tuple(binding.index for binding in tensor.bindings) for tensor in component.tensors],
            component.output_indices,
        )
    return expected, raw_values


def _evaluate_partial(partial, raw_values):
    values = {}
    evaluations = {}

    def ports(value_id):
        if value_id in partial.primitive_values:
            return partial.primitive_values[value_id].ports
        return partial.intermediates[value_id].output_indices

    def value(value_id):
        if value_id in values:
            return values[value_id]
        if value_id in partial.primitive_values:
            primitive = partial.primitive_values[value_id]
            result = raw_values[(primitive.kind, primitive.resource_id, primitive.placement, primitive.ports)]
        else:
            intermediate = partial.intermediates[value_id]
            arrays = []
            indices = []
            for input_use in intermediate.inputs:
                bindings = {binding.port: binding.index for binding in input_use.bindings}
                arrays.append(value(input_use.value_id))
                indices.append(tuple(bindings[port] for port in ports(input_use.value_id)))
            result = _contract(arrays, indices, intermediate.output_indices)
            evaluations[value_id] = evaluations.get(value_id, 0) + 1
        values[value_id] = result
        return result

    actual = {}
    for component_id, component_plan in partial.component_plans.items():
        bindings = {binding.port: binding.index for binding in component_plan.result_bindings}
        indices = tuple(bindings[port] for port in ports(component_plan.result_value_id))
        component = partial.source_plan.source_dag.components[component_id]
        actual[component_id] = _contract(
            [value(component_plan.result_value_id)], [indices], component.output_indices,
        )
    assert all(count == 1 for count in evaluations.values())
    return actual


def _assert_results(actual, expected):
    assert set(actual) == set(expected)
    for name, reference in expected.items():
        assert np.max(np.abs(reference)) > 0.1
        np.testing.assert_allclose(actual[name], reference, rtol=1e-12, atol=1e-12)
        scale = np.max(np.abs(reference))
        assert np.max(np.abs(actual[name] - reference)) / scale < 1e-13


def test_recursive_reuse_survives_dummy_renaming_operand_and_output_permutations():
    pairs = [
        _chain("short", ("A", "B", "C")),
        _chain("long", ("A", "B", "C", "D"), reverse=True, transpose=True),
    ]
    order = _order(pairs)
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)
    actual = _evaluate_partial(partial, raw_values)

    _assert_results(actual, expected)
    assert partial.statistics.intermediate_occurrences == 5
    assert partial.statistics.unique_intermediates == 3
    assert partial.statistics.estimated_flops_saved == 108
    short = partial.component_plans["short"].step_uses
    long = partial.component_plans["long"].step_uses
    assert short[0].intermediate_id == long[0].intermediate_id
    assert short[1].intermediate_id == long[1].intermediate_id


@pytest.mark.parametrize("reverse,transpose", ((False, False), (True, False), (False, True), (True, True)))
def test_reordered_component_nodes_keep_exact_subtree_reuse(reverse, transpose):
    first = _chain("first", ("A", "B", "C"))
    second_component, second_plan = _chain("second", ("A", "B", "C"), reverse=reverse, transpose=transpose)
    second = (replace(second_component, tensors=second_component.tensors[::-1]), second_plan)
    order = _order((first, second))
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)

    _assert_results(_evaluate_partial(partial, raw_values), expected)
    assert partial.statistics.unique_intermediates == 2
    assert partial.statistics.intermediate_reuses == 2


@pytest.mark.parametrize("separation", ("placement", "arity"))
def test_different_placements_or_operator_arities_are_numerically_separate(separation):
    first = _chain("first", ("A", "B"), operator_id=1, arity=2)
    second = _chain(
        "second", ("A", "B"), operator_id=0 if separation == "placement" else 1,
        arity=2 if separation == "placement" else 3,
    )
    order = _order((first, second))
    expected, raw_values = _references(order)
    assert np.max(np.abs(expected["first"] - expected["second"])) > 0.1

    partial = build_partial_intermediate_cse(order)

    _assert_results(_evaluate_partial(partial, raw_values), expected)
    assert partial.statistics.unique_primitive_values == 4
    assert partial.statistics.unique_intermediates == 2


def test_repeated_primitive_value_remains_two_factors_in_one_intermediate():
    order = _order((_chain("square", ("A", "A")),))
    expected, raw_values = _references(order)
    assert np.max(np.abs(expected["square"] - next(iter(raw_values.values())))) > 0.1

    partial = build_partial_intermediate_cse(order)

    _assert_results(_evaluate_partial(partial, raw_values), expected)
    assert partial.statistics.unique_primitive_values == 1
    intermediate = next(iter(partial.intermediates.values()))
    assert len(intermediate.inputs) == 2
    assert intermediate.inputs[0].value_id == intermediate.inputs[1].value_id


def test_different_port_connectivity_cannot_reuse_the_same_intermediate():
    first = _chain("normal", ("A", "B"))
    component, _ = _chain("changed", ("A", "B"))
    changed_tensor = component.tensors[1]
    changed_tensor = replace(changed_tensor, bindings=(
        replace(changed_tensor.bindings[0], index=changed_tensor.bindings[1].index),
        replace(changed_tensor.bindings[1], index=changed_tensor.bindings[0].index),
    ))
    component = replace(component, tensors=(component.tensors[0], changed_tensor))
    inputs = tuple(tuple(binding.index for binding in tensor.bindings) for tensor in component.tensors)
    step = _step("changed:step", tuple(tensor.slot for tensor in component.tensors), inputs, component.output_indices)
    order = _order((first, (component, _plan(component, (step,)))))
    expected, raw_values = _references(order)
    assert np.max(np.abs(expected["normal"] - expected["changed"])) > 0.1

    partial = build_partial_intermediate_cse(order)

    _assert_results(_evaluate_partial(partial, raw_values), expected)
    assert partial.statistics.unique_intermediates == 2


@pytest.mark.parametrize("count", (3, 6))
def test_multi_input_steps_reuse_under_operand_and_output_permutations(count):
    order = _order((
        _single_step_chain("first", count),
        _single_step_chain("second", count, reverse=True, transpose=True),
    ))
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)

    _assert_results(_evaluate_partial(partial, raw_values), expected)
    assert partial.statistics.unique_intermediates == 1
    assert partial.statistics.intermediate_reuses == 1


def test_exposed_and_contracted_interfaces_cannot_share_one_intermediate():
    closed_component, closed_plan = _chain("closed", ("A", "B"))
    component, _ = _chain("open", ("A", "B"))
    inputs = tuple(tuple(binding.index for binding in tensor.bindings) for tensor in component.tensors)
    outputs = (inputs[0][0], inputs[0][1], inputs[1][1])
    component = replace(component, output_indices=outputs)
    step = _step("open:step", tuple(tensor.slot for tensor in component.tensors), inputs, outputs)
    order = _order(((closed_component, closed_plan), (component, _plan(component, (step,)))))
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)
    actual = _evaluate_partial(partial, raw_values)

    _assert_results(actual, expected)
    assert actual["closed"].shape == (3, 3)
    assert actual["open"].shape == (3, 3, 3)
    np.testing.assert_allclose(actual["open"].sum(axis=1), actual["closed"], rtol=1e-13, atol=1e-13)
    assert partial.statistics.unique_intermediates == 2


def test_trace_diagonal_and_no_step_results_preserve_their_interfaces():
    pairs = []
    for name, inputs, outputs in (
        ("trace", ("a", "a"), ()),
        ("diagonal", ("b", "b"), ("b",)),
        ("unchanged", ("left", "right"), ("left", "right")),
    ):
        tensor = _tensor(name + ":tensor", "A", inputs)
        component = ComponentComputation(name, json.dumps({"arity": 1}), (tensor,), outputs)
        steps = () if name == "unchanged" else (_step(name + ":step", (tensor.slot,), (inputs,), outputs),)
        pairs.append((component, _plan(component, steps)))
    order = _order(pairs)
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)
    actual = _evaluate_partial(partial, raw_values)

    _assert_results(actual, expected)
    assert actual["trace"].shape == ()
    assert actual["diagonal"].shape == (3,)
    assert actual["unchanged"].shape == (3, 3)
    assert partial.statistics.unique_primitive_values == 1
    assert partial.statistics.unique_intermediates == 2


def test_structurally_equal_intermediates_reject_inconsistent_cost_metadata():
    first = _chain("first", ("A", "B"))
    component, plan = _chain("second", ("A", "B"))
    changed_step = replace(plan.steps[0], estimated_flops=plan.steps[0].estimated_flops + 1)
    plan = replace(plan, steps=(changed_step,), estimated_flops=changed_step.estimated_flops)

    with pytest.raises(RuntimeError, match="inconsistent cost metadata"):
        build_partial_intermediate_cse(_order((first, (component, plan))))


def test_partial_cse_rejects_mismatched_component_map_identity():
    component, plan = _chain("actual", ("A", "B"))
    order = _order(((component, plan),))
    dag = replace(order.source_dag, components={"wrong_key": component})
    order = replace(order, source_dag=dag, component_plans={"wrong_key": plan})

    with pytest.raises(ValueError, match="component.*key"):
        build_partial_intermediate_cse(order)


@pytest.mark.parametrize("arity", (0, 1, True, -1, 2.0, "2", None))
def test_partial_cse_rejects_inconsistent_explicit_operator_arity(arity):
    pair = _chain("invalid", ("A", "B"), operator_id=1, arity=arity)

    with pytest.raises(ValueError, match="arity"):
        build_partial_intermediate_cse(_order((pair,)))


@pytest.mark.parametrize("signature", ("hand-built", "{}", "null", "[]"))
def test_hand_built_components_without_explicit_arity_keep_placement_inference(signature):
    component, plan = _chain("inferred", ("A", "B"), operator_id=1, arity=2)
    component = replace(component, signature=signature)

    partial = build_partial_intermediate_cse(_order(((component, plan),)))

    assert all(primitive.placement[:2] == ("operator_arity", 2) for primitive in partial.primitive_values.values())


def test_declared_arity_may_exceed_the_placements_in_one_component():
    pair = _chain("subcomponent", ("A", "B"), operator_id=1, arity=4)
    order = _order((pair,))
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)

    _assert_results(_evaluate_partial(partial, raw_values), expected)
    assert all(primitive.placement[:2] == ("operator_arity", 4) for primitive in partial.primitive_values.values())


@pytest.mark.parametrize("count", (7, 8))
def test_memory_limited_large_steps_reuse_only_the_same_operand_order(count):
    pairs = []
    for name, reverse in (("first", False), ("renamed", False), ("reordered", True)):
        indices = tuple(f"{name}:index{number}" for number in range(count))
        tensors = tuple(
            _tensor(
                f"{name}:tensor{number}", f"resource{number}",
                (indices[number], indices[(number + 1) % count]),
            )
            for number in range(count)
        )
        component = ComponentComputation(name, json.dumps({"arity": 1}), tensors, ())
        plan = optimize_component_contraction_order(
            component, ContractionSizeModel(3), memory_limit=1, strategy="greedy",
        )
        assert len(plan.steps) == 1
        assert len(plan.steps[0].input_refs) > 6
        if reverse:
            step = plan.steps[0]
            plan = replace(plan, steps=(replace(
                step, input_refs=step.input_refs[::-1], input_indices=step.input_indices[::-1],
            ),))
        pairs.append((component, plan))
    order = _order(pairs)
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)

    _assert_results(_evaluate_partial(partial, raw_values), expected)
    assert partial.statistics.unique_intermediates == 2
    assert partial.component_plans["first"].result_value_id == partial.component_plans["renamed"].result_value_id
    assert partial.component_plans["first"].result_value_id != partial.component_plans["reordered"].result_value_id


def test_large_step_preserves_reordered_open_output_interface():
    order = _order((
        _single_step_chain("first", 7),
        _single_step_chain("second", 7, transpose=True),
    ))
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)

    _assert_results(_evaluate_partial(partial, raw_values), expected)
    assert partial.statistics.unique_intermediates == 1


@pytest.mark.parametrize("case", ("meson", "baryon"))
def test_physics_derived_networks_match_direct_components_before_partial_cse(case):
    from tests.plan_costs import _baryon_correlator, _meson_correlator
    from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators

    builder = _meson_correlator if case == "meson" else _baryon_correlator
    compilation = compile_correlators(
        {case: builder()},
        RuntimeCompilationConfig(mode_size=2, shared_orthonormal_laph_basis=False),
    )
    order = compilation.contraction_order
    expected, raw_values = _references(order)

    partial = build_partial_intermediate_cse(order)

    assert partial == compilation.intermediate_dag
    _assert_results(_evaluate_partial(partial, raw_values), expected)
