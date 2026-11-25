"""Independent numerical replay and safety checks for contraction paths."""

from dataclasses import replace
from math import prod

import numpy as np
import pytest

from concord_lqcd.compiler.contraction_graph import TensorLegBinding
from concord_lqcd.compiler.contraction_order import (
    ComponentContractionPlan,
    ContractionSizeModel,
    ContractionStep,
    optimize_component_contraction_order,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.global_cse import (
    ComponentComputation,
    ComponentTensorUse,
    GlobalCSEStatistics,
    GlobalComputationDAG,
)


def _component(rows, outputs):
    tensors = []
    for number, (kind, indices) in enumerate(rows):
        if kind == "propagation":
            ports = ("sink_mode", "sink_spin", "source_mode", "source_spin")
        else:
            ports = tuple(f"axis_{axis}" for axis in range(len(indices)))
        tensors.append(ComponentTensorUse(
            slot=f"t{number}",
            kind=kind,
            resource_id=f"resource_{number}",
            placement=(),
            bindings=tuple(
                TensorLegBinding(port, index)
                for port, index in zip(ports, indices, strict=True)
            ),
        ))
    return ComponentComputation("numerical", "independent_fixture", tuple(tensors), tuple(outputs))


def _case(name):
    if name == "cycle":
        return _component((("spin", ("a", "b")), ("spin", ("b", "c")), ("spin", ("c", "a"))), ())
    if name == "open_chain":
        return _component((("spin", ("a", "b")), ("spin", ("b", "c")), ("spin", ("c", "d"))), ("d", "a"))
    if name == "trace":
        return _component((("spin", ("a", "a")),), ())
    if name == "diagonal":
        return _component((("spin", ("a", "a")),), ("a",))
    if name == "transpose":
        return _component((("spin", ("a", "b")),), ("b", "a"))
    if name == "scalar":
        return _component((("kernel", ()),), ())
    if name == "mixed_mode_spin":
        return _component((
            ("kernel", ("a", "b")),
            ("kernel", ("c", "d")),
            ("spin", ("s", "t")),
            ("spin", ("u", "v")),
            ("propagation", ("b", "t", "c", "u")),
            ("propagation", ("d", "v", "a", "s")),
        ), ())
    raise AssertionError(name)


def _inputs(component):
    sizes = {}
    for tensor in component.tensors:
        for binding in tensor.bindings:
            space = "mode" if tensor.kind == "kernel" or binding.port.endswith("_mode") else "spin"
            dimension = 3 if space == "mode" else 4
            if binding.index in sizes:
                assert sizes[binding.index] == dimension
            sizes[binding.index] = dimension
    generator = np.random.default_rng(7138)
    arrays = {}
    for tensor in component.tensors:
        shape = tuple(sizes[binding.index] for binding in tensor.bindings)
        arrays[tensor.slot] = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    return arrays, sizes


def _contract(arrays, indices, outputs):
    labels = sorted({index for legs in indices for index in legs})
    label_numbers = {label: number for number, label in enumerate(labels)}
    arguments = []
    for array, legs in zip(arrays, indices, strict=True):
        arguments.extend((array, [label_numbers[index] for index in legs]))
    arguments.append([label_numbers[index] for index in outputs])
    return np.einsum(*arguments, optimize=False)


def _direct(component, arrays):
    return _contract(
        [arrays[tensor.slot] for tensor in component.tensors],
        [tuple(binding.index for binding in tensor.bindings) for tensor in component.tensors],
        component.output_indices,
    )


def _replay(plan, arrays):
    active = dict(arrays)
    for step in plan.steps:
        result = _contract([active[ref] for ref in step.input_refs], step.input_indices, step.output_indices)
        for ref in step.input_refs:
            del active[ref]
        active[step.step_id] = result
    assert tuple(active) == (plan.result_ref,)
    return active[plan.result_ref]


@pytest.mark.parametrize("strategy", ("greedy", "optimal", "dynamic-programming", "branch-1", "branch-2", "auto"))
@pytest.mark.parametrize("case", ("cycle", "open_chain", "trace", "diagonal", "transpose", "scalar", "mixed_mode_spin"))
def test_generated_paths_preserve_numerics_and_accounting(case, strategy):
    component = _case(case)
    arrays, sizes = _inputs(component)
    expected = _direct(component, arrays)

    plan = optimize_component_contraction_order(
        component, ContractionSizeModel(mode_size=3), strategy=strategy,
    )
    actual = _replay(plan, arrays)

    assert np.max(np.abs(expected)) > 0.1
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert plan.estimated_flops == sum(step.estimated_flops for step in plan.steps)
    assert plan.largest_intermediate_elements == max(step.output_elements for step in plan.steps)
    for step in plan.steps:
        output_size = prod(sizes[index] for index in step.output_indices)
        assert step.output_elements == output_size
        involved = {index for legs in step.input_indices for index in legs}
        removed = involved - set(step.output_indices)
        assert set(step.contracted_indices) == removed
        work_size = prod(sizes[index] for index in involved)
        factor = max(1, len(step.input_refs) - 1) + bool(removed)
        assert step.estimated_flops == factor * work_size
    plan.validate(component)


def test_tight_search_memory_limit_can_produce_a_correct_nary_step():
    component = _case("cycle")
    arrays, _ = _inputs(component)
    expected = _direct(component, arrays)

    plan = optimize_component_contraction_order(
        component, ContractionSizeModel(mode_size=3), strategy="greedy", memory_limit=1,
    )

    assert any(len(step.input_refs) > 2 for step in plan.steps)
    assert plan.largest_intermediate_elements == 1
    np.testing.assert_allclose(_replay(plan, arrays), expected, rtol=1e-13, atol=1e-13)



def _three_baryon_network():
    """24-tensor three-baryon network: auto-hq uses opt_einsum's random-greedy search."""
    slots = [(baryon, quark) for baryon in range(3) for quark in range(3)]
    targets = [slots[i] for i in (4, 0, 8, 1, 6, 2, 7, 5, 3)]
    rows = []
    for baryon in range(3):
        rows.append(("kernel", tuple(f"m{baryon}{quark}" for quark in range(3))))
        rows.append(("kernel", tuple(f"n{baryon}{quark}" for quark in range(3))))
        rows.append(("spin", (f"s{baryon}0", f"s{baryon}1")))
        rows.append(("spin", (f"r{baryon}0", f"r{baryon}1")))
        rows.append(("spin", (f"s{baryon}2", f"r{baryon}2")))
    for number, (baryon, quark) in enumerate(slots):
        target_baryon, target_quark = targets[number]
        rows.append(("propagation", (
            f"m{baryon}{quark}", f"s{baryon}{quark}",
            f"n{target_baryon}{target_quark}", f"r{target_baryon}{target_quark}",
        )))
    return _component(rows, ())


def test_auto_hq_path_is_replaced_when_it_violates_the_memory_limit():
    component = _three_baryon_network()
    model = ContractionSizeModel(mode_size=3)
    unlimited = optimize_component_contraction_order(component, model, strategy="auto-hq")
    limit = unlimited.largest_intermediate_elements - 1

    plan = optimize_component_contraction_order(component, model, strategy="auto-hq", memory_limit=limit)

    assert plan.strategy == "greedy"
    assert plan.largest_intermediate_elements <= limit
    assert max(step.output_elements for step in plan.steps) <= limit
    arrays, _ = _inputs(component)
    expected = _replay(unlimited, arrays)
    assert abs(expected) > 1.0
    np.testing.assert_allclose(_replay(plan, arrays), expected, rtol=1e-12, atol=0)


def test_paths_within_the_memory_limit_keep_the_requested_strategy():
    component = _three_baryon_network()
    model = ContractionSizeModel(mode_size=3)
    unlimited = optimize_component_contraction_order(component, model, strategy="auto-hq")

    plan = optimize_component_contraction_order(
        component, model, strategy="auto-hq", memory_limit=unlimited.largest_intermediate_elements,
    )

    assert plan.strategy == "auto-hq"
    assert plan.steps == unlimited.steps


def test_memory_limit_below_the_component_output_is_rejected():
    with pytest.raises(ValueError, match="memory_limit=1 elements cannot be met"):
        optimize_component_contraction_order(
            _case("open_chain"), ContractionSizeModel(mode_size=3), strategy="greedy", memory_limit=1,
        )

@pytest.mark.parametrize("field", ("mode_size", "spin_size"))
@pytest.mark.parametrize("value", (True, False, 3.0, 2.5, float("nan"), np.bool_(True), "3"))
def test_dimensions_reject_non_integer_inputs(field, value):
    arguments = {"mode_size": 3, field: value}
    with pytest.raises(TypeError, match=field):
        ContractionSizeModel(**arguments)


def test_numpy_integer_dimensions_are_normalized_without_rounding():
    model = ContractionSizeModel(np.int64(3), np.int32(4))

    assert type(model.mode_size) is int and model.mode_size == 3
    assert type(model.spin_size) is int and model.spin_size == 4


@pytest.mark.parametrize("value", (True, 2.5, float("nan"), "3"))
def test_search_memory_limit_rejects_non_integer_inputs(value):
    with pytest.raises(TypeError, match="memory_limit"):
        optimize_component_contraction_order(_case("cycle"), ContractionSizeModel(3), memory_limit=value)


@pytest.mark.parametrize("value", (0, -1, True, 2.5))
def test_empty_global_dag_still_validates_its_search_memory_limit(value):
    dag = GlobalComputationDAG({}, {}, (), GlobalCSEStatistics(0, 0, 0, 0))
    expected_error = ValueError if type(value) is int else TypeError
    with pytest.raises(expected_error, match="memory_limit"):
        optimize_global_contraction_order(dag, ContractionSizeModel(3), memory_limit=value)


def _step(name, refs, inputs, contracted, outputs):
    return ContractionStep(name, tuple(refs), tuple(inputs), tuple(contracted), tuple(outputs), 0, 1)


def _plan(component, steps):
    return ComponentContractionPlan(
        component.computation_id, "manual", "fixed", tuple(steps), steps[-1].step_id,
        component.output_indices, 0, 1,
    )


def test_saved_plan_cannot_sum_an_index_needed_by_an_unconsumed_tensor():
    component = _component((("spin", ("a", "b")), ("spin", ("b", "c"))), ("a", "c"))
    arrays, _ = _inputs(component)
    expected = _direct(component, arrays)
    plan = _plan(component, (
        _step("early", ("t0",), (("a", "b"),), ("b",), ("a",)),
        _step("final", ("early", "t1"), (("a",), ("b", "c")), ("b",), ("a", "c")),
    ))
    assert np.max(np.abs(_replay(plan, arrays) - expected)) > 0.1

    with pytest.raises(ValueError, match="still needed"):
        plan.validate(component)


def test_saved_plan_cannot_create_output_indices_absent_from_its_inputs():
    component = _case("transpose")
    plan = _plan(component, (
        _step("invent", ("t0",), (("a", "b"),), ("b",), ("a", "invented")),
        _step("restore", ("invent",), (("a", "invented"),), ("invented",), ("b", "a")),
    ))

    with pytest.raises(ValueError, match="absent"):
        plan.validate(component)


def test_saved_plan_cannot_repeat_an_input_reference_in_one_step():
    component = _case("transpose")
    plan = _plan(component, (
        _step("duplicate", ("t0", "t0"), (("a", "b"), ("a", "b")), (), ("b", "a")),
    ))

    with pytest.raises(ValueError, match="repeated input"):
        plan.validate(component)


def test_saved_plan_cannot_repeat_an_output_index():
    component = _case("transpose")
    plan = _plan(component, (
        _step("duplicate", ("t0",), (("a", "b"),), (), ("a", "a", "b")),
        _step("final", ("duplicate",), (("a", "a", "b"),), (), ("b", "a")),
    ))

    with pytest.raises(ValueError, match="repeated output"):
        plan.validate(component)


def test_saved_plan_contracted_metadata_must_match_the_reduction():
    component = _case("trace")
    plan = _plan(component, (_step("trace", ("t0",), (("a", "a"),), (), ()),))

    with pytest.raises(ValueError, match="contracted indices"):
        plan.validate(component)


def test_saved_plan_cannot_overwrite_a_duplicate_tensor_slot():
    component = _component((("spin", ("a", "b")), ("spin", ("b", "c"))), ("a", "c"))
    component = replace(component, tensors=(
        component.tensors[0], replace(component.tensors[1], slot="t0"),
    ))
    plan = _plan(component, (_step("final", ("t0",), (("b", "c"),), ("b",), ("a", "c")),))

    with pytest.raises(ValueError, match="duplicate tensor slots"):
        plan.validate(component)


def test_saved_plan_cannot_drop_a_free_output_even_before_the_final_step():
    component = _case("transpose")
    plan = _plan(component, (
        _step("early", ("t0",), (("a", "b"),), ("b",), ("a",)),
        _step("final", ("early",), (("a",),), (), ("b", "a")),
    ))

    with pytest.raises(ValueError, match="still needed"):
        plan.validate(component)
