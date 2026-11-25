"""Independent dense-tensor checks of primitive graph CSE, not runtime speed."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from concord_lqcd.compiler.contraction_graph import (
    PrimitiveTensorNode,
    TensorLegBinding,
    WickContractionGraph,
)
from concord_lqcd.compiler.global_cse import build_global_cse
from concord_lqcd.compiler.physics_ir import SpinTensorRecord


def _matrices():
    generator = np.random.default_rng(6042)
    return {
        kind: generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
        for kind in ("matrix:A", "matrix:B", "matrix:C")
    }


def _node(name, kind, left, right, *, operator_id=0, resource_id=None):
    return PrimitiveTensorNode(
        node_id=name,
        kind="spin",
        resource_id=resource_id or kind,
        bindings=(TensorLegBinding("left", left), TensorLegBinding("right", right)),
        payload=SpinTensorRecord(kind, left, right, operator_id),
    )


def _contract(tensors, indices, outputs):
    labels = sorted({label for legs in indices for label in legs})
    label_numbers = {label: number for number, label in enumerate(labels)}
    arguments = []
    for tensor, legs in zip(tensors, indices, strict=True):
        arguments.extend((tensor, [label_numbers[label] for label in legs]))
    if not arguments:
        assert not outputs
        return np.asarray(1.0 + 0j)
    arguments.append([label_numbers[label] for label in outputs])
    return np.einsum(*arguments, optimize=False)


def _direct(graph, matrices):
    indices = [node.legs for node in graph.nodes]
    counts = Counter(label for legs in indices for label in legs)
    outputs = sorted(label for label, count in counts.items() if count == 1)
    tensors = [matrices[node.payload.kind] for node in graph.nodes]
    return graph.coefficient * _contract(tensors, indices, outputs)


def _after_cse(dag, matrices):
    values = {}
    for component_id, component in dag.components.items():
        tensors = [
            matrices[dag.resources[tensor.resource_id].payload.kind]
            for tensor in component.tensors
        ]
        indices = [
            tuple(binding.index for binding in tensor.bindings)
            for tensor in component.tensors
        ]
        values[component_id] = _contract(tensors, indices, component.output_indices)
    results = {}
    for correlator in dag.correlators:
        result = np.asarray(0.0 + 0j)
        for term in correlator.terms:
            tensors = []
            indices = []
            for use in term.components:
                component = dag.components[use.computation_id]
                bindings = {binding.port: binding.index for binding in use.output_bindings}
                tensors.append(values[use.computation_id])
                indices.append(tuple(bindings[label] for label in component.output_indices))
            outputs = sorted({label for legs in indices for label in legs})
            result = result + term.coefficient * _contract(tensors, indices, outputs)
        results[correlator.correlator_id] = result
    return results


def _loop(prefix):
    return (
        _node(prefix + "A", "matrix:A", prefix + "a", prefix + "b"),
        _node(prefix + "B", "matrix:B", prefix + "b", prefix + "c"),
        _node(prefix + "C", "matrix:C", prefix + "c", prefix + "a"),
    )


def test_random_relabeling_preserves_numerical_graphs_and_exact_term_sum():
    matrices = _matrices()
    graph = WickContractionGraph(2.0 - 0.5j, _loop("original:"))
    reference = _direct(graph, matrices)
    generator = np.random.default_rng(3891)
    terms = [graph]
    for trial in range(12):
        relabeled = tuple(
            replace(
                node,
                node_id=f"trial{trial}:{node.node_id}",
                bindings=tuple(
                    replace(binding, index=f"dummy{trial}:{binding.index}")
                    for binding in node.bindings
                ),
            )
            for node in graph.nodes
        )
        order = generator.permutation(len(relabeled))
        terms.append(WickContractionGraph(-0.125 + 0.25j, tuple(relabeled[index] for index in order)))
    opposite = replace(graph, coefficient=-graph.coefficient)
    expected = sum(_direct(term, matrices) for term in terms)

    dag = build_global_cse({"sum": tuple(terms), "cancel": (graph, opposite)})
    actual = _after_cse(dag, matrices)

    assert abs(reference) > 0.1
    assert dag.statistics.unique_components == 1
    assert len(dag.correlators[0].terms) == 1
    assert dag.correlators[1].terms == ()
    np.testing.assert_allclose(actual["sum"], expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(actual["cancel"], 0.0)


def test_reused_disconnected_component_is_multiplied_twice():
    matrices = _matrices()
    graph = WickContractionGraph(1.0, _loop("first:") + _loop("second:"))
    loop = np.trace(matrices["matrix:A"] @ matrices["matrix:B"] @ matrices["matrix:C"])
    expected = _direct(graph, matrices)

    dag = build_global_cse({"squared": (graph,)})
    actual = _after_cse(dag, matrices)["squared"]

    assert dag.statistics.resource_occurrences == 6
    assert dag.statistics.unique_resources == 3
    assert dag.statistics.component_occurrences == 2
    assert dag.statistics.unique_components == 1
    uses = dag.correlators[0].terms[0].components
    assert len(uses) == 2 and uses[0] == uses[1]
    assert abs(expected - loop) > 0.1
    np.testing.assert_allclose(expected, loop * loop, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_open_output_bindings_keep_transposition_and_prevent_false_cancellation():
    matrices = _matrices()
    graph = WickContractionGraph(1.0, (
        _node("A", "matrix:A", "external_a", "dummy"),
        _node("B", "matrix:B", "dummy", "external_b"),
    ))
    swapped = replace(graph, coefficient=-1.0, nodes=(
        _node("otherA", "matrix:A", "external_b", "otherdummy"),
        _node("otherB", "matrix:B", "otherdummy", "external_a"),
    ))
    expected = _direct(graph, matrices) + _direct(swapped, matrices)

    dag = build_global_cse({"antisymmetric": (graph, swapped)})
    actual = _after_cse(dag, matrices)["antisymmetric"]

    assert np.max(np.abs(expected)) > 0.1
    assert dag.statistics.unique_components == 1
    assert len(dag.correlators[0].terms) == 2
    product = matrices["matrix:A"] @ matrices["matrix:B"]
    np.testing.assert_allclose(expected, product - product.T, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_different_port_connectivity_is_not_reused_as_the_same_component():
    matrices = _matrices()
    graph = WickContractionGraph(1.0, _loop("first:"))
    middle = graph.nodes[1]
    reversed_middle = replace(middle, bindings=(
        TensorLegBinding("left", middle.legs[1]),
        TensorLegBinding("right", middle.legs[0]),
    ))
    changed = replace(graph, nodes=(graph.nodes[0], reversed_middle, graph.nodes[2]))
    expected = {"original": _direct(graph, matrices), "changed": _direct(changed, matrices)}

    dag = build_global_cse({"original": (graph,), "changed": (changed,)})
    actual = _after_cse(dag, matrices)

    assert abs(expected["original"] - expected["changed"]) > 0.1
    assert dag.statistics.unique_components == 2
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-13, atol=1e-13)


def test_self_trace_and_open_shaped_zero_are_preserved():
    matrices = _matrices()
    trace = WickContractionGraph(1.0, (_node("trace", "matrix:A", "loop", "loop"),))
    open_graph = WickContractionGraph(1.0, (_node("open", "matrix:B", "left", "right"),))
    opposite = replace(open_graph, coefficient=-1.0)
    expected_trace = _direct(trace, matrices)
    expected_zero = _direct(open_graph, matrices) + _direct(opposite, matrices)

    dag = build_global_cse({"trace": (trace,), "zero": (open_graph, opposite)})
    actual = _after_cse(dag, matrices)

    assert abs(expected_trace) > 0.1
    assert actual["zero"].shape == (4, 4)
    np.testing.assert_allclose(actual["trace"], expected_trace, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(actual["zero"], expected_zero)


def test_conflicting_spin_payloads_under_one_resource_id_are_rejected():
    matrices = _matrices()
    original = WickContractionGraph(1.0, (
        _node("original", "matrix:A", "loop", "loop", resource_id="alias"),
    ))
    conflicting = WickContractionGraph(1.0, (
        _node("conflicting", "matrix:B", "otherloop", "otherloop", resource_id="alias"),
    ))
    assert abs(_direct(original, matrices) - _direct(conflicting, matrices)) > 0.1

    with pytest.raises(ValueError, match="incompatible.*payload"):
        build_global_cse({"original": (original,), "conflicting": (conflicting,)})
