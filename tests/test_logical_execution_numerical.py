"""Independent numerical checks of complete Wick-term/correlator assembly."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from concord_lqcd.compiler.contraction_graph import PrimitiveTensorNode, TensorLegBinding, WickContractionGraph
from concord_lqcd.compiler.contraction_order import ContractionSizeModel, optimize_global_contraction_order
from concord_lqcd.compiler.global_cse import build_global_cse
from concord_lqcd.compiler.logical_execution import (
    LogicalContraction,
    LogicalCorrelatorSum,
    LogicalPrimitiveEvaluation,
    LogicalWickTermProduct,
    build_logical_execution_dag,
)
from concord_lqcd.compiler.partial_cse import build_partial_intermediate_cse
from concord_lqcd.compiler.physics_ir import SpinTensorRecord


def _matrices():
    generator = np.random.default_rng(20261008)
    return {
        (kind, placement): generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4))
        for kind in ("matrix:A", "matrix:B", "matrix:C", "matrix:D")
        for placement in (0, 1)
    }


def _node(name, kind, left, right, *, placement=0):
    return PrimitiveTensorNode(
        name, "spin", kind,
        (TensorLegBinding("left", left), TensorLegBinding("right", right)),
        SpinTensorRecord(kind, left, right, placement),
    )


def _loop(prefix, last="matrix:C", placement=0):
    return (
        _node(prefix + "A", "matrix:A", prefix + "a", prefix + "b", placement=placement),
        _node(prefix + "B", "matrix:B", prefix + "b", prefix + "c", placement=placement),
        _node(prefix + "C", last, prefix + "c", prefix + "a", placement=placement),
    )


def _open(prefix, outputs=("external_a", "external_b"), placement=0):
    return (
        _node(prefix + "A", "matrix:A", outputs[0], prefix + "dummy", placement=placement),
        _node(prefix + "B", "matrix:B", prefix + "dummy", outputs[1], placement=placement),
    )


def _contract(arrays, indices, outputs):
    if not arrays:
        assert not outputs
        return np.asarray(1.0 + 0j)
    labels = sorted({label for legs in indices for label in legs})
    numbers = {label: number for number, label in enumerate(labels)}
    arguments = []
    for array, legs in zip(arrays, indices, strict=True):
        arguments.extend((array, [numbers[label] for label in legs]))
    arguments.append([numbers[label] for label in outputs])
    return np.einsum(*arguments, optimize=False)


def _direct(collection, matrices):
    expected = {}
    for name, graphs in collection.items():
        terms = []
        for graph in graphs:
            indices = [node.legs for node in graph.nodes]
            counts = Counter(label for legs in indices for label in legs)
            outputs = tuple(sorted(label for label, count in counts.items() if count == 1))
            arrays = [matrices[(node.payload.kind, node.payload.operator_id)] for node in graph.nodes]
            terms.append(graph.coefficient * _contract(arrays, indices, outputs))
        expected[name] = sum(terms, np.asarray(0.0 + 0j))
    return expected


def _partial(collection):
    global_dag = build_global_cse(collection)
    order = optimize_global_contraction_order(global_dag, ContractionSizeModel(3), strategy="greedy")
    return build_partial_intermediate_cse(order)


def _evaluate(logical, matrices):
    values = {}
    ports = {}
    counts = Counter()

    def inputs(uses):
        arrays = []
        indices = []
        for use in uses:
            bindings = {binding.port: binding.index for binding in use.bindings}
            arrays.append(values[use.value_id])
            indices.append(tuple(bindings[port] for port in ports[use.value_id]))
        return arrays, indices

    for node_id in logical.topological_order():
        node = logical.node(node_id)
        if isinstance(node, LogicalPrimitiveEvaluation):
            record = logical.source_dag.source_plan.source_dag.resources[node.resource_id].payload
            operator_id = node.placement[2][1]
            values[node_id] = matrices[(record.kind, operator_id)]
            ports[node_id] = node.output_ports
        elif isinstance(node, LogicalContraction):
            arrays, indices = inputs(node.inputs)
            values[node_id] = _contract(arrays, indices, node.output_indices)
            ports[node_id] = node.output_indices
        elif isinstance(node, LogicalWickTermProduct):
            arrays, indices = inputs(node.inputs)
            values[node_id] = node.coefficient * _contract(arrays, indices, node.output_indices)
            ports[node_id] = node.output_indices
        elif isinstance(node, LogicalCorrelatorSum):
            values[node_id] = sum((values[term_id] for term_id in node.term_ids), np.asarray(0.0 + 0j))
            ports[node_id] = node.output_indices
        else:
            raise AssertionError(type(node))
        counts[node_id] += 1
    assert all(count == 1 for count in counts.values())
    assert set(counts) == set(logical.node_ids)
    return {name: values[node_id] for name, node_id in logical.outputs.items()}


def _assert_results(actual, expected, *, zeros=()):
    assert set(actual) == set(expected)
    for name, reference in expected.items():
        if name in zeros:
            np.testing.assert_array_equal(actual[name], reference)
        else:
            scale = np.max(np.abs(reference))
            assert scale > 0.1
            np.testing.assert_allclose(actual[name], reference, rtol=1e-12, atol=1e-12)
            assert np.max(np.abs(actual[name] - reference)) / scale < 1e-13


def _case():
    matrices = _matrices()
    collection = {
        "scalar": (WickContractionGraph(2 - 3j, _loop("scalar:")),),
        "matrix": (WickContractionGraph(1 + 0.5j, _open("open:")),),
    }
    return build_logical_execution_dag(_partial(collection)), matrices


def _replace_value_reference(logical, old, new):
    def uses(inputs):
        return tuple(replace(use, value_id=new) if use.value_id == old else use for use in inputs)

    return replace(
        logical,
        contractions={key: replace(node, inputs=uses(node.inputs)) for key, node in logical.contractions.items()},
        terms=tuple(replace(term, inputs=uses(term.inputs)) for term in logical.terms),
    )


def test_complex_coefficients_and_repeated_disconnected_factors_survive_assembly():
    matrices = _matrices()
    collection = {
        "cubed": (WickContractionGraph(2 - 3j, _loop("first:") + _loop("second:") + _loop("third:")),),
        "mixed": (
            WickContractionGraph(-0.5 + 2j, _loop("mixed:") + _loop("other:", last="matrix:D")),
            WickContractionGraph(0.25 - 0.75j, _loop("last:")),
        ),
        "another_channel": (WickContractionGraph(-3 + 4j, _loop("renamed:")),),
    }
    expected = _direct(collection, matrices)
    loop = np.trace(matrices[("matrix:A", 0)] @ matrices[("matrix:B", 0)] @ matrices[("matrix:C", 0)])
    assert abs(expected["cubed"] - (2 - 3j) * loop) > 0.1
    assert abs(expected["cubed"] - (2 + 3j) * loop**3) > 0.1
    original = {key: array.copy() for key, array in matrices.items()}

    logical = build_logical_execution_dag(_partial(collection))

    _assert_results(_evaluate(logical, matrices), expected)
    term = next(term for term in logical.terms if term.correlator_id == "cubed")
    assert len(term.inputs) == 3
    assert len(set(term.dependencies)) == 1
    for key, array in matrices.items():
        np.testing.assert_array_equal(array, original[key])


def test_component_output_maps_preserve_transposes_and_disconnected_outer_products():
    matrices = _matrices()
    collection = {
        "antisymmetric": (
            WickContractionGraph(1, _open("first:")),
            WickContractionGraph(-1, _open("transpose:", outputs=("external_b", "external_a"))),
        ),
        "outer": (WickContractionGraph(0.5 - 0.25j,
            _open("left:", outputs=("a1", "b1")) + _open("right:", outputs=("a2", "b2"))),),
        "matrix_loop": (WickContractionGraph(-2j, _open("matrix:") + _loop("loop:")),),
    }
    expected = _direct(collection, matrices)

    logical = build_logical_execution_dag(_partial(collection))
    actual = _evaluate(logical, matrices)

    _assert_results(actual, expected)
    assert actual["outer"].shape == (4, 4, 4, 4)
    outer = next(term for term in logical.terms if term.correlator_id == "outer")
    assert outer.inputs[0].value_id == outer.inputs[1].value_id
    assert outer.inputs[0].bindings != outer.inputs[1].bindings
    assert np.max(np.abs(expected["antisymmetric"])) > 0.1


def test_empty_products_zero_sums_and_open_zero_terms_keep_their_meaning():
    matrices = _matrices()
    loop = WickContractionGraph(1, _loop("loop:"))
    matrix = WickContractionGraph(1, (_node("matrix", "matrix:A", "external_a", "external_b"),))
    collection = {
        "constant": (WickContractionGraph(2 - 3j, ()),),
        "empty": (),
        "cancelled_scalar": (loop, replace(loop, coefficient=-1)),
        "cancelled_matrix": (matrix, replace(matrix, coefficient=-1)),
        "primitive_result": (replace(matrix, coefficient=0.25 - 0.5j),),
    }
    expected = _direct(collection, matrices)

    logical = build_logical_execution_dag(_partial(collection))
    actual = _evaluate(logical, matrices)

    _assert_results(actual, expected, zeros=("empty", "cancelled_scalar", "cancelled_matrix"))
    assert actual["cancelled_matrix"].shape == (4, 4)
    assert actual["empty"].shape == ()
    assert any(not term.inputs and term.coefficient == 2 - 3j for term in logical.terms)


@pytest.mark.parametrize("seed", range(5))
def test_dummy_renaming_and_node_order_keep_the_complete_output(seed):
    matrices = _matrices()
    generator = np.random.default_rng(seed)
    first = _open("first:") + _loop("loop:", placement=1)
    indices = {index for node in first for index in node.legs} - {"external_a", "external_b"}
    renamed = {index: f"dummy{seed}:{number}" for number, index in enumerate(sorted(indices))}
    nodes = tuple(replace(
        node, node_id=f"trial{seed}:{node.node_id}",
        bindings=tuple(replace(binding, index=renamed.get(binding.index, binding.index)) for binding in node.bindings),
    ) for node in first)
    shuffled = tuple(nodes[number] for number in generator.permutation(len(nodes)))
    collection = {"sum": (WickContractionGraph(1 + 2j, first), WickContractionGraph(-0.5 - 1j, shuffled))}
    expected = _direct(collection, matrices)

    logical = build_logical_execution_dag(_partial(collection))

    _assert_results(_evaluate(logical, matrices), expected)
    assert len(logical.terms) == 1


@pytest.mark.parametrize("kind", ("primitive", "contraction"))
def test_logical_node_map_key_must_match_its_record_identity(kind):
    logical, _ = _case()
    mapping = logical.primitives if kind == "primitive" else logical.contractions
    old = next(iter(mapping))
    new = "wrong_map_key"
    changed = {new if key == old else key: node for key, node in mapping.items()}
    logical = replace(logical, **{("primitives" if kind == "primitive" else "contractions"): changed})
    logical = _replace_value_reference(logical, old, new)

    with pytest.raises(ValueError, match="key.*identity"):
        logical.validate()


@pytest.mark.parametrize("kind", ("primitive", "intermediate", "component_plan"))
def test_logical_builder_rejects_inconsistent_source_record_identity(kind):
    logical, _ = _case()
    partial = logical.source_dag
    if kind == "primitive":
        key = next(iter(partial.primitive_values))
        values = dict(partial.primitive_values)
        values[key] = replace(values[key], value_id="wrong_record")
        partial = replace(partial, primitive_values=values)
    elif kind == "intermediate":
        key = next(iter(partial.intermediates))
        values = dict(partial.intermediates)
        values[key] = replace(values[key], intermediate_id="wrong_record")
        partial = replace(partial, intermediates=values)
    else:
        key = next(iter(partial.component_plans))
        plans = dict(partial.component_plans)
        plans[key] = replace(plans[key], computation_id="wrong_record")
        partial = replace(partial, component_plans=plans)

    with pytest.raises(ValueError, match="identity"):
        build_logical_execution_dag(partial)


@pytest.mark.parametrize("coefficient", (np.inf, -np.inf, np.nan, complex(1, np.inf), complex(np.nan, 2)))
def test_logical_assembly_rejects_nonfinite_wick_coefficients(coefficient):
    logical, _ = _case()
    partial = logical.source_dag
    global_dag = partial.source_plan.source_dag
    first = global_dag.correlators[0]
    changed = replace(first, terms=(replace(first.terms[0], coefficient=coefficient),))
    global_dag = replace(global_dag, correlators=(changed,) + global_dag.correlators[1:])
    partial = replace(partial, source_plan=replace(partial.source_plan, source_dag=global_dag))

    with pytest.raises(ValueError, match="finite"):
        build_logical_execution_dag(partial)


def test_logical_sum_cannot_repeat_a_complete_wick_term():
    logical, _ = _case()
    first = logical.correlators[0]
    changed = replace(first, term_ids=first.term_ids + first.term_ids)
    logical = replace(logical, correlators=(changed,) + logical.correlators[1:])

    with pytest.raises(ValueError, match="Wick term"):
        logical.validate()


def test_logical_sum_cannot_omit_its_wick_terms():
    logical, _ = _case()
    first = logical.correlators[0]
    changed = replace(first, term_ids=())
    logical = replace(logical, correlators=(changed,) + logical.correlators[1:])

    with pytest.raises(ValueError, match="Wick term"):
        logical.validate()


def test_changed_coefficient_cannot_disagree_with_the_attached_source():
    logical, matrices = _case()
    expected = _evaluate(logical, matrices)
    changed = replace(logical.terms[0], coefficient=logical.terms[0].coefficient.conjugate())
    logical = replace(logical, terms=(changed,) + logical.terms[1:])
    assert abs(_evaluate(logical, matrices)["scalar"] - expected["scalar"]) > 0.1

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_changed_output_binding_cannot_disagree_with_the_attached_source():
    logical, matrices = _case()
    expected = _evaluate(logical, matrices)
    position = next(number for number, term in enumerate(logical.terms) if term.correlator_id == "matrix")
    term = logical.terms[position]
    use = term.inputs[0]
    assert len(use.bindings) == 2
    changed = replace(use, bindings=tuple(
        replace(binding, index=use.bindings[1 - number].index)
        for number, binding in enumerate(use.bindings)
    ))
    terms = list(logical.terms)
    terms[position] = replace(term, inputs=(changed,))
    logical = replace(logical, terms=tuple(terms))
    assert np.max(np.abs(_evaluate(logical, matrices)["matrix"] - expected["matrix"])) > 0.1

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_changed_contraction_connectivity_cannot_disagree_with_the_attached_source():
    logical, matrices = _case()
    expected = _evaluate(logical, matrices)
    key, node = next((key, node) for key, node in logical.contractions.items() if len(node.inputs[0].bindings) == 2)
    use = node.inputs[0]
    changed = replace(use, bindings=tuple(
        replace(binding, index=use.bindings[1 - number].index)
        for number, binding in enumerate(use.bindings)
    ))
    nodes = dict(logical.contractions)
    nodes[key] = replace(node, inputs=(changed,) + node.inputs[1:])
    logical = replace(logical, contractions=nodes)
    actual = _evaluate(logical, matrices)
    assert any(np.max(np.abs(actual[name] - expected[name])) > 0.1 for name in expected)

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_tiny_finite_coefficients_are_preserved_without_tolerance_based_elimination():
    matrices = _matrices()
    coefficient = 1e-30 - 2e-30j
    collection = {"tiny": (WickContractionGraph(coefficient, _loop("tiny:")),)}
    expected = _direct(collection, matrices)

    logical = build_logical_execution_dag(_partial(collection))
    actual = _evaluate(logical, matrices)

    assert logical.terms[0].coefficient == coefficient
    assert abs(expected["tiny"]) > 0
    np.testing.assert_allclose(actual["tiny"], expected["tiny"], rtol=1e-13, atol=0)


def test_source_agreement_is_exact_not_a_numerical_tolerance():
    logical, _ = _case()
    original = logical.terms[0]
    logical = replace(logical, terms=(
        replace(original, coefficient=original.coefficient + 1e-12),
    ) + logical.terms[1:])

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_logical_assembly_cannot_omit_a_complete_source_term():
    logical, _ = _case()
    first = logical.terms[0]
    logical = replace(
        logical,
        terms=logical.terms[1:],
        correlators=tuple(replace(
            correlator, term_ids=tuple(term_id for term_id in correlator.term_ids if term_id != first.node_id),
        ) for correlator in logical.correlators),
    )

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_logical_assembly_cannot_omit_a_complete_source_correlator():
    logical, _ = _case()
    first = logical.correlators[0]
    logical = replace(
        logical,
        terms=tuple(term for term in logical.terms if term.correlator_id != first.correlator_id),
        correlators=logical.correlators[1:],
        outputs={name: node_id for name, node_id in logical.outputs.items() if name != first.correlator_id},
    )

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_source_agreement_preserves_correlator_sum_order():
    collection = {"sum": (
        WickContractionGraph(1 + 2j, _loop("first:")),
        WickContractionGraph(2 - 3j, _loop("second:", last="matrix:D")),
    )}
    logical = build_logical_execution_dag(_partial(collection))
    correlator = logical.correlators[0]
    assert len(correlator.term_ids) == 2
    logical = replace(logical, correlators=(replace(correlator, term_ids=correlator.term_ids[::-1]),))

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_changed_primitive_placement_cannot_disagree_with_the_attached_source():
    logical, _ = _case()
    key, node = next(iter(logical.primitives.items()))
    primitives = dict(logical.primitives)
    primitives[key] = replace(node, placement=("operator_arity", 2, ("spin_op", 1)))
    logical = replace(logical, primitives=primitives)

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_changed_contraction_size_cannot_disagree_with_the_attached_source():
    logical, _ = _case()
    key, node = next(iter(logical.contractions.items()))
    contractions = dict(logical.contractions)
    contractions[key] = replace(node, output_elements=node.output_elements + 1)
    logical = replace(logical, contractions=contractions)

    with pytest.raises(ValueError, match="source"):
        logical.validate()


def test_empty_source_correlator_cannot_invent_an_open_output_interface():
    logical = build_logical_execution_dag(_partial({"empty": ()}))
    logical = replace(logical, correlators=(
        replace(logical.correlators[0], output_indices=("invented_axis",)),
    ))

    with pytest.raises(ValueError, match="scalar output"):
        logical.validate()
