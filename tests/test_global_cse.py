"""Focused tests for global resource, component, and complete-term CSE."""

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
    PrimitiveTensorNode,
    TensorLegBinding,
    WickContractionGraph,
    aggregate_global_wick_terms,
    build_global_cse,
    lower_physics_ir_collection_to_global_cse,
    lower_physics_ir_to_contraction_graphs,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_ir import SpinTensorRecord
from concord_lqcd.compiler.distillation_lowering import identify_kernel_resource
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import (
    Correlator,
    Operator,
    c,
    cbar,
    gamma5,
    gammaX,
    point,
    reset_registry,
    s,
    sbar,
    u,
    ubar,
)
from concord_lqcd.core.wick import wick_contract


def _physics_ir(correlator):
    return lower_to_physics_ir(correlator, wick_contract(correlator))


def _single_graph(correlator):
    results = lower_physics_ir_to_contraction_graphs(_physics_ir(correlator))
    assert len(results) == 1
    return results[0].graph


def _meson_two_point(sink_gamma=gamma5):
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * sink_gamma("s1", "s2")
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


def _disconnected_three_point(hadron_sink_gamma):
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * hadron_sink_gamma("s1", "s2")
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


def _rename_dummy_indices(graph, prefix):
    nodes = []
    for node in reversed(graph.nodes):
        nodes.append(PrimitiveTensorNode(
            node_id=f"renamed_{node.node_id}",
            kind=node.kind,
            resource_id=node.resource_id,
            bindings=tuple(
                TensorLegBinding(binding.port, f"{prefix}{binding.index}")
                for binding in node.bindings
            ),
            payload=node.payload,
        ))
    renamed = WickContractionGraph(coefficient=graph.coefficient, nodes=tuple(nodes))
    renamed.validate()
    return renamed


def test_global_cse_reuses_identical_connected_graph_across_correlators():
    graph = _single_graph(_meson_two_point())
    dag = build_global_cse({"C00": (graph,), "C11": (graph,)})

    assert dag.statistics.component_occurrences == 2
    assert dag.statistics.unique_components == 1
    assert dag.statistics.component_reuses == 1
    assert len(dag.correlators) == 2
    assert (
        dag.correlators[0].terms[0].components[0].computation_id
        == dag.correlators[1].terms[0].components[0].computation_id
    )


def test_component_canonicalization_ignores_node_ids_order_and_dummy_indices():
    graph = _single_graph(_meson_two_point())
    renamed = _rename_dummy_indices(graph, "alpha:")

    dag = build_global_cse({"original": (graph,), "renamed": (renamed,)})

    assert dag.statistics.unique_components == 1
    assert dag.statistics.component_reuses == 1


def test_global_cse_preserves_coefficients_while_reusing_computation():
    graph = _single_graph(_meson_two_point())
    scaled = WickContractionGraph(coefficient=-3.0 + 0.25j, nodes=graph.nodes)

    dag = build_global_cse({"A": (graph,), "B": (scaled,)})

    assert dag.statistics.unique_components == 1
    assert dag.correlators[0].terms[0].coefficient == graph.coefficient
    assert dag.correlators[1].terms[0].coefficient == -3.0 + 0.25j


def test_identical_complete_terms_are_combined_before_contraction_ordering():
    graph = _single_graph(_meson_two_point())
    scaled = WickContractionGraph(coefficient=-3.0 + 0.25j, nodes=graph.nodes)

    dag = build_global_cse({"C": (graph, scaled)})

    assert len(dag.correlators[0].terms) == 1
    assert dag.correlators[0].terms[0].term_index == 0
    assert dag.correlators[0].terms[0].coefficient == graph.coefficient - 3.0 + 0.25j


def test_exact_complete_term_cancellation_prunes_dead_work():
    graph = _single_graph(_meson_two_point())
    opposite = WickContractionGraph(coefficient=-graph.coefficient, nodes=graph.nodes)

    dag = build_global_cse({"C": (graph, opposite)})

    assert dag.correlators[0].terms == ()
    assert dag.components == {}
    assert dag.resources == {}
    assert dag.statistics.unique_components == 0
    assert dag.statistics.unique_resources == 0

    order = optimize_global_contraction_order(
        dag,
        ContractionSizeModel(mode_size=2),
    )
    assert order.component_plans == {}


def test_complete_term_aggregation_uses_exact_not_tolerant_zero():
    graph = _single_graph(_meson_two_point())
    near_opposite = WickContractionGraph(
        coefficient=-graph.coefficient + 1.0e-12,
        nodes=graph.nodes,
    )

    dag = build_global_cse({"C": (graph, near_opposite)})

    assert len(dag.correlators[0].terms) == 1
    expected = graph.coefficient + near_opposite.coefficient
    assert expected != 0
    assert dag.correlators[0].terms[0].coefficient == expected


def test_complete_term_aggregation_preserves_component_multiplicity():
    resource = GlobalTensorResource("R", "spin", object())
    tensor = ComponentTensorUse(
        slot="t0",
        kind="spin",
        resource_id="R",
        placement=("spin_op", 0),
        bindings=(),
    )
    component = ComponentComputation("component", "signature", (tensor,), ())
    use = ComponentUse("component")
    dag = GlobalComputationDAG(
        resources={"R": resource},
        components={"component": component},
        correlators=(GlobalCorrelatorUse("C", (
            GlobalWickTermUse(4, 2.0, (use,)),
            GlobalWickTermUse(9, 3.0, (use, use)),
        )),),
        statistics=GlobalCSEStatistics(2, 1, 2, 1),
    )

    aggregated = aggregate_global_wick_terms(dag)

    assert tuple(term.term_index for term in aggregated.correlators[0].terms) == (0, 1)
    assert tuple(term.coefficient for term in aggregated.correlators[0].terms) == (2.0, 3.0)


def test_complete_term_aggregation_treats_component_product_as_commutative():
    resources = {
        resource_id: GlobalTensorResource(resource_id, "spin", object())
        for resource_id in ("R_A", "R_B")
    }
    components = {
        name: ComponentComputation(
            name,
            f"signature_{name}",
            (ComponentTensorUse(
                slot="t0",
                kind="spin",
                resource_id=resource_id,
                placement=("spin_op", operator_id),
                bindings=(),
            ),),
            (),
        )
        for name, resource_id, operator_id in (
            ("component_a", "R_A", 0),
            ("component_b", "R_B", 1),
        )
    }
    use_a = ComponentUse("component_a")
    use_b = ComponentUse("component_b")
    dag = GlobalComputationDAG(
        resources=resources,
        components=components,
        correlators=(GlobalCorrelatorUse("C", (
            GlobalWickTermUse(0, 2.0, (use_b, use_a)),
            GlobalWickTermUse(1, 3.0, (use_a, use_b)),
        )),),
        statistics=GlobalCSEStatistics(4, 2, 4, 2),
    )

    aggregated = aggregate_global_wick_terms(dag)

    assert len(aggregated.correlators[0].terms) == 1
    assert aggregated.correlators[0].terms[0].coefficient == 5.0
    assert aggregated.correlators[0].terms[0].components == (use_a, use_b)


def test_complete_term_aggregation_preserves_external_output_bindings():
    resource = GlobalTensorResource("R", "spin", object())
    tensor = ComponentTensorUse(
        slot="t0",
        kind="spin",
        resource_id="R",
        placement=("spin_op", 0),
        bindings=(TensorLegBinding("left", "o0"),),
    )
    component = ComponentComputation("component", "signature", (tensor,), ("o0",))
    dag = GlobalComputationDAG(
        resources={"R": resource},
        components={"component": component},
        correlators=(GlobalCorrelatorUse("C", (
            GlobalWickTermUse(
                0,
                1.0,
                (ComponentUse(
                    "component", (TensorLegBinding("o0", "external_a"),)
                ),),
            ),
            GlobalWickTermUse(
                1,
                -1.0,
                (ComponentUse(
                    "component", (TensorLegBinding("o0", "external_b"),)
                ),),
            ),
        )),),
        statistics=GlobalCSEStatistics(2, 1, 2, 1),
    )

    aggregated = aggregate_global_wick_terms(dag)

    assert len(aggregated.correlators[0].terms) == 2


def test_open_index_exact_cancellation_retains_zero_term_interface():
    resource = GlobalTensorResource("R", "spin", object())
    tensor = ComponentTensorUse(
        slot="t0",
        kind="spin",
        resource_id="R",
        placement=("spin_op", 0),
        bindings=(TensorLegBinding("left", "o0"),),
    )
    component = ComponentComputation("component", "signature", (tensor,), ("o0",))
    use = ComponentUse("component", (TensorLegBinding("o0", "external"),))
    dag = GlobalComputationDAG(
        resources={"R": resource},
        components={"component": component},
        correlators=(GlobalCorrelatorUse("C", (
            GlobalWickTermUse(0, 1.0, (use,)),
            GlobalWickTermUse(1, -1.0, (use,)),
        )),),
        statistics=GlobalCSEStatistics(2, 1, 2, 1),
    )

    aggregated = aggregate_global_wick_terms(dag)

    assert len(aggregated.correlators[0].terms) == 1
    assert aggregated.correlators[0].terms[0].coefficient == 0
    assert aggregated.correlators[0].terms[0].components == (use,)
    assert set(aggregated.components) == {"component"}
    assert set(aggregated.resources) == {"R"}


def test_complete_term_aggregation_rejects_missing_component():
    dag = GlobalComputationDAG(
        resources={},
        components={},
        correlators=(GlobalCorrelatorUse("C", (
            GlobalWickTermUse(0, 1.0, (ComponentUse("missing"),)),
        )),),
        statistics=GlobalCSEStatistics(0, 0, 1, 0),
    )

    with pytest.raises(ValueError, match="missing from the global DAG"):
        aggregate_global_wick_terms(dag)


def test_complete_term_cancellation_cannot_hide_a_missing_component():
    missing_use = ComponentUse("missing")
    dag = GlobalComputationDAG(
        resources={},
        components={},
        correlators=(GlobalCorrelatorUse("C", (
            GlobalWickTermUse(0, 1.0, (missing_use,)),
            GlobalWickTermUse(1, -1.0, (missing_use,)),
        )),),
        statistics=GlobalCSEStatistics(0, 0, 2, 0),
    )

    with pytest.raises(ValueError, match="missing from the global DAG"):
        aggregate_global_wick_terms(dag)


def test_complete_term_aggregation_rejects_missing_resource():
    tensor = ComponentTensorUse(
        slot="t0",
        kind="spin",
        resource_id="missing_resource",
        placement=("spin_op", 0),
        bindings=(),
    )
    dag = GlobalComputationDAG(
        resources={},
        components={
            "component": ComponentComputation(
                "component", "signature", (tensor,), ()
            )
        },
        correlators=(GlobalCorrelatorUse("C", (
            GlobalWickTermUse(0, 1.0, (ComponentUse("component"),)),
        )),),
        statistics=GlobalCSEStatistics(1, 0, 1, 1),
    )

    with pytest.raises(ValueError, match="resources missing from the global DAG"):
        aggregate_global_wick_terms(dag)


def test_global_resource_registry_deduplicates_without_merging_occurrences():
    graph = _single_graph(_meson_two_point())
    dag = build_global_cse({"C": (graph,)})

    assert dag.statistics.resource_occurrences == len(graph.nodes)
    assert dag.statistics.unique_resources < dag.statistics.resource_occurrences

    component = next(iter(dag.components.values()))
    assert len(component.tensors) == len(graph.nodes)
    assert len(component.resource_ids) == len(graph.nodes)


@pytest.mark.parametrize("kind", ("kernel", "propagation"))
def test_global_resource_registry_rejects_conflicting_payloads(kind):
    graph = _single_graph(_meson_two_point())
    original = next(node for node in graph.nodes if node.kind == kind)
    if kind == "kernel":
        momenta = dict(original.payload.momenta)
        momenta[original.payload.position_labels[0]] = (1, 0, 0)
        changed_payload = replace(original.payload, momenta=momenta)
        assert identify_kernel_resource(changed_payload) != original.resource_id
    else:
        changed_payload = replace(original.payload, segment_flavors=("s",))
        assert changed_payload.resource_id != original.resource_id
    conflicting = replace(original, payload=changed_payload)
    changed_graph = replace(graph, nodes=tuple(
        conflicting if node is original else node for node in graph.nodes
    ))

    with pytest.raises(ValueError, match="incompatible.*payload"):
        build_global_cse({"original": (graph,), "conflicting": (changed_graph,)})


def test_ordinary_resource_reuse_excludes_boundary_kernel_details():
    graph = _single_graph(_meson_two_point())
    original = next(node for node in graph.nodes if node.kind == "propagation")
    boundary = original.payload.start_kernel
    momenta = dict(boundary.momenta)
    momenta[boundary.position_labels[0]] = (1, 0, 0)
    changed_payload = replace(original.payload, start_kernel=replace(boundary, momenta=momenta))
    assert changed_payload.resource_signature == original.payload.resource_signature
    changed_graph = replace(graph, nodes=tuple(
        replace(node, payload=changed_payload) if node is original else node
        for node in graph.nodes
    ))

    dag = build_global_cse({"original": (graph,), "same_tau": (changed_graph,)})

    assert dag.statistics.unique_components == 1


def test_generalized_resource_registry_checks_inserted_spin_structure():
    reset_registry()
    sink = Operator(cbar("y", "a", "s1") * gamma5("s1", "s2") * s("y", "a", "s2"))
    current = Operator(
        point(sbar("z", "b", "s3")) * gammaX("s3", "s4") * point(u("z", "b", "s4")),
        momentum=(1, 0, 0),
    )
    source = Operator(ubar("x", "c", "s5") * gamma5("s5", "s6") * c("x", "c", "s6"))
    graph = _single_graph(Correlator(sink, current, source))
    original = next(
        node for node in graph.nodes
        if node.kind == "propagation" and node.payload.is_generalized
    )
    insertion = original.payload.insertion_specs[0]
    changed_insertion = replace(insertion, spin_tensors=tuple(
        replace(tensor, kind="gamma5") for tensor in insertion.spin_tensors
    ))
    changed_payload = replace(original.payload, insertion_specs=(changed_insertion,))
    assert changed_payload.resource_signature != original.payload.resource_signature
    changed_graph = replace(graph, nodes=tuple(
        replace(node, payload=changed_payload) if node is original else node
        for node in graph.nodes
    ))

    with pytest.raises(ValueError, match="incompatible.*payload"):
        build_global_cse({"original": (graph,), "conflicting": (changed_graph,)})


def test_different_spin_resource_prevents_false_component_cse():
    gamma5_graph = _single_graph(_meson_two_point(gamma5))
    gamma_x_graph = _single_graph(_meson_two_point(gammaX))

    dag = build_global_cse({"g5": (gamma5_graph,), "gx": (gamma_x_graph,)})

    assert dag.statistics.component_occurrences == 2
    assert dag.statistics.unique_components == 2
    assert dag.statistics.component_reuses == 0


def test_repeated_disconnected_loop_is_reused_across_different_hadron_graphs():
    first = _single_graph(_disconnected_three_point(gamma5))
    second = _single_graph(_disconnected_three_point(gammaX))

    dag = build_global_cse({"A": (first,), "B": (second,)})

    # Each graph has a hadronic component and a local strange-loop component.
    # Changing only the hadron sink gamma changes the hadronic component, while
    # the disconnected strange loop is exactly reusable.
    assert dag.statistics.component_occurrences == 4
    assert dag.statistics.unique_components == 3
    assert dag.statistics.component_reuses == 1


def test_placement_is_part_of_component_identity_not_resource_identity():
    node_op0 = PrimitiveTensorNode(
        node_id="s0",
        kind="spin",
        resource_id="spin_tensor_shared",
        bindings=(TensorLegBinding("left", "a"), TensorLegBinding("right", "b")),
        payload=SpinTensorRecord("gamma5", "a", "b", operator_id=0),
    )
    node_op1 = PrimitiveTensorNode(
        node_id="s1",
        kind="spin",
        resource_id="spin_tensor_shared",
        bindings=(TensorLegBinding("left", "c"), TensorLegBinding("right", "d")),
        payload=SpinTensorRecord("gamma5", "c", "d", operator_id=1),
    )
    graph0 = WickContractionGraph(1.0, (node_op0,))
    graph1 = WickContractionGraph(1.0, (node_op1,))

    dag = build_global_cse({"op0": (graph0,), "op1": (graph1,)})

    assert dag.statistics.unique_resources == 1
    assert dag.statistics.unique_components == 2


def test_graph_arity_prevents_reuse_across_different_time_slot_conventions():
    shared_op1_a = PrimitiveTensorNode(
        node_id="a",
        kind="spin",
        resource_id="spin_tensor_shared",
        bindings=(TensorLegBinding("left", "a1"), TensorLegBinding("right", "a2")),
        payload=SpinTensorRecord("gamma5", "a1", "a2", operator_id=1),
    )
    graph_two_point = WickContractionGraph(1.0, (shared_op1_a,))

    shared_op1_b = PrimitiveTensorNode(
        node_id="b",
        kind="spin",
        resource_id="spin_tensor_shared",
        bindings=(TensorLegBinding("left", "b1"), TensorLegBinding("right", "b2")),
        payload=SpinTensorRecord("gamma5", "b1", "b2", operator_id=1),
    )
    extra_op2 = PrimitiveTensorNode(
        node_id="c",
        kind="spin",
        resource_id="spin_tensor_extra",
        bindings=(TensorLegBinding("left", "c1"), TensorLegBinding("right", "c2")),
        payload=SpinTensorRecord("gammaX", "c1", "c2", operator_id=2),
    )
    graph_three_point = WickContractionGraph(1.0, (shared_op1_b, extra_op2))

    dag = build_global_cse({"two": (graph_two_point,), "three": (graph_three_point,)})

    # The op1 tensor is the same resource and same local placement number, but
    # op1 means a different time slot in a 2-point versus 3-point graph.
    assert dag.statistics.component_occurrences == 3
    assert dag.statistics.unique_components == 3


def test_physics_ir_collection_entry_point_reuses_across_cij():
    corr_a = _meson_two_point()
    physics_a = _physics_ir(corr_a)
    corr_b = _meson_two_point()
    physics_b = _physics_ir(corr_b)

    dag = lower_physics_ir_collection_to_global_cse({"C00": physics_a, "C01": physics_b})

    assert dag.statistics.component_occurrences == 2
    assert dag.statistics.unique_components == 1
    assert {corr.correlator_id for corr in dag.correlators} == {"C00", "C01"}
