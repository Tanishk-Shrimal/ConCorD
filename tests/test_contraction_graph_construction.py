"""Focused tests for primitive per-Wick-term contraction graph construction."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime.context import ExecutionContext
from concord_lqcd.backend.runtime.kernels import evaluate_kernel_program
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider
from concord_lqcd.compiler import (
    lower_logical_correlator_to_contraction_graph,
    lower_physics_ir_to_contraction_graphs,
)
from concord_lqcd.compiler.contraction_graph import _canonicalize_kernel_axes
from concord_lqcd.compiler.distillation_lowering import (
    KernelRecipe,
    ProjectedField,
    ProjectionContribution,
    emit_backend_kernel_program,
    identify_kernel_resource,
)
from concord_lqcd.compiler.physics_ir import (
    ColorTensorRecord,
    LogicalCorrelator,
    LogicalQuarkLine,
    OperationKind,
    OperationRecord,
    QuarkRecord,
    SpinTensorRecord,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import (
    Correlator,
    Operator,
    build_baryon_operator,
    c,
    cbar,
    gamma5,
    gammaX,
    point,
    reset_registry,
    s,
    sbar,
    spin_matrix,
    u,
    ubar,
)
from concord_lqcd.core.wick import wick_contract


def _frontend_mesonic_two_point():
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


def _frontend_connected_point_current():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * s("y", "a", "s2"),
        momentum=(0, 0, 0),
    )
    current = Operator(
        point(sbar("z", "b", "s3"))
        * gammaX("s3", "s4")
        * point(u("z", "b", "s4")),
        momentum=(1, 0, 0),
    )
    source = Operator(
        ubar("x", "c", "s5")
        * gamma5("s5", "s6")
        * c("x", "c", "s6"),
        momentum=(-1, 0, 0),
    )
    return Correlator(sink, current, source)


def _frontend_disconnected_current(*, point_current: bool):
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * u("y", "a", "s2"),
    )
    if point_current:
        current_expr = (
            point(sbar("z", "b", "s3"))
            * gammaX("s3", "s4")
            * point(s("z", "b", "s4"))
        )
    else:
        current_expr = (
            sbar("z", "b", "s3")
            * gammaX("s3", "s4")
            * s("z", "b", "s4")
        )
    current = Operator(current_expr, momentum=(1, 0, 0))
    source = Operator(
        ubar("x", "c", "s5")
        * gamma5("s5", "s6")
        * c("x", "c", "s6"),
    )
    return Correlator(sink, current, source)


def _first_graph(correlator):
    terms = wick_contract(correlator)
    assert len(terms) == 1
    physics_ir = lower_to_physics_ir(correlator, terms)
    results = lower_physics_ir_to_contraction_graphs(physics_ir)
    assert len(results) == 1
    return results[0]


def test_exchangeable_kernel_fields_share_resource_with_local_axis_mapping():
    position = "c0:op1:x"
    down = ProjectedField(
        QuarkRecord("d", True, position, "b", "sd", operator_id=1),
        ProjectionContribution.EIGENVECTOR_DAGGER,
    )
    up_a = ProjectedField(
        QuarkRecord("u", True, position, "a", "sua", operator_id=1),
        ProjectionContribution.EIGENVECTOR_DAGGER,
    )
    up_c = ProjectedField(
        QuarkRecord("u", True, position, "c", "suc", operator_id=1),
        ProjectionContribution.EIGENVECTOR_DAGGER,
    )
    epsilon = ColorTensorRecord("epsilon", ("a", "b", "c"), operator_id=1)

    def recipe(fields):
        return KernelRecipe(
            vertex_id=position,
            position_labels=(position,),
            momenta={position: (0, 0, 0)},
            projected_fields=fields,
            color_tensors=(epsilon,),
            open_ports=("mode_0", "mode_1", "mode_2"),
        )

    first = recipe((down, up_c, up_a))
    second = recipe((down, up_a, up_c))
    assert identify_kernel_resource(first) != identify_kernel_resource(second)

    canonical_first, first_axes = _canonicalize_kernel_axes(first)
    canonical_second, second_axes = _canonicalize_kernel_axes(second)

    assert canonical_first == canonical_second
    assert identify_kernel_resource(canonical_first) == identify_kernel_resource(
        canonical_second
    )
    assert first_axes != second_axes
    assert canonical_first.projected_fields == tuple(
        first.projected_fields[index] for index in first_axes
    )
    assert canonical_second.projected_fields == tuple(
        second.projected_fields[index] for index in second_axes
    )


def test_exchangeable_kernel_axis_cse_is_integrated_into_baryon_graphs():
    reset_registry()
    base = build_baryon_operator(
        ("u", "d", "u"),
        diquark_gamma="Cg5",
        momentum=(0, 0, 0),
    )
    operator = spin_matrix("Pplus", "projected_spin", "de") * base
    correlator = Correlator(operator, operator.dagger())
    terms = wick_contract(correlator)
    physics_ir = lower_to_physics_ir(correlator, terms)
    graphs = tuple(
        result.graph
        for result in lower_physics_ir_to_contraction_graphs(physics_ir)
    )

    assert len(graphs) == 2
    assert len({
        node.resource_id
        for graph in graphs
        for node in graph.nodes_of_kind("kernel")
    }) == 2

    source_nodes = tuple(graph.nodes_of_kind("kernel")[1] for graph in graphs)
    assert source_nodes[0].resource_id == source_nodes[1].resource_id
    assert source_nodes[0].payload == source_nodes[1].payload
    assert source_nodes[0].bindings != source_nodes[1].bindings


def test_kernel_axis_canonicalization_reuses_permuted_distinct_operations():
    position = "c0:op1:x"
    derivative = OperationRecord(
        OperationKind.DERIVATIVE,
        {"direction": "+x"},
    )
    first = ProjectedField(
        QuarkRecord("u", True, position, "a", "s0", operator_id=1),
        ProjectionContribution.EIGENVECTOR_DAGGER,
        (derivative,),
    )
    second = ProjectedField(
        QuarkRecord("u", True, position, "b", "s1", operator_id=1),
        ProjectionContribution.EIGENVECTOR_DAGGER,
    )
    recipe = KernelRecipe(
        vertex_id=position,
        position_labels=(position,),
        momenta={position: (0, 0, 0)},
        projected_fields=(second, first),
        open_ports=("mode_0", "mode_1"),
    )

    reversed_recipe = replace(recipe, projected_fields=tuple(reversed(recipe.projected_fields)))
    canonical, axes = _canonicalize_kernel_axes(recipe)
    reversed_canonical, reversed_axes = _canonicalize_kernel_axes(reversed_recipe)

    assert identify_kernel_resource(canonical) == identify_kernel_resource(reversed_canonical)
    assert canonical.projected_fields == tuple(recipe.projected_fields[index] for index in axes)
    assert reversed_canonical.projected_fields == tuple(
        reversed_recipe.projected_fields[index] for index in reversed_axes
    )
    assert axes != reversed_axes
    without_derivative = replace(first, operations=())
    different = replace(recipe, projected_fields=(second, without_derivative))
    assert identify_kernel_resource(_canonicalize_kernel_axes(different)[0]) != (
        identify_kernel_resource(canonical)
    )


def test_kernel_axis_canonicalization_reuses_permuted_distinct_flavors():
    position = "c0:op1:x"
    fields = (
        ProjectedField(
            QuarkRecord("u", False, position, "a", "su", operator_id=1),
            ProjectionContribution.EIGENVECTOR,
        ),
        ProjectedField(
            QuarkRecord("d", True, position, "a", "sd", operator_id=1),
            ProjectionContribution.EIGENVECTOR_DAGGER,
        ),
    )
    recipe = KernelRecipe(
        vertex_id=position,
        position_labels=(position,),
        momenta={position: (1, 0, 0)},
        projected_fields=fields,
        open_ports=("mode_0", "mode_1"),
    )
    swapped = replace(recipe, projected_fields=tuple(reversed(fields)))
    canonical, axes = _canonicalize_kernel_axes(recipe)
    swapped_canonical, swapped_axes = _canonicalize_kernel_axes(swapped)

    assert identify_kernel_resource(canonical) == identify_kernel_resource(swapped_canonical)
    assert axes != swapped_axes
    assert canonical.projected_fields == tuple(fields[index] for index in axes)
    assert swapped_canonical.projected_fields == tuple(
        swapped.projected_fields[index] for index in swapped_axes
    )
    changed_flavor = replace(
        recipe,
        projected_fields=(
            replace(fields[0], record=replace(fields[0].record, flavor="d")),
            fields[1],
        ),
    )
    assert identify_kernel_resource(_canonicalize_kernel_axes(changed_flavor)[0]) != (
        identify_kernel_resource(canonical)
    )


def test_kernel_axis_permutation_preserves_epsilon_sign():
    position = "c0:op0:x"
    fields = tuple(
        ProjectedField(
            QuarkRecord(flavor, False, position, color, spin, operator_id=0),
            ProjectionContribution.EIGENVECTOR,
        )
        for flavor, color, spin in (
            ("u", "a", "su"), ("d", "b", "sd"), ("s", "c", "ss")
        )
    )
    recipe = KernelRecipe(
        vertex_id=position,
        position_labels=(position,),
        momenta={position: (0, 0, 0)},
        projected_fields=fields,
        color_tensors=(ColorTensorRecord("epsilon", ("a", "b", "c"), operator_id=0),),
        open_ports=("mode_0", "mode_1", "mode_2"),
    )
    order = (2, 0, 1)
    reordered = replace(recipe, projected_fields=tuple(fields[index] for index in order))
    opposite_sign = replace(
        recipe,
        color_tensors=(ColorTensorRecord("epsilon", ("b", "a", "c"), operator_id=0),),
    )
    canonical = _canonicalize_kernel_axes(recipe)[0]
    assert identify_kernel_resource(canonical) == identify_kernel_resource(
        _canonicalize_kernel_axes(reordered)[0]
    )
    assert identify_kernel_resource(canonical) != identify_kernel_resource(
        _canonicalize_kernel_axes(opposite_sign)[0]
    )

    generator = np.random.default_rng(20260929)
    vectors = {
        flavor: torch.from_numpy(
            generator.normal(size=(1, 2, 3, 2))
            + 1j * generator.normal(size=(1, 2, 3, 2))
        )
        for flavor in ("u", "d", "s")
    }
    provider = DistillationTensorProvider(vectors, vectors, {})
    context = ExecutionContext.create(
        device=torch.device("cpu"),
        t_src=0,
        t_snk_list=(0,),
        dtype=torch.float64,
        complex_dtype=torch.complex128,
    )

    def evaluate(value):
        return evaluate_kernel_program(emit_backend_kernel_program(value), provider, context)

    original = evaluate(recipe)
    torch.testing.assert_close(evaluate(reordered), original.permute(order), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(evaluate(opposite_sign), -original, rtol=1e-13, atol=1e-13)


def test_kernel_axis_canonicalization_keeps_nonstandard_ports_unchanged():
    position = "c0:op1:x"
    fields = tuple(
        ProjectedField(
            QuarkRecord("u", True, position, color, spin, operator_id=1),
            ProjectionContribution.EIGENVECTOR_DAGGER,
        )
        for color, spin in (("a", "s0"), ("b", "s1"), ("c", "s2"))
    )
    recipe = KernelRecipe(
        vertex_id=position,
        position_labels=(position,),
        momenta={position: (0, 0, 0)},
        projected_fields=fields,
        open_ports=("left", "middle", "right"),
    )

    canonical, axes = _canonicalize_kernel_axes(recipe)

    assert canonical == recipe
    assert axes == (0, 1, 2)


def test_kernel_axis_canonicalization_respects_permutation_budget():
    position = "c0:op1:x"
    fields = tuple(
        ProjectedField(
            QuarkRecord(
                "u",
                True,
                position,
                f"c{index}",
                f"s{index}",
                operator_id=1,
            ),
            ProjectionContribution.EIGENVECTOR_DAGGER,
        )
        for index in range(7)
    )
    recipe = KernelRecipe(
        vertex_id=position,
        position_labels=(position,),
        momenta={position: (0, 0, 0)},
        projected_fields=fields,
        open_ports=tuple(f"mode_{index}" for index in range(7)),
    )

    canonical, axes = _canonicalize_kernel_axes(recipe)

    assert canonical == recipe
    assert axes == tuple(range(7))


def test_mesonic_two_point_becomes_one_scalar_primitive_tensor_graph():
    result = _first_graph(_frontend_mesonic_two_point())
    graph = result.graph

    assert result.topology.num_open_paths == 0
    assert result.topology.num_closed_cycles == 1
    assert len(graph.nodes_of_kind("kernel")) == 2
    assert len(graph.nodes_of_kind("propagation")) == 2
    assert len(graph.nodes_of_kind("spin")) == 2
    assert {node.payload.flavor for node in graph.nodes_of_kind("propagation")} == {"u", "c"}
    assert all(not node.payload.is_generalized for node in graph.nodes_of_kind("propagation"))
    assert graph.free_indices == ()
    assert graph.is_closed_scalar


def test_connected_point_current_is_absorbed_once_inside_generalized_propagation():
    result = _first_graph(_frontend_connected_point_current())
    graph = result.graph

    # The full mesonic fermion line is a cycle after source/sink closure, but
    # the POINT current itself belongs inside one generalized propagation edge.
    assert result.topology.num_closed_cycles == 1

    kernel_nodes = graph.nodes_of_kind("kernel")
    propagation_nodes = graph.nodes_of_kind("propagation")
    spin_nodes = graph.nodes_of_kind("spin")

    assert len(kernel_nodes) == 2  # sink + source; POINT current kernel is absorbed
    assert len(propagation_nodes) == 2  # active generalized path + spectator
    assert len(spin_nodes) == 2  # sink/source gammas; current gamma is absorbed

    generalized = [node.payload for node in propagation_nodes if node.payload.is_generalized]
    ordinary = [node.payload for node in propagation_nodes if not node.payload.is_generalized]
    assert len(generalized) == 1
    assert len(ordinary) == 1
    assert generalized[0].segment_flavors == ("u", "s")
    assert generalized[0].resource_kind == "generalized_perambulator"
    assert [tensor.kind for tensor in generalized[0].insertion_specs[0].spin_tensors] == [
        "gammaX"
    ]
    assert graph.free_indices == ()
    assert graph.is_closed_scalar


def test_laph_disconnected_current_is_explicit_kernel_spin_and_ordinary_tau_cycle():
    result = _first_graph(_frontend_disconnected_current(point_current=False))
    graph = result.graph

    assert result.topology.num_closed_cycles == 2
    assert any(cycle.is_local for cycle in result.topology.closed_cycles)

    assert len(graph.nodes_of_kind("kernel")) == 3
    assert len(graph.nodes_of_kind("propagation")) == 3
    assert len(graph.nodes_of_kind("spin")) == 3
    assert all(
        not node.payload.is_generalized
        for node in graph.nodes_of_kind("propagation")
    )
    assert {node.payload.flavor for node in graph.nodes_of_kind("propagation")} == {
        "u", "c", "s"
    }
    assert {node.payload.kind for node in graph.nodes_of_kind("spin")} == {
        "gamma5", "gammaX"
    }
    assert graph.free_indices == ()
    assert graph.is_closed_scalar


def test_genuine_point_disconnected_current_is_rejected_from_exact_tau_graph():
    correlator = _frontend_disconnected_current(point_current=True)
    physics_ir = lower_to_physics_ir(correlator, wick_contract(correlator))

    with pytest.raises(NotImplementedError, match="POINT.*disconnected|POINT transitions"):
        lower_physics_ir_to_contraction_graphs(physics_ir)


def test_same_resource_id_does_not_merge_distinct_propagation_occurrences():
    forward = LogicalQuarkLine(
        quark=QuarkRecord("u", False, "c0:op0:y", "b0", "q0", operator_id=0),
        antiquark=QuarkRecord("u", True, "c0:op1:x", "a1", "a1s", operator_id=1),
    )
    backward = LogicalQuarkLine(
        quark=QuarkRecord("u", False, "c0:op1:x", "b1", "q1", operator_id=1),
        antiquark=QuarkRecord("u", True, "c0:op0:y", "a0", "a0s", operator_id=0),
    )
    logical = LogicalCorrelator(
        coefficient=-2.0 + 0.5j,
        quark_lines=(forward, backward),
        spin_tensors=(
            SpinTensorRecord("gamma5", "a0s", "q0", operator_id=0),
            SpinTensorRecord("gamma5", "a1s", "q1", operator_id=1),
        ),
    )

    result = lower_logical_correlator_to_contraction_graph(logical)
    graph = result.graph
    propagations = graph.nodes_of_kind("propagation")

    assert graph.coefficient == -2.0 + 0.5j
    assert len(propagations) == 2
    assert propagations[0].node_id != propagations[1].node_id
    assert propagations[0].resource_id == propagations[1].resource_id
    assert graph.is_closed_scalar


def test_physics_ir_produces_exactly_one_graph_per_wick_term():
    correlator = _frontend_mesonic_two_point()
    terms = wick_contract(correlator)
    physics_ir = lower_to_physics_ir(correlator, terms)

    results = lower_physics_ir_to_contraction_graphs(physics_ir)

    assert len(results) == len(physics_ir.correlators) == len(terms)
    assert [result.graph.coefficient for result in results] == [
        logical.coefficient for logical in physics_ir.correlators
    ]
