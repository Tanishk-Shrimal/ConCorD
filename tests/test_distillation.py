"""Tests for the distillation compiler lowering pipeline."""

import pytest
from concord_lqcd.core import Operator, u, ubar, d, dbar, s, sbar, c, epsilon, gamma5, gammaX, Correlator, CorrMatrix, build_baryon_operator, reset_registry
from concord_lqcd.core.wick import wick_contract
from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.compiler import (
    lower_correlator_to_physics_ir,
    lower_correlator_to_kernel_recipes_and_propagations,
    lower_physics_ir_to_contraction_graphs,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.compiler.distillation_lowering import (
    identify_operator_vertices,
    realize_projected_fields,
    construct_kernel_recipe,
    identify_kernel_resource,
)
def _contraction_graphs(correlator):
    physics_ir = lower_correlator_to_physics_ir(correlator)
    return tuple(
        result.graph
        for result in lower_physics_ir_to_contraction_graphs(physics_ir)
    )


def test_elemental_recipe_canonicalization():
    op1_snk = Operator(dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β'), momenta={'x': (1,0,0)})
    op1_src = Operator(ubar('x', 'a', 'α') * gamma5('α', 'β') * d('x', 'a', 'β'), momenta={'x': (-1,0,0)})

    op2_snk = Operator(dbar('y', 'b', 'γ') * gamma5('γ', 'δ') * u('y', 'b', 'δ'), momenta={'y': (1,0,0)})
    op2_src = Operator(ubar('y', 'b', 'γ') * gamma5('γ', 'δ') * d('y', 'b', 'δ'), momenta={'y': (-1,0,0)})

    corr1 = Correlator(op1_snk, op1_src)
    corr2 = Correlator(op2_snk, op2_src)

    terms1 = wick_contract(corr1)
    terms2 = wick_contract(corr2)

    phys1 = lower_to_physics_ir(corr1, terms1)
    phys2 = lower_to_physics_ir(corr2, terms2)

    v1 = identify_operator_vertices(phys1.correlators[0])[0]
    v2 = identify_operator_vertices(phys2.correlators[0])[0]

    kr1 = construct_kernel_recipe(realize_projected_fields(v1))
    kr2 = construct_kernel_recipe(realize_projected_fields(v2))

    hash1 = identify_kernel_resource(kr1)
    hash2 = identify_kernel_resource(kr2)

    assert hash1 == hash2
    assert kr1.num_quarks == 2
    assert kr2.num_quarks == 2


def test_wick_keeps_explicit_light_flavors_distinct():
    """u and d may share numerical data, but they are distinct Wick flavors."""
    reset_registry()
    dmeson = Operator(
        dbar('x1', 'a1', 'alpha1') * gamma5('alpha1', 'beta1') * c('x1', 'a2', 'beta1'),
        momentum=(0, 0, 0),
    )
    kmeson = Operator(
        sbar('x2', 'b1', 'alpha2') * gamma5('alpha2', 'beta2') * u('x2', 'b2', 'beta2'),
        momentum=(0, 0, 0),
    )

    composite = dmeson * kmeson
    terms = wick_contract(Correlator(composite, composite.dagger()))

    assert len(terms) == 1
    assert all(q.flavor == aq.flavor for q, aq in terms[0].pairing)


def test_compiler_deduplicates_elementals():
    op1 = Operator(dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β'), momenta={'x': (0,0,1)})
    op2 = Operator(dbar('y', 'b', 'γ') * gamma5('γ', 'δ') * u('y', 'b', 'δ'), momenta={'y': (0,0,0)})

    matrix = CorrMatrix([op1, op2])
    compilation = compile_correlators(
        matrix,
        RuntimeCompilationConfig(mode_size=2),
    )

    assert compilation.correlator_ids == ("C[0,0]", "C[0,1]", "C[1,0]", "C[1,1]")
    kernel_occurrences = sum(
        len(result.graph.nodes_of_kind("kernel"))
        for physics_ir in compilation.physics_ir.values()
        for result in lower_physics_ir_to_contraction_graphs(physics_ir)
    )
    kernel_resources = {
        resource_id
        for resource_id, resource in compilation.global_cse.resources.items()
        if resource.kind == "kernel"
    }
    assert len(kernel_resources) < kernel_occurrences
    assert set(compilation.required_kernel_resource_ids) == kernel_resources


def test_partition_multi_hadron():
    pi1 = Operator(dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β'), momenta={'x': (0,0,1)})
    pi2 = Operator(dbar('y', 'b', 'γ') * gamma5('γ', 'δ') * u('y', 'b', 'δ'), momenta={'y': (0,0,-1)})
    dipi = pi1 * pi2

    graphs = _contraction_graphs(CorrMatrix([dipi])[0, 0])

    assert graphs
    assert all(len(graph.nodes_of_kind("kernel")) == 4 for graph in graphs)
    assert all(len(graph.nodes_of_kind("propagation")) == 4 for graph in graphs)


def test_partition_tetraquark():
    snk = Operator(
        dbar('x', 'a', 'α') * u('x', 'b', 'β') * ubar('x', 'c', 'γ') * d('x', 'd', 'δ'),
        momenta={'x': (0,0,0)}
    )
    src = Operator(
        ubar('y', 'a', 'α') * d('y', 'b', 'β') * dbar('y', 'c', 'γ') * u('y', 'd', 'δ'),
        momenta={'y': (0,0,0)}
    )

    corr = Correlator(snk, src)
    terms = wick_contract(corr)
    phys = lower_to_physics_ir(corr, terms)
    vertices = identify_operator_vertices(phys.correlators[0])

    assert len(vertices) == 2
    assert vertices[0].num_quarks == 4


def test_partition_multi_baryon():
    b1 = Operator(epsilon('c1','c2','c3') * u('x','c1','α') * u('x','c2','β') * d('x','c3','γ'), momenta={'x': (0,0,1)})
    b2 = Operator(epsilon('c4','c5','c6') * u('y','c4','δ') * u('y','c5','ε') * d('y','c6','ζ'), momenta={'y': (0,0,-1)})
    dibaryon = b1 * b2

    graphs = _contraction_graphs(CorrMatrix([dibaryon])[0, 0])

    assert graphs
    assert all(len(graph.nodes_of_kind("kernel")) == 4 for graph in graphs)
    assert all(len(graph.nodes_of_kind("propagation")) == 6 for graph in graphs)


def test_quark_derivatives_are_preserved_in_elemental_recipe():
    from concord_lqcd.compiler.physics_ir import QuarkRecord, LogicalQuarkLine, LogicalCorrelator, OperationRecord, OperationKind, Representation

    snk_u = QuarkRecord("u", False, "c0:op0:x", "b", "beta", representation=Representation.LAPH)
    src_d = QuarkRecord("d", True, "c0:op0:x", "a", "alpha", representation=Representation.LAPH)
    ops_d = (OperationRecord(OperationKind.DERIVATIVE, {"direction": "+x"}),)
    line = LogicalQuarkLine(quark=snk_u, quark_operations=(), antiquark=src_d, antiquark_operations=ops_d)
    correlator = LogicalCorrelator(coefficient=1.0, quark_lines=(line,))

    vertices = identify_operator_vertices(correlator)
    kr = construct_kernel_recipe(realize_projected_fields(vertices[0]))

    assert kr.num_quarks == 2
    assert kr.projected_fields[1].operations == ops_d


def test_generic_multiquark_recipe_keeps_per_field_flavors():
    snk = Operator(
        dbar('x', 'a', 'α') * u('x', 'b', 'β') * ubar('x', 'c', 'γ') * d('x', 'd', 'δ'),
        momenta={'x': (0, 0, 0)}
    )
    src = Operator(
        ubar('y', 'a', 'α') * d('y', 'b', 'β') * dbar('y', 'c', 'γ') * u('y', 'd', 'δ'),
        momenta={'y': (0, 0, 0)}
    )

    corr = Correlator(snk, src)
    terms = wick_contract(corr)
    phys = lower_to_physics_ir(corr, terms)
    vertices = identify_operator_vertices(phys.correlators[0])
    kr = construct_kernel_recipe(realize_projected_fields(vertices[0]))

    flavors = sorted([pf.record.flavor for pf in kr.projected_fields])
    assert flavors == ['d', 'd', 'u', 'u']


def test_baryon_factory_builds_three_quark_operator_and_wick_terms():
    baryon = build_baryon_operator(['l', 'l', 's'], 'Cg5', momentum=(0, 0, 0))
    graphs = _contraction_graphs(CorrMatrix([baryon])[0, 0])

    assert len(graphs) == 2
    assert all(len(graph.nodes_of_kind("propagation")) == 3 for graph in graphs)
