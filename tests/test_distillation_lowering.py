"""Unit tests for distillation vertex and kernel lowering."""

from concord_lqcd.compiler.physics_ir import (
    OperationKind,
    Representation,
    OperationRecord,
    QuarkRecord,
    LogicalQuarkLine,
    LogicalCorrelator,
)
from concord_lqcd.compiler.distillation_lowering import (
    ProjectionContribution,
    ProjectedField,
    RealizedOperatorVertex,
    KernelRecipe,
    OperatorVertex,
    identify_operator_vertices,
    realize_projected_fields,
    construct_kernel_recipe,
    identify_kernel_resource,
    emit_backend_kernel_program,
)
from concord_lqcd.compiler import (
    lower_correlator_to_kernel_recipes_and_propagations,
)


def test_identify_operator_vertices_local_meson():
    snk_u = QuarkRecord("u", False, "c0:op0:y", "b", "beta", representation=Representation.LAPH)
    src_u = QuarkRecord("u", True, "c0:op1:x", "a", "alpha", representation=Representation.LAPH)
    ops_u = (OperationRecord(OperationKind.DERIVATIVE, {"direction": "+x"}),)

    line = LogicalQuarkLine(quark=snk_u, quark_operations=ops_u, antiquark=src_u, antiquark_operations=())
    correlator = LogicalCorrelator(coefficient=1.0, quark_lines=(line,))

    vertices = identify_operator_vertices(correlator)

    assert len(vertices) == 2
    v_labels = {v.pos_label for v in vertices}
    assert v_labels == {"x", "y"}

    v_y = next(v for v in vertices if v.vertex_id == "c0:op0:y")
    assert v_y.num_quarks == 1
    assert v_y.endpoints[0].operations == ops_u


def test_realize_projected_fields_laph_and_point():
    snk_u = QuarkRecord("u", False, "c0:op0:y", "b", "beta", representation=Representation.LAPH)
    src_u = QuarkRecord("u", True, "c0:op1:x", "a", "alpha", representation=Representation.POINT)

    line = LogicalQuarkLine(quark=snk_u, quark_operations=(), antiquark=src_u, antiquark_operations=())
    correlator = LogicalCorrelator(coefficient=1.0, quark_lines=(line,))

    vertices = identify_operator_vertices(correlator)
    v_y = next(v for v in vertices if v.vertex_id == "c0:op0:y")
    v_x = next(v for v in vertices if v.vertex_id == "c0:op1:x")

    rv_y = realize_projected_fields(v_y)
    assert len(rv_y.projected_fields) == 1
    assert rv_y.projected_fields[0].contribution == ProjectionContribution.EIGENVECTOR

    rv_x = realize_projected_fields(v_x)
    assert len(rv_x.projected_fields) == 1
    assert rv_x.projected_fields[0].contribution == ProjectionContribution.POINT


def test_canonical_topology_lowers_flavor_changing_point_current():
    # 3-point function: src(x) ---> insertion J(z) ---> snk(y)
    q_ins = QuarkRecord("u", False, "c0:op1:z", "c2", "s2", representation=Representation.POINT)
    q_src = QuarkRecord("u", True, "c0:op2:x", "c3", "s3", representation=Representation.POINT)
    line1 = LogicalQuarkLine(quark=q_ins, quark_operations=(), antiquark=q_src, antiquark_operations=())

    q_snk = QuarkRecord("u", False, "c0:op0:y", "c1", "s1", representation=Representation.LAPH)
    q_ins_bar = QuarkRecord("u", True, "c0:op1:z", "c2", "s2", representation=Representation.POINT)
    ops_ins = (OperationRecord(OperationKind.DERIVATIVE, {"direction": "+z"}),)
    line2 = LogicalQuarkLine(quark=q_snk, quark_operations=(), antiquark=q_ins_bar, antiquark_operations=ops_ins)

    correlator = LogicalCorrelator(coefficient=1.0, quark_lines=(line1, line2))

    kernel_recipes, propagation_recipes = lower_correlator_to_kernel_recipes_and_propagations(
        correlator
    )
    assert len(kernel_recipes) == 3
    generalized = [recipe for recipe in propagation_recipes if recipe.is_generalized]
    assert len(generalized) == 1
    prop_recipe = generalized[0]
    assert prop_recipe.is_generalized
    assert prop_recipe.start_kernel.vertex_id == "c0:op2:x"
    assert prop_recipe.end_kernel.vertex_id == "c0:op0:y"
    assert len(prop_recipe.inserted_kernels) == 1
    assert prop_recipe.inserted_kernels[0].vertex_id == "c0:op1:z"
    assert prop_recipe.flavor == "u"


def test_stage6_identify_kernel_resource_and_full_lowering():
    # 2-point function: src(x) ---> snk(y)
    snk_u = QuarkRecord("u", False, "c0:op0:y", "b", "beta", representation=Representation.LAPH)
    src_u = QuarkRecord("u", True, "c0:op1:x", "a", "alpha", representation=Representation.LAPH)
    ops = (OperationRecord(OperationKind.DERIVATIVE, {"direction": "+x"}),)
    line = LogicalQuarkLine(quark=snk_u, quark_operations=ops, antiquark=src_u, antiquark_operations=())
    correlator = LogicalCorrelator(coefficient=1.0, quark_lines=(line,))

    kernel_recipes, prop_recipes = lower_correlator_to_kernel_recipes_and_propagations(correlator)

    assert len(kernel_recipes) == 2
    assert len(prop_recipes) == 1

    res_id_y = identify_kernel_resource(kernel_recipes[0])
    res_id_x = identify_kernel_resource(kernel_recipes[1])

    assert res_id_y.startswith("spatial_kernel_")
    assert res_id_x.startswith("spatial_kernel_")
    assert prop_recipes[0].start_kernel == kernel_recipes[1]
    assert prop_recipes[0].end_kernel == kernel_recipes[0]
    assert not prop_recipes[0].is_generalized


def test_stage7_emit_backend_kernel_program():
    snk_u = QuarkRecord("u", False, "c0:op0:x", "b", "beta", representation=Representation.LAPH)
    src_d = QuarkRecord("d", True, "c0:op0:x", "a", "alpha", representation=Representation.LAPH)
    ops_d = (OperationRecord(OperationKind.DERIVATIVE, {"direction": "+x"}),)
    line = LogicalQuarkLine(quark=snk_u, quark_operations=(), antiquark=src_d, antiquark_operations=ops_d)
    correlator = LogicalCorrelator(coefficient=1.0, quark_lines=(line,))

    vertices = identify_operator_vertices(correlator)
    rv_x = realize_projected_fields(vertices[0])
    recipe_x = construct_kernel_recipe(rv_x)

    program = emit_backend_kernel_program(recipe_x)
    assert program.num_quarks == 2
    assert len(program.operations) >= 2
    assert any(op.kind == "eigenvector" for op in program.operations)
    assert any(op.kind == "phase" for op in program.operations)


def test_stage3_construct_kernel_recipe_meson_and_baryon():
    # Meson operator vertex: qbar_u q_d with derivative on q_d
    snk_u = QuarkRecord("u", False, "c0:op0:x", "b", "beta", representation=Representation.LAPH)
    src_d = QuarkRecord("d", True, "c0:op0:x", "a", "alpha", representation=Representation.LAPH)
    ops_d = (OperationRecord(OperationKind.DERIVATIVE, {"direction": "+x"}),)

    line = LogicalQuarkLine(quark=snk_u, quark_operations=(), antiquark=src_d, antiquark_operations=ops_d)
    correlator = LogicalCorrelator(coefficient=1.0, quark_lines=(line,))

    vertices = identify_operator_vertices(correlator)
    v_x = vertices[0]
    rv_x = realize_projected_fields(v_x)
    recipe_x = construct_kernel_recipe(rv_x)

    assert recipe_x.num_quarks == 2
    assert recipe_x.pos_label == "x"
    assert len(recipe_x.open_ports) == 2
    assert recipe_x.projected_fields[0].contribution == ProjectionContribution.EIGENVECTOR
    assert recipe_x.projected_fields[1].contribution == ProjectionContribution.EIGENVECTOR_DAGGER
    assert recipe_x.projected_fields[1].operations == ops_d
