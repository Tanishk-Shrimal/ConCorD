"""Focused tests for Stage-3 generic spatial/color kernel construction."""

from concord_lqcd.compiler.physics_ir import (
    ColorTensorRecord,
    GaugeTensorRecord,
    LogicalCorrelator,
    LogicalQuarkLine,
    QuarkRecord,
    Representation,
    SpinTensorRecord,
)
from concord_lqcd.compiler.distillation_lowering import (
    ProjectionContribution,
    construct_kernel_recipe,
    identify_kernel_resource,
    identify_operator_vertices,
    realize_projected_fields,
)


def _q(flavor, is_anti, pos, color, spin, operator_id, rep=Representation.LAPH):
    return QuarkRecord(
        flavor=flavor,
        is_anti=is_anti,
        pos_label=pos,
        color_label=color,
        spin_label=spin,
        representation=rep,
        operator_id=operator_id,
    )


def _recipe_for_position(correlator, pos):
    vertex = next(
        vertex
        for vertex in identify_operator_vertices(correlator)
        if vertex.position_labels == (pos,)
    )
    return construct_kernel_recipe(realize_projected_fields(vertex))


def test_stage3_generic_rank_is_inferred_from_realized_fields():
    for rank in (2, 3, 4):
        sink = tuple(
            _q("u", False, "c0:op0:x", f"x_c{i}", f"x_s{i}", 0)
            for i in range(rank)
        )
        source = tuple(
            _q("u", True, "c0:op1:y", f"y_c{i}", f"y_s{i}", 1)
            for i in range(rank)
        )
        lines = tuple(
            LogicalQuarkLine(quark=q, antiquark=aq)
            for q, aq in zip(sink, source)
        )
        correlator = LogicalCorrelator(coefficient=1.0, quark_lines=lines)

        recipe = _recipe_for_position(correlator, "c0:op0:x")

        assert recipe.num_quarks == rank
        assert recipe.open_ports == tuple(f"mode_{i}" for i in range(rank))
        assert all(
            pf.contribution == ProjectionContribution.EIGENVECTOR
            for pf in recipe.projected_fields
        )


def test_stage3_color_tensor_is_part_of_generic_kernel():
    sink = (
        _q("u", False, "c0:op0:x", "a", "s1", 0),
        _q("u", False, "c0:op0:x", "b", "s2", 0),
        _q("d", False, "c0:op0:x", "c", "s3", 0),
    )
    source = (
        _q("u", True, "c0:op1:y", "A", "S1", 1),
        _q("u", True, "c0:op1:y", "B", "S2", 1),
        _q("d", True, "c0:op1:y", "C", "S3", 1),
    )
    epsilon = ColorTensorRecord(
        "epsilon",
        ("a", "b", "c"),
        operator_id=0,
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=tuple(
            LogicalQuarkLine(quark=q, antiquark=aq)
            for q, aq in zip(sink, source)
        ),
        color_tensors=(epsilon,),
        momenta={"c0:op0:x": (1, 0, 0)},
    )

    recipe = _recipe_for_position(correlator, "c0:op0:x")

    assert recipe.color_tensors == (epsilon,)
    assert recipe.momenta == {"c0:op0:x": (1, 0, 0)}
    assert recipe.open_ports == ("mode_0", "mode_1", "mode_2")


def test_stage3_shared_color_index_needs_no_object_specific_rule():
    q = _q("u", False, "c0:op0:x", "a", "s1", 0)
    qbar = _q("d", True, "c0:op0:x", "a", "s2", 0)
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
    )

    recipe = _recipe_for_position(correlator, "c0:op0:x")

    assert recipe.color_tensors == ()
    assert tuple(
        pf.record.color_label for pf in recipe.projected_fields
    ) == ("a", "a")


def test_stage3_spin_tensors_do_not_enter_kernel_or_resource_identity():
    q = _q("u", False, "c0:op0:x", "a", "s1", 0)
    qbar = _q("d", True, "c0:op0:x", "b", "s2", 0)
    delta = ColorTensorRecord("delta", ("a", "b"), operator_id=0)

    corr_g5 = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
        spin_tensors=(
            SpinTensorRecord("gamma5", "s1", "s2", operator_id=0),
        ),
        color_tensors=(delta,),
        momenta={"c0:op0:x": (0, 0, 1)},
    )
    corr_gx = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
        spin_tensors=(
            SpinTensorRecord("gammaX", "s1", "s2", operator_id=0),
        ),
        color_tensors=(delta,),
        momenta={"c0:op0:x": (0, 0, 1)},
    )

    recipe_g5 = _recipe_for_position(corr_g5, "c0:op0:x")
    recipe_gx = _recipe_for_position(corr_gx, "c0:op0:x")

    assert not hasattr(recipe_g5, "spin_tensors")
    assert not hasattr(recipe_gx, "spin_tensors")
    assert identify_kernel_resource(recipe_g5) == identify_kernel_resource(recipe_gx)


def test_stage3_momentum_changes_spatial_kernel_identity():
    q = _q("u", False, "c0:op0:x", "a", "s1", 0)
    qbar = _q("d", True, "c0:op0:x", "b", "s2", 0)
    delta = ColorTensorRecord("delta", ("a", "b"), operator_id=0)

    corr_p0 = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
        color_tensors=(delta,),
        momenta={"c0:op0:x": (0, 0, 0)},
    )
    corr_p1 = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
        color_tensors=(delta,),
        momenta={"c0:op0:x": (1, 0, 0)},
    )

    recipe_p0 = _recipe_for_position(corr_p0, "c0:op0:x")
    recipe_p1 = _recipe_for_position(corr_p1, "c0:op0:x")

    assert identify_kernel_resource(recipe_p0) != identify_kernel_resource(recipe_p1)


def test_stage3_explicit_gauge_connectivity_allows_multi_position_kernel():
    qbar = _q("u", True, "c0:op0:x", "a", "s1", 0)
    q = _q("u", False, "c0:op0:y", "b", "s2", 0)
    gauge = GaugeTensorRecord(
        start_pos="c0:op0:x",
        end_pos="c0:op0:y",
        start_color="a",
        end_color="b",
        operator_id=0,
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
        gauge_tensors=(gauge,),
        momenta={
            "c0:op0:x": (1, 0, 0),
            "c0:op0:y": (-1, 0, 0),
        },
    )

    vertices = identify_operator_vertices(correlator)
    assert len(vertices) == 1

    recipe = construct_kernel_recipe(realize_projected_fields(vertices[0]))

    assert set(recipe.position_labels) == {"c0:op0:x", "c0:op0:y"}
    assert recipe.gauge_tensors == (gauge,)
    assert recipe.momenta == correlator.momenta
    assert recipe.open_ports == ("mode_0", "mode_1")


def test_stage3_point_contribution_is_not_relabelled_as_mode():
    q = _q(
        "u",
        False,
        "c0:op0:x",
        "a",
        "s1",
        0,
        rep=Representation.POINT,
    )
    qbar = _q("d", True, "c0:op0:x", "b", "s2", 0)
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
    )

    recipe = _recipe_for_position(correlator, "c0:op0:x")

    assert recipe.projected_fields[0].contribution == ProjectionContribution.POINT
    assert recipe.open_ports[0] == "color_0"
    assert recipe.open_ports[1] == "mode_1"


def test_stage3_spatial_kernel_emission_contains_no_spin_matrix():
    from concord_lqcd.compiler.distillation_lowering import emit_backend_kernel_program

    q = _q("u", False, "c0:op0:x", "a", "s1", 0)
    qbar = _q("d", True, "c0:op0:x", "b", "s2", 0)
    delta = ColorTensorRecord("delta", ("a", "b"), operator_id=0)
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
        spin_tensors=(
            SpinTensorRecord("gamma5", "s1", "s2", operator_id=0),
        ),
        color_tensors=(delta,),
        momenta={"c0:op0:x": (0, 0, 0)},
    )

    recipe = _recipe_for_position(correlator, "c0:op0:x")
    program = emit_backend_kernel_program(recipe)

    assert all(op.kind != "spin_matrix" for op in program.operations)
    assert program.spin_ports == []
    assert program.mode_ports == ["mode_0", "mode_1"]
    program.validate()
