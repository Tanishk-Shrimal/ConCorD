"""Focused tests for Stage-1 spatial/color component identification."""

from concord_lqcd.compiler.physics_ir import (
    ColorTensorRecord,
    GaugeTensorRecord,
    LogicalCorrelator,
    LogicalQuarkLine,
    QuarkRecord,
    Representation,
    SpinTensorRecord,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.compiler.distillation_lowering import identify_operator_vertices
from concord_lqcd.core import (
    Correlator,
    Operator,
    d,
    dbar,
    delta,
    epsilon,
    gamma5,
    link,
    reset_registry,
    u,
    ubar,
)
from concord_lqcd.core.wick import wick_contract


def _q(flavor, is_anti, pos, color, spin, operator_id):
    return QuarkRecord(
        flavor=flavor,
        is_anti=is_anti,
        pos_label=pos,
        color_label=color,
        spin_label=spin,
        representation=Representation.LAPH,
        operator_id=operator_id,
    )


def test_stage1_local_component_keeps_single_position_compatibility():
    q = _q("u", False, "c0:op0:x", "c0:op0:a", "c0:op0:s1", 0)
    qbar = _q("d", True, "c0:op0:x", "c0:op0:b", "c0:op0:s2", 0)
    delta_record = ColorTensorRecord(
        kind="delta",
        indices=("c0:op0:a", "c0:op0:b"),
        operator_id=0,
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
        color_tensors=(delta_record,),
        momenta={"c0:op0:x": (1, 0, 0)},
    )

    vertices = identify_operator_vertices(correlator)

    assert len(vertices) == 1
    vertex = vertices[0]
    assert vertex.vertex_id == "c0:op0:x"
    assert vertex.position_labels == ("c0:op0:x",)
    assert vertex.pos_label == "x"
    assert vertex.momentum == (1, 0, 0)
    assert vertex.color_tensors == (delta_record,)


def test_stage1_multibaryon_partitions_color_tensors_by_connectivity():
    sink_x = (
        _q("u", False, "c0:op0:x", "c0:op0:x_a", "c0:op0:x_s1", 0),
        _q("u", False, "c0:op0:x", "c0:op0:x_b", "c0:op0:x_s2", 0),
        _q("d", False, "c0:op0:x", "c0:op0:x_c", "c0:op0:x_s3", 0),
    )
    sink_y = (
        _q("u", False, "c0:op0:y", "c0:op0:y_a", "c0:op0:y_s1", 0),
        _q("u", False, "c0:op0:y", "c0:op0:y_b", "c0:op0:y_s2", 0),
        _q("d", False, "c0:op0:y", "c0:op0:y_c", "c0:op0:y_s3", 0),
    )
    src_x = (
        _q("u", True, "c0:op1:X", "c0:op1:X_a", "c0:op1:X_s1", 1),
        _q("u", True, "c0:op1:X", "c0:op1:X_b", "c0:op1:X_s2", 1),
        _q("d", True, "c0:op1:X", "c0:op1:X_c", "c0:op1:X_s3", 1),
    )
    src_y = (
        _q("u", True, "c0:op1:Y", "c0:op1:Y_a", "c0:op1:Y_s1", 1),
        _q("u", True, "c0:op1:Y", "c0:op1:Y_b", "c0:op1:Y_s2", 1),
        _q("d", True, "c0:op1:Y", "c0:op1:Y_c", "c0:op1:Y_s3", 1),
    )

    lines = tuple(
        LogicalQuarkLine(quark=sink, antiquark=source)
        for sink, source in zip(sink_x + sink_y, src_x + src_y)
    )
    eps_x = ColorTensorRecord("epsilon", tuple(q.color_label for q in sink_x), operator_id=0)
    eps_y = ColorTensorRecord("epsilon", tuple(q.color_label for q in sink_y), operator_id=0)
    eps_X = ColorTensorRecord("epsilon", tuple(q.color_label for q in src_x), operator_id=1)
    eps_Y = ColorTensorRecord("epsilon", tuple(q.color_label for q in src_y), operator_id=1)

    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=lines,
        color_tensors=(eps_x, eps_y, eps_X, eps_Y),
    )

    vertices = identify_operator_vertices(correlator)

    assert len(vertices) == 4
    by_position = {vertex.position_labels[0]: vertex for vertex in vertices}
    assert by_position["c0:op0:x"].color_tensors == (eps_x,)
    assert by_position["c0:op0:y"].color_tensors == (eps_y,)
    assert by_position["c0:op1:X"].color_tensors == (eps_X,)
    assert by_position["c0:op1:Y"].color_tensors == (eps_Y,)
    assert all(vertex.num_quarks == 3 for vertex in vertices)


def test_stage1_explicit_gauge_tensor_connects_multiple_positions():
    qbar = _q("u", True, "c0:op0:x", "c0:op0:a", "c0:op0:s1", 0)
    q = _q("u", False, "c0:op0:y", "c0:op0:b", "c0:op0:s2", 0)
    gauge = GaugeTensorRecord(
        start_pos="c0:op0:x",
        end_pos="c0:op0:y",
        start_color="c0:op0:a",
        end_color="c0:op0:b",
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
    vertex = vertices[0]
    assert set(vertex.position_labels) == {"c0:op0:x", "c0:op0:y"}
    assert vertex.gauge_tensors == (gauge,)
    assert vertex.momenta == {
        "c0:op0:x": (1, 0, 0),
        "c0:op0:y": (-1, 0, 0),
    }


def test_stage1_spin_connectivity_does_not_merge_spatial_components():
    qbar = _q("u", True, "c0:op0:x", "c0:op0:a", "c0:op0:sx", 0)
    q = _q("u", False, "c0:op0:y", "c0:op0:b", "c0:op0:sy", 0)
    gamma = SpinTensorRecord(
        kind="gamma5",
        left_spin="c0:op0:sx",
        right_spin="c0:op0:sy",
        operator_id=0,
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=q, antiquark=qbar),),
        spin_tensors=(gamma,),
    )

    vertices = identify_operator_vertices(correlator)

    assert len(vertices) == 2
    assert {vertex.pos_label for vertex in vertices} == {"x", "y"}


def test_stage1_frontend_multimeson_pipeline_partitions_local_color_components():
    """Real frontend -> Wick -> PhysicsIR keeps independent local kernels separate."""
    reset_registry()
    meson_x = Operator(
        dbar("x", "xa", "xs1")
        * gamma5("xs1", "xs2")
        * u("x", "xb", "xs2")
        * delta("xa", "xb"),
        momenta={"x": (1, 0, 0)},
    )
    meson_y = Operator(
        dbar("y", "ya", "ys1")
        * gamma5("ys1", "ys2")
        * u("y", "yb", "ys2")
        * delta("ya", "yb"),
        momenta={"y": (-1, 0, 0)},
    )
    operator = meson_x * meson_y
    corr = Correlator(operator, operator.dagger())

    physics_ir = lower_to_physics_ir(corr, wick_contract(corr))
    assert physics_ir.correlators

    for logical in physics_ir.correlators:
        vertices = identify_operator_vertices(logical)
        assert len(vertices) == 4
        assert all(len(vertex.position_labels) == 1 for vertex in vertices)
        assert all(vertex.num_quarks == 2 for vertex in vertices)
        assert all(len(vertex.color_tensors) == 1 for vertex in vertices)

        sink_x = next(
            v
            for v in vertices
            if v.pos_label == "x"
            and {ep.record.operator_id for ep in v.endpoints} == {0}
        )
        sink_y = next(
            v
            for v in vertices
            if v.pos_label == "y"
            and {ep.record.operator_id for ep in v.endpoints} == {0}
        )
        assert sink_x.momentum == (1, 0, 0)
        assert sink_y.momentum == (-1, 0, 0)


def test_stage1_frontend_multibaryon_pipeline_partitions_epsilons():
    """Real baryon products are partitioned from actual frontend index connectivity."""
    reset_registry()
    baryon_x = Operator(
        epsilon("xa", "xb", "xc")
        * u("x", "xa", "xs1")
        * u("x", "xb", "xs2")
        * d("x", "xc", "xs3"),
        momenta={"x": (1, 0, 0)},
    )
    baryon_y = Operator(
        epsilon("ya", "yb", "yc")
        * u("y", "ya", "ys1")
        * u("y", "yb", "ys2")
        * d("y", "yc", "ys3"),
        momenta={"y": (-1, 0, 0)},
    )
    operator = baryon_x * baryon_y
    corr = Correlator(operator, operator.dagger())

    physics_ir = lower_to_physics_ir(corr, wick_contract(corr))
    assert physics_ir.correlators

    for logical in physics_ir.correlators:
        vertices = identify_operator_vertices(logical)
        assert len(vertices) == 4
        assert all(len(vertex.position_labels) == 1 for vertex in vertices)
        assert all(vertex.num_quarks == 3 for vertex in vertices)
        assert all(len(vertex.color_tensors) == 1 for vertex in vertices)
        assert all(vertex.color_tensors[0].kind == "epsilon" for vertex in vertices)


def test_stage1_frontend_explicit_gauge_link_connects_positions():
    """Frontend GaugeLink connectivity survives Wick/PhysicsIR into Stage 1."""
    reset_registry()
    nonlocal_op = Operator(
        ubar("x", "a", "s1")
        * link("x", "y", "a", "b")
        * u("y", "b", "s2"),
        momenta={
            "x": (1, 0, 0),
            "y": (-1, 0, 0),
        },
    )
    corr = Correlator(nonlocal_op, nonlocal_op.dagger())

    physics_ir = lower_to_physics_ir(corr, wick_contract(corr))
    assert physics_ir.correlators

    for logical in physics_ir.correlators:
        vertices = identify_operator_vertices(logical)
        assert len(vertices) == 2
        assert all(len(vertex.position_labels) == 2 for vertex in vertices)
        assert all(vertex.num_quarks == 2 for vertex in vertices)
        assert all(len(vertex.gauge_tensors) == 1 for vertex in vertices)
