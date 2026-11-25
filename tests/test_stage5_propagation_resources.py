"""Focused tests for Stage-5 external propagation-resource identification."""

import hashlib
import json

from concord_lqcd.compiler import lower_correlator_to_kernel_recipes_and_propagations
from concord_lqcd.compiler.physics_ir import (
    LogicalCorrelator,
    LogicalQuarkLine,
    QuarkRecord,
    Representation,
    SpinTensorRecord,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import (
    Correlator,
    Operator,
    c,
    cbar,
    derivative,
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


def _active_generalized_recipe(correlator):
    physics_ir = lower_to_physics_ir(correlator, wick_contract(correlator))
    logical = physics_ir.correlators[0]
    _, recipes = lower_correlator_to_kernel_recipes_and_propagations(logical)
    active = [recipe for recipe in recipes if recipe.is_generalized]
    assert len(active) == 1
    return active[0]


def _three_point_current(gamma_factory, momentum=(1, 0, 0), derivative_direction=None):
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * s("y", "a", "s2"),
        momentum=(0, 0, 0),
    )

    current_quark = point(u("z", "b", "s4"))
    if derivative_direction is not None:
        current_quark = derivative(current_quark, derivative_direction)
    current = Operator(
        point(sbar("z", "b", "s3"))
        * gamma_factory("s3", "s4")
        * current_quark,
        momentum=momentum,
    )

    source = Operator(
        ubar("x", "c", "s5")
        * gamma5("s5", "s6")
        * c("x", "c", "s6"),
        momentum=(-1, 0, 0),
    )
    return Correlator(sink, current, source)


def test_stage5_ordinary_perambulator_has_exact_external_ports():
    sink = _q("u", False, "c0:op0:y", "a", "s1", 0)
    source = _q("u", True, "c0:op1:x", "A", "S1", 1)
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(LogicalQuarkLine(quark=sink, antiquark=source),),
    )

    _, recipes = lower_correlator_to_kernel_recipes_and_propagations(correlator)

    assert len(recipes) == 1
    recipe = recipes[0]
    assert not recipe.is_generalized
    assert recipe.resource_kind == "perambulator"
    assert recipe.segment_flavors == ("u",)
    assert recipe.flavor == "u"
    assert recipe.start_port == "mode_0"
    assert recipe.end_port == "mode_0"
    assert recipe.resource_id.startswith("perambulator_")


def test_stage5_repeated_same_position_lines_keep_distinct_kernel_ports():
    lines = (
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op0:y", "a", "s1", 0),
            antiquark=_q("u", True, "c0:op1:x", "A", "S1", 1),
        ),
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op0:y", "b", "s2", 0),
            antiquark=_q("u", True, "c0:op1:x", "B", "S2", 1),
        ),
    )
    correlator = LogicalCorrelator(coefficient=1.0, quark_lines=lines)

    _, recipes = lower_correlator_to_kernel_recipes_and_propagations(correlator)

    assert len(recipes) == 2
    assert {(r.start_port, r.end_port) for r in recipes} == {
        ("mode_0", "mode_0"),
        ("mode_1", "mode_1"),
    }
    assert recipes[0].resource_id == recipes[1].resource_id


def test_stage5_point_current_selects_flavor_changing_generalized_resource():
    recipe = _active_generalized_recipe(_three_point_current(gammaX))

    assert recipe.resource_kind == "generalized_perambulator"
    assert recipe.segment_flavors == ("u", "s")
    assert recipe.incoming_flavor == "u"
    assert recipe.outgoing_flavor == "s"
    assert recipe.resource_id.startswith("generalized_perambulator_")
    assert len(recipe.insertion_specs) == 1

    insertion = recipe.insertion_specs[0]
    assert insertion.incoming_flavor == "u"
    assert insertion.outgoing_flavor == "s"
    assert insertion.incoming_port.startswith("color_")
    assert insertion.outgoing_port.startswith("color_")
    assert [tensor.kind for tensor in insertion.spin_tensors] == ["gammaX"]


def test_stage5_resource_signature_preserves_generalized_resource_id_encoding():
    recipe = _active_generalized_recipe(_three_point_current(gammaX))
    insertion = recipe.insertion_specs[0]
    payload = {
        "kind": "generalized_perambulator",
        "source_representation": recipe.source_representation.value,
        "sink_representation": recipe.sink_representation.value,
        "segment_flavors": ["u", "s"],
        "insertions": [{
            "incoming_flavor": "u",
            "outgoing_flavor": "s",
            "spatial_resource": insertion.spatial_resource_id,
            "incoming_spatial_leg": insertion.kernel.open_ports.index(insertion.incoming_port),
            "outgoing_spatial_leg": insertion.kernel.open_ports.index(insertion.outgoing_port),
            "spin": {
                "outgoing_endpoint": "s0",
                "incoming_endpoint": "s1",
                "tensors": [{"kind": "gammaX", "left": "s0", "right": "s1"}],
            },
        }],
    }
    signature = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]

    assert recipe.resource_signature == signature
    assert recipe.resource_id == f"generalized_perambulator_{digest}"


def test_stage5_gamma_changes_generalized_resource_identity():
    gamma_x = _active_generalized_recipe(_three_point_current(gammaX))
    gamma_5 = _active_generalized_recipe(_three_point_current(gamma5))

    assert gamma_x.insertion_specs[0].spatial_resource_id == gamma_5.insertion_specs[0].spatial_resource_id
    assert gamma_x.resource_id != gamma_5.resource_id


def test_stage5_current_momentum_changes_generalized_resource_identity():
    p1 = _active_generalized_recipe(
        _three_point_current(gammaX, momentum=(1, 0, 0))
    )
    p2 = _active_generalized_recipe(
        _three_point_current(gammaX, momentum=(0, 1, 0))
    )

    assert p1.insertion_specs[0].spatial_resource_id != p2.insertion_specs[0].spatial_resource_id
    assert p1.resource_id != p2.resource_id


def test_stage5_current_derivative_changes_generalized_resource_identity():
    plain = _active_generalized_recipe(_three_point_current(gammaX))
    deriv = _active_generalized_recipe(
        _three_point_current(gammaX, derivative_direction="+x")
    )

    assert plain.insertion_specs[0].spatial_resource_id != deriv.insertion_specs[0].spatial_resource_id
    assert plain.resource_id != deriv.resource_id


def test_stage5_preserves_ordered_multiple_flavor_changing_insertions():
    q_u_at_j2 = _q(
        "u", False, "c0:op2:z2", "b2", "u_in", 2, rep=Representation.POINT
    )
    src_ubar = _q("u", True, "c0:op3:x", "A", "u_src", 3)

    q_s_at_j1 = _q(
        "s", False, "c0:op1:z1", "b1", "s_in", 1, rep=Representation.POINT
    )
    sbar_at_j2 = _q(
        "s", True, "c0:op2:z2", "a2", "s_out", 2, rep=Representation.POINT
    )

    sink_c = _q("c", False, "c0:op0:y", "C", "c_sink", 0)
    cbar_at_j1 = _q(
        "c", True, "c0:op1:z1", "a1", "c_out", 1, rep=Representation.POINT
    )

    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(
            LogicalQuarkLine(quark=q_u_at_j2, antiquark=src_ubar),
            LogicalQuarkLine(quark=q_s_at_j1, antiquark=sbar_at_j2),
            LogicalQuarkLine(quark=sink_c, antiquark=cbar_at_j1),
        ),
        spin_tensors=(
            SpinTensorRecord("gammaX", "s_out", "u_in", operator_id=2),
            SpinTensorRecord("gamma5", "c_out", "s_in", operator_id=1),
        ),
        momenta={
            "c0:op2:z2": (1, 0, 0),
            "c0:op1:z1": (0, 1, 0),
        },
    )

    _, recipes = lower_correlator_to_kernel_recipes_and_propagations(correlator)

    assert len(recipes) == 1
    recipe = recipes[0]
    assert recipe.is_generalized
    assert recipe.segment_flavors == ("u", "s", "c")
    assert [(s.incoming_flavor, s.outgoing_flavor) for s in recipe.insertion_specs] == [
        ("u", "s"),
        ("s", "c"),
    ]
    assert [s.spin_tensors[0].kind for s in recipe.insertion_specs] == [
        "gammaX",
        "gamma5",
    ]
    assert len(recipe.inserted_kernels) == 2


def test_laph_intermediate_current_remains_an_explicit_kernel_transition():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * u("y", "a", "s2")
    )
    current = Operator(
        ubar("z", "b", "s3") * gammaX("s3", "s4") * u("z", "b", "s4")
    )
    source = Operator(
        ubar("x", "c", "s5") * gamma5("s5", "s6") * c("x", "c", "s6")
    )
    correlator = Correlator(sink, current, source)
    logical = lower_to_physics_ir(correlator, wick_contract(correlator)).correlators[0]

    _, recipes = lower_correlator_to_kernel_recipes_and_propagations(logical)

    assert recipes
    assert all(not recipe.is_generalized for recipe in recipes)
