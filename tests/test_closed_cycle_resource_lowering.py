"""Focused tests for exact-distillation closed-cycle resource lowering."""

import pytest

from concord_lqcd.compiler.closed_cycle_lowering import lower_correlator_closed_cycles
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
    gamma5,
    gammaX,
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


def test_local_laph_disconnected_loop_uses_one_ordinary_equal_time_perambulator():
    loop_line = LogicalQuarkLine(
        quark=_q("u", False, "c0:op0:z", "b", "q", 0),
        antiquark=_q("u", True, "c0:op0:z", "a", "aq", 0),
    )
    gamma = SpinTensorRecord("gammaX", "aq", "q", operator_id=0)
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(loop_line,),
        spin_tensors=(gamma,),
    )

    result = lower_correlator_closed_cycles(correlator)

    assert result.topology.num_open_paths == 0
    assert result.topology.num_closed_cycles == 1
    assert len(result.closed_cycles) == 1

    cycle = result.closed_cycles[0]
    assert cycle.is_local
    assert cycle.flavors == ("u",)
    assert len(cycle.propagations) == 1
    assert len(cycle.transitions) == 1

    propagation = cycle.propagations[0]
    assert not propagation.is_generalized
    assert propagation.resource_kind == "perambulator"
    assert propagation.resource_id.startswith("perambulator_")
    assert propagation.source_representation == Representation.LAPH
    assert propagation.sink_representation == Representation.LAPH
    assert propagation.flavor == "u"

    transition = cycle.transitions[0]
    assert transition.is_single_kernel_transition
    assert transition.incoming_flavor == "u"
    assert transition.outgoing_flavor == "u"
    assert transition.spin_tensors == (gamma,)


def test_genuinely_point_disconnected_loop_is_not_reconstructed_from_tau():
    loop_line = LogicalQuarkLine(
        quark=_q(
            "u", False, "c0:op0:z", "b", "q", 0,
            rep=Representation.POINT,
        ),
        antiquark=_q(
            "u", True, "c0:op0:z", "a", "aq", 0,
            rep=Representation.POINT,
        ),
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(loop_line,),
        spin_tensors=(SpinTensorRecord("gammaX", "aq", "q", operator_id=0),),
    )

    with pytest.raises(NotImplementedError, match="POINT disconnected loop|LapH/distillation"):
        lower_correlator_closed_cycles(correlator)


def test_flavor_changing_mesonic_cycle_requests_one_ordinary_perambulator_per_segment():
    u_segment = LogicalQuarkLine(
        quark=_q("u", False, "c0:op0:y", "bu", "qu", 0),
        antiquark=_q("u", True, "c0:op1:x", "au", "auq", 1),
    )
    c_segment = LogicalQuarkLine(
        quark=_q("c", False, "c0:op1:x", "bc", "qc", 1),
        antiquark=_q("c", True, "c0:op0:y", "ac", "acq", 0),
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(u_segment, c_segment),
        spin_tensors=(
            SpinTensorRecord("gamma5", "acq", "qu", operator_id=0),
            SpinTensorRecord("gamma5", "auq", "qc", operator_id=1),
        ),
    )

    result = lower_correlator_closed_cycles(correlator)

    assert result.topology.num_closed_cycles == 1
    cycle = result.closed_cycles[0]
    assert cycle.flavors == ("u", "c")
    assert len(cycle.propagations) == 2
    assert all(not propagation.is_generalized for propagation in cycle.propagations)
    assert all(
        propagation.resource_kind == "perambulator"
        for propagation in cycle.propagations
    )
    assert {propagation.flavor for propagation in cycle.propagations} == {"u", "c"}
    assert [transition.incoming_flavor for transition in cycle.transitions] == ["u", "c"]
    assert [transition.outgoing_flavor for transition in cycle.transitions] == ["c", "u"]


def test_same_flavor_cycle_reuses_same_all_time_perambulator_resource():
    forward = LogicalQuarkLine(
        quark=_q("u", False, "c0:op0:y", "b0", "q0", 0),
        antiquark=_q("u", True, "c0:op1:x", "a1", "a1s", 1),
    )
    backward = LogicalQuarkLine(
        quark=_q("u", False, "c0:op1:x", "b1", "q1", 1),
        antiquark=_q("u", True, "c0:op0:y", "a0", "a0s", 0),
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(forward, backward),
        spin_tensors=(
            SpinTensorRecord("gamma5", "a0s", "q0", operator_id=0),
            SpinTensorRecord("gamma5", "a1s", "q1", operator_id=1),
        ),
    )

    cycle = lower_correlator_closed_cycles(correlator).closed_cycles[0]

    assert len(cycle.propagation_resource_ids) == 2
    assert cycle.propagation_resource_ids[0] == cycle.propagation_resource_ids[1]


def test_baryon_meson_like_term_lowers_only_mesonic_closed_cycle_here():
    baryon_lines = (
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op0:y", "ub", "us", 0),
            antiquark=_q("u", True, "c0:op1:x", "uB", "uS", 1),
        ),
        LogicalQuarkLine(
            quark=_q("d", False, "c0:op0:y", "db", "ds", 0),
            antiquark=_q("d", True, "c0:op1:x", "dB", "dS", 1),
        ),
        LogicalQuarkLine(
            quark=_q("c", False, "c0:op0:y", "cb", "cs", 0),
            antiquark=_q("c", True, "c0:op1:x", "cB", "cS", 1),
        ),
    )
    meson_forward = LogicalQuarkLine(
        quark=_q("s", False, "c0:op0:y", "mb0", "mq0", 0),
        antiquark=_q("s", True, "c0:op1:x", "ma1", "ma1s", 1),
    )
    meson_backward = LogicalQuarkLine(
        quark=_q("s", False, "c0:op1:x", "mb1", "mq1", 1),
        antiquark=_q("s", True, "c0:op0:y", "ma0", "ma0s", 0),
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=baryon_lines + (meson_forward, meson_backward),
        spin_tensors=(
            SpinTensorRecord("gamma5", "ma0s", "mq0", operator_id=0),
            SpinTensorRecord("gamma5", "ma1s", "mq1", operator_id=1),
        ),
    )

    result = lower_correlator_closed_cycles(correlator)

    assert result.topology.num_open_paths == 3
    assert result.topology.num_closed_cycles == 1
    assert len(result.closed_cycles) == 1
    assert result.closed_cycles[0].flavors == ("s", "s")
    assert len(result.closed_cycles[0].propagations) == 2


def test_frontend_laph_disconnected_current_lowers_to_ordinary_perambulator_cycle():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * u("y", "a", "s2"),
        momentum=(0, 0, 0),
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
        momentum=(-1, 0, 0),
    )
    correlator = Correlator(sink, current, source)

    wick_terms = wick_contract(correlator)
    assert wick_terms
    physics_ir = lower_to_physics_ir(correlator, wick_terms)
    assert physics_ir.correlators

    for logical in physics_ir.correlators:
        result = lower_correlator_closed_cycles(logical)
        local_cycles = [cycle for cycle in result.closed_cycles if cycle.is_local]
        strange_cycles = [cycle for cycle in local_cycles if cycle.flavors == ("s",)]
        assert len(strange_cycles) == 1

        current_cycle = strange_cycles[0]
        assert len(current_cycle.propagations) == 1
        assert current_cycle.propagations[0].resource_kind == "perambulator"
        assert current_cycle.propagations[0].flavor == "s"
        assert current_cycle.propagations[0].source_representation == Representation.LAPH
        assert current_cycle.propagations[0].sink_representation == Representation.LAPH
        assert any(
            tensor.kind == "gammaX"
            for tensor in current_cycle.transitions[0].spin_tensors
        )
