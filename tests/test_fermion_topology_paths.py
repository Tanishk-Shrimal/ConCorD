"""Focused tests for open paths in the canonical fermion topology."""

from concord_lqcd.compiler.fermion_topology import reconstruct_fermion_topology
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


def test_repeated_positions_remain_distinct_open_fermion_paths():
    lines = (
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op0:y", "ua", "us", 0),
            antiquark=_q("u", True, "c0:op1:x", "uA", "uS", 1),
        ),
        LogicalQuarkLine(
            quark=_q("d", False, "c0:op0:y", "da", "ds", 0),
            antiquark=_q("d", True, "c0:op1:x", "dA", "dS", 1),
        ),
    )

    topology = reconstruct_fermion_topology(
        LogicalCorrelator(coefficient=1.0, quark_lines=lines)
    )

    assert len(topology.open_paths) == 2
    assert sorted(path.flavors for path in topology.open_paths) == [("d",), ("u",)]
    assert all(path.start_endpoint.pos_label == "c0:op1:x" for path in topology.open_paths)
    assert all(path.end_endpoint.pos_label == "c0:op0:y" for path in topology.open_paths)


def test_flavor_changing_current_is_an_explicit_topology_transition():
    incoming = _q("u", False, "c0:op1:z", "b", "sin", 1, Representation.POINT)
    source = _q("u", True, "c0:op2:x", "A", "S", 2)
    sink = _q("s", False, "c0:op0:y", "C", "T", 0)
    current_anti = _q("s", True, "c0:op1:z", "a", "sout", 1, Representation.POINT)
    current_gamma = SpinTensorRecord("gammaX", "sout", "sin", operator_id=1)
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(
            LogicalQuarkLine(quark=incoming, antiquark=source),
            LogicalQuarkLine(quark=sink, antiquark=current_anti),
        ),
        spin_tensors=(current_gamma,),
    )

    topology = reconstruct_fermion_topology(correlator)

    assert len(topology.open_paths) == 1
    path = topology.open_paths[0]
    assert path.flavors == ("u", "s")
    assert len(path.transitions) == 1
    transition = path.transitions[0]
    assert transition.operator_id == 1
    assert (transition.incoming_flavor, transition.outgoing_flavor) == ("u", "s")
    assert transition.quark_endpoint.record == incoming
    assert transition.antiquark_endpoint.record == current_anti
    assert transition.spin_tensors == (current_gamma,)


def test_npoint_path_uses_wick_connectivity_and_preserves_transition_order():
    lines = (
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op2:z2", "b2", "u_in", 2, Representation.POINT),
            antiquark=_q("u", True, "c0:op3:x", "A", "u_src", 3),
        ),
        LogicalQuarkLine(
            quark=_q("s", False, "c0:op1:z1", "b1", "s_in", 1, Representation.POINT),
            antiquark=_q("s", True, "c0:op2:z2", "a2", "s_out", 2, Representation.POINT),
        ),
        LogicalQuarkLine(
            quark=_q("c", False, "c0:op0:y", "C", "c_sink", 0),
            antiquark=_q("c", True, "c0:op1:z1", "a1", "c_out", 1, Representation.POINT),
        ),
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=lines,
        spin_tensors=(
            SpinTensorRecord("gammaX", "s_out", "u_in", operator_id=2),
            SpinTensorRecord("gamma5", "c_out", "s_in", operator_id=1),
        ),
    )

    topology = reconstruct_fermion_topology(correlator)

    assert len(topology.open_paths) == 1
    path = topology.open_paths[0]
    assert path.flavors == ("u", "s", "c")
    assert [transition.operator_id for transition in path.transitions] == [2, 1]
    assert [transition.incoming_flavor for transition in path.transitions] == ["u", "s"]
    assert [transition.outgoing_flavor for transition in path.transitions] == ["s", "c"]


def test_ambiguous_operator_connectivity_is_not_guessed():
    lines = (
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op1:z", "a", "q1", 1),
            antiquark=_q("u", True, "c0:op2:x", "A", "s1", 2),
        ),
        LogicalQuarkLine(
            quark=_q("d", False, "c0:op1:z", "b", "q2", 1),
            antiquark=_q("d", True, "c0:op2:x", "B", "s2", 2),
        ),
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op0:y", "C", "t1", 0),
            antiquark=_q("u", True, "c0:op1:z", "c", "a1", 1),
        ),
        LogicalQuarkLine(
            quark=_q("d", False, "c0:op0:y", "D", "t2", 0),
            antiquark=_q("d", True, "c0:op1:z", "d", "a2", 1),
        ),
    )

    topology = reconstruct_fermion_topology(
        LogicalCorrelator(coefficient=1.0, quark_lines=lines)
    )

    assert len(topology.open_paths) == 4
    assert all(len(path.segments) == 1 for path in topology.open_paths)
    assert all(not path.transitions for path in topology.open_paths)


def test_frontend_point_current_appears_in_canonical_topology():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * s("y", "a", "s2"),
        momentum=(0, 0, 0),
    )
    current = Operator(
        point(sbar("z", "b", "s3"))
        * gammaX("s3", "s4")
        * point(u("z", "b", "s4")),
        momentum=(1, 0, 0),
    )
    source = Operator(
        ubar("x", "c", "s5") * gamma5("s5", "s6") * c("x", "c", "s6"),
        momentum=(-1, 0, 0),
    )
    correlator = Correlator(sink, current, source)
    physics_ir = lower_to_physics_ir(correlator, wick_contract(correlator))

    transitions = []
    for logical in physics_ir.correlators:
        topology = reconstruct_fermion_topology(logical)
        transitions.extend(
            transition
            for path in topology.open_paths
            for transition in path.transitions
        )
        transitions.extend(
            transition
            for cycle in topology.closed_cycles
            for transition in cycle.transitions
        )

    current_transitions = [
        transition
        for transition in transitions
        if (transition.incoming_flavor, transition.outgoing_flavor) == ("u", "s")
    ]
    assert len(current_transitions) == 1
    transition = current_transitions[0]
    assert transition.quark_endpoint.representation == Representation.POINT
    assert transition.antiquark_endpoint.representation == Representation.POINT
    assert any(tensor.kind == "gammaX" for tensor in transition.spin_tensors)
