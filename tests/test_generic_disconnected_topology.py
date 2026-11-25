"""Focused tests for generic open-path / closed-cycle fermion topology."""

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
    dbar,
    gamma5,
    gammaX,
    point,
    reset_registry,
    s,
    sbar,
    u,
    ubar,
)
from concord_lqcd.core.wick import WickTerm, wick_contract


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


def test_mesonic_two_point_is_one_closed_fermion_cycle():
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

    topology = reconstruct_fermion_topology(correlator)

    assert topology.num_open_paths == 0
    assert topology.num_closed_cycles == 1
    cycle = topology.closed_cycles[0]
    assert len(cycle.segments) == 2
    assert len(cycle.transitions) == 2
    assert cycle.operator_ids == frozenset({0, 1})
    assert not cycle.is_local


def test_same_side_mesonic_contractions_become_two_local_closed_cycles():
    sink_loop = LogicalQuarkLine(
        quark=_q("u", False, "c0:op0:y", "b0", "q0", 0),
        antiquark=_q("u", True, "c0:op0:y", "a0", "a0s", 0),
    )
    source_loop = LogicalQuarkLine(
        quark=_q("u", False, "c0:op1:x", "b1", "q1", 1),
        antiquark=_q("u", True, "c0:op1:x", "a1", "a1s", 1),
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(sink_loop, source_loop),
        spin_tensors=(
            SpinTensorRecord("gamma5", "a0s", "q0", operator_id=0),
            SpinTensorRecord("gamma5", "a1s", "q1", operator_id=1),
        ),
    )

    topology = reconstruct_fermion_topology(correlator)

    assert topology.num_open_paths == 0
    assert topology.num_closed_cycles == 2
    assert all(len(cycle.segments) == 1 for cycle in topology.closed_cycles)
    assert all(cycle.is_local for cycle in topology.closed_cycles)
    assert {cycle.operator_ids for cycle in topology.closed_cycles} == {
        frozenset({0}),
        frozenset({1}),
    }


def test_two_pion_exchange_is_not_classified_by_coarse_wick_topology():
    reset_registry()
    sink = Operator(
        dbar("x", "a", "s1") * gamma5("s1", "s2") * u("x", "a", "s2")
        * dbar("y", "b", "s3") * gamma5("s3", "s4") * u("y", "b", "s4"),
        momenta={"x": (0, 0, 0), "y": (0, 0, 0)},
    )
    correlator = Correlator(sink, sink.dagger())
    terms = wick_contract(correlator)

    assert len(terms) == 4
    assert {term.topology for term in terms} == {"connected"}
    assert not hasattr(WickTerm, "factorized")
    assert sorted(len(term.components) for term in terms) == [1, 1, 2, 2]

    physics_ir = lower_to_physics_ir(correlator, terms)
    cycle_lengths = []
    for logical in physics_ir.correlators:
        topology = reconstruct_fermion_topology(logical)
        assert topology.num_open_paths == 0
        cycle_lengths.append(tuple(sorted(len(cycle.segments) for cycle in topology.closed_cycles)))
    assert sorted(cycle_lengths) == [(2, 2), (2, 2), (4,), (4,)]


def test_baryon_meson_like_two_point_has_open_paths_plus_closed_cycle():
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

    topology = reconstruct_fermion_topology(correlator)

    assert topology.num_open_paths == 3
    assert sorted(path.flavors for path in topology.open_paths) == [
        ("c",),
        ("d",),
        ("u",),
    ]
    assert topology.num_closed_cycles == 1
    assert topology.closed_cycles[0].flavors == ("s", "s")
    assert topology.closed_cycles[0].operator_ids == frozenset({0, 1})


def test_factorized_two_meson_like_term_is_two_independent_cycles():
    lines = (
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op0:y", "b0u", "q0u", 0),
            antiquark=_q("u", True, "c0:op1:x", "a1u", "a1us", 1),
        ),
        LogicalQuarkLine(
            quark=_q("u", False, "c0:op1:x", "b1u", "q1u", 1),
            antiquark=_q("u", True, "c0:op0:y", "a0u", "a0us", 0),
        ),
        LogicalQuarkLine(
            quark=_q("s", False, "c0:op0:y", "b0s", "q0s", 0),
            antiquark=_q("s", True, "c0:op1:x", "a1s", "a1ss", 1),
        ),
        LogicalQuarkLine(
            quark=_q("s", False, "c0:op1:x", "b1s", "q1s", 1),
            antiquark=_q("s", True, "c0:op0:y", "a0s", "a0ss", 0),
        ),
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=lines,
        spin_tensors=(
            SpinTensorRecord("gamma5", "a0us", "q0u", operator_id=0),
            SpinTensorRecord("gamma5", "a1us", "q1u", operator_id=1),
            SpinTensorRecord("gammaX", "a0ss", "q0s", operator_id=0),
            SpinTensorRecord("gammaX", "a1ss", "q1s", operator_id=1),
        ),
    )

    topology = reconstruct_fermion_topology(correlator)

    assert topology.num_open_paths == 0
    assert topology.num_closed_cycles == 2
    assert sorted(cycle.flavors for cycle in topology.closed_cycles) == [
        ("s", "s"),
        ("u", "u"),
    ]


def test_disconnected_local_current_adds_local_cycle_to_three_point_topology():
    hadron_forward = LogicalQuarkLine(
        quark=_q("u", False, "c0:op0:y", "b0", "q0", 0),
        antiquark=_q("u", True, "c0:op2:x", "a2", "a2s", 2),
    )
    hadron_backward = LogicalQuarkLine(
        quark=_q("u", False, "c0:op2:x", "b2", "q2", 2),
        antiquark=_q("u", True, "c0:op0:y", "a0", "a0s", 0),
    )
    current_loop = LogicalQuarkLine(
        quark=_q(
            "s", False, "c0:op1:z", "jb", "jq", 1,
            rep=Representation.POINT,
        ),
        antiquark=_q(
            "s", True, "c0:op1:z", "ja", "jas", 1,
            rep=Representation.POINT,
        ),
    )
    correlator = LogicalCorrelator(
        coefficient=1.0,
        quark_lines=(hadron_forward, hadron_backward, current_loop),
        spin_tensors=(
            SpinTensorRecord("gamma5", "a0s", "q0", operator_id=0),
            SpinTensorRecord("gammaX", "jas", "jq", operator_id=1),
            SpinTensorRecord("gamma5", "a2s", "q2", operator_id=2),
        ),
    )

    topology = reconstruct_fermion_topology(correlator)

    assert topology.num_open_paths == 0
    assert topology.num_closed_cycles == 2
    local_cycles = [cycle for cycle in topology.closed_cycles if cycle.is_local]
    spanning_cycles = [cycle for cycle in topology.closed_cycles if not cycle.is_local]
    assert len(local_cycles) == 1
    assert len(spanning_cycles) == 1
    assert local_cycles[0].flavors == ("s",)
    assert local_cycles[0].operator_ids == frozenset({1})
    assert spanning_cycles[0].operator_ids == frozenset({0, 2})


def test_frontend_three_point_strange_current_produces_disconnected_local_cycle():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * u("y", "a", "s2"),
        momentum=(0, 0, 0),
    )
    current = Operator(
        point(sbar("z", "b", "s3"))
        * gammaX("s3", "s4")
        * point(s("z", "b", "s4")),
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
    assert all(term.topology in {"disconnected", "mixed"} for term in wick_terms)

    physics_ir = lower_to_physics_ir(correlator, wick_terms)
    assert physics_ir.correlators

    for logical in physics_ir.correlators:
        topology = reconstruct_fermion_topology(logical)
        local_cycles = [cycle for cycle in topology.closed_cycles if cycle.is_local]
        assert any(
            cycle.operator_ids == frozenset({1}) and cycle.flavors == ("s",)
            for cycle in local_cycles
        )
