"""Exact symbolic whole-network matches, certificates, and portable bounds."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
import signal
import sys
from time import monotonic

import pytest

from concord_lqcd.compiler import _tensor_isomorphism as matching
from concord_lqcd.compiler._tensor_isomorphism import (
    Factor, Network, SearchAborted, SearchBudgetExceeded, find_witness, graph,
    index_kinds, invariant, parity, verify_witness,
)


def _find(left, right, **options):
    return find_witness(left, right, deadline=monotonic() + 30, max_states=100000, **options)


def _anchored(symmetry="alternating"):
    return Network((
        Factor("epsilon", ("a", "b", "c"), ("mode",) * 3, symmetry),
        Factor("tau_u_1_0", ("a", "alpha", "b", "beta"),
               ("mode", "spin", "mode", "spin")),
        Factor("tail", ("c", "rho"), ("mode", "spin")),
    ), ("alpha", "beta", "rho"))


def _renamed(value, names):
    return Network(tuple(replace(factor, labels=tuple(names.get(label, label) for label in factor.labels))
                         for factor in value.factors),
                   tuple(names.get(label, label) for label in value.outputs))


def _cycles(lengths):
    factors = []
    offset = 0
    for length in lengths:
        for position in range(length):
            labels = (f"i{offset + position}", f"i{offset + (position + 1) % length}")
            factors.append(Factor("edge", labels, ("mode", "mode"), "symmetric"))
        offset += length
    return Network(tuple(factors), ())


@pytest.mark.parametrize("symmetry", ["symmetric", "alternating"])
@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_renamed_dummies_permuted_factors_and_exact_phase(symmetry, order):
    left = _anchored(symmetry)
    names = {"a": "z", "b": "x", "c": "y"}
    right = _renamed(left, names)
    elemental = replace(right.factors[0], labels=tuple(right.factors[0].labels[port] for port in order))
    right = replace(right, factors=(right.factors[2], elemental, right.factors[1]))
    expected = (-1 if sum(order[first] > order[second] for first in range(3)
                         for second in range(first + 1, 3)) % 2 else 1)
    if symmetry == "symmetric":
        expected = 1
    left_graph, right_graph = graph(left), graph(right)
    assert invariant(left_graph) == invariant(right_graph)
    witness = _find(left, right, left_graph=left_graph, right_graph=right_graph)
    assert witness is not None
    assert witness["indices"] == {**names, "alpha": "alpha", "beta": "beta", "rho": "rho"}
    assert witness["phase"] == expected
    assert verify_witness(left, right, witness) == expected
    del witness["phase"]
    assert verify_witness(left, right, witness) == expected


def test_identical_factor_occurrences_are_permuted_bijectively_and_deterministically():
    left = Network((
        Factor("start", ("a",), ("mode",)),
        Factor("link", ("a", "b"), ("mode", "mode")),
        Factor("link", ("a", "b"), ("mode", "mode")),
        Factor("end", ("b",), ("mode",)),
    ), ())
    right = _renamed(replace(left, factors=tuple(reversed(left.factors))), {"a": "x", "b": "y"})
    witness = _find(left, right)
    assert witness == _find(left, right)
    assert verify_witness(left, right, witness) == 1
    assert sorted(record["target"] for record in witness["factors"]) == [0, 1, 2, 3]


@pytest.mark.parametrize("symmetry,expected", [("symmetric", 1), ("alternating", -1)])
def test_exchangeable_ports_preserve_kinds_not_original_kind_order(symmetry, expected):
    left = Network((Factor("mixed", ("m", "s"), ("mode", "spin"), symmetry),), ("m", "s"))
    right = Network((Factor("mixed", ("s", "m"), ("spin", "mode"), symmetry),), ("m", "s"))
    assert invariant(graph(left)) == invariant(graph(right))
    assert verify_witness(left, right, _find(left, right)) == expected


def test_multiple_alternating_factor_signs_multiply():
    left = Network((Factor("first", ("a", "b"), ("spin", "spin"), "alternating"),
                    Factor("second", ("a", "b"), ("spin", "spin"), "alternating")), ("a", "b"))
    right = replace(left, factors=tuple(replace(factor, labels=("b", "a")) for factor in left.factors))
    assert verify_witness(left, right, _find(left, right)) == 1


def test_wrong_claimed_phase_is_rejected():
    left = _anchored()
    right = replace(left, factors=(replace(left.factors[0], labels=("b", "a", "c")), *left.factors[1:]))
    witness = _find(left, right)
    assert witness["phase"] == -1
    witness["phase"] = 1
    with pytest.raises(ValueError, match="claimed phase"):
        verify_witness(left, right, witness)


@pytest.mark.parametrize("claim", [True, 0, 2, 1.0, "1", None])
def test_claimed_phase_must_be_an_integer_sign(claim):
    value = _anchored()
    witness = _find(value, value)
    witness["phase"] = claim
    with pytest.raises(ValueError, match="claimed phase"):
        verify_witness(value, value, witness)


def test_output_names_are_fixed_not_just_output_ordinals():
    left = _anchored()
    right = _renamed(left, {"alpha": "other_alpha"})
    assert _find(left, right) is None
    witness = _find(left, left)
    witness["indices"]["alpha"] = "other_alpha"
    with pytest.raises(ValueError, match="named external"):
        verify_witness(left, right, witness)


def test_output_order_and_named_output_mapping_are_fixed():
    value = _anchored()
    assert _find(value, replace(value, outputs=tuple(reversed(value.outputs)))) is None
    witness = _find(value, value)
    witness["indices"]["alpha"], witness["indices"]["beta"] = "beta", "alpha"
    with pytest.raises(ValueError, match="named external"):
        verify_witness(value, value, witness)


@pytest.mark.parametrize("order", [(2, 3, 0, 1), (0, 3, 2, 1), (2, 1, 0, 3)])
def test_all_ordered_propagation_ports_are_pinned(order):
    left = _anchored()
    left = replace(left, factors=(*left.factors, Factor("pin_modes", ("a", "b"), ("mode", "mode"))))
    propagation = left.factors[1]
    right = replace(left, factors=(left.factors[0],
                                  replace(propagation, labels=tuple(propagation.labels[port] for port in order)),
                                  *left.factors[2:]))
    assert _find(left, right) is None
    witness = _find(left, left)
    witness["factors"][1]["ports"] = list(order)
    with pytest.raises(ValueError, match="ordered tensor"):
        verify_witness(left, left, witness)


def test_valid_dummy_swap_changes_epsilon_sign_without_permuting_tau_ports():
    left = _anchored()
    right = replace(left, factors=(left.factors[0],
                                  replace(left.factors[1], labels=("b", "alpha", "a", "beta")),
                                  left.factors[2]))
    witness = _find(left, right)
    assert witness is not None
    assert witness["indices"]["a"] == "b"
    assert witness["indices"]["b"] == "a"
    assert witness["factors"][0]["ports"] == [1, 0, 2]
    assert witness["factors"][1]["ports"] == [0, 1, 2, 3]
    assert verify_witness(left, right, witness) == -1


@pytest.mark.parametrize("resource", ["tau_d_1_0", "tau_u_0_1", "other_recipe"])
def test_resource_flavor_and_source_sink_placement_cannot_change(resource):
    left = _anchored()
    right = replace(left, factors=(left.factors[0], replace(left.factors[1], resource=resource), left.factors[2]))
    assert _find(left, right) is None
    with pytest.raises(ValueError, match="resource"):
        verify_witness(left, right, _find(left, left))


@pytest.mark.parametrize("change", ["missing", "extra", "same_count_wrong_multiplicity"])
def test_factor_multiplicity_is_exact(change):
    left = _anchored()
    if change == "missing":
        right = replace(left, factors=left.factors[:2])
        right = replace(right, outputs=("alpha", "beta"))
    elif change == "extra":
        right = replace(left, factors=(*left.factors, left.factors[0]))
    else:
        left = replace(left, factors=(*left.factors, left.factors[0]))
        right = replace(left, factors=(*left.factors[:3], left.factors[1]))
    assert _find(left, right) is None


def test_missing_and_repeated_factor_records_are_rejected():
    value = _anchored()
    witness = _find(value, value)
    missing = deepcopy(witness)
    missing["factors"].pop()
    repeated = deepcopy(witness)
    repeated["factors"][1]["target"] = 0
    for invalid in (missing, repeated):
        with pytest.raises(ValueError, match="factor map"):
            verify_witness(value, value, invalid)


@pytest.mark.parametrize("target", [-1, 3, True, 1.0, "0", None])
def test_malformed_factor_targets_are_rejected(target):
    value = _anchored()
    witness = _find(value, value)
    witness["factors"][0]["target"] = target
    with pytest.raises(ValueError, match="factor map"):
        verify_witness(value, value, witness)


@pytest.mark.parametrize("ports", [[0, 0, 2], [0, 1], [0, 1, 3], [False, 1, 2], [0.0, 1, 2], None])
def test_malformed_port_permutations_are_rejected(ports):
    value = _anchored()
    witness = _find(value, value)
    witness["factors"][0]["ports"] = ports
    with pytest.raises(ValueError, match="port map"):
        verify_witness(value, value, witness)


def test_dummy_map_must_be_complete_bijective_and_kind_preserving():
    value = _anchored()
    witness = _find(value, value)
    missing = deepcopy(witness)
    del missing["indices"]["a"]
    extra = deepcopy(witness)
    extra["indices"]["extra"] = "a"
    repeated = deepcopy(witness)
    repeated["indices"]["b"] = "a"
    unhashable = deepcopy(witness)
    unhashable["indices"]["a"] = []
    for invalid in (missing, extra, repeated, unhashable):
        with pytest.raises(ValueError, match="index map"):
            verify_witness(value, value, invalid)
    witness["indices"]["a"], witness["indices"]["alpha"] = "alpha", "a"
    with pytest.raises(ValueError, match="index kinds"):
        verify_witness(value, value, witness)


def test_repeated_index_occurrences_are_not_collapsed():
    left = Network((Factor("trace", ("a", "a"), ("mode", "mode"), "symmetric"),), ())
    right = _renamed(left, {"a": "x"})
    assert len(graph(left)) == 4
    assert verify_witness(left, right, _find(left, right)) == 1
    different = Network((Factor("trace", ("x", "y"), ("mode", "mode"), "symmetric"),), ())
    assert _find(left, different) is None
    with pytest.raises(ValueError, match="bijection"):
        verify_witness(different, right, {"indices": {"x": "x", "y": "x"},
                                          "factors": [{"target": 0, "ports": [0, 1]}]})


def test_empty_and_scalar_networks():
    empty = Network((), ())
    scalar = Network((Factor("scalar", (), ()),), ())
    assert _find(empty, empty) == {"indices": {}, "factors": [], "phase": 1}
    assert verify_witness(scalar, scalar, _find(scalar, scalar)) == 1
    assert _find(empty, scalar) is None
    assert _find(scalar, replace(scalar, factors=scalar.factors * 2)) is None
    assert isinstance(hash(invariant(graph(empty))), int)


@pytest.mark.parametrize("value", [
    Network((Factor("bad", ("a",), ()),), ()),
    Network((Factor("bad", ("a",), ("mode",), "unknown"),), ()),
    Network((Factor("bad", ("a", "a"), ("mode", "spin")),), ()),
    Network((Factor("bad", ("a",), ("mode",)),), ("missing",)),
    Network((Factor("bad", ("a",), ("mode",)),), ("a", "a")),
])
def test_invalid_networks_fail_validation(value):
    with pytest.raises(ValueError):
        index_kinds(value)
    with pytest.raises(ValueError):
        graph(value)
    with pytest.raises(ValueError):
        verify_witness(value, value, {})


def test_real_refinement_collision_is_not_an_equality_certificate():
    left, right = _cycles((6,)), _cycles((3, 3))
    key = invariant(graph(left))
    assert key == invariant(graph(right))
    assert {key: "candidate"}[invariant(graph(right))] == "candidate"
    assert _find(left, right) is None
    with pytest.raises(ValueError, match="incidence"):
        verify_witness(left, right, _find(left, left))


def test_forced_candidate_key_collision_cannot_authorize_a_witness(monkeypatch):
    left, right = _cycles((6,)), _cycles((3, 3))
    monkeypatch.setattr(matching, "invariant", lambda *args, **kwargs: "collision")
    assert matching.invariant(graph(left)) == matching.invariant(graph(right))
    assert _find(left, right) is None
    with pytest.raises(ValueError):
        verify_witness(left, right, _find(left, left))


def test_successful_search_backtracks_after_a_wrong_component_choice():
    left, right = _cycles((3, 4)), _cycles((4, 3))
    witness = _find(left, right)
    assert witness is not None
    assert verify_witness(left, right, witness) == 1


def test_verifier_is_independent_of_matching_refinement_and_graph_library(monkeypatch):
    value = _anchored()
    witness = _find(value, value)

    def forbidden(*args, **kwargs):
        pytest.fail("The independent verifier called graph machinery")

    for name in ("graph", "_graph", "invariant", "_refine", "find_witness"):
        monkeypatch.setattr(matching, name, forbidden)
    monkeypatch.setattr(matching.nx, "Graph", forbidden)
    assert verify_witness(value, value, witness) == 1


def test_supplied_graphs_are_validated_not_trusted():
    value = _anchored()
    supplied = graph(value)
    assert verify_witness(value, value, _find(value, value, left_graph=supplied.copy())) == 1
    supplied.nodes["index", "a"]["color"] = "forged"
    with pytest.raises(ValueError, match="supplied graph"):
        _find(value, value, left_graph=supplied)
    supplied = graph(value)
    supplied.remove_edge(("factor", 0), ("leg", 0, 0))
    supplied.add_edge(("factor", 0), ("leg", 1, 0))
    with pytest.raises(ValueError, match="supplied graph"):
        _find(value, value, right_graph=supplied)


@pytest.mark.parametrize("value", [Network((), ()), _anchored()])
def test_zero_state_budget_always_exhausts(value):
    with pytest.raises(SearchBudgetExceeded, match="state budget"):
        find_witness(value, value, deadline=monotonic() + 30, max_states=0)


def test_state_exhaustion_never_returns_a_partial_or_negative_witness():
    left, right = _cycles((6,)), _cycles((3, 3))
    with pytest.raises(SearchBudgetExceeded, match="state budget"):
        find_witness(left, right, deadline=monotonic() + 30, max_states=2)
    assert _find(left, right) is None


def test_expired_deadline_is_a_hard_abort_not_an_unproved_result():
    value = _anchored()
    expired = monotonic() - 1
    with pytest.raises(SearchAborted, match="deadline"):
        find_witness(value, value, deadline=expired, max_states=10000)
    with pytest.raises(SearchAborted, match="deadline"):
        invariant(graph(value), deadline=expired)
    with pytest.raises(SearchAborted, match="deadline"):
        graph(value, deadline=expired)
    assert not issubclass(SearchAborted, SearchBudgetExceeded)
    assert verify_witness(value, value, find_witness(value, value, max_states=10000)) == 1


def test_deadline_is_checked_during_preprocessing(monkeypatch):
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return calls

    monkeypatch.setattr(matching, "monotonic", clock)
    with pytest.raises(SearchAborted, match="deadline"):
        find_witness(_anchored(), _anchored(), deadline=10, max_states=10000)
    value_graph = graph(_cycles((8,)))
    calls = 0
    with pytest.raises(SearchAborted, match="deadline"):
        invariant(value_graph, deadline=10)


def _vectors(count, prefix):
    return Network(tuple(Factor("vector", (f"{prefix}{ordinal}",), ("mode",)) for ordinal in range(count)), ())


def test_deadline_is_checked_during_iterative_search(monkeypatch):
    current = 0
    calls = 0
    original = matching._individualize

    def individualize(*args):
        nonlocal current, calls
        result = original(*args)
        calls += 1
        if calls == 3:
            current = 2
        return result

    monkeypatch.setattr(matching, "monotonic", lambda: current)
    monkeypatch.setattr(matching, "_individualize", individualize)
    with pytest.raises(SearchAborted, match="deadline"):
        find_witness(_vectors(6, "left"), _vectors(6, "right"), deadline=1, max_states=10000)
    assert calls == 3


def test_outcome_depends_on_states_not_on_the_clock(monkeypatch):
    left, right = _cycles((3, 4)), _cycles((4, 3))
    expected = find_witness(left, right, max_states=10000)
    ticks = iter(range(10 ** 9))
    monkeypatch.setattr(matching, "monotonic", lambda: next(ticks))
    assert find_witness(left, right, deadline=10 ** 9, max_states=10000) == expected
    assert find_witness(_cycles((6,)), _cycles((3, 3)), deadline=10 ** 9, max_states=10000) is None


def test_refinement_proves_regular_nonisomorphism_in_few_states():
    with pytest.raises(SearchBudgetExceeded, match="state budget"):
        find_witness(_cycles((12,)), _cycles((6, 6)), max_states=2)
    assert find_witness(_cycles((12,)), _cycles((6, 6)), max_states=13) is None


def test_witness_is_reproducible_for_highly_symmetric_networks():
    left, right = _vectors(8, "left"), _vectors(8, "right")
    witnesses = {repr(find_witness(left, right, max_states=100)) for _ in range(3)}
    assert len(witnesses) == 1
    assert verify_witness(left, right, find_witness(left, right, max_states=100)) == 1


@pytest.mark.parametrize("max_states", [-1, True, 1.0, None])
def test_invalid_state_budgets(max_states):
    with pytest.raises(ValueError, match="max_states"):
        find_witness(_anchored(), _anchored(), deadline=monotonic() + 30, max_states=max_states)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, "soon"])
def test_invalid_deadlines(deadline):
    with pytest.raises(ValueError, match="deadline"):
        find_witness(_anchored(), _anchored(), deadline=deadline, max_states=10000)


def test_iterative_search_exceeds_python_recursion_depth_without_changing_it():
    recursion_limit = sys.getrecursionlimit()
    count = recursion_limit // 3 + 2
    left = Network(tuple(Factor("vector", (f"left{ordinal}",), ("mode",)) for ordinal in range(count)), ())
    right = Network(tuple(Factor("vector", (f"right{ordinal}",), ("mode",)) for ordinal in range(count)), ())
    assert len(graph(left)) > recursion_limit
    witness = _find(left, right)
    assert verify_witness(left, right, witness) == 1
    assert sys.getrecursionlimit() == recursion_limit


def test_no_signal_or_uncontrolled_library_matcher_effects_even_in_worker_thread(monkeypatch):
    handler = signal.getsignal(signal.SIGALRM) if hasattr(signal, "SIGALRM") else None
    timer = signal.getitimer(signal.ITIMER_REAL) if hasattr(signal, "getitimer") else None

    def forbidden(*args, **kwargs):
        pytest.fail("Matching used process signals or an uncontrolled library matcher")

    for name in ("signal", "setitimer", "alarm"):
        if hasattr(signal, name):
            monkeypatch.setattr(signal, name, forbidden)
    for name in ("vf2pp_isomorphism", "is_isomorphic"):
        if hasattr(matching.nx, name):
            monkeypatch.setattr(matching.nx, name, forbidden)

    def run():
        value = _anchored()
        assert verify_witness(value, value, _find(value, value)) == 1
        with pytest.raises(SearchAborted):
            find_witness(value, value, deadline=monotonic() - 1, max_states=10000)
        with pytest.raises(SearchBudgetExceeded):
            find_witness(value, value, deadline=monotonic() + 30, max_states=0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(run).result(timeout=30)
    if hasattr(signal, "SIGALRM"):
        assert signal.getsignal(signal.SIGALRM) == handler
    if timer is not None:
        assert signal.getitimer(signal.ITIMER_REAL) == timer


@pytest.mark.parametrize("rank", range(5))
def test_parity_matches_independent_inversion_count(rank):
    for order in permutations(range(rank)):
        inversions = sum(order[first] > order[second] for first in range(rank)
                         for second in range(first + 1, rank))
        assert parity(order) == (-1 if inversions % 2 else 1)


@pytest.mark.parametrize("order", [(1,), (0, 0), (-1, 0), (False,), (0.0,)])
def test_parity_rejects_nonpermutations(order):
    with pytest.raises(ValueError, match="permutation"):
        parity(order)
