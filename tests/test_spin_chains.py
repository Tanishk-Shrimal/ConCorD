"""Exact fixed-spin path identities, independent replay and conservative limits."""

from dataclasses import replace
from fractions import Fraction
from time import monotonic

import numpy as np
import pytest

from concord_lqcd.compiler._spin_chains import (
    _ExactComplex, _canonical, _matrix, _product, _scalar_replay,
    discover_spin_chains, replay_spin_chains,
)
from concord_lqcd.compiler._tensor_isomorphism import Factor, Network, SearchAborted


def _spin(values, labels):
    return Factor("fixed-spin:" + np.asarray(values, dtype="<c16").tobytes().hex(),
                  labels, ("spin", "spin"))


def _chain():
    first = np.array([[1, 2j, 0, 0], [3, 0, 4, 0], [0, 0, 1j, 0], [2, 0, 0, 1]])
    second = np.array([[0, 1, 0, 0], [2j, 0, 0, 0], [0, 0, 3, 2], [1, 0, 0, 4]])
    return Network((_spin(first, ("a", "middle")), _spin(second, ("b", "middle"))), ("a", "b"))


def test_exact_product_and_independent_scalar_replay_agree_with_integer_numpy():
    network = _chain()
    first = _matrix(network.factors[0].resource)
    second = tuple(zip(*_matrix(network.factors[1].resource), strict=True))
    actual = _product((first, second))
    assert actual == _scalar_replay((first, second))
    first_array = np.array([[complex(value) for value in row] for row in first])
    second_array = np.array([[complex(value) for value in row] for row in second])
    np.testing.assert_array_equal(np.array([[complex(value) for value in row] for row in actual]),
                                  first_array @ second_array)
    folded, phase, paths = discover_spin_chains(network, deadline=monotonic() + 5)
    assert replay_spin_chains(network, paths) == (folded, phase)
    assert folded.outputs == network.outputs
    assert len(folded.factors) == 1


@pytest.mark.parametrize("paths", [[], (((-1, False),),), (((True, False),),), (((0, 0),),),
                                   (((0, False), (0, False)),), (((0, True), (1, False)),),
                                   (((0, False), (1, False)),)])
def test_invalid_saved_paths_rejected(paths):
    with pytest.raises(ValueError):
        replay_spin_chains(_chain(), paths)


def test_source_conjugation_and_field_adjoints_are_not_omitted():
    first = np.array([[1, 2j, 0, 0], [3, 0, 4, 0], [0, 0, 1j, 0], [2, 0, 0, 1]])
    temporal = np.diag([1, 1, -1, -1])
    network = Network((_spin(temporal, ("a", "r")), _spin(first.conj(), ("r", "s")),
                       _spin(temporal, ("s", "b"))), ("a", "b"))
    folded, phase, paths = discover_spin_chains(network, deadline=monotonic() + 5)
    assert replay_spin_chains(network, paths) == (folded, phase)
    actual = _scalar_replay(tuple(_matrix(factor.resource) for factor in network.factors))
    np.testing.assert_array_equal(np.array([[complex(value) for value in row] for row in actual]),
                                  temporal @ first.conj() @ temporal)


def test_tiny_stored_asymmetry_is_preserved_exactly():
    matrix = np.eye(4)
    symmetric = _matrix(_spin(matrix, ("a", "b")).resource)
    matrix[0, 1] = 2.0 ** -60
    asymmetric = _matrix(_spin(matrix, ("a", "b")).resource)
    assert asymmetric[0][1].real == Fraction(1, 2**60)
    assert _canonical(symmetric)[3] == "symmetric"
    assert _canonical(asymmetric)[3] == "ordered"
    assert _canonical(symmetric)[0] != _canonical(asymmetric)[0]


def test_closed_cycles_and_long_paths_keep_original_factors():
    matrix = np.eye(4)
    cycle = Network((_spin(matrix, ("a", "b")), _spin(matrix, ("b", "a"))), ())
    long_path = Network(tuple(_spin(matrix, (f"spin{index}", f"spin{index + 1}"))
                              for index in range(7)), ("spin0", "spin7"))
    for network in (cycle, long_path):
        folded, phase, paths = discover_spin_chains(network, deadline=monotonic() + 5)
        assert folded == network
        assert phase == _ExactComplex.from_complex(1)
        assert paths == ()


def test_public_internal_axis_is_not_hidden():
    network = replace(_chain(), outputs=("a", "b", "middle"))
    folded, phase, paths = discover_spin_chains(network, deadline=monotonic() + 5)
    assert folded == network and paths == ()
    with pytest.raises(ValueError):
        replay_spin_chains(network, (((0, False), (1, True)),))


def test_propagator_ports_remain_untouched_and_cannot_join_a_spin_path():
    propagation = Factor("tau", ("r", "s"), ("spin", "spin"))
    network = Network((_spin(np.eye(4), ("a", "r")), propagation,
                       _spin(np.eye(4), ("s", "b"))), ("a", "b"))
    folded, phase, paths = discover_spin_chains(network, deadline=monotonic() + 5)
    assert propagation in folded.factors and len(folded.factors) == 3
    assert replay_spin_chains(network, paths) == (folded, phase)
    with pytest.raises(ValueError):
        replay_spin_chains(network, (((0, False), (1, False), (2, False)),))


def test_expired_discovery_deadline_aborts_instead_of_authorizing_a_rewrite():
    with pytest.raises(SearchAborted):
        discover_spin_chains(_chain(), deadline=monotonic() - 1)


@pytest.mark.parametrize("values", [np.ones(15), np.full(16, np.nan), np.full(16, np.inf)])
def test_invalid_constant_is_rejected(values):
    resource = "fixed-spin:" + np.asarray(values, dtype="<c16").tobytes().hex()
    with pytest.raises(ValueError):
        _matrix(resource)
