"""Tests for concord_lqcd.core.gamma — gamma matrix factories."""

import pytest

from concord_lqcd.core import (
    GammaMatrix,
    gamma5, gammaX, gammaY, gammaZ, gammaT, gamma_identity,
    reset_registry, SPIN,
)


# ── gamma5 basic ──────────────────────────────────────────────────────

def test_gamma5_creates_gamma_matrix():
    reset_registry()
    g = gamma5('α', 'β')
    assert isinstance(g, GammaMatrix)
    assert g.kind == 'gamma5'


def test_gamma5_spin_indices():
    reset_registry()
    g = gamma5('α', 'β')
    assert g.left.label == 'α'
    assert g.left.space == SPIN
    assert g.right.label == 'β'
    assert g.right.space == SPIN


def test_gamma5_indices_tuple():
    reset_registry()
    g = gamma5('α', 'β')
    assert len(g.indices) == 2
    assert g.indices == (g.left, g.right)


# ── all 6 factories ──────────────────────────────────────────────────

_GAMMA_FACTORIES = [
    (gamma5,         'gamma5'),
    (gammaX,         'gammaX'),
    (gammaY,         'gammaY'),
    (gammaZ,         'gammaZ'),
    (gammaT,         'gammaT'),
    (gamma_identity, 'gamma_id'),
]


@pytest.mark.parametrize("factory,expected_kind", _GAMMA_FACTORIES,
                         ids=[name for _, name in _GAMMA_FACTORIES])
def test_all_gamma_factories(factory, expected_kind):
    reset_registry()
    g = factory('s1', 's2')
    assert isinstance(g, GammaMatrix)
    assert g.kind == expected_kind
    assert g.left.space == SPIN
    assert g.right.space == SPIN
    assert g.left.label == 's1'
    assert g.right.label == 's2'


# ── repr ──────────────────────────────────────────────────────────────

def test_gamma5_repr():
    reset_registry()
    g = gamma5('α', 'β')
    assert repr(g) == 'gamma5(α,β)'


def test_gammaX_repr():
    reset_registry()
    g = gammaX('a', 'b')
    assert repr(g) == 'gammaX(a,b)'


def test_gamma_identity_repr():
    reset_registry()
    g = gamma_identity('x', 'y')
    assert repr(g) == 'gamma_id(x,y)'
