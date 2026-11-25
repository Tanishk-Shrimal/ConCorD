"""Tests for concord_lqcd.core.fields — quark / antiquark factories."""

import pytest

from concord_lqcd.core import (
    QuarkField, Index,
    u, ubar, d, dbar, s, sbar, c, cbar, b, bbar, t, tbar,
    reset_registry, get_registry, COLOR, SPIN, POSITION,
)


# ── basic quark creation ──────────────────────────────────────────────

def test_u_creates_quark_field():
    reset_registry()
    q = u('x', 'a', 'α')
    assert isinstance(q, QuarkField)
    assert q.flavor == 'u'
    assert q.is_anti is False
    assert q.pos.label == 'x'
    assert q.pos.space == POSITION
    assert q.color.label == 'a'
    assert q.color.space == COLOR
    assert q.spin.label == 'α'
    assert q.spin.space == SPIN


def test_ubar_creates_antiquark_field():
    reset_registry()
    q = ubar('x', 'a', 'α')
    assert isinstance(q, QuarkField)
    assert q.flavor == 'u'
    assert q.is_anti is True
    assert q.pos.space == POSITION
    assert q.color.space == COLOR
    assert q.spin.space == SPIN


# ── all 12 factory functions ──────────────────────────────────────────

_FACTORIES = [
    (u,    'u', False),
    (ubar, 'u', True),
    (d,    'd', False),
    (dbar, 'd', True),
    (s,    's', False),
    (sbar, 's', True),
    (c,    'c', False),
    (cbar, 'c', True),
    (b,    'b', False),
    (bbar, 'b', True),
    (t,    't', False),
    (tbar, 't', True),
]


@pytest.mark.parametrize("factory,expected_flavor,expected_anti", _FACTORIES,
                         ids=[f.__name__ for f, _, _ in _FACTORIES])
def test_all_quark_factories(factory, expected_flavor, expected_anti):
    reset_registry()
    q = factory('x1', 'c1', 's1')
    assert isinstance(q, QuarkField)
    assert q.flavor == expected_flavor
    assert q.is_anti is expected_anti
    assert q.pos.label == 'x1'
    assert q.color.label == 'c1'
    assert q.spin.label == 's1'


# ── index identity ────────────────────────────────────────────────────

def test_same_label_and_space_give_equal_index_values():
    """Typed value equality, not Python object identity, identifies an index."""
    reset_registry()
    q1 = u('x', 'a', 'α')
    q2 = d('x', 'a', 'β')
    assert q1.color == q2.color
    assert hash(q1.color) == hash(q2.color)
    assert q1.pos == q2.pos


def test_different_labels_give_different_indices():
    reset_registry()
    q1 = u('x', 'a', 'α')
    q2 = d('y', 'b', 'β')
    assert q1.color != q2.color
    assert q1.spin != q2.spin
    assert q1.pos != q2.pos


# ── indices property ──────────────────────────────────────────────────

def test_indices_returns_pos_color_spin():
    reset_registry()
    q = s('x', 'a', 'α')
    idx = q.indices
    assert len(idx) == 3
    assert idx[0].space == POSITION
    assert idx[1].space == COLOR
    assert idx[2].space == SPIN


# ── repr ──────────────────────────────────────────────────────────────

def test_quark_repr():
    reset_registry()
    q = u('x', 'a', 'α')
    assert 'u(' in repr(q)
    assert 'x' in repr(q)
    assert 'a' in repr(q)
    assert 'α' in repr(q)


def test_antiquark_repr():
    reset_registry()
    q = dbar('x', 'b', 'β')
    assert 'dbar(' in repr(q)
