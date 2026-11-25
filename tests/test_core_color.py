"""Tests for concord_lqcd.core.color — color tensor factories."""

from concord_lqcd.core import (
    ColorDelta, ColorEpsilon,
    delta, epsilon,
    reset_registry, COLOR,
)


# ── ColorDelta ────────────────────────────────────────────────────────

def test_delta_creates_color_delta():
    reset_registry()
    d = delta('a', 'b')
    assert isinstance(d, ColorDelta)


def test_delta_color_indices():
    reset_registry()
    d = delta('a', 'b')
    assert d.left.label == 'a'
    assert d.left.space == COLOR
    assert d.right.label == 'b'
    assert d.right.space == COLOR


def test_delta_indices_tuple():
    reset_registry()
    d = delta('a', 'b')
    assert len(d.indices) == 2
    assert d.indices == (d.left, d.right)


# ── ColorEpsilon ──────────────────────────────────────────────────────

def test_epsilon_creates_color_epsilon():
    reset_registry()
    e = epsilon('a', 'b', 'c')
    assert isinstance(e, ColorEpsilon)


def test_epsilon_color_indices():
    reset_registry()
    e = epsilon('a', 'b', 'c')
    assert e.i.label == 'a'
    assert e.i.space == COLOR
    assert e.j.label == 'b'
    assert e.j.space == COLOR
    assert e.k.label == 'c'
    assert e.k.space == COLOR


def test_epsilon_indices_tuple():
    reset_registry()
    e = epsilon('a', 'b', 'c')
    assert len(e.indices) == 3
    assert e.indices == (e.i, e.j, e.k)


# ── repr uses Unicode δ and ε ─────────────────────────────────────────

def test_delta_repr_uses_delta_symbol():
    reset_registry()
    d = delta('a', 'b')
    r = repr(d)
    assert r == 'δ(a,b)'
    assert 'δ' in r


def test_epsilon_repr_uses_epsilon_symbol():
    reset_registry()
    e = epsilon('a', 'b', 'c')
    r = repr(e)
    assert r == 'ε(a,b,c)'
    assert 'ε' in r
