"""Tests for concord_lqcd.core.expr — Product algebra."""

from concord_lqcd.core import (
    Product,
    u, dbar, gamma5, delta,
    reset_registry,
)


# ── Product basics ────────────────────────────────────────────────────

def test_field_times_gamma_creates_product():
    reset_registry()
    f = u('x', 'a', 'α')
    g = gamma5('α', 'β')
    p = f * g
    assert isinstance(p, Product)
    assert len(p) == 2


def test_three_way_product_is_flat():
    reset_registry()
    f1 = dbar('x', 'a', 'α')
    g = gamma5('α', 'β')
    f2 = u('x', 'a', 'β')
    p = f1 * g * f2
    assert isinstance(p, Product)
    assert len(p) == 3
    # No nested Products
    for factor in p:
        assert not isinstance(factor, Product)


def test_product_times_product_flattens():
    reset_registry()
    f1 = dbar('x', 'a', 'α')
    g = gamma5('α', 'β')
    f2 = u('x', 'b', 'β')
    d_ = delta('a', 'b')
    p1 = f1 * g
    p2 = f2 * d_
    combined = p1 * p2
    assert isinstance(combined, Product)
    assert len(combined) == 4


# ── repr ──────────────────────────────────────────────────────────────

def test_product_repr_uses_star():
    reset_registry()
    f = u('x', 'a', 'α')
    g = gamma5('α', 'β')
    p = f * g
    r = repr(p)
    assert ' * ' in r


# ── len and indexing ──────────────────────────────────────────────────

def test_product_len():
    reset_registry()
    p = dbar('x', 'a', 'α') * gamma5('α', 'β') * u('x', 'a', 'β')
    assert len(p) == 3


def test_product_getitem():
    reset_registry()
    f = dbar('x', 'a', 'α')
    g = gamma5('α', 'β')
    q = u('x', 'a', 'β')
    p = f * g * q
    assert p[0] is f
    assert p[1] is g
    assert p[2] is q


# ── indices aggregation ──────────────────────────────────────────────

def test_product_indices_aggregates_all():
    """Product.indices collects all indices from all factors."""
    reset_registry()
    f = dbar('x', 'a', 'α')
    g = gamma5('α', 'β')
    q = u('x', 'b', 'β')
    d_ = delta('a', 'b')
    p = f * g * q * d_
    all_idx = p.indices
    # dbar: (x_pos, a_color, α_spin), gamma5: (α_spin, β_spin),
    # u: (x_pos, b_color, β_spin), delta: (a_color, b_color)
    assert len(all_idx) == 10
    labels = [i.label for i in all_idx]
    assert labels.count('x') == 2
    assert labels.count('a') == 2
    assert labels.count('b') == 2
    assert labels.count('α') == 2
    assert labels.count('β') == 2
