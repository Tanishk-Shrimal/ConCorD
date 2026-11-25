"""Convenience constructors for common baryon interpolators."""

from __future__ import annotations

from collections.abc import Sequence

from .color import epsilon
from .fields import b, c, d, l, s, t, u, QuarkField
from .gamma import spin_matrix
from .operator import Momentum, Operator

_FIELD_FACTORIES = {
    "u": u,
    "d": d,
    "l": l,
    "s": s,
    "c": c,
    "b": b,
    "t": t,
}


def _quark_factory(flavor: str):
    try:
        return _FIELD_FACTORIES[flavor]
    except KeyError as exc:
        known = ", ".join(sorted(_FIELD_FACTORIES))
        raise ValueError(f"Unknown quark flavor {flavor!r}; expected one of: {known}.") from exc


def build_baryon_operator(
    flavors: Sequence[str],
    diquark_gamma: str = "Cg5",
    momentum: Momentum = (0, 0, 0),
    pos_label: str = "x",
    *,
    color_labels: tuple[str, str, str] = ("a", "b", "c"),
    spin_labels: tuple[str, str, str] = ("al", "be", "de"),
    name: str | None = None,
) -> Operator:
    """Build a local three-quark baryon interpolating operator.

    The returned operator has the form::

        ε_abc (Γ_D)_αβ q0(x,a,α) q1(x,b,β) q2(x,c,δ)

    where ``Γ_D`` is selected by ``diquark_gamma`` and the third quark carries
    the spectator spin index.  Eigenvectors remain spinless in the backend; the
    spin matrix is represented symbolically and folded only by lowering passes.
    """
    if len(flavors) != 3:
        raise ValueError("A baryon operator requires exactly three quark flavors.")
    if len(color_labels) != 3 or len(spin_labels) != 3:
        raise ValueError("color_labels and spin_labels must each contain exactly three labels.")

    q_fields: list[QuarkField] = [
        _quark_factory(flavor)(pos_label, color_labels[i], spin_labels[i])
        for i, flavor in enumerate(flavors)
    ]

    expr = (
        epsilon(*color_labels)
        * spin_matrix(diquark_gamma, spin_labels[0], spin_labels[1])
        * q_fields[0]
        * q_fields[1]
        * q_fields[2]
    )
    metadata = {"name": name} if name is not None else {}
    return Operator(expr, momenta={pos_label: momentum}, **metadata)
