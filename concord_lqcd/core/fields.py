"""Symbolic indexed quark-field leaves and flavor factories."""

from __future__ import annotations

from typing import Any

from .expr import Expr, Product
from .indices import Index, POSITION, COLOR, SPIN


def quark_fields(expr: Expr) -> list[QuarkField]:
    """Return all quark-field leaves in encounter order.

    Products are flat by construction; ``Apply`` nodes are transparent for
    field discovery because the transformation acts on the contained field.
    """
    from .operations import Apply

    if isinstance(expr, Apply):
        return quark_fields(expr.operand)
    if isinstance(expr, Product):
        result: list[QuarkField] = []
        for factor in expr.factors:
            result.extend(quark_fields(factor))
        return result
    if isinstance(expr, QuarkField):
        return [expr]
    return []


class QuarkField(Expr):
    """An indexed quark or antiquark field leaf."""

    __slots__ = ("_flavor", "_is_anti", "_pos", "_color", "_spin")

    def __init__(
        self,
        flavor: str,
        is_anti: bool,
        pos: Index,
        color: Index,
        spin: Index,
    ) -> None:
        self._flavor = flavor
        self._is_anti = is_anti
        self._pos = pos
        self._color = color
        self._spin = spin

    @property
    def flavor(self) -> str:
        return self._flavor

    @property
    def is_anti(self) -> bool:
        return self._is_anti

    @property
    def pos(self) -> Index:
        return self._pos

    @property
    def color(self) -> Index:
        return self._color

    @property
    def spin(self) -> Index:
        return self._spin

    @property
    def indices(self) -> tuple[Index, ...]:
        return (self._pos, self._color, self._spin)

    @property
    def output_type(self) -> Any:
        from .operations import FieldType

        return FieldType.QUARK

    def __repr__(self) -> str:
        name = self._flavor + ("bar" if self._is_anti else "")
        return f"{name}({self._pos.label},{self._color.label},{self._spin.label})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuarkField):
            return NotImplemented
        return (
            self._flavor == other._flavor
            and self._is_anti == other._is_anti
            and self._pos == other._pos
            and self._color == other._color
            and self._spin == other._spin
        )

    def __hash__(self) -> int:
        return hash((self._flavor, self._is_anti, self._pos, self._color, self._spin))


def _make_quark(
    flavor: str,
    is_anti: bool,
    pos_label: str,
    color_label: str,
    spin_label: str,
) -> QuarkField:
    return QuarkField(
        flavor,
        is_anti,
        Index(pos_label, POSITION),
        Index(color_label, COLOR),
        Index(spin_label, SPIN),
    )


# Named field factories keep operator definitions close to conventional notation.
def u(pos: str, color: str, spin: str) -> QuarkField:
    """Up quark field: ``u(pos, color, spin)``."""
    return _make_quark("u", False, pos, color, spin)


def ubar(pos: str, color: str, spin: str) -> QuarkField:
    """Up antiquark field."""
    return _make_quark("u", True, pos, color, spin)


def d(pos: str, color: str, spin: str) -> QuarkField:
    """Down quark field."""
    return _make_quark("d", False, pos, color, spin)


def dbar(pos: str, color: str, spin: str) -> QuarkField:
    """Down antiquark field."""
    return _make_quark("d", True, pos, color, spin)


def s(pos: str, color: str, spin: str) -> QuarkField:
    """Strange quark field."""
    return _make_quark("s", False, pos, color, spin)


def sbar(pos: str, color: str, spin: str) -> QuarkField:
    """Strange antiquark field."""
    return _make_quark("s", True, pos, color, spin)


def l(pos: str, color: str, spin: str) -> QuarkField:
    """Generic light quark field."""
    return _make_quark("l", False, pos, color, spin)


def lbar(pos: str, color: str, spin: str) -> QuarkField:
    """Generic light antiquark field."""
    return _make_quark("l", True, pos, color, spin)


def c(pos: str, color: str, spin: str) -> QuarkField:
    """Charm quark field."""
    return _make_quark("c", False, pos, color, spin)


def cbar(pos: str, color: str, spin: str) -> QuarkField:
    """Charm antiquark field."""
    return _make_quark("c", True, pos, color, spin)


def b(pos: str, color: str, spin: str) -> QuarkField:
    """Bottom quark field."""
    return _make_quark("b", False, pos, color, spin)


def bbar(pos: str, color: str, spin: str) -> QuarkField:
    """Bottom antiquark field."""
    return _make_quark("b", True, pos, color, spin)


def t(pos: str, color: str, spin: str) -> QuarkField:
    """Top quark field."""
    return _make_quark("t", False, pos, color, spin)


def tbar(pos: str, color: str, spin: str) -> QuarkField:
    """Top antiquark field."""
    return _make_quark("t", True, pos, color, spin)
