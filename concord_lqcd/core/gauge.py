"""Gauge-link (Wilson-line) constructors."""

from __future__ import annotations

from typing import Any

from .expr import Expr
from .indices import Index, POSITION, COLOR


class GaugeLink(Expr):
    """A gauge link / Wilson line ``U(x, y)_{a,b}``."""

    __slots__ = ("_start_pos", "_end_pos", "_start_color", "_end_color")

    def __init__(
        self,
        start_pos: Index,
        end_pos: Index,
        start_color: Index,
        end_color: Index,
    ) -> None:
        self._start_pos = start_pos
        self._end_pos = end_pos
        self._start_color = start_color
        self._end_color = end_color

    @property
    def output_type(self) -> Any:
        from .operations import FieldType

        return FieldType.GAUGE

    @property
    def start_pos(self) -> Index:
        return self._start_pos

    @property
    def end_pos(self) -> Index:
        return self._end_pos

    @property
    def start_color(self) -> Index:
        return self._start_color

    @property
    def end_color(self) -> Index:
        return self._end_color

    @property
    def indices(self) -> tuple[Index, ...]:
        return (self._start_pos, self._end_pos, self._start_color, self._end_color)

    def __repr__(self) -> str:
        pos = f"{self._start_pos.label}→{self._end_pos.label}"
        color = f"{self._start_color.label},{self._end_color.label}"
        return f"U({pos})_{{{color}}}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaugeLink):
            return NotImplemented
        return (
            self._start_pos == other._start_pos
            and self._end_pos == other._end_pos
            and self._start_color == other._start_color
            and self._end_color == other._end_color
        )

    def __hash__(self) -> int:
        return hash((
            "gauge_link",
            self._start_pos,
            self._end_pos,
            self._start_color,
            self._end_color,
        ))


def link(start_pos: str, end_pos: str, start_color: str, end_color: str) -> GaugeLink:
    """Construct a gauge link ``U(start_pos, end_pos)`` with open color legs."""
    return GaugeLink(
        Index(start_pos, POSITION),
        Index(end_pos, POSITION),
        Index(start_color, COLOR),
        Index(end_color, COLOR),
    )
