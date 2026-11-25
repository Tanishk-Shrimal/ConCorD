"""Internal propagator node produced by exact Wick contraction."""

from __future__ import annotations

from .expr import Expr
from .indices import Index

class Propagator(Expr):
    """Internal ``S(sink; source)`` node; users define quark fields, not propagators."""

    __slots__ = (
        "_flavor",
        "_sink_x", "_sink_c", "_sink_s",
        "_source_x", "_source_c", "_source_s"
    )

    def __init__(
        self,
        flavor: str,
        sink_pos: Index,
        sink_color: Index,
        sink_spin: Index,
        source_pos: Index,
        source_color: Index,
        source_spin: Index,
    ) -> None:
        self._flavor = flavor
        self._sink_x = sink_pos
        self._sink_c = sink_color
        self._sink_s = sink_spin
        self._source_x = source_pos
        self._source_c = source_color
        self._source_s = source_spin

    @property
    def flavor(self) -> str:
        return self._flavor

    @property
    def sink_pos(self) -> Index:
        return self._sink_x

    @property
    def sink_color(self) -> Index:
        return self._sink_c

    @property
    def sink_spin(self) -> Index:
        return self._sink_s

    @property
    def source_pos(self) -> Index:
        return self._source_x

    @property
    def source_color(self) -> Index:
        return self._source_c

    @property
    def source_spin(self) -> Index:
        return self._source_s

    @property
    def indices(self) -> tuple[Index, ...]:
        return (
            self._sink_x, self._sink_c, self._sink_s,
            self._source_x, self._source_c, self._source_s
        )

    def __repr__(self) -> str:
        sink = f"{self._sink_x.label},{self._sink_c.label},{self._sink_s.label}"
        src = f"{self._source_x.label},{self._source_c.label},{self._source_s.label}"
        return f"S_{self._flavor}({sink} | {src})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Propagator):
            return NotImplemented
        return (
            self._flavor == other._flavor and
            self._sink_x == other._sink_x and
            self._sink_c == other._sink_c and
            self._sink_s == other._sink_s and
            self._source_x == other._source_x and
            self._source_c == other._source_c and
            self._source_s == other._source_s
        )

    def __hash__(self) -> int:
        return hash((
            "propagator", self._flavor,
            self._sink_x, self._sink_c, self._sink_s,
            self._source_x, self._source_c, self._source_s
        ))
