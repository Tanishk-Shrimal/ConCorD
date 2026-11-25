"""Physical transformation algebra for the ConCorD symbolic frontend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .expr import Expr
from .indices import Index


class Space(Enum):
    COORDINATE = "coordinate"
    COLOR = "color"
    SPIN = "spin"


class Locality(Enum):
    LOCAL = "local"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    NONLOCAL = "nonlocal"


class Axis(Enum):
    X = "x"
    Y = "y"
    Z = "z"
    T = "t"

    def __neg__(self) -> str:
        """Return the negative directed hop associated with this axis."""
        return f"-{self.value}"


class FieldType(Enum):
    QUARK = "quark"
    GAUGE = "gauge"
    TENSOR = "tensor"


_VALID_HOPS = frozenset({
    "+x", "-x",
    "+y", "-y",
    "+z", "-z",
    "+t", "-t",
})
_OPPOSITE_HOP = {
    "+x": "-x", "-x": "+x",
    "+y": "-y", "-y": "+y",
    "+z": "-z", "-z": "+z",
    "+t": "-t", "-t": "+t",
}


def _normalize_hop(hop: str | Axis) -> str:
    """Return one canonical signed lattice direction."""
    if isinstance(hop, Axis):
        normalized = f"+{hop.value}"
    elif isinstance(hop, str):
        normalized = hop.strip()
        if not normalized.startswith(("+", "-")):
            normalized = f"+{normalized}"
    else:
        raise TypeError(f"A path hop must be str or Axis, got {type(hop).__name__}.")

    if normalized not in _VALID_HOPS:
        raise ValueError(
            f"Invalid lattice hop {hop!r}; expected one of {sorted(_VALID_HOPS)}."
        )
    return normalized


@dataclass(frozen=True)
class Path:
    """An ordered path of signed lattice hops, e.g. ``(+x, +x, -y)``."""

    hops: tuple[str, ...]

    def __init__(self, hops: tuple[str | Axis, ...] | list[str | Axis]) -> None:
        object.__setattr__(self, "hops", tuple(_normalize_hop(hop) for hop in hops))

    def reverse(self) -> Path:
        """Reverse the path and negate every hop."""
        return Path(tuple(_OPPOSITE_HOP[hop] for hop in reversed(self.hops)))

    def __add__(self, other: Path) -> Path:
        if not isinstance(other, Path):
            return NotImplemented
        return Path(self.hops + other.hops)

    def simplify(self) -> Path:
        """Cancel adjacent inverse hops, e.g. ``(+x, -x) -> ()``."""
        stack: list[str] = []
        for hop in self.hops:
            if stack and _OPPOSITE_HOP[hop] == stack[-1]:
                stack.pop()
            else:
                stack.append(hop)
        return Path(tuple(stack))

    @property
    def locality(self) -> Locality:
        if not self.hops:
            return Locality.LOCAL
        if len(self.hops) == 1:
            return Locality.NEAREST_NEIGHBOR
        return Locality.NONLOCAL


class LinearOperator(ABC):
    """Abstract base class for field-level linear operators."""

    @property
    @abstractmethod
    def acts_on(self) -> frozenset[Space]:
        """Spaces acted upon by this operator."""
        ...

    @property
    @abstractmethod
    def locality(self) -> Locality:
        """Locality classification of this operator."""
        ...

    @property
    def input_type(self) -> FieldType:
        return FieldType.QUARK

    @property
    def output_type(self) -> FieldType:
        return FieldType.QUARK

    @abstractmethod
    def dagger(self) -> LinearOperator:
        """Return the adjoint linear operator."""
        ...


class Shift(LinearOperator):
    """Pure coordinate displacement without gauge-link transport."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @property
    def acts_on(self) -> frozenset[Space]:
        return frozenset({Space.COORDINATE})

    @property
    def locality(self) -> Locality:
        return self._path.locality

    def dagger(self) -> Shift:
        return Shift(self._path.reverse())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shift):
            return NotImplemented
        return self._path == other._path

    def __hash__(self) -> int:
        return hash(("Shift", self._path))

    def __repr__(self) -> str:
        return f"Shift({self._path.hops})"


class ParallelTransport(LinearOperator):
    """Gauge-covariant displacement along a path of Wilson-link hops."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @property
    def acts_on(self) -> frozenset[Space]:
        return frozenset({Space.COORDINATE, Space.COLOR})

    @property
    def locality(self) -> Locality:
        return self._path.locality

    def dagger(self) -> ParallelTransport:
        return ParallelTransport(self._path.reverse())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParallelTransport):
            return NotImplemented
        return self._path == other._path

    def __hash__(self) -> int:
        return hash(("ParallelTransport", self._path))

    def __repr__(self) -> str:
        return f"ParallelTransport({self._path.hops})"


class CovariantDerivative(LinearOperator):
    """Symbolic gauge-covariant derivative along one directed axis."""

    def __init__(self, direction: str | Axis) -> None:
        self._direction = _normalize_hop(direction)

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def acts_on(self) -> frozenset[Space]:
        return frozenset({Space.COORDINATE, Space.COLOR})

    @property
    def locality(self) -> Locality:
        return Locality.NEAREST_NEIGHBOR

    def dagger(self) -> CovariantDerivative:
        return CovariantDerivative(_OPPOSITE_HOP[self._direction])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CovariantDerivative):
            return NotImplemented
        return self._direction == other._direction

    def __hash__(self) -> int:
        return hash(("CovariantDerivative", self._direction))

    def __repr__(self) -> str:
        return f"D_{{{self._direction}}}"


class Projection(LinearOperator):
    """Symbolic spatial representation/projection applied to a quark field."""

    def __init__(self, projector: str = "laph") -> None:
        if not isinstance(projector, str) or not projector.strip():
            raise ValueError("projector must be a non-empty string.")
        self._projector = projector.strip()

    @property
    def projector(self) -> str:
        return self._projector

    @property
    def acts_on(self) -> frozenset[Space]:
        return frozenset({Space.COORDINATE, Space.COLOR})

    @property
    def locality(self) -> Locality:
        return Locality.NONLOCAL

    def dagger(self) -> Projection:
        return Projection(self._projector)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Projection):
            return NotImplemented
        return self._projector == other._projector

    def __hash__(self) -> int:
        return hash(("Projection", self._projector))

    def __repr__(self) -> str:
        return f"Projection({self._projector})"


class Apply(Expr):
    """Symbolic application of a ``LinearOperator`` to one field expression."""

    __slots__ = ("_operator", "_operand")

    def __init__(self, operator: LinearOperator, operand: Expr) -> None:
        if not isinstance(operator, LinearOperator):
            raise TypeError(
                f"Apply operator must be a LinearOperator instance, got {type(operator).__name__}."
            )
        if not isinstance(operand, Expr):
            raise TypeError(
                f"Apply operand must be an Expr instance, got {type(operand).__name__}."
            )
        from .expr import Product

        if isinstance(operand, Product):
            raise TypeError(
                "Apply operand cannot be a Product expression. Leibniz expansion "
                "over Product expressions is a compiler pass, not legal frontend syntax."
            )
        operand_out = getattr(operand, "output_type", FieldType.QUARK)
        if operator.input_type != operand_out:
            raise TypeError(
                f"Operator {type(operator).__name__} expects input_type "
                f"{operator.input_type.value}, but operand has output_type {operand_out.value}."
            )
        self._operator = operator
        self._operand = operand

    @property
    def operator(self) -> LinearOperator:
        return self._operator

    @property
    def operand(self) -> Expr:
        return self._operand

    @property
    def output_type(self) -> FieldType:
        return self._operator.output_type

    @property
    def indices(self) -> tuple[Index, ...]:
        return self._operand.indices

    def __repr__(self) -> str:
        return f"{self._operator!r}({self._operand!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Apply):
            return NotImplemented
        return self._operator == other._operator and self._operand == other._operand

    def __hash__(self) -> int:
        return hash(("Apply", self._operator, self._operand))
