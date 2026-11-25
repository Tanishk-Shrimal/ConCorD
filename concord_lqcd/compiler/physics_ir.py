"""Backend-independent records for the ConCorD physics IR.

This module defines the data contract that separates symbolic/Wick physics from
backend realization.  Frontend AST traversal belongs in
``concord_lqcd.compiler.physics_lowering``; no frontend expression class is needed to
construct or consume the records defined here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from numbers import Number


class OperationKind(Enum):
    DERIVATIVE = "derivative"
    TRANSPORT = "transport"
    SHIFT = "shift"


class Representation(Enum):
    POINT = "point"
    LAPH = "laph"


@dataclass(frozen=True)
class OperationRecord:
    """Backend-independent compiler record for a field operation.

    Operation tuples in PhysicsIR are stored in field-application order:
    innermost transformation first, outermost transformation last. Thus
    ``D_x(U_y(q))`` is represented as ``(U_y, D_x)``.
    """

    kind: OperationKind
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QuarkRecord:
    """A quark-field endpoint carrying only lowered physical information."""

    flavor: str
    is_anti: bool
    pos_label: str
    color_label: str
    spin_label: str
    representation: Representation = Representation.LAPH
    operator_id: int | None = None


@dataclass(frozen=True)
class SpinTensorRecord:
    """Typed compiler record for a spin matrix."""

    kind: str
    left_spin: str
    right_spin: str
    operator_id: int | None = None


@dataclass(frozen=True)
class ColorTensorRecord:
    """Typed compiler record for a color delta or epsilon."""

    kind: str
    indices: tuple[str, ...]
    operator_id: int | None = None


@dataclass(frozen=True)
class GaugeTensorRecord:
    """A symbolic gauge-link tensor retained by the physics IR."""

    start_pos: str
    end_pos: str
    start_color: str
    end_color: str
    operator_id: int | None = None


@dataclass(frozen=True)
class LogicalQuarkLine:
    """A contracted quark-antiquark line with endpoint-local transformations.

    ``quark_operations`` and ``antiquark_operations`` are ordered from the
    transformation closest to the field leaf to the outermost transformation.
    Projection operators are represented by ``QuarkRecord.representation`` and
    are therefore not repeated in these operation tuples.
    """

    quark: QuarkRecord
    antiquark: QuarkRecord
    quark_operations: tuple[OperationRecord, ...] = ()
    antiquark_operations: tuple[OperationRecord, ...] = ()


@dataclass(frozen=True)
class LogicalCorrelator:
    """A single Wick contribution in backend-independent compiler IR."""

    coefficient: complex
    quark_lines: tuple[LogicalQuarkLine, ...]
    spin_tensors: tuple[SpinTensorRecord, ...] = ()
    color_tensors: tuple[ColorTensorRecord, ...] = ()
    gauge_tensors: tuple[GaugeTensorRecord, ...] = ()
    momenta: Mapping[str, tuple[int, int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicsIR:
    """Complete backend-independent representation of a correlator collection."""

    correlators: tuple[LogicalCorrelator, ...]
    momenta: dict[str, tuple[int, int, int]] = field(default_factory=dict)


_DIRECTIONS = frozenset({
    "+x", "-x", "+y", "-y", "+z", "-z", "+t", "-t",
})


def _require_label(value: object, context: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string.")


def _validate_coefficient(value: object, context: str) -> None:
    if not isinstance(value, Number):
        raise TypeError(f"{context} coefficient must be numeric.")
    try:
        coefficient = complex(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{context} coefficient must be finite and representable as a complex scalar."
        ) from error
    if not isfinite(coefficient.real) or not isfinite(coefficient.imag):
        raise ValueError(f"{context} coefficient must be finite.")


def _validate_operator_id(value: object, context: str, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"{context} is missing operator_id provenance.")
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context} operator_id must be a non-negative integer.")


def _validate_momenta(
    momenta: object,
    context: str,
) -> None:
    if not isinstance(momenta, Mapping):
        raise TypeError(f"{context} momenta must be a mapping.")
    for label, momentum in momenta.items():
        _require_label(label, f"{context} momentum label")
        if (
            not isinstance(momentum, tuple)
            or len(momentum) != 3
            or any(not isinstance(component, int) or isinstance(component, bool) for component in momentum)
        ):
            raise TypeError(
                f"{context} momentum for {label!r} must be a three-integer tuple."
            )


def _validate_operation(record: object, context: str) -> None:
    if not isinstance(record, OperationRecord):
        raise TypeError(f"{context} must contain only OperationRecord values.")
    if not isinstance(record.kind, OperationKind):
        raise TypeError(f"{context} operation kind must be OperationKind.")
    if not isinstance(record.parameters, Mapping):
        raise TypeError(f"{context} operation parameters must be a mapping.")
    if any(not isinstance(key, str) for key in record.parameters):
        raise TypeError(f"{context} operation parameter names must be strings.")

    parameters = dict(record.parameters)
    if record.kind is OperationKind.DERIVATIVE:
        if set(parameters) != {"direction"} or parameters["direction"] not in _DIRECTIONS:
            raise ValueError(
                f"{context} derivative must contain one canonical direction parameter."
            )
        return

    if set(parameters) != {"path"}:
        raise ValueError(f"{context} {record.kind.value} must contain one path parameter.")
    path = parameters["path"]
    if (
        not isinstance(path, tuple)
        or any(not isinstance(hop, str) or hop not in _DIRECTIONS for hop in path)
    ):
        raise ValueError(f"{context} {record.kind.value} path must contain canonical hops.")


def _validate_quark(record: object, context: str, *, expect_anti: bool) -> None:
    if not isinstance(record, QuarkRecord):
        raise TypeError(f"{context} must be a QuarkRecord.")
    _require_label(record.flavor, f"{context} flavor")
    _require_label(record.pos_label, f"{context} position label")
    _require_label(record.color_label, f"{context} color label")
    _require_label(record.spin_label, f"{context} spin label")
    if record.is_anti is not expect_anti:
        expected = "antiquark" if expect_anti else "quark"
        raise ValueError(f"{context} must describe a {expected} endpoint.")
    if not isinstance(record.representation, Representation):
        raise TypeError(f"{context} representation must be Representation.")
    _validate_operator_id(record.operator_id, context, required=True)


def _validate_spin_tensor(record: object, context: str) -> None:
    if not isinstance(record, SpinTensorRecord):
        raise TypeError(f"{context} must contain only SpinTensorRecord values.")
    _require_label(record.kind, f"{context} spin-tensor kind")
    _require_label(record.left_spin, f"{context} left-spin label")
    _require_label(record.right_spin, f"{context} right-spin label")
    _validate_operator_id(record.operator_id, context, required=False)


def _validate_color_tensor(record: object, context: str) -> None:
    if not isinstance(record, ColorTensorRecord):
        raise TypeError(f"{context} must contain only ColorTensorRecord values.")
    expected_rank = {"delta": 2, "epsilon": 3}.get(record.kind)
    if expected_rank is None:
        raise ValueError(f"{context} color-tensor kind must be 'delta' or 'epsilon'.")
    if not isinstance(record.indices, tuple) or len(record.indices) != expected_rank:
        raise ValueError(f"{context} {record.kind} must have rank {expected_rank}.")
    for label in record.indices:
        _require_label(label, f"{context} color-index label")
    _validate_operator_id(record.operator_id, context, required=False)


def _validate_gauge_tensor(record: object, context: str) -> None:
    if not isinstance(record, GaugeTensorRecord):
        raise TypeError(f"{context} must contain only GaugeTensorRecord values.")
    _require_label(record.start_pos, f"{context} start-position label")
    _require_label(record.end_pos, f"{context} end-position label")
    _require_label(record.start_color, f"{context} start-color label")
    _require_label(record.end_color, f"{context} end-color label")
    _validate_operator_id(record.operator_id, context, required=False)


def validate_physics_ir(physics_ir: PhysicsIR) -> None:
    """Validate the frontend/Wick-to-PhysicsIR boundary contract.

    Validation is intentionally independent of distillation, numerical tensor
    shapes, resource identities, schedules, devices, files, and buffers.
    """

    if not isinstance(physics_ir, PhysicsIR):
        raise TypeError("physics_ir must be a PhysicsIR instance.")
    if not isinstance(physics_ir.correlators, tuple):
        raise TypeError("PhysicsIR correlators must be a tuple.")
    _validate_momenta(physics_ir.momenta, "PhysicsIR")

    for corr_index, correlator in enumerate(physics_ir.correlators):
        context = f"PhysicsIR correlator {corr_index}"
        if not isinstance(correlator, LogicalCorrelator):
            raise TypeError(f"{context} must be a LogicalCorrelator.")
        _validate_coefficient(correlator.coefficient, context)
        if not isinstance(correlator.quark_lines, tuple):
            raise TypeError(f"{context} quark_lines must be a tuple.")
        _validate_momenta(correlator.momenta, context)
        for label, momentum in correlator.momenta.items():
            if physics_ir.momenta.get(label) != momentum:
                raise ValueError(
                    f"{context} momentum {label!r} disagrees with the enclosing PhysicsIR."
                )

        for line_index, line in enumerate(correlator.quark_lines):
            line_context = f"{context} quark line {line_index}"
            if not isinstance(line, LogicalQuarkLine):
                raise TypeError(f"{line_context} must be a LogicalQuarkLine.")
            _validate_quark(line.quark, f"{line_context} quark", expect_anti=False)
            _validate_quark(line.antiquark, f"{line_context} antiquark", expect_anti=True)
            if line.quark.flavor != line.antiquark.flavor:
                raise ValueError(f"{line_context} violates Wick flavor conservation.")
            if not isinstance(line.quark_operations, tuple):
                raise TypeError(f"{line_context} quark operations must be a tuple.")
            if not isinstance(line.antiquark_operations, tuple):
                raise TypeError(f"{line_context} antiquark operations must be a tuple.")
            for operation_index, operation in enumerate(line.quark_operations):
                _validate_operation(operation, f"{line_context} quark operation {operation_index}")
            for operation_index, operation in enumerate(line.antiquark_operations):
                _validate_operation(
                    operation,
                    f"{line_context} antiquark operation {operation_index}",
                )

        if not isinstance(correlator.spin_tensors, tuple):
            raise TypeError(f"{context} spin_tensors must be a tuple.")
        if not isinstance(correlator.color_tensors, tuple):
            raise TypeError(f"{context} color_tensors must be a tuple.")
        if not isinstance(correlator.gauge_tensors, tuple):
            raise TypeError(f"{context} gauge_tensors must be a tuple.")
        for tensor_index, tensor in enumerate(correlator.spin_tensors):
            _validate_spin_tensor(tensor, f"{context} spin tensor {tensor_index}")
        for tensor_index, tensor in enumerate(correlator.color_tensors):
            _validate_color_tensor(tensor, f"{context} color tensor {tensor_index}")
        for tensor_index, tensor in enumerate(correlator.gauge_tensors):
            _validate_gauge_tensor(tensor, f"{context} gauge tensor {tensor_index}")


__all__ = [
    "OperationKind",
    "Representation",
    "OperationRecord",
    "QuarkRecord",
    "SpinTensorRecord",
    "ColorTensorRecord",
    "GaugeTensorRecord",
    "LogicalQuarkLine",
    "LogicalCorrelator",
    "PhysicsIR",
    "validate_physics_ir",
]
