"""Lower normalized symbolic/Wick objects into backend-independent PhysicsIR."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from numbers import Integral

from concord_lqcd.compiler.normalization import CompositeOperator
from concord_lqcd.compiler.physics_ir import (
    ColorTensorRecord,
    GaugeTensorRecord,
    LogicalCorrelator,
    LogicalQuarkLine,
    OperationKind,
    OperationRecord,
    PhysicsIR,
    QuarkRecord,
    Representation,
    SpinTensorRecord,
    _validate_coefficient,
    validate_physics_ir,
)
from concord_lqcd.core.color import ColorDelta, ColorEpsilon
from concord_lqcd.core.correlator import Correlator
from concord_lqcd.core.expr import Expr, Product
from concord_lqcd.core.fields import QuarkField
from concord_lqcd.core.gamma import GammaMatrix
from concord_lqcd.core.gauge import GaugeLink
from concord_lqcd.core.operations import (
    Apply,
    CovariantDerivative,
    LinearOperator,
    ParallelTransport,
    Projection,
    Shift,
)
from concord_lqcd.core.wick import WickTerm


def _primitive_ops_in_application_order(
    operator: LinearOperator,
) -> Iterator[LinearOperator]:
    """Yield primitive operators nearest-to-field first."""

    if isinstance(operator, CompositeOperator):
        for child in reversed(operator.operators):
            yield from _primitive_ops_in_application_order(child)
        return
    yield operator


def extract_operations(
    expr: Expr,
) -> tuple[QuarkField, list[OperationRecord], Representation]:
    """Extract a field leaf, its ordered operations, and its representation.

    Returned operations are in application order (innermost first). A
    ``Projection`` selects the field representation and is not emitted as an
    ``OperationRecord``. With nested projections the outermost projection wins,
    matching ordinary operator composition.

    Unsupported operators and projectors fail here. Lowering must never guess a
    physical meaning for a frontend extension it does not understand.
    """

    records: list[OperationRecord] = []
    representation = Representation.LAPH
    current = expr
    outer_to_inner: list[LinearOperator] = []

    while isinstance(current, Apply):
        outer_to_inner.append(current.operator)
        current = current.operand

    if not isinstance(current, QuarkField):
        raise TypeError(
            "Operation chain must terminate on a QuarkField, got "
            f"{type(current).__name__}."
        )

    def process_operator(operator: LinearOperator) -> None:
        nonlocal representation

        if isinstance(operator, CovariantDerivative):
            records.append(OperationRecord(
                kind=OperationKind.DERIVATIVE,
                parameters={"direction": operator.direction},
            ))
            return
        if isinstance(operator, ParallelTransport):
            records.append(OperationRecord(
                kind=OperationKind.TRANSPORT,
                parameters={"path": tuple(operator.path.hops)},
            ))
            return
        if isinstance(operator, Shift):
            records.append(OperationRecord(
                kind=OperationKind.SHIFT,
                parameters={"path": tuple(operator.path.hops)},
            ))
            return
        if isinstance(operator, Projection):
            if operator.projector == "point":
                representation = Representation.POINT
                return
            if operator.projector == "laph":
                representation = Representation.LAPH
                return
            raise NotImplementedError(
                f"Unsupported field projection {operator.projector!r} at PhysicsIR lowering."
            )
        raise NotImplementedError(
            "Unsupported field operator at PhysicsIR lowering: "
            f"{type(operator).__name__}."
        )

    for operator in reversed(outer_to_inner):
        for primitive in _primitive_ops_in_application_order(operator):
            process_operator(primitive)

    return current, records, representation


def _operator_id_from_label(label: str) -> int | None:
    """Recover operator provenance from the collision-free correlator namespace."""

    for part in label.split(":"):
        if part.startswith("op") and part[2:].isdigit():
            return int(part[2:])
    return None


def _required_operator_id(label: str, context: str) -> int:
    operator_id = _operator_id_from_label(label)
    if operator_id is None:
        raise ValueError(
            f"{context} label {label!r} has no collision-free operator provenance."
        )
    return operator_id


FieldContext = dict[int, tuple[tuple[OperationRecord, ...], Representation]]


def _collect_field_context(expr: Expr, context: FieldContext) -> None:
    """Index every field leaf by the complete operation chain acting on it."""

    if isinstance(expr, Apply):
        leaf, operations, representation = extract_operations(expr)
        lowered = (tuple(operations), representation)
        previous = context.get(id(leaf))
        if previous is not None and previous != lowered:
            raise ValueError(
                "One QuarkField object occurs under conflicting operation chains; "
                "use distinct field occurrences in the symbolic operator."
            )
        context[id(leaf)] = lowered
        return
    if isinstance(expr, Product):
        for factor in expr.factors:
            _collect_field_context(factor, context)
        return
    if isinstance(expr, QuarkField):
        lowered = ((), Representation.LAPH)
        previous = context.get(id(expr))
        if previous is not None and previous != lowered:
            raise ValueError(
                "One QuarkField object occurs both bare and transformed; use "
                "distinct field occurrences in the symbolic operator."
            )
        context[id(expr)] = lowered


def _collect_tensor_records(
    expr: Expr,
    spin_tensors: list[SpinTensorRecord],
    color_tensors: list[ColorTensorRecord],
    gauge_tensors: list[GaugeTensorRecord],
) -> None:
    """Collect explicit tensor nodes without interpreting or contracting them."""

    if isinstance(expr, GammaMatrix):
        spin_tensors.append(SpinTensorRecord(
            kind=expr.kind,
            left_spin=expr.left.label,
            right_spin=expr.right.label,
            operator_id=_operator_id_from_label(expr.left.label),
        ))
        return
    if isinstance(expr, ColorDelta):
        color_tensors.append(ColorTensorRecord(
            kind="delta",
            indices=tuple(index.label for index in expr.indices),
            operator_id=_operator_id_from_label(expr.indices[0].label),
        ))
        return
    if isinstance(expr, ColorEpsilon):
        color_tensors.append(ColorTensorRecord(
            kind="epsilon",
            indices=tuple(index.label for index in expr.indices),
            operator_id=_operator_id_from_label(expr.indices[0].label),
        ))
        return
    if isinstance(expr, GaugeLink):
        gauge_tensors.append(GaugeTensorRecord(
            start_pos=expr.start_pos.label,
            end_pos=expr.end_pos.label,
            start_color=expr.start_color.label,
            end_color=expr.end_color.label,
            operator_id=_operator_id_from_label(expr.start_pos.label),
        ))
        return
    if isinstance(expr, Apply):
        _collect_tensor_records(
            expr.operand,
            spin_tensors,
            color_tensors,
            gauge_tensors,
        )
        return
    if isinstance(expr, Product):
        for factor in expr.factors:
            _collect_tensor_records(
                factor,
                spin_tensors,
                color_tensors,
                gauge_tensors,
            )


def _require_field_context(
    field: QuarkField,
    context: FieldContext,
    role: str,
) -> tuple[tuple[OperationRecord, ...], Representation]:
    try:
        return context[id(field)]
    except KeyError as exc:
        raise ValueError(
            f"Wick pairing {role} field is absent from the normalized correlator; "
            "PhysicsIR lowering requires WickTerms produced from that exact basis correlator."
        ) from exc


def _validate_wick_term(term: WickTerm, field_context: FieldContext, context: str) -> None:
    if isinstance(term.sign, bool) or not isinstance(term.sign, Integral):
        raise TypeError(f"{context} fermionic sign must be an integer equal to -1 or 1.")
    if term.sign not in (-1, 1):
        raise ValueError(f"{context} fermionic sign must equal -1 or 1.")
    _validate_coefficient(term.coefficient, context)
    if not isinstance(term.pairing, tuple):
        raise TypeError(f"{context} Wick pairing must be a tuple of field pairs.")

    seen_fields: set[int] = set()
    for pair in term.pairing:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{context} Wick pairing must contain two-field tuples.")
        quark, antiquark = pair
        if not isinstance(quark, QuarkField) or not isinstance(antiquark, QuarkField):
            raise TypeError(f"{context} Wick pairing must contain QuarkField endpoints.")
        _require_field_context(quark, field_context, "quark")
        _require_field_context(antiquark, field_context, "antiquark")
        if quark.is_anti or not antiquark.is_anti:
            raise ValueError(f"{context} Wick pairing must be ordered as quark, antiquark.")
        if quark.flavor != antiquark.flavor:
            raise ValueError(f"{context} Wick pairing violates flavor conservation.")
        for endpoint in pair:
            if id(endpoint) in seen_fields:
                raise ValueError(f"{context} Wick pairing reuses a field occurrence.")
            seen_fields.add(id(endpoint))
    if seen_fields != field_context.keys():
        raise ValueError(f"{context} Wick pairing must cover every original field exactly once.")


def lower_to_physics_ir(
    correlator: Correlator,
    wick_terms: Sequence[WickTerm],
) -> PhysicsIR:
    """Lower one basis Correlator and its exact WickTerms into PhysicsIR."""

    if not isinstance(correlator, Correlator):
        raise TypeError("correlator must be a Correlator instance.")
    if not correlator.is_basis:
        raise ValueError(
            "lower_to_physics_ir() requires a basis Correlator; expand linear "
            "combinations before lowering."
        )
    if not isinstance(wick_terms, Sequence):
        raise TypeError("wick_terms must be a sequence of WickTerm objects.")
    if any(not isinstance(term, WickTerm) for term in wick_terms):
        raise TypeError("wick_terms must contain only WickTerm objects.")

    logical_correlators: list[LogicalCorrelator] = []
    combined_momenta: dict[str, tuple[int, int, int]] = {}

    for operator in correlator.symbolic_operators:
        combined_momenta.update(operator.momenta)

    field_context: FieldContext = {}
    basis_spin_tensors: list[SpinTensorRecord] = []
    basis_color_tensors: list[ColorTensorRecord] = []
    basis_gauge_tensors: list[GaugeTensorRecord] = []
    for operator in correlator.symbolic_operators:
        _collect_field_context(operator.expr, field_context)
        _collect_tensor_records(
            operator.expr,
            basis_spin_tensors,
            basis_color_tensors,
            basis_gauge_tensors,
        )
    basis_tensors = (
        tuple(basis_spin_tensors),
        tuple(basis_color_tensors),
        tuple(basis_gauge_tensors),
    )

    for term_index, term in enumerate(wick_terms):
        context = f"Wick term {term_index}"
        _validate_wick_term(term, field_context, context)
        lines: list[LogicalQuarkLine] = []
        spin_tensors: list[SpinTensorRecord] = []
        color_tensors: list[ColorTensorRecord] = []
        gauge_tensors: list[GaugeTensorRecord] = []

        _collect_tensor_records(
            term.expr,
            spin_tensors,
            color_tensors,
            gauge_tensors,
        )
        if (tuple(spin_tensors), tuple(color_tensors), tuple(gauge_tensors)) != basis_tensors:
            raise ValueError(
                f"{context} tensor bookkeeping disagrees with the original operator context."
            )

        for quark_field, antiquark_field in term.pairing:
            quark_operations, quark_representation = _require_field_context(
                quark_field,
                field_context,
                "quark",
            )
            antiquark_operations, antiquark_representation = _require_field_context(
                antiquark_field,
                field_context,
                "antiquark",
            )

            quark_record = QuarkRecord(
                flavor=quark_field.flavor,
                is_anti=quark_field.is_anti,
                pos_label=quark_field.pos.label,
                color_label=quark_field.color.label,
                spin_label=quark_field.spin.label,
                representation=quark_representation,
                operator_id=_required_operator_id(
                    quark_field.pos.label,
                    "Wick pairing quark position",
                ),
            )
            antiquark_record = QuarkRecord(
                flavor=antiquark_field.flavor,
                is_anti=antiquark_field.is_anti,
                pos_label=antiquark_field.pos.label,
                color_label=antiquark_field.color.label,
                spin_label=antiquark_field.spin.label,
                representation=antiquark_representation,
                operator_id=_required_operator_id(
                    antiquark_field.pos.label,
                    "Wick pairing antiquark position",
                ),
            )

            lines.append(LogicalQuarkLine(
                quark=quark_record,
                antiquark=antiquark_record,
                quark_operations=quark_operations,
                antiquark_operations=antiquark_operations,
            ))

        logical_correlators.append(LogicalCorrelator(
            coefficient=complex(term.coefficient * term.sign),
            quark_lines=tuple(lines),
            spin_tensors=tuple(spin_tensors),
            color_tensors=tuple(color_tensors),
            gauge_tensors=tuple(gauge_tensors),
            momenta=combined_momenta,
        ))

    physics_ir = PhysicsIR(
        correlators=tuple(logical_correlators),
        momenta=combined_momenta,
    )
    validate_physics_ir(physics_ir)
    return physics_ir


__all__ = ["extract_operations", "lower_to_physics_ir"]
