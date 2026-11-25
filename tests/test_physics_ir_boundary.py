"""Regression tests for the symbolic/Wick-to-PhysicsIR boundary."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest

from concord_lqcd.compiler.physics_ir import (
    LogicalCorrelator,
    LogicalQuarkLine,
    OperationKind,
    OperationRecord,
    PhysicsIR,
    QuarkRecord,
    validate_physics_ir,
)
from concord_lqcd.compiler.physics_lowering import extract_operations, lower_to_physics_ir
from concord_lqcd.compiler.prewick import lower_correlator_to_physics_ir
from concord_lqcd.core import (
    Apply,
    Correlator,
    Operator,
    dbar,
    delta,
    derivative,
    gamma5,
    link,
    projected,
    reset_registry,
    transport,
    u,
)
from concord_lqcd.core.expr import Expr, Product
from concord_lqcd.core.fields import QuarkField, quark_fields, ubar
from concord_lqcd.core.operations import LinearOperator, Locality, Space
from concord_lqcd.core.wick import WickTerm, wick_contract


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_symbolic_registry():
    reset_registry()


def _walk_payload(value):
    yield value
    if is_dataclass(value):
        for item in fields(value):
            yield from _walk_payload(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_payload(key)
            yield from _walk_payload(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_payload(item)


class _UnsupportedFieldOperator(LinearOperator):
    @property
    def acts_on(self) -> frozenset[Space]:
        return frozenset({Space.COORDINATE})

    @property
    def locality(self) -> Locality:
        return Locality.LOCAL

    def dagger(self):
        return self


def test_physics_ir_record_module_has_no_frontend_imports():
    source = (ROOT / "concord_lqcd" / "compiler" / "physics_ir.py").read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("concord_lqcd.core") for module in imported_modules)
    assert "concord_lqcd.compiler.normalization" not in imported_modules


def test_lowered_physics_ir_contains_no_frontend_objects():
    sink = Operator(
        dbar("x", "a", "s1")
        * gamma5("s1", "s2")
        * derivative(transport(u("x", "b", "s2"), "+y"), "+x")
        * delta("a", "b"),
        momenta={"x": (1, 0, 0)},
    )
    correlator = Correlator(sink, sink.dagger())

    physics_ir = lower_to_physics_ir(correlator, wick_contract(correlator))
    validate_physics_ir(physics_ir)

    assert not any(
        isinstance(value, (Expr, LinearOperator))
        for value in _walk_payload(physics_ir)
    )


def test_wrapped_field_bookkeeping_preserves_exact_pairing_and_endpoint_operation():
    sink = Operator(
        dbar("x", "a", "s1")
        * gamma5("s1", "s2")
        * derivative(u("x", "a", "s2"), "+x"),
        momentum=(0, 0, 0),
    )
    source = Operator(
        dbar("y", "b", "r1") * gamma5("r1", "r2") * u("y", "b", "r2"),
        momentum=(0, 0, 0),
    ).dagger()
    correlator = Correlator(sink, source)

    terms = wick_contract(correlator)

    assert len(terms) == 1
    term = terms[0]
    remaining_fields = quark_fields(term.expr)
    assert len(remaining_fields) == 1
    assert any(remaining_fields[0] is quark for quark, _ in term.pairing)
    assert len(term.pairing) == 2

    physics_ir = lower_to_physics_ir(correlator, terms)
    logical = physics_ir.correlators[0]
    assert logical.coefficient == term.sign * term.coefficient
    assert len(logical.quark_lines) == 2
    assert len(logical.spin_tensors) == 4
    assert {(tensor.kind, tensor.operator_id) for tensor in logical.spin_tensors} == {
        ("gamma5", 0),
        ("gamma5_dag", 1),
        ("gammaT", 1),
    }
    derivative_lines = [line for line in logical.quark_lines if line.quark_operations]
    assert len(derivative_lines) == 1
    line = derivative_lines[0]
    assert line.quark.flavor == "u"
    assert line.quark.operator_id == 0
    assert len(line.quark_operations) == 1
    assert line.quark_operations[0].kind is OperationKind.DERIVATIVE
    assert line.quark_operations[0].parameters == {"direction": "+x"}
    assert all(not line.antiquark_operations for line in logical.quark_lines)


def test_aggregated_linear_combination_physics_ir_validates():
    sink_rest = Operator(
        dbar("x", "a", "s1") * gamma5("s1", "s2") * u("x", "a", "s2"),
        momentum=(0, 0, 0),
    )
    sink_moving = Operator(
        dbar("x", "a", "s1") * gamma5("s1", "s2") * u("x", "a", "s2"),
        momentum=(1, 0, 0),
    )
    correlator = Correlator(sink_rest + sink_moving, sink_rest.dagger())

    physics_ir = lower_correlator_to_physics_ir(correlator)

    assert len(physics_ir.correlators) >= 2
    validate_physics_ir(physics_ir)


def test_lowering_rejects_unknown_field_operator_instead_of_guessing_kind():
    field = u("x", "a", "s")

    with pytest.raises(NotImplementedError, match="Unsupported field operator"):
        extract_operations(Apply(_UnsupportedFieldOperator(), field))


def test_lowering_rejects_unknown_projection_instead_of_defaulting_to_laph():
    field = u("x", "a", "s")

    with pytest.raises(NotImplementedError, match="Unsupported field projection"):
        extract_operations(projected(field, "wall"))


def test_lowering_rejects_wick_pairing_from_a_different_correlator_context():
    sink = Operator(
        dbar("x", "a", "s1")
        * gamma5("s1", "s2")
        * u("x", "a", "s2"),
        momenta={"x": (0, 0, 0)},
    )
    correlator = Correlator(sink, sink.dagger())
    term = wick_contract(correlator)[0]
    quark, antiquark = term.pairing[0]
    foreign_quark = QuarkField(
        flavor=quark.flavor,
        is_anti=quark.is_anti,
        pos=quark.pos,
        color=quark.color,
        spin=quark.spin,
    )
    mismatched_term = WickTerm(
        sign=term.sign,
        coefficient=term.coefficient,
        expr=term.expr,
        components=term.components,
        pairing=((foreign_quark, antiquark),) + term.pairing[1:],
    )

    with pytest.raises(ValueError, match="absent from the normalized correlator"):
        lower_to_physics_ir(correlator, (mismatched_term,))


def _charged_wick_context():
    sink = Operator(
        dbar("x", "a", "s1") * gamma5("s1", "s2") * u("x", "a", "s2"),
        momentum=(0, 0, 0),
    )
    correlator = Correlator(sink, sink.dagger())
    return correlator, wick_contract(correlator)[0]


@pytest.mark.parametrize("case", ("missing", "duplicate", "extra", "flavor", "orientation"))
def test_lowering_rejects_incomplete_or_reused_wick_endpoints(case):
    correlator, term = _charged_wick_context()
    first, second = term.pairing
    malformed = {
        "missing": (first,),
        "duplicate": (first, first),
        "extra": (first, second, first),
        "flavor": ((first[0], second[1]), (second[0], first[1])),
        "orientation": ((first[1], first[0]), second),
    }

    with pytest.raises(ValueError, match="Wick pairing"):
        lower_to_physics_ir(correlator, (replace(term, pairing=malformed[case]),))


@pytest.mark.parametrize("pairing", (None, [], ((),), ((None, None),)))
def test_lowering_rejects_malformed_wick_pairing_payload(pairing):
    correlator, term = _charged_wick_context()

    with pytest.raises(TypeError, match="Wick pairing"):
        lower_to_physics_ir(correlator, (replace(term, pairing=pairing),))


@pytest.mark.parametrize("sign", (0, 2, -2, True, 1.0, None))
def test_lowering_rejects_invalid_fermionic_sign(sign):
    correlator, term = _charged_wick_context()

    with pytest.raises((TypeError, ValueError), match="fermionic sign"):
        lower_to_physics_ir(correlator, (replace(term, sign=sign),))


@pytest.mark.parametrize(
    "coefficient",
    (float("nan"), float("inf"), -float("inf"), complex(1, float("nan")), complex(1, float("inf")), 10 ** 400),
)
def test_physics_ir_rejects_nonfinite_or_unrepresentable_coefficients(coefficient):
    physics_ir = PhysicsIR((LogicalCorrelator(coefficient, ()),))

    with pytest.raises(ValueError, match="coefficient.*finite"):
        validate_physics_ir(physics_ir)


@pytest.mark.parametrize("coefficient", (float("nan"), complex(1, float("inf")), "1"))
def test_lowering_rejects_invalid_wick_coefficients(coefficient):
    correlator, term = _charged_wick_context()

    with pytest.raises((TypeError, ValueError), match="coefficient"):
        lower_to_physics_ir(correlator, (replace(term, coefficient=coefficient),))


@pytest.mark.parametrize("coefficient", (0, -1, 2 + 3j, 1e-250 + 1e-250j))
def test_lowering_preserves_finite_wick_coefficients_exactly(coefficient):
    correlator, term = _charged_wick_context()
    weighted = replace(term, coefficient=coefficient)

    physics_ir = lower_to_physics_ir(correlator, (weighted,))

    assert physics_ir.correlators[0].coefficient == complex(term.sign * coefficient)


@pytest.mark.parametrize("tensor_kind", ("spin", "color", "gauge"))
def test_lowering_rejects_wick_tensor_context_mismatch(tensor_kind):
    if tensor_kind == "spin":
        correlator, term = _charged_wick_context()
    else:
        if tensor_kind == "color":
            expression = dbar("x", "a", "s1") * gamma5("s1", "s2") * u("x", "b", "s2") * delta("a", "b")
        else:
            expression = dbar("x", "a", "s1") * gamma5("s1", "s2") * link("x", "y", "a", "b") * u("y", "b", "s2")
        sink = Operator(expression)
        correlator = Correlator(sink, sink.dagger())
        term = wick_contract(correlator)[0]

    with pytest.raises(ValueError, match="tensor.*context"):
        lower_to_physics_ir(correlator, (replace(term, expr=Product(())),))


def test_lowering_allows_complete_term_subsets_and_flavor_unbalanced_zero():
    neutral = Operator(ubar("x", "a", "s1") * gamma5("s1", "s2") * u("x", "a", "s2"))
    correlator = Correlator(neutral, neutral.dagger())
    terms = wick_contract(correlator)

    assert len(terms) == 2
    assert len(lower_to_physics_ir(correlator, terms[:1]).correlators) == 1
    unmatched = Correlator(Operator(u("x", "a", "s1") * dbar("x", "a", "s2")))
    assert lower_to_physics_ir(unmatched, wick_contract(unmatched)).correlators == ()


def test_physics_ir_validation_rejects_untyped_operation_payload():
    line = LogicalQuarkLine(
        quark=QuarkRecord("u", False, "sink", "a", "s1", operator_id=0),
        antiquark=QuarkRecord("u", True, "source", "b", "s2", operator_id=1),
        quark_operations=(
            OperationRecord(OperationKind.DERIVATIVE, {"op_type": object()}),
        ),
    )
    physics_ir = PhysicsIR(
        correlators=(LogicalCorrelator(1.0, (line,), momenta={}),),
        momenta={},
    )

    with pytest.raises(ValueError, match="canonical direction"):
        validate_physics_ir(physics_ir)
