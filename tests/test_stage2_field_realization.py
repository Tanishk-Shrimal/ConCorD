"""Focused tests for Stage-2 projected-field realization semantics."""

from concord_lqcd.compiler.normalization import normalize_expr
from concord_lqcd.compiler.physics_ir import (
    OperationKind,
    Representation,
)
from concord_lqcd.compiler.physics_lowering import extract_operations, lower_to_physics_ir
from concord_lqcd.compiler.distillation_lowering import (
    ProjectionContribution,
    identify_operator_vertices,
    realize_projected_fields,
)
from concord_lqcd.core import (
    Correlator,
    Operator,
    dbar,
    delta,
    derivative,
    gamma5,
    laph,
    point,
    reset_registry,
    transport,
    u,
)
from concord_lqcd.core.operator import _dagger_expr
from concord_lqcd.core.wick import wick_contract
from concord_lqcd.core.expr import Product
from concord_lqcd.core.operations import Apply


def _operation_signature(records):
    return tuple((record.kind, dict(record.parameters)) for record in records)


def test_stage2_operation_records_follow_field_application_order():
    """D_x(U_y q) is recorded as U_y first, then D_x."""
    reset_registry()
    q = u("x", "a", "s")

    expr = derivative(transport(q, "+y"), "+x")
    leaf, records, representation = extract_operations(expr)

    assert leaf == q
    assert representation == Representation.LAPH
    assert _operation_signature(records) == (
        (OperationKind.TRANSPORT, {"path": ("+y",)}),
        (OperationKind.DERIVATIVE, {"direction": "+x"}),
    )

    # Reversing the symbolic nesting gives a genuinely different operation chain.
    expr_reversed = transport(derivative(q, "+y"), "+x")
    _, reversed_records, _ = extract_operations(expr_reversed)
    assert _operation_signature(reversed_records) == (
        (OperationKind.DERIVATIVE, {"direction": "+y"}),
        (OperationKind.TRANSPORT, {"path": ("+x",)}),
    )


def test_stage2_operation_order_survives_composite_normalization():
    """Compiler CompositeOperator normalization must not change application order."""
    reset_registry()
    q = u("x", "a", "s")
    expr = derivative(transport(q, "+y"), "+x")

    _, direct_records, direct_rep = extract_operations(expr)
    normalized = normalize_expr(expr)
    _, normalized_records, normalized_rep = extract_operations(normalized)

    assert normalized_rep == direct_rep == Representation.LAPH
    assert _operation_signature(normalized_records) == _operation_signature(direct_records)


def test_stage2_projection_selects_representation_not_kernel_operation():
    """Projection metadata is separated from derivative/transport operations."""
    reset_registry()
    q = u("x", "a", "s")

    expr = derivative(point(transport(q, "+z")), "+x")
    _, records, representation = extract_operations(expr)

    assert representation == Representation.POINT
    assert _operation_signature(records) == (
        (OperationKind.TRANSPORT, {"path": ("+z",)}),
        (OperationKind.DERIVATIVE, {"direction": "+x"}),
    )
    assert all(record.kind in {OperationKind.TRANSPORT, OperationKind.DERIVATIVE} for record in records)

    # With nested projections the outermost projection is the final representation.
    _, records_point, rep_point = extract_operations(point(laph(q)))
    assert records_point == []
    assert rep_point == Representation.POINT

    _, records_laph, rep_laph = extract_operations(laph(point(q)))
    assert records_laph == []
    assert rep_laph == Representation.LAPH


def test_stage2_daggered_field_keeps_original_transformations():
    """Stage 2 retains the field chain that antiquark realization conjugates."""
    reset_registry()
    q = u("x", "a", "s")
    expr = derivative(transport(q, "+y"), "+x")

    daggered = _dagger_expr(expr)
    assert isinstance(daggered, Product)
    transformed_antiquark = daggered.factors[0]
    assert isinstance(transformed_antiquark, Apply)

    leaf, records, representation = extract_operations(transformed_antiquark)

    assert leaf.is_anti
    assert representation == Representation.LAPH
    assert _operation_signature(records) == (
        (OperationKind.TRANSPORT, {"path": ("+y",)}),
        (OperationKind.DERIVATIVE, {"direction": "+x"}),
    )


def test_stage2_frontend_pipeline_realizes_v_and_vdag_with_ordered_operations():
    """Frontend -> Wick -> PhysicsIR -> Stage 1 -> Stage 2 keeps field identity and operations."""
    reset_registry()
    sink = Operator(
        transport(dbar("x", "a", "s1"), "+z")
        * gamma5("s1", "s2")
        * derivative(transport(u("x", "b", "s2"), "+y"), "+x")
        * delta("a", "b"),
        momenta={"x": (1, 0, 0)},
    )
    corr = Correlator(sink, sink.dagger())

    physics_ir = lower_to_physics_ir(corr, wick_contract(corr))
    assert physics_ir.correlators

    for logical in physics_ir.correlators:
        sink_vertex = next(
            vertex
            for vertex in identify_operator_vertices(logical)
            if vertex.pos_label == "x"
            and {ep.record.operator_id for ep in vertex.endpoints} == {0}
        )
        realized = realize_projected_fields(sink_vertex)

        quark = next(field for field in realized.projected_fields if not field.record.is_anti)
        antiquark = next(field for field in realized.projected_fields if field.record.is_anti)

        assert quark.contribution == ProjectionContribution.EIGENVECTOR
        assert antiquark.contribution == ProjectionContribution.EIGENVECTOR_DAGGER
        assert _operation_signature(quark.operations) == (
            (OperationKind.TRANSPORT, {"path": ("+y",)}),
            (OperationKind.DERIVATIVE, {"direction": "+x"}),
        )
        assert _operation_signature(antiquark.operations) == (
            (OperationKind.TRANSPORT, {"path": ("+z",)}),
        )


def test_stage2_frontend_point_projection_realizes_point_contribution():
    """Point projection changes realization without becoming a field operation."""
    reset_registry()
    sink = Operator(
        dbar("x", "a", "s1")
        * gamma5("s1", "s2")
        * point(u("x", "b", "s2"))
        * delta("a", "b"),
        momenta={"x": (0, 0, 0)},
    )
    corr = Correlator(sink, sink.dagger())

    logical = lower_to_physics_ir(corr, wick_contract(corr)).correlators[0]
    sink_vertex = next(
        vertex
        for vertex in identify_operator_vertices(logical)
        if vertex.pos_label == "x"
        and {ep.record.operator_id for ep in vertex.endpoints} == {0}
    )
    realized = realize_projected_fields(sink_vertex)

    point_field = next(field for field in realized.projected_fields if not field.record.is_anti)
    assert point_field.record.representation == Representation.POINT
    assert point_field.contribution == ProjectionContribution.POINT
    assert point_field.operations == ()
