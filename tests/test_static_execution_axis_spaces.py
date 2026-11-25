"""Regression tests for static execution index-space metadata."""

from dataclasses import replace

import pytest

from concord_lqcd.compiler import (
    ContractionSizeModel,
    IntermediateComputation,
    IntermediateInputUse,
    TensorLegBinding,
    analyze_static_execution,
    build_logical_execution_dag,
    build_partial_intermediate_cse,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import Correlator, Operator, c, cbar, gamma5, reset_registry, u, ubar
from concord_lqcd.core.wick import wick_contract


def _logical(mode_size=4):
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * u("y", "a", "s2")
    )
    source = Operator(
        ubar("x", "b", "s3") * gamma5("s3", "s4") * c("x", "b", "s4")
    )
    correlator = Correlator(sink, source)
    physics_ir = lower_to_physics_ir(correlator, wick_contract(correlator))
    global_dag = lower_physics_ir_collection_to_global_cse({"C00": physics_ir})
    order = optimize_global_contraction_order(
        global_dag,
        ContractionSizeModel(mode_size=mode_size),
        strategy="greedy",
    )
    return build_logical_execution_dag(build_partial_intermediate_cse(order))


def test_static_descriptors_preserve_axis_spaces_even_when_dimensions_match():
    logical = _logical(mode_size=4)
    analysis = analyze_static_execution(logical)

    propagation = next(
        analysis.value(node.node_id)
        for node in logical.primitives.values()
        if node.kind == "propagation"
    )
    assert propagation.shape == (4, 4, 4, 4)
    assert propagation.axis_spaces == ("mode", "spin", "mode", "spin")

    spin = next(
        analysis.value(node.node_id)
        for node in logical.primitives.values()
        if node.kind == "spin"
    )
    assert spin.axis_spaces == ("spin", "spin")


def test_static_analysis_rejects_mode_spin_join_even_when_dimensions_equal():
    logical = _logical(mode_size=4)
    propagation_id, propagation = next(
        (node_id, node)
        for node_id, node in logical.primitives.items()
        if node.kind == "propagation"
    )
    spin_id, spin = next(
        (node_id, node)
        for node_id, node in logical.primitives.items()
        if node.kind == "spin"
    )

    bad = IntermediateComputation(
        intermediate_id="manual_bad_join",
        signature="manual_bad_mode_spin_join",
        inputs=(
            IntermediateInputUse(
                propagation_id,
                (
                    TensorLegBinding("sink_mode", "x"),
                    TensorLegBinding("sink_spin", "a"),
                    TensorLegBinding("source_mode", "b"),
                    TensorLegBinding("source_spin", "c"),
                ),
            ),
            IntermediateInputUse(
                spin_id,
                (
                    TensorLegBinding("left", "x"),
                    TensorLegBinding("right", "d"),
                ),
            ),
        ),
        contracted_indices=("x",),
        output_indices=("a", "b", "c", "d"),
        estimated_flops=1,
        output_elements=4**4,
    )
    contractions = dict(logical.source_dag.intermediates)
    contractions[bad.intermediate_id] = bad
    broken = build_logical_execution_dag(replace(logical.source_dag, intermediates=contractions))

    with pytest.raises(ValueError, match="incompatible index spaces"):
        analyze_static_execution(broken)
