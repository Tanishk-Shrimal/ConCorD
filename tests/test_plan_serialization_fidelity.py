"""Additional fidelity tests for the V1 compiled-plan boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from concord_lqcd.job import (
    JobSpec,
    StalePlanError,
    compile_job_plan,
    load_concord_plan,
    save_concord_plan,
    validate_plan_for_job,
)


def _spec(tmp_path: Path) -> JobSpec:
    operator_file = tmp_path / "operators.py"
    operator_file.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, c, cbar, u, ubar, gamma5",
                "def build():",
                "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * u('y','a','s2'))",
                "    source = Operator(ubar('x','b','s3') * gamma5('s3','s4') * c('x','b','s4'))",
                "    return {'D': Correlator(sink, source)}",
            ]
        ),
        encoding="utf-8",
    )
    return JobSpec.model_validate(
        {
            "schema_version": 1,
            "job": {"name": "fidelity"},
            "operators": {"file": str(operator_file), "factory": "build"},
            "plan": {"file": str(tmp_path / "fidelity.concord")},
            "lattice": {"spatial_extent": [4, 4, 4], "temporal_extent": 8},
            "compile": {"mode_size": 2},
            "run": {
                "device": "cpu",
                "configurations": {"values": [0]},
                "times": {
                    "generator": {
                        "type": "two_point",
                        "source": 1,
                        "sinks": {"values": [3]},
                    }
                },
            },
            "data": {
                "eigenvectors": {
                    "provider": "hdf5_distillation",
                    "path": str(tmp_path / "evec_{configuration}.h5"),
                    "dataset": "/evec",
                    "axes": ["time", "z", "y", "x", "color", "mode"],
                },
                "propagation": {
                    "provider": "hdf5_perambulator",
                    "ordinary": {
                        "path": str(tmp_path / "peram_{flavor}_{configuration}.h5"),
                        "dataset": "/tau_{source_time}",
                        "axes": [
                            "sink_time",
                            "sink_spin",
                            "source_spin",
                            "sink_mode",
                            "source_mode",
                        ],
                    },
                },
            },
            "output": {"file": str(tmp_path / "result.h5")},
        }
    )


def test_roundtrip_preserves_global_cse_contraction_order_and_partial_cse(tmp_path):
    spec = _spec(tmp_path)
    original = compile_job_plan(spec)
    save_concord_plan(original, spec.plan.file)
    loaded = load_concord_plan(spec.plan.file)

    assert loaded.global_cse.resources == original.global_cse.resources
    assert loaded.global_cse.components == original.global_cse.components
    assert loaded.global_cse.correlators == original.global_cse.correlators
    assert loaded.global_cse.statistics == original.global_cse.statistics

    assert loaded.contraction_order.size_model == original.contraction_order.size_model
    assert loaded.contraction_order.strategy == original.contraction_order.strategy
    assert loaded.contraction_order.memory_limit == original.contraction_order.memory_limit
    assert loaded.contraction_order.component_plans == original.contraction_order.component_plans

    assert loaded.intermediate_dag.primitive_values == original.intermediate_dag.primitive_values
    assert loaded.intermediate_dag.intermediates == original.intermediate_dag.intermediates
    assert loaded.intermediate_dag.component_plans == original.intermediate_dag.component_plans
    assert loaded.intermediate_dag.statistics == original.intermediate_dag.statistics
    assert loaded.cost_report == original.cost_report
    assert loaded.diagnostics["cost_report"] == original.diagnostics["cost_report"]


def test_runtime_only_job_changes_do_not_stale_plan_but_compile_changes_do(tmp_path):
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)

    runtime_changed = spec.model_copy(
        update={
            "run": spec.run.model_copy(update={"device": "cuda:1"}),
            "output": spec.output.model_copy(update={"file": tmp_path / "other.h5"}),
        }
    )
    validate_plan_for_job(plan, runtime_changed)

    compile_changed = spec.model_copy(
        update={
            "compile": spec.compile.model_copy(update={"mode_size": 3}),
        }
    )
    with pytest.raises(StalePlanError, match="stale"):
        validate_plan_for_job(plan, compile_changed)


def test_cost_report_handles_saved_plan_without_new_diagnostics(tmp_path):
    plan = compile_job_plan(_spec(tmp_path))
    older_plan = replace(plan, diagnostics={})

    report = older_plan.cost_report

    assert report.input_wick_terms is None
    assert report.surviving_wick_terms == plan.cost_report.surviving_wick_terms
    assert (
        report.scheduled_contraction_flops
        == plan.cost_report.scheduled_contraction_flops
    )


def test_reuse_aware_component_strategy_survives_plan_roundtrip(tmp_path):
    spec = _spec(tmp_path)
    spec = spec.model_copy(
        update={
            "compile": spec.compile.model_copy(
                update={
                    "contraction": spec.compile.contraction.model_copy(
                        update={"strategy": "reuse-aware"}
                    )
                }
            )
        }
    )
    original = compile_job_plan(spec)
    save_concord_plan(original, spec.plan.file)
    loaded = load_concord_plan(spec.plan.file)

    assert loaded.config.contraction_strategy == "reuse-aware"
    assert loaded.contraction_order.strategy == "reuse-aware"
    assert loaded.contraction_order.component_plans == (
        original.contraction_order.component_plans
    )
    assert loaded.diagnostics["contraction_order"]["selected_strategies"] == list(
        original.contraction_order.selected_strategies
    )
    loaded.validate()
