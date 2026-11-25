"""Regression tests for deterministic compilation cost reporting."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tests.plan_costs import collect_plan_costs
from concord_lqcd.backend.runtime.metrics import CompilationCostReport


_BASELINE = (
    Path(__file__).parent
    / "baselines"
    / "plan_costs_nv4.json"
)
_SCALING_BASELINE = (
    Path(__file__).parent
    / "baselines"
    / "plan_cost_scaling.json"
)


@pytest.fixture(scope="module")
def plan_costs_by_mode():
    return {
        str(mode_size): collect_plan_costs(mode_size=mode_size)
        for mode_size in (4, 45, 75)
    }


def test_representative_plan_costs_match_frozen_nv4_baseline(plan_costs_by_mode):
    expected = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert plan_costs_by_mode["4"] == expected


def test_representative_plan_cost_scaling_matches_frozen_baseline(
    plan_costs_by_mode,
):
    expected = json.loads(_SCALING_BASELINE.read_text(encoding="utf-8"))
    actual = {
        "scaling_schema_version": 1,
        "mode_sizes": {
            mode_size: {
                name: {
                    metric: report[metric]
                    for metric in (
                        "scheduled_contraction_flops",
                        "logical_peak_live_bytes",
                        "largest_value_bytes",
                    )
                }
                for name, report in payload["cases"].items()
            }
            for mode_size, payload in plan_costs_by_mode.items()
        },
    }

    assert actual == expected


def test_cost_report_exposes_reduction_and_parallelism_metrics(plan_costs_by_mode):
    report_payload = plan_costs_by_mode["4"]["cases"]["baryon"]
    report = CompilationCostReport(**report_payload)

    report.validate()
    assert report.wick_terms_eliminated == 0
    assert report.resource_occurrences_eliminated > 0
    assert report.contraction_steps_eliminated > 0
    assert report.estimated_flops_saved > 0
    assert report.dead_contraction_flops_skipped == 0
    assert report.max_ready_width > 1


def test_cost_report_rejects_internally_inconsistent_counts(plan_costs_by_mode):
    report_payload = plan_costs_by_mode["4"]["cases"]["meson"]
    report = CompilationCostReport(**report_payload)
    invalid = replace(
        report,
        unique_resources=report.resource_occurrences + 1,
    )

    with pytest.raises(ValueError, match="Unique resources exceed"):
        invalid.validate()
