"""Reuse-aware path selection with declared shared-basis cost fixtures."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import concord_lqcd.backend.runtime.pipeline as runtime_pipeline
from tests.plan_costs import (
    _baryon_correlator,
    _meson_correlator,
    _tetraquark_correlator,
)
from concord_lqcd.backend.runtime import (
    RuntimeCompilationConfig,
    compile_correlators,
    estimate_execution_memory,
)


_BUILDERS = {
    "meson": _meson_correlator,
    "baryon": _baryon_correlator,
    "tetraquark": _tetraquark_correlator,
}
_BASELINE = (
    Path(__file__).parent
    / "baselines"
    / "reuse_aware_plan_costs.json"
)


def _online_pool_capacity(schedule, reuse_strategy="best_fit"):
    capacities = []
    assignments = {}
    available = set()
    for step in schedule.steps:
        descriptor = schedule.source_analysis.value(step.node_id)
        compatible = [
            index
            for index in available
            if reuse_strategy != "none"
            and capacities[index][0] == descriptor.scalar_type
            and capacities[index][1] >= descriptor.size_bytes
            and (
                reuse_strategy != "size_matched"
                or capacities[index][1] == descriptor.size_bytes
            )
        ]
        if compatible:
            index = min(
                compatible, key=lambda item: (capacities[item][1], f"buffer_{item}")
            )
            available.remove(index)
        else:
            index = len(capacities)
            capacities.append((descriptor.scalar_type, descriptor.size_bytes))
        assignments[step.node_id] = index
        available.update(assignments[value_id] for value_id in step.released_values)
    return sum(capacity for _, capacity in capacities)


@pytest.fixture(scope="module")
def nv45_comparisons():
    comparisons = {}
    for name, builder in _BUILDERS.items():
        greedy = compile_correlators(
            {name: builder()},
            RuntimeCompilationConfig(
                mode_size=45,
                contraction_strategy="greedy",
                shared_orthonormal_laph_basis=True,
            ),
        )
        reuse_aware = compile_correlators(
            {name: builder()},
            RuntimeCompilationConfig(
                mode_size=45,
                contraction_strategy="reuse-aware",
                shared_orthonormal_laph_basis=True,
            ),
        )
        comparisons[name] = (greedy, reuse_aware)
    return comparisons


def test_reuse_aware_selection_improves_cost_within_online_greedy_ceiling(
    nv45_comparisons,
):
    expected_component_strategies = {
        "meson": {"dynamic-programming"},
        "baryon": {"branch-1"},
        "tetraquark": {"branch-1"},
    }

    for name, (greedy, reuse_aware) in nv45_comparisons.items():
        greedy_cost = greedy.cost_report
        selected_cost = reuse_aware.cost_report

        reuse_aware.validate()
        assert reuse_aware.contraction_order.strategy == "reuse-aware"
        assert {
            plan.strategy
            for plan in reuse_aware.contraction_order.component_plans.values()
        } == expected_component_strategies[name]
        assert (
            selected_cost.scheduled_contraction_flops
            < greedy_cost.scheduled_contraction_flops
        )
        assert (
            selected_cost.total_buffer_capacity_bytes
            <= _online_pool_capacity(greedy.schedule)
        )


def test_lifetime_packing_does_not_tighten_default_path_admission(nv45_comparisons):
    greedy, selected = nv45_comparisons["tetraquark"]
    online_ceiling = _online_pool_capacity(greedy.schedule)

    assert online_ceiling == 19_315_142_176
    assert greedy.cost_report.total_buffer_capacity_bytes == 18_916_962_048
    assert (
        greedy.cost_report.total_buffer_capacity_bytes
        < selected.cost_report.total_buffer_capacity_bytes
        <= online_ceiling
    )
    assert selected.cost_report.scheduled_contraction_flops == 1_619_452_440_000


def test_explicit_packed_greedy_budget_stays_strict(nv45_comparisons, monkeypatch):
    greedy, selected = nv45_comparisons["tetraquark"]
    limit = greedy.cost_report.total_buffer_capacity_bytes

    def reject_implicit_ceiling(*arguments, **keywords):
        raise AssertionError("An explicit budget must not use the implicit ceiling")

    monkeypatch.setattr(
        runtime_pipeline, "_online_buffer_capacity_bytes", reject_implicit_ceiling
    )
    limited = compile_correlators(
        {"tetraquark": _tetraquark_correlator()},
        RuntimeCompilationConfig(
            mode_size=45,
            contraction_strategy="reuse-aware",
            shared_orthonormal_laph_basis=True,
            physical_memory_limit_bytes=limit,
        ),
    )

    assert limited.cost_report.total_buffer_capacity_bytes <= limit
    assert limited.contraction_order.selected_strategies == ("greedy",)
    assert (
        limited.cost_report.scheduled_contraction_flops
        > selected.cost_report.scheduled_contraction_flops
    )


@pytest.mark.parametrize("reuse_strategy", ("size_matched", "none"))
def test_other_reuse_policies_keep_their_original_ceiling(reuse_strategy, monkeypatch):
    def reject_lifetime_ceiling(*arguments, **keywords):
        raise AssertionError("Unchanged reuse policies must use their original plan")

    monkeypatch.setattr(
        runtime_pipeline, "_online_buffer_capacity_bytes", reject_lifetime_ceiling
    )
    greedy = compile_correlators(
        {"baryon": _baryon_correlator()},
        RuntimeCompilationConfig(
            mode_size=4,
            shared_orthonormal_laph_basis=True,
            reuse_strategy=reuse_strategy,
        ),
    )
    selected = compile_correlators(
        {"baryon": _baryon_correlator()},
        RuntimeCompilationConfig(
            mode_size=4,
            contraction_strategy="reuse-aware",
            shared_orthonormal_laph_basis=True,
            reuse_strategy=reuse_strategy,
        ),
    )

    assert greedy.cost_report.total_buffer_capacity_bytes == _online_pool_capacity(
        greedy.schedule, reuse_strategy
    )
    assert (
        selected.cost_report.total_buffer_capacity_bytes
        <= greedy.cost_report.total_buffer_capacity_bytes
    )


def test_reuse_aware_selection_is_deterministic(nv45_comparisons):
    first = nv45_comparisons["baryon"][1]
    second = compile_correlators(
        {"baryon": _baryon_correlator()},
        RuntimeCompilationConfig(
            mode_size=45,
            contraction_strategy="reuse-aware",
        ),
    )

    assert (
        second.contraction_order.component_plans
        == first.contraction_order.component_plans
    )
    assert asdict(second.cost_report) == asdict(first.cost_report)


def test_reuse_aware_selection_is_stable_across_hash_seeds():
    code = "\n".join(
        (
            "import json",
            "from dataclasses import asdict",
            "from tests.plan_costs import _baryon_correlator",
            "from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators",
            "compilation = compile_correlators(",
            "    {'baryon': _baryon_correlator()},",
            "    RuntimeCompilationConfig(mode_size=45, contraction_strategy='reuse-aware'),",
            ")",
            "payload = {",
            "    'plans': {",
            "        key: asdict(value)",
            "        for key, value in compilation.contraction_order.component_plans.items()",
            "    },",
            "    'cost': asdict(compilation.cost_report),",
            "}",
            "print(json.dumps(payload, sort_keys=True))",
        )
    )

    outputs = []
    for seed in ("1", "101"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).parents[1],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]


def test_reuse_aware_scaling_matches_frozen_baseline(nv45_comparisons):
    compilations_by_mode = {
        "45": {
            name: comparison[1]
            for name, comparison in nv45_comparisons.items()
        }
    }
    for mode_size in (4, 75):
        compilations_by_mode[str(mode_size)] = {
            name: compile_correlators(
                {name: builder()},
                RuntimeCompilationConfig(
                    mode_size=mode_size,
                    contraction_strategy="reuse-aware",
                    shared_orthonormal_laph_basis=True,
                ),
            )
            for name, builder in _BUILDERS.items()
        }

    actual = {
        "comparison_schema_version": 2,
        "mode_sizes": {
            mode_size: {
                name: {
                    "component_strategies": sorted(
                        {
                            plan.strategy
                            for plan in (
                                compilation.contraction_order.component_plans.values()
                            )
                        }
                    ),
                    "scheduled_contraction_flops": (
                        compilation.cost_report.scheduled_contraction_flops
                    ),
                    "logical_peak_live_bytes": (
                        compilation.cost_report.logical_peak_live_bytes
                    ),
                        "total_buffer_capacity_bytes": (
                            compilation.cost_report.total_buffer_capacity_bytes
                        ),
                        "largest_transient_value_bytes": (
                            estimate_execution_memory(
                                compilation.physical_plan
                            ).largest_transient_value_bytes
                        ),
                        "known_working_set_bytes": (
                            estimate_execution_memory(
                                compilation.physical_plan
                            ).known_working_set_bytes
                        ),
                }
                for name, compilation in compilations.items()
            }
            for mode_size, compilations in compilations_by_mode.items()
        },
    }
    expected = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert actual == expected


def test_physical_budget_controls_tetraquark_flop_memory_tradeoff():
    safe_limit = 19_200_000_000
    roomy_limit = 36_000_000_000
    safe = compile_correlators(
        {"tetraquark": _tetraquark_correlator()},
        RuntimeCompilationConfig(
            mode_size=45,
            contraction_strategy="reuse-aware",
            physical_memory_limit_bytes=safe_limit,
        ),
    )
    roomy = compile_correlators(
        {"tetraquark": _tetraquark_correlator()},
        RuntimeCompilationConfig(
            mode_size=45,
            contraction_strategy="reuse-aware",
            physical_memory_limit_bytes=roomy_limit,
        ),
    )

    assert {
        plan.strategy for plan in safe.contraction_order.component_plans.values()
    } == {"branch-1"}
    assert {
        plan.strategy for plan in roomy.contraction_order.component_plans.values()
    } == {"dynamic-programming"}
    assert safe.cost_report.total_buffer_capacity_bytes <= safe_limit
    assert roomy.cost_report.total_buffer_capacity_bytes <= roomy_limit
    assert (
        roomy.cost_report.scheduled_contraction_flops
        < safe.cost_report.scheduled_contraction_flops
    )
    assert (
        roomy.cost_report.total_buffer_capacity_bytes
        > safe.cost_report.total_buffer_capacity_bytes
    )


def test_reuse_aware_selection_rejects_budget_when_no_candidate_fits():
    with pytest.raises(MemoryError, match="No reuse-aware.*minimum required"):
        compile_correlators(
            {"tetraquark": _tetraquark_correlator()},
            RuntimeCompilationConfig(
                mode_size=45,
                contraction_strategy="reuse-aware",
                physical_memory_limit_bytes=1,
            ),
        )


def test_optional_candidate_failure_preserves_greedy_fallback(monkeypatch):
    original = runtime_pipeline.optimize_global_contraction_order

    def fail_optional_candidates(
        dag,
        size_model,
        *,
        strategy,
        memory_limit,
    ):
        if strategy != "greedy":
            raise RuntimeError("synthetic optional optimizer failure")
        return original(
            dag,
            size_model,
            strategy=strategy,
            memory_limit=memory_limit,
        )

    monkeypatch.setattr(
        runtime_pipeline,
        "optimize_global_contraction_order",
        fail_optional_candidates,
    )

    compilation = compile_correlators(
        {"meson": _meson_correlator()},
        RuntimeCompilationConfig(
            mode_size=4,
            contraction_strategy="reuse-aware",
        ),
    )

    assert compilation.contraction_order.strategy == "reuse-aware"
    assert compilation.contraction_order.selected_strategies == ("greedy",)


def test_reuse_aware_candidates_support_minimum_opt_einsum_api(monkeypatch):
    original = runtime_pipeline.optimize_global_contraction_order
    supported_by_opt_einsum_3_3 = {
        "greedy",
        "dynamic-programming",
        "branch-1",
        "branch-2",
    }
    observed = set()

    def enforce_supported_candidates(
        dag,
        size_model,
        *,
        strategy,
        memory_limit,
    ):
        assert strategy in supported_by_opt_einsum_3_3
        observed.add(strategy)
        return original(
            dag,
            size_model,
            strategy=strategy,
            memory_limit=memory_limit,
        )

    monkeypatch.setattr(
        runtime_pipeline,
        "optimize_global_contraction_order",
        enforce_supported_candidates,
    )

    compile_correlators(
        {"meson": _meson_correlator()},
        RuntimeCompilationConfig(
            mode_size=4,
            contraction_strategy="reuse-aware",
        ),
    )

    assert observed == supported_by_opt_einsum_3_3
