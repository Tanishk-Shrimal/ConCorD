"""Focused tests for the new end-to-end ConCorD runtime orchestration."""

from __future__ import annotations

from math import prod

import pytest
import torch

from concord_lqcd.backend.runtime import (
    KernelMaterializationProvider,
    PropagationSourceHandler,
    RuntimeCompilationConfig,
    RuntimeExecutionConfig,
    bind_compilation_resources,
    compile_correlators,
    execute_compilation,
    run_correlators,
)
from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource
from concord_lqcd.compiler.propagation_lowering import PropagationRecipe
from concord_lqcd.core import (
    Correlator,
    Operator,
    build_baryon_operator,
    c,
    cbar,
    d,
    dbar,
    gamma5,
    reset_registry,
    u,
    ubar,
)
from concord_lqcd.core.correlator import CorrMatrix


def _operators():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * u("y", "a", "s2")
    )
    source = Operator(
        ubar("x", "b", "s3") * gamma5("s3", "s4") * c("x", "b", "s4")
    )
    return sink, source


def _two_point():
    sink, source = _operators()
    return Correlator(sink, source)


def _baryon_two_point():
    reset_registry()
    baryon = build_baryon_operator(
        ["l", "l", "s"],
        "Cg5",
        momentum=(0, 0, 0),
    )
    return CorrMatrix([baryon])[0, 0]


def _tetraquark_two_point():
    reset_registry()
    sink = Operator(
        dbar("x", "a", "s1")
        * u("x", "b", "s2")
        * ubar("x", "c", "s3")
        * d("x", "d", "s4"),
        momenta={"x": (0, 0, 0)},
    )
    source = Operator(
        ubar("y", "a", "s5")
        * d("y", "b", "s6")
        * dbar("y", "c", "s7")
        * u("y", "d", "s8"),
        momenta={"y": (0, 0, 0)},
    )
    return Correlator(sink, source)


class _DeterministicKernelProvider(KernelMaterializationProvider):
    def materialize(self, request):
        count = prod(request.expected_shape) if request.expected_shape else 1
        offset = sum(ord(char) for char in request.resource_id) % 17
        offset += sum(request.operator_times)
        values = torch.arange(count, dtype=torch.float64).reshape(request.expected_shape)
        return (values + 1 + offset).to(torch.complex128)


class _DeterministicPropagationHandler(PropagationSourceHandler):
    def materialize(self, request):
        count = prod(request.expected_shape)
        offset = sum(ord(char) for char in request.resource_id) % 19
        offset += 10 * request.source_time + request.sink_time
        offset += sum(
            (index + 1) * value
            for index, value in enumerate(request.insertion_times)
        )
        values = torch.arange(count, dtype=torch.float64).reshape(request.expected_shape)
        return (values / 23.0 + 1 + offset).to(torch.complex128)


def _source_resolver(resource_id: str, recipe: PropagationRecipe) -> ExternalTensorSource:
    assert recipe.resource_id == resource_id
    return ExternalTensorSource(
        source_id=f"synthetic:{resource_id}",
        kind="synthetic",
        location=resource_id,
    )


def _compilation_config(
    *,
    reuse_strategy="best_fit",
    contraction_strategy="greedy",
):
    return RuntimeCompilationConfig(
        mode_size=2,
        contraction_strategy=contraction_strategy,
        scalar_type="complex128",
        element_bytes=16,
        schedule_strategy="memory_aware",
        reuse_strategy=reuse_strategy,
    )


def _execution_config():
    return RuntimeExecutionConfig(
        operator_times={2: (3, 1)},
        propagation_source_resolver=_source_resolver,
        kernel_providers={"distillation": _DeterministicKernelProvider()},
        source_handlers={"synthetic": _DeterministicPropagationHandler()},
        device="cpu",
    )


def test_compile_correlators_exposes_complete_frozen_pipeline_and_exact_resources():
    compilation = compile_correlators(
        {"C0": _two_point()},
        _compilation_config(),
    )

    compilation.validate()
    assert compilation.correlator_ids == ("C0",)
    assert tuple(compilation.physics_ir) == ("C0",)
    assert compilation.contraction_order.source_dag is compilation.global_cse
    assert compilation.intermediate_dag.source_plan is compilation.contraction_order
    assert compilation.logical_dag.source_dag is compilation.intermediate_dag
    assert compilation.static_analysis.source_dag is compilation.logical_dag
    assert compilation.schedule.source_analysis is compilation.static_analysis
    assert compilation.physical_plan.source_schedule is compilation.schedule

    required = compilation.required_propagation_resources
    assert required
    assert all(resource_id == recipe.resource_id for resource_id, recipe in required.items())
    assert len(compilation.required_kernel_resource_ids) == 2


def test_resource_binding_resolver_receives_exact_scheduled_resource_ids_and_recipes():
    compilation = compile_correlators(
        {"C0": _two_point()},
        _compilation_config(),
    )
    seen = []

    def resolver(resource_id, recipe):
        seen.append((resource_id, recipe))
        return ExternalTensorSource(
            source_id=f"resolved:{resource_id}",
            kind="synthetic",
            location=resource_id,
        )

    config = RuntimeExecutionConfig(
        operator_times={2: (3, 1)},
        propagation_source_resolver=resolver,
        kernel_providers={"distillation": _DeterministicKernelProvider()},
        source_handlers={"synthetic": _DeterministicPropagationHandler()},
    )
    binding = bind_compilation_resources(compilation, config)

    required = compilation.required_propagation_resources
    assert {resource_id for resource_id, _ in seen} == set(required)
    for resource_id, recipe in seen:
        assert recipe is required[resource_id]
        assert binding.bindings[resource_id].source.location == resource_id


def test_run_correlators_executes_full_new_pipeline_without_legacy_runtime_objects():
    correlator = _two_point()
    result = run_correlators(
        {"C0": correlator, "C1": correlator},
        _compilation_config(),
        _execution_config(),
    )

    result.validate()
    assert set(result.outputs) == {"C0", "C1"}
    assert result.numerical_result.executed_steps == len(result.compilation.physical_plan.steps)
    assert torch.allclose(result.output("C0"), result.output("C1"))
    assert torch.isfinite(result.output("C0").real).all()
    assert torch.isfinite(result.output("C0").imag).all()


def test_split_compile_execute_matches_one_call_entry_point():
    correlator = _two_point()
    compilation_config = _compilation_config()
    execution_config = _execution_config()

    compilation = compile_correlators({"C0": correlator}, compilation_config)
    split = execute_compilation(compilation, execution_config)
    direct = run_correlators(
        {"C0": correlator},
        compilation_config,
        execution_config,
    )

    assert torch.allclose(split.output("C0"), direct.output("C0"))


def test_corr_matrix_input_gets_stable_matrix_entry_ids():
    sink, source = _operators()
    matrix = CorrMatrix([sink], [source])

    compilation = compile_correlators(matrix, _compilation_config())

    assert compilation.correlator_ids == ("C[0,0]",)
    assert tuple(compilation.physics_ir) == ("C[0,0]",)
    assert set(compilation.physical_plan.outputs) == {"C[0,0]"}


def test_single_correlator_input_uses_default_output_id():
    compilation = compile_correlators(_two_point(), _compilation_config())

    assert compilation.correlator_ids == ("C0",)
    assert set(compilation.physical_plan.outputs) == {"C0"}


def test_flavor_like_source_key_cannot_replace_exact_propagation_resource_binding():
    compilation = compile_correlators({"C0": _two_point()}, _compilation_config())
    fake = ExternalTensorSource(
        source_id="flavor:u",
        kind="synthetic",
        location="u",
    )
    config = RuntimeExecutionConfig(
        operator_times={2: (3, 1)},
        propagation_sources={"u": fake},
        kernel_providers={"distillation": _DeterministicKernelProvider()},
        source_handlers={"synthetic": _DeterministicPropagationHandler()},
    )

    with pytest.raises(ValueError, match="No exact external tensor source"):
        bind_compilation_resources(compilation, config)


def test_end_to_end_reuse_and_no_reuse_are_numerically_identical():
    correlator = _two_point()
    execution = _execution_config()

    reused = run_correlators(
        {"C0": correlator, "C1": correlator},
        _compilation_config(reuse_strategy="best_fit"),
        execution,
    )
    separate = run_correlators(
        {"C0": correlator, "C1": correlator},
        _compilation_config(reuse_strategy="none"),
        execution,
    )

    assert reused.compilation.physical_plan.statistics.buffer_reuses > 0
    for correlator_id in ("C0", "C1"):
        assert torch.allclose(reused.output(correlator_id), separate.output(correlator_id))


@pytest.mark.parametrize(
    ("builder", "expects_distinct_plans"),
    (
        (_two_point, True),
        (_baryon_two_point, True),
        (_tetraquark_two_point, True),
    ),
    ids=("meson", "baryon", "tetraquark"),
)
def test_end_to_end_reuse_aware_and_greedy_paths_are_numerically_identical(
    builder,
    expects_distinct_plans,
):
    execution = _execution_config()

    greedy = run_correlators(
        {"C0": builder()},
        _compilation_config(contraction_strategy="greedy"),
        execution,
    )
    reuse_aware = run_correlators(
        {"C0": builder()},
        _compilation_config(contraction_strategy="reuse-aware"),
        execution,
    )

    plans_are_distinct = greedy.compilation.contraction_order.component_plans != (
        reuse_aware.compilation.contraction_order.component_plans
    )
    assert plans_are_distinct is expects_distinct_plans
    assert torch.allclose(greedy.output("C0"), reuse_aware.output("C0"))
