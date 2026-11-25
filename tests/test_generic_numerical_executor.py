"""Focused tests for the new schedule/physical-plan numerical executor."""

from __future__ import annotations

from dataclasses import replace
from math import prod

import pytest
import torch

from tests.plan_costs import _baryon_correlator, _tetraquark_correlator
from concord_lqcd.backend.planner import build_physical_execution_plan
from concord_lqcd.backend.planner.physical import MaterializePrimitiveOperation
from concord_lqcd.backend.runtime.materialization import (
    InMemoryTensorSourceHandler,
    KernelMaterializationProvider,
    PrimitiveMaterializer,
)
from concord_lqcd.backend.runtime.batched_executor import BatchedNumericalExecutor
from concord_lqcd.backend.runtime.numerical_executor import (
    GenericNumericalExecutor,
    RuntimeBufferPool,
)
from concord_lqcd.backend.runtime.pipeline import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.backend.runtime.residency import ResidentExecutionSession
from concord_lqcd.backend.runtime.tiling import (
    TilingPlanningError,
    plan_tiled_execution,
)
from concord_lqcd.backend.runtime.resource_binding import (
    ExternalTensorSource,
    KernelMaterializationRequest,
    PropagationMaterializationRequest,
    build_resource_binding_plan,
)
from concord_lqcd.compiler import (
    ContractionSizeModel,
    analyze_static_execution,
    build_logical_execution_dag,
    build_logical_schedule,
    build_partial_intermediate_cse,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import Correlator, Operator, c, cbar, gamma5, reset_registry, u, ubar
from concord_lqcd.core.wick import wick_contract


def _physics_ir():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * u("y", "a", "s2")
    )
    source = Operator(
        ubar("x", "b", "s3") * gamma5("s3", "s4") * c("x", "b", "s4")
    )
    correlator = Correlator(sink, source)
    return lower_to_physics_ir(correlator, wick_contract(correlator))


def _schedule(
    correlator_ids=("C00",),
    *,
    mode_size=2,
    scalar_type="complex128",
    element_bytes=16,
):
    physics_ir = _physics_ir()
    global_dag = lower_physics_ir_collection_to_global_cse({
        correlator_id: physics_ir for correlator_id in correlator_ids
    })
    order = optimize_global_contraction_order(
        global_dag,
        ContractionSizeModel(mode_size=mode_size),
        strategy="greedy",
    )
    logical = build_logical_execution_dag(build_partial_intermediate_cse(order))
    analysis = analyze_static_execution(
        logical,
        scalar_type=scalar_type,
        element_bytes=element_bytes,
    )
    return build_logical_schedule(analysis, strategy="memory_aware")


def _propagation_sources(plan):
    resource_ids = tuple(dict.fromkeys(
        step.operation.resource_id
        for step in plan.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
        and step.operation.kind == "propagation"
    ))
    return {
        resource_id: ExternalTensorSource(
            source_id=f"memory:{resource_id}",
            kind="memory",
            location=resource_id,
        )
        for resource_id in resource_ids
    }


class _DeterministicKernelProvider(KernelMaterializationProvider):
    def materialize(self, request):
        count = prod(request.expected_shape) if request.expected_shape else 1
        resource_offset = sum(ord(char) for char in request.resource_id) % 11
        time_offset = sum(request.operator_times)
        values = torch.arange(count, dtype=torch.float64).reshape(request.expected_shape)
        return (values + 1 + resource_offset + time_offset).to(torch.complex128)


class _CountingKernelProvider(_DeterministicKernelProvider):
    def __init__(self, calls, *, cacheable):
        self.calls = calls
        self.cacheable = cacheable

    def cache_identity(self):
        return ("counting-kernel", 1) if self.cacheable else None

    def materialize(self, request):
        self.calls.append((request.resource_id, request.operator_times))
        return super().materialize(request)


def _seeded_complex_tensor(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imaginary)


class _SeededComplexKernelProvider(KernelMaterializationProvider):
    def __init__(self, seed):
        self.seed = seed

    def materialize(self, request):
        flavor_seed = sum(
            (index + 1) * sum(ord(character) for character in field.record.flavor)
            for index, field in enumerate(request.recipe.projected_fields)
        )
        time_seed = sum(
            (index + 1) * value for index, value in enumerate(request.operator_times)
        )
        return _seeded_complex_tensor(
            request.expected_shape, 20261006 + self.seed + flavor_seed + 1000 * time_seed
        )


def _complex_propagation_loader(request):
    flavor_seed = sum(
        (index + 1) * sum(ord(character) for character in flavor)
        for index, flavor in enumerate(request.recipe.segment_flavors)
    )
    time_seed = 10 * request.source_time + request.sink_time
    time_seed += sum(
        (index + 1) * value for index, value in enumerate(request.insertion_times)
    )
    return _seeded_complex_tensor(
        request.expected_shape, 20261007 + flavor_seed + 1000 * time_seed
    )


def _propagation_loader(request):
    count = prod(request.expected_shape)
    resource_offset = sum(ord(char) for char in request.resource_id) % 13
    time_offset = 10 * request.source_time + request.sink_time
    time_offset += sum((idx + 1) * value for idx, value in enumerate(request.insertion_times))
    values = torch.arange(count, dtype=torch.float64).reshape(request.expected_shape)
    return (values / 17.0 + 1 + resource_offset + time_offset).to(torch.complex128)


def _materializer_for(
    plan, *, operator_times=None, kernel_provider=None, propagation_loader=_propagation_loader
):
    sources = _propagation_sources(plan)
    binding = build_resource_binding_plan(
        plan,
        operator_times=operator_times or {2: (3, 1)},
        propagation_sources=sources,
    )
    registry = {source.location: propagation_loader for source in sources.values()}
    return PrimitiveMaterializer(
        binding,
        kernel_providers={
            "distillation": kernel_provider or _DeterministicKernelProvider()
        },
        source_handlers={"memory": InMemoryTensorSourceHandler(registry)},
    )


def _executor_for(plan, *, tiled_plan=None, compute_device="cpu"):
    return GenericNumericalExecutor(
        _materializer_for(plan),
        device="cpu",
        compute_device=compute_device,
        tiled_plan=tiled_plan,
    )


def test_full_executor_produces_requested_outputs_with_planned_interface():
    schedule = _schedule(("C0", "C1"), mode_size=2)
    plan = build_physical_execution_plan(schedule, reuse_strategy="best_fit")

    result = _executor_for(plan).execute()

    assert result.executed_steps == len(plan.steps)
    assert set(result.outputs) == {"C0", "C1"}
    for correlator_id, output in result.outputs.items():
        physical_output = plan.outputs[correlator_id]
        binding = plan.binding(physical_output.value_id)
        assert output.shape == torch.Size(binding.shape)
        assert output.dtype == torch.complex128
        assert output.device.type == "cpu"
        assert torch.isfinite(output.real).all()
        assert torch.isfinite(output.imag).all()

    # The two requested correlators are identical and share the globally reused
    # component computation; only their assembly/output nodes are distinct.
    assert torch.allclose(result.outputs["C0"], result.outputs["C1"])


def test_buffer_reuse_and_no_reuse_execution_are_numerically_identical():
    schedule = _schedule(("C0", "C1"), mode_size=2)
    reused = build_physical_execution_plan(schedule, reuse_strategy="best_fit")
    size_matched = build_physical_execution_plan(schedule, reuse_strategy="size_matched")
    separate = build_physical_execution_plan(schedule, reuse_strategy="none")

    assert reused.statistics.buffer_reuses > 0
    assert reused.statistics.unique_buffers < separate.statistics.unique_buffers

    reused_result = _executor_for(reused).execute()
    size_matched_result = _executor_for(size_matched).execute()
    separate_result = _executor_for(separate).execute()

    for correlator_id in ("C0", "C1"):
        assert torch.allclose(
            reused_result.outputs[correlator_id],
            separate_result.outputs[correlator_id],
        )
        assert torch.allclose(
            size_matched_result.outputs[correlator_id],
            separate_result.outputs[correlator_id],
        )


def test_leading_axis_batch_matches_each_independent_execution():
    schedule = _schedule(("C0", "C1"), mode_size=2)
    plan = build_physical_execution_plan(schedule, reuse_strategy="best_fit")
    contexts = (
        {2: (3, 1)},
        {2: (4, 1)},
        {2: (5, 2)},
    )
    materializers = tuple(
        _materializer_for(plan, operator_times=context)
        for context in contexts
    )

    batched = BatchedNumericalExecutor(materializers, device="cpu").execute()

    assert batched.batch_size == len(contexts)
    assert batched.executed_steps == len(plan.steps)
    for batch_index, context in enumerate(contexts):
        sequential = GenericNumericalExecutor(
            _materializer_for(plan, operator_times=context),
            device="cpu",
        ).execute()
        for correlator_id in ("C0", "C1"):
            output = batched.output(correlator_id)
            binding = plan.binding(plan.outputs[correlator_id].value_id)
            assert output.shape == torch.Size((len(contexts), *binding.shape))
            torch.testing.assert_close(
                batched.outputs_at(batch_index)[correlator_id],
                sequential.output(correlator_id),
                rtol=1e-12,
                atol=1e-10,
            )


@pytest.mark.parametrize(
    ("cacheable", "expected_multiplier"),
    ((True, 1), (False, 3)),
    ids=("exact-identity-broadcast", "uncacheable-independent-fallback"),
)
def test_batched_kernel_materialization_uses_only_proven_invariants(
    cacheable,
    expected_multiplier,
):
    plan = build_physical_execution_plan(_schedule(mode_size=2))
    calls = []
    provider = _CountingKernelProvider(calls, cacheable=cacheable)
    context = {2: (3, 1)}
    materializers = tuple(
        _materializer_for(
            plan,
            operator_times=context,
            kernel_provider=provider,
        )
        for _ in range(3)
    )
    kernel_operations = tuple(
        step.operation
        for step in plan.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
        and step.operation.kind == "kernel"
    )

    BatchedNumericalExecutor(materializers, device="cpu").execute()

    assert len(calls) == expected_multiplier * len(kernel_operations)


@pytest.mark.parametrize(
    "correlator_factory",
    (_baryon_correlator, _tetraquark_correlator),
    ids=("baryon", "tetraquark"),
)
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_leading_axis_batch_is_topology_agnostic(correlator_factory, seed):
    compilation = compile_correlators(
        {"C": correlator_factory()},
        RuntimeCompilationConfig(mode_size=2),
    )
    plan = compilation.physical_plan
    contexts = ({2: (3, 1)}, {2: (4, 1)})
    batched = BatchedNumericalExecutor(
        tuple(
            _materializer_for(
                plan,
                operator_times=context,
                kernel_provider=_SeededComplexKernelProvider(seed),
                propagation_loader=_complex_propagation_loader,
            )
            for context in contexts
        ),
        device="cpu",
    ).execute()

    for batch_index, context in enumerate(contexts):
        sequential = GenericNumericalExecutor(
            _materializer_for(
                plan,
                operator_times=context,
                kernel_provider=_SeededComplexKernelProvider(seed),
                propagation_loader=_complex_propagation_loader,
            ),
            device="cpu",
        ).execute()
        torch.testing.assert_close(
            batched.outputs_at(batch_index)["C"],
            sequential.output("C"),
            rtol=1e-12,
            atol=1e-10,
        )
        assert torch.max(torch.abs(sequential.output("C"))).item() > 1e-8
    assert not torch.equal(batched.outputs_at(0)["C"], batched.outputs_at(1)["C"])


def test_complex_batch_fixtures_ignore_opaque_resource_ids():
    compilation = compile_correlators(
        {"C": _baryon_correlator()}, RuntimeCompilationConfig(mode_size=2)
    )
    materializer = _materializer_for(compilation.physical_plan)
    provider = _SeededComplexKernelProvider(0)
    for request in materializer.binding_plan.materializations.values():
        renamed = replace(request, resource_id="unrelated-opaque-resource-id")
        if isinstance(request, KernelMaterializationRequest):
            original = provider.materialize(request)
            actual = provider.materialize(renamed)
        elif isinstance(request, PropagationMaterializationRequest):
            original = _complex_propagation_loader(request)
            actual = _complex_propagation_loader(renamed)
        else:
            continue
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_batched_executor_rejects_single_work_unit():
    plan = build_physical_execution_plan(_schedule(mode_size=2))

    with pytest.raises(ValueError, match="at least two"):
        BatchedNumericalExecutor((_materializer_for(plan),), device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_resident_cuda_batch_matches_independent_cuda_execution():
    plan = build_physical_execution_plan(
        _schedule(("C0", "C1"), mode_size=3),
        reuse_strategy="best_fit",
    )
    contexts = ({2: (3, 1)}, {2: (4, 1)}, {2: (5, 2)})
    batched = BatchedNumericalExecutor(
        tuple(
            _materializer_for(plan, operator_times=context)
            for context in contexts
        ),
        device="cuda:0",
    ).execute()

    assert batched.device.type == "cuda"
    for batch_index, context in enumerate(contexts):
        sequential = GenericNumericalExecutor(
            _materializer_for(plan, operator_times=context),
            device="cuda:0",
        ).execute()
        for correlator_id in ("C0", "C1"):
            torch.testing.assert_close(
                batched.outputs_at(batch_index)[correlator_id],
                sequential.output(correlator_id),
                rtol=1e-12,
                atol=1e-10,
            )


def test_runtime_buffer_pool_scales_capacity_and_shapes_for_batch_axis():
    plan = build_physical_execution_plan(_schedule(mode_size=2))
    pool = RuntimeBufferPool(plan, device="cpu", batch_size=3)
    assert pool.active_capacity_bytes == 0

    for buffer_id, slot in plan.buffers.items():
        assert pool.storage[buffer_id].numel() * pool.storage[buffer_id].element_size() == (
            3 * slot.capacity_bytes
        )
    primitive = next(
        step.operation.value_id
        for step in plan.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
    )
    binding = plan.binding(primitive)
    full_view = pool.reserve(primitive)
    full_pointer = full_view.data_ptr()
    assert full_view.shape == torch.Size((3, *binding.shape))
    assert pool.active_capacity_bytes == 3 * plan.buffer(binding.buffer_id).capacity_bytes
    other = next(
        item for item in plan.value_bindings.values()
        if item.buffer_id != binding.buffer_id
    )
    pool.reserve(other.value_id)
    assert pool.active_capacity_bytes == 3 * (
        plan.buffer(binding.buffer_id).capacity_bytes
        + plan.buffer(other.buffer_id).capacity_bytes
    )
    pool.release(other.value_id)
    assert pool.active_capacity_bytes == 3 * plan.buffer(binding.buffer_id).capacity_bytes

    pool.reset(batch_size=1)
    assert pool.active_capacity_bytes == 0
    short_view = pool.reserve(primitive)
    assert short_view.shape == torch.Size(binding.shape)
    assert short_view.data_ptr() == full_pointer
    assert pool.active_capacity_bytes == plan.buffer(binding.buffer_id).capacity_bytes
    pool.release(primitive)
    assert pool.active_capacity_bytes == 0

    with pytest.raises(ValueError, match="positive"):
        RuntimeBufferPool(plan, batch_size=0)


def test_resident_session_reuses_one_pool_for_full_and_short_batches():
    plan = build_physical_execution_plan(
        _schedule(("C0", "C1"), mode_size=2),
        reuse_strategy="best_fit",
    )
    contexts = (
        {2: (3, 1)},
        {2: (4, 1)},
        {2: (5, 2)},
    )
    materializers = tuple(
        _materializer_for(plan, operator_times=context)
        for context in contexts
    )

    session = ResidentExecutionSession(
        plan,
        device="cpu",
        maximum_batch_size=2,
    )
    storage_pointers = {
        buffer_id: tensor.data_ptr()
        for buffer_id, tensor in session.pool.storage.items()
    }

    first = session.execute(materializers[:2])
    observed = [
        {
            correlator_id: first.outputs_at(batch_index)[correlator_id].clone()
            for correlator_id in ("C0", "C1")
        }
        for batch_index in range(2)
    ]
    with pytest.raises(RuntimeError, match="consumed and released"):
        session.execute(materializers[2:])
    session.release()

    final = session.execute(materializers[2:])
    observed.append(
        {
            correlator_id: final.output(correlator_id).clone()
            for correlator_id in ("C0", "C1")
        }
    )
    session.release()

    assert {
        buffer_id: tensor.data_ptr()
        for buffer_id, tensor in session.pool.storage.items()
    } == storage_pointers
    statistics = session.statistics
    assert statistics.pool_allocations == 1
    assert statistics.executions == 2
    assert statistics.released_executions == 2
    assert statistics.maximum_batch_size == 2
    assert statistics.pool_allocation_bytes == sum(
        2 * slot.capacity_bytes for slot in plan.buffers.values()
    )
    session.close()

    for work_index, context in enumerate(contexts):
        expected = GenericNumericalExecutor(
            _materializer_for(plan, operator_times=context),
            device="cpu",
        ).execute()
        for correlator_id in ("C0", "C1"):
            torch.testing.assert_close(
                observed[work_index][correlator_id],
                expected.output(correlator_id),
                rtol=1e-12,
                atol=1e-10,
            )


def test_host_staged_index_tiles_match_resident_execution():
    schedule = _schedule(("C0", "C1"), mode_size=5)
    plan = build_physical_execution_plan(schedule, reuse_strategy="best_fit")
    full = plan_tiled_execution(
        plan,
        device_memory_limit_bytes=10**18,
    )
    assert full.uses_index_tiling is False

    # One byte below the largest full-operation footprint forces the planner
    # to tile at least one LapH-mode index while leaving the physical DAG and
    # complete host buffer pool unchanged.
    tiled = plan_tiled_execution(
        plan,
        device_memory_limit_bytes=full.full_peak_device_bytes - 1,
    )
    assert tiled.uses_index_tiling
    assert tiled.peak_device_bytes <= tiled.device_memory_limit_bytes
    assert tiled.total_contraction_calls > len(tiled.operations)
    assert all(
        operation.space_by_index[index] == "mode"
        for operation in tiled.operations.values()
        for index in operation.tiled_indices
    )

    resident_result = _executor_for(plan).execute()
    tiled_result = _executor_for(plan, tiled_plan=tiled).execute()

    assert tiled_result.device.type == "cpu"
    assert tiled_result.compute_device.type == "cpu"
    for correlator_id in ("C0", "C1"):
        torch.testing.assert_close(
            tiled_result.output(correlator_id),
            resident_result.output(correlator_id),
            rtol=1e-12,
            atol=1e-10,
        )


def test_tiling_reports_when_even_unit_mode_tiles_exceed_the_limit():
    schedule = _schedule(("C0",), mode_size=2)
    plan = build_physical_execution_plan(schedule)

    with pytest.raises(TilingPlanningError, match="safe limit|tile budget"):
        plan_tiled_execution(plan, device_memory_limit_bytes=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_host_staged_cuda_tiles_return_cpu_resident_outputs():
    schedule = _schedule(("C0",), mode_size=5)
    plan = build_physical_execution_plan(schedule, reuse_strategy="best_fit")
    full = plan_tiled_execution(
        plan,
        device_memory_limit_bytes=10**18,
    )
    tiled = plan_tiled_execution(
        plan,
        device_memory_limit_bytes=full.full_peak_device_bytes - 1,
    )

    expected = _executor_for(plan).execute().output("C0")
    actual_result = _executor_for(
        plan,
        tiled_plan=tiled,
        compute_device="cuda:0",
    ).execute()

    assert actual_result.device.type == "cpu"
    assert actual_result.compute_device.type == "cuda"
    torch.testing.assert_close(
        actual_result.output("C0"),
        expected,
        rtol=1e-12,
        atol=1e-10,
    )


def test_runtime_buffer_pool_reuses_the_same_storage_only_after_release():
    schedule = _schedule(("C0", "C1"), mode_size=2)
    plan = build_physical_execution_plan(schedule, reuse_strategy="best_fit")
    pool = RuntimeBufferPool(plan)

    by_buffer = {}
    for binding in plan.value_bindings.values():
        by_buffer.setdefault(binding.buffer_id, []).append(binding)
    reused_bindings = next(
        sorted(items, key=lambda item: item.produced_at)
        for items in by_buffer.values()
        if len(items) > 1
    )
    first, second = reused_bindings[:2]

    first_value = torch.full(first.shape, 2 + 0j, dtype=torch.complex128)
    second_value = torch.full(second.shape, 7 + 0j, dtype=torch.complex128)

    first_view = pool.write(first.value_id, first_value)
    first_ptr = first_view.data_ptr()
    assert pool.active_capacity_bytes == plan.buffer(first.buffer_id).capacity_bytes
    with pytest.raises(RuntimeError, match="overwrite active buffer"):
        pool.write(second.value_id, second_value)
    assert pool.active_capacity_bytes == plan.buffer(first.buffer_id).capacity_bytes

    pool.release(first.value_id)
    assert pool.active_capacity_bytes == 0
    second_view = pool.write(second.value_id, second_value)
    second_ptr = second_view.data_ptr()
    assert pool.active_capacity_bytes == plan.buffer(second.buffer_id).capacity_bytes

    assert first_ptr == second_ptr
    assert torch.all(second_view == 7)


def test_executor_rejects_static_scalar_byte_model_incompatible_with_torch_dtype():
    schedule = _schedule(
        ("C00",),
        mode_size=2,
        scalar_type="complex128",
        element_bytes=8,
    )
    plan = build_physical_execution_plan(schedule)
    sources = _propagation_sources(plan)
    binding = build_resource_binding_plan(
        plan,
        operator_times={2: (3, 1)},
        propagation_sources=sources,
    )
    materializer = PrimitiveMaterializer(
        binding,
        kernel_providers={"distillation": _DeterministicKernelProvider()},
        source_handlers={
            "memory": InMemoryTensorSourceHandler(
                {source.location: _propagation_loader for source in sources.values()}
            )
        },
    )

    with pytest.raises(ValueError, match="runtime dtype"):
        GenericNumericalExecutor(materializer).execute()
