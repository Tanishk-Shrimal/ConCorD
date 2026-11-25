"""Focused tests for request-scoped primitive tensor materialization."""

from __future__ import annotations

from contextlib import contextmanager

import h5py
import numpy as np
import pytest
import torch

import concord_lqcd.backend.runtime.materialization as runtime_materialization
from concord_lqcd.backend.planner import build_physical_execution_plan
from concord_lqcd.backend.planner.physical import MaterializePrimitiveOperation
from concord_lqcd.backend.runtime.materialization import (
    CompiledKernelMaterializationProvider,
    HDF5FilePool,
    HDF5TensorSourceHandler,
    InMemoryTensorSourceHandler,
    KernelBatchMaterialization,
    KernelMaterializationProvider,
    PrimitiveMaterializer,
    PrimitiveTensorCache,
    PropagationSourceHandler,
    materialize_primitive_batch,
    plan_primitive_tensor_cache,
)
from concord_lqcd.backend.runtime.hardware import (
    HardwareDeviceProfile,
    HardwareProfile,
    detect_hardware_profile,
)
from concord_lqcd.backend.runtime.prefetch import (
    PropagationPrefetchSession,
    plan_propagation_prefetch,
)
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider
from concord_lqcd.backend.runtime.resource_planning import resolve_runtime_device
from concord_lqcd.backend.runtime.resource_binding import (
    ExternalTensorSource,
    KernelMaterializationRequest,
    PropagationMaterializationRequest,
    SpinMaterializationRequest,
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
from concord_lqcd.core import (
    Correlator,
    Operator,
    c,
    cbar,
    gamma5,
    gammaX,
    point,
    reset_registry,
    s,
    sbar,
    u,
    ubar,
)
from concord_lqcd.core.dirac_basis import get_spin_matrices
from concord_lqcd.core.wick import wick_contract


def _physical_plan(
    correlators,
    *,
    mode_size=3,
    scalar_type="complex128",
    element_bytes=16,
):
    physics = {
        key: lower_to_physics_ir(correlator, wick_contract(correlator))
        for key, correlator in correlators.items()
    }
    global_dag = lower_physics_ir_collection_to_global_cse(physics)
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
    schedule = build_logical_schedule(analysis, strategy="memory_aware")
    return build_physical_execution_plan(schedule)


def _two_point():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * u("y", "a", "s2")
    )
    source = Operator(
        ubar("x", "b", "s3") * gamma5("s3", "s4") * c("x", "b", "s4")
    )
    return Correlator(sink, source)


def _three_point():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * s("y", "a", "s2"),
        momentum=(0, 0, 0),
    )
    current = Operator(
        point(sbar("z", "b", "s3"))
        * gammaX("s3", "s4")
        * point(u("z", "b", "s4")),
        momentum=(1, 0, 0),
    )
    source = Operator(
        ubar("x", "c", "s5")
        * gamma5("s5", "s6")
        * c("x", "c", "s6"),
        momentum=(-1, 0, 0),
    )
    return Correlator(sink, current, source)


def _primitive_operations(plan, kind=None):
    operations = [
        step.operation
        for step in plan.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
    ]
    if kind is not None:
        operations = [operation for operation in operations if operation.kind == kind]
    return operations


def _propagation_sources(plan, *, kind="memory", location=None, dataset=None, metadata=None):
    return {
        operation.resource_id: ExternalTensorSource(
            source_id=f"source:{operation.resource_id}",
            kind=kind,
            location=(location(operation.resource_id) if callable(location) else location)
            or operation.resource_id,
            dataset=dataset,
            metadata=metadata or {},
        )
        for operation in _primitive_operations(plan, "propagation")
    }


def _binding_plan(
    physical,
    *,
    operator_times,
    propagation_sources=None,
):
    sources = propagation_sources or _propagation_sources(physical)
    return build_resource_binding_plan(
        physical,
        operator_times=operator_times,
        propagation_sources=sources,
    )


def _hdf5_prefetch_materializers(tmp_path, *, count=2):
    physical = _physical_plan({"C00": _two_point()}, mode_size=2)
    path = tmp_path / "prefetch_perambulator.h5"
    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(path),
        dataset="tau",
    )
    bindings = tuple(
        _binding_plan(
            physical,
            operator_times={2: (7 + index, 1)},
            propagation_sources=sources,
        )
        for index in range(count)
    )
    requests = [
        request
        for request in bindings[0].materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
    ]
    assert requests
    assert len({request.expected_shape for request in requests}) == 1
    shape = requests[0].expected_shape
    raw = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tau", data=(raw + 1j * raw / 13).astype(np.complex128))
    materializers = tuple(PrimitiveMaterializer(binding) for binding in bindings)
    return physical, materializers


def _fixed_source_batch_materializers(
    physical,
    filepath,
    *,
    fixed_source_time,
    other_times,
    reverse,
    handler,
):
    materializers = []
    requests = []
    for other_time in other_times:
        sources = _propagation_sources(
            physical,
            kind="hdf5",
            location=str(filepath),
            dataset="perambulators/src_t_{fixed_source_time}/data",
            metadata={
                "fixed_source_time": fixed_source_time,
                "axes": (
                    "sink_time",
                    "source_spin",
                    "sink_spin",
                    "source_mode",
                    "sink_mode",
                ),
            },
        )
        operator_times = (
            {2: (fixed_source_time, other_time)}
            if reverse
            else {2: (other_time, fixed_source_time)}
        )
        binding = _binding_plan(
            physical,
            operator_times=operator_times,
            propagation_sources=sources,
        )
        request = next(
            request
            for request in binding.materializations.values()
            if isinstance(request, PropagationMaterializationRequest)
            and (
                request.sink_time == fixed_source_time
                if reverse
                else request.source_time == fixed_source_time
            )
            and (
                request.source_time == other_time
                if reverse
                else request.sink_time == other_time
            )
        )
        materializers.append(
            PrimitiveMaterializer(
                binding,
                source_handlers={"hdf5": handler},
            )
        )
        requests.append(request)
    return tuple(materializers), tuple(requests)


def _large_cpu_execution_plan(physical):
    gib = 1024**3
    hardware = HardwareProfile(
        host=HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name="Prefetch Test CPU",
            total_memory_bytes=8 * gib,
            available_memory_bytes=6 * gib,
            logical_processors=4,
        ),
        system="TestOS",
        machine="prefetch-test-machine",
    )
    return resolve_runtime_device(physical, "cpu", hardware=hardware)


class _FunctionKernelProvider(KernelMaterializationProvider):
    def __init__(self, function):
        self.function = function

    def materialize(self, request):
        return self.function(request)


class _CountingCacheableKernelProvider(KernelMaterializationProvider):
    def __init__(self, calls):
        self.calls = calls

    def cache_identity(self):
        return ("counting-test-kernel", 1)

    def materialize(self, request):
        self.calls.append((request.resource_id, request.operator_times))
        value = 1 + sum(request.operator_times)
        return torch.full(
            request.expected_shape,
            value,
            dtype=torch.complex128,
        )


class _BatchingKernelProvider(KernelMaterializationProvider):
    def __init__(self):
        self.independent_calls = []
        self.batch_calls = []

    @staticmethod
    def _tensor(request):
        return torch.full(
            request.expected_shape,
            1 + sum(request.operator_times),
            dtype=torch.complex128,
        )

    def materialize(self, request):
        self.independent_calls.append(request.operator_times)
        return self._tensor(request)

    def cache_identity(self):
        return ("test-batching-kernel", 1)

    def materialize_batch_with_limit(self, requests, *, maximum_staging_bytes):
        values = tuple(requests)
        self.batch_calls.append(
            (tuple(request.operator_times for request in values), maximum_staging_bytes)
        )
        return KernelBatchMaterialization(
            tensor=torch.stack(tuple(self._tensor(request) for request in values)),
            mode="test_kernel_batch",
        )


class _UnboundedBatchingKernelProvider(_BatchingKernelProvider):
    def materialize_batch(self, requests):
        values = tuple(requests)
        self.batch_calls.append(
            (tuple(request.operator_times for request in values), None)
        )
        return KernelBatchMaterialization(
            tensor=torch.stack(tuple(self._tensor(request) for request in values)),
            mode="test_unbounded_kernel_batch",
        )

    def materialize_batch_with_limit(self, requests, *, maximum_staging_bytes):
        return KernelMaterializationProvider.materialize_batch_with_limit(
            self,
            requests,
            maximum_staging_bytes=maximum_staging_bytes,
        )


def _kernel_batch_materializers(physical, provider, *, limits=(4096, 4096)):
    value_id = _primitive_operations(physical, "kernel")[0].value_id
    bindings = tuple(
        _binding_plan(physical, operator_times={2: (sink_time, 1)})
        for sink_time in (7, 8)
    )
    registry = {
        request.source.location: (
            lambda request: torch.zeros(
                request.expected_shape,
                dtype=torch.complex128,
            )
        )
        for request in bindings[0].materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
    }
    source_handler = InMemoryTensorSourceHandler(registry)
    materializers = tuple(
        PrimitiveMaterializer(
            binding,
            kernel_providers={"distillation": provider},
            source_handlers={"memory": source_handler},
            kernel_batch_staging_limit_bytes=limit,
        )
        for binding, limit in zip(bindings, limits, strict=True)
    )
    return value_id, materializers


def test_kernel_provider_can_opt_into_exact_batched_materialization():
    physical = _physical_plan({"C00": _two_point()}, mode_size=2)
    provider = _BatchingKernelProvider()
    value_id, materializers = _kernel_batch_materializers(physical, provider)

    result = materialize_primitive_batch(materializers, value_id)

    assert result is not None
    assert result.mode == "test_kernel_batch"
    assert result.tensor.shape[0] == 2
    assert provider.independent_calls == []
    assert provider.batch_calls == [(((7,), (8,)), 4096)]
    assert torch.all(result.tensor[0] == 8)
    assert torch.all(result.tensor[1] == 9)


def test_kernel_batch_fills_and_reuses_one_shared_exact_cache():
    physical = _physical_plan({"C00": _two_point()}, mode_size=2)
    provider = _BatchingKernelProvider()
    value_id, uncached = _kernel_batch_materializers(physical, provider)
    cache_plan = plan_primitive_tensor_cache(
        (*uncached, *uncached),
        max_bytes=1024**2,
    )
    cache = PrimitiveTensorCache(cache_plan, device="cpu")
    materializers = tuple(
        PrimitiveMaterializer(
            materializer.binding_plan,
            kernel_providers={"distillation": provider},
            source_handlers=materializer.source_handlers,
            tensor_cache=cache,
            kernel_batch_staging_limit_bytes=4096,
        )
        for materializer in uncached
    )

    first = materialize_primitive_batch(materializers, value_id)
    second = materialize_primitive_batch(materializers, value_id)

    assert first is not None
    assert first.mode == "test_kernel_batch"
    assert second is not None
    assert second.mode == "primitive_cache_stack"
    torch.testing.assert_close(first.tensor, second.tensor, rtol=0.0, atol=0.0)
    assert len(provider.batch_calls) == 1
    assert cache.statistics.stores == 2
    assert cache.statistics.hits == 2


def test_kernel_batch_preserves_independent_fallback_without_provider_opt_in():
    physical = _physical_plan({"C00": _two_point()}, mode_size=2)
    provider = _FunctionKernelProvider(
        lambda request: torch.zeros(request.expected_shape, dtype=torch.complex128)
    )
    value_id, materializers = _kernel_batch_materializers(physical, provider)

    assert materialize_primitive_batch(materializers, value_id) is None


def test_kernel_batch_requires_explicit_bounded_staging_opt_in():
    physical = _physical_plan({"C00": _two_point()}, mode_size=2)
    provider = _UnboundedBatchingKernelProvider()
    value_id, bounded_materializers = _kernel_batch_materializers(
        physical,
        provider,
    )

    assert materialize_primitive_batch(bounded_materializers, value_id) is None
    assert provider.batch_calls == []

    _, unbounded_materializers = _kernel_batch_materializers(
        physical,
        provider,
        limits=(None, None),
    )
    result = materialize_primitive_batch(unbounded_materializers, value_id)

    assert result is not None
    assert result.mode == "test_unbounded_kernel_batch"
    assert provider.batch_calls == [(((7,), (8,)), None)]


def test_kernel_batch_requires_one_provider_instance_and_staging_limit():
    physical = _physical_plan({"C00": _two_point()}, mode_size=2)
    first = _BatchingKernelProvider()
    value_id, materializers = _kernel_batch_materializers(physical, first)
    second = _BatchingKernelProvider()
    distinct_provider_materializer = PrimitiveMaterializer(
        materializers[1].binding_plan,
        kernel_providers={"distillation": second},
        kernel_batch_staging_limit_bytes=4096,
    )
    assert materialize_primitive_batch(
        (materializers[0], distinct_provider_materializer),
        value_id,
    ) is None

    _, mismatched_limits = _kernel_batch_materializers(
        physical,
        first,
        limits=(4096, 2048),
    )
    with pytest.raises(ValueError, match="staging limit"):
        materialize_primitive_batch(mismatched_limits, value_id)


def test_exact_primitive_cache_reuses_only_opted_in_provider_requests():
    physical = _physical_plan({"C00": _two_point()}, mode_size=3)
    first_binding = _binding_plan(physical, operator_times={2: (7, 1)})
    repeated_binding = _binding_plan(physical, operator_times={2: (7, 1)})
    different_binding = _binding_plan(physical, operator_times={2: (8, 1)})
    request = next(
        request
        for request in first_binding.materializations.values()
        if isinstance(request, KernelMaterializationRequest)
        and 7 in request.operator_times
    )
    calls = []
    provider = _CountingCacheableKernelProvider(calls)

    def materializer(binding, *, tensor_cache=None):
        propagation_registry = {
            actual.source.location: (
                lambda request: torch.zeros(
                    request.expected_shape,
                    dtype=torch.complex128,
                )
            )
            for actual in binding.materializations.values()
            if isinstance(actual, PropagationMaterializationRequest)
        }
        return PrimitiveMaterializer(
            binding,
            kernel_providers={request.provider_id: provider},
            source_handlers={
                "memory": InMemoryTensorSourceHandler(propagation_registry)
            },
            tensor_cache=tensor_cache,
        )

    candidates = tuple(
        materializer(binding)
        for binding in (first_binding, repeated_binding, different_binding)
    )
    disabled_plan = plan_primitive_tensor_cache(candidates, max_bytes=0)
    assert disabled_plan.entries == ()
    assert disabled_plan.repeated_keys > 0
    assert "no safe cache headroom" in disabled_plan.reason
    cache_plan = plan_primitive_tensor_cache(candidates, max_bytes=10**9)
    assert any(entry.key[0] == "kernel-v1" for entry in cache_plan.entries)
    assert "fit the cache byte budget" in cache_plan.reason
    cache = PrimitiveTensorCache(cache_plan, device="cpu")

    first = materializer(first_binding, tensor_cache=cache).materialize(
        request.value_id
    )
    repeated = materializer(repeated_binding, tensor_cache=cache).materialize(
        request.value_id
    )
    different = materializer(different_binding, tensor_cache=cache).materialize(
        request.value_id
    )

    assert first.tensor.data_ptr() == repeated.tensor.data_ptr()
    assert first.tensor.data_ptr() != different.tensor.data_ptr()
    assert len(calls) == 2
    assert cache.statistics.hits == 1
    assert cache.statistics.misses == 1
    assert cache.statistics.stores == 1

    uncacheable = PrimitiveMaterializer(
        first_binding,
        kernel_providers={
            request.provider_id: _FunctionKernelProvider(
                lambda actual: torch.zeros(
                    actual.expected_shape,
                    dtype=torch.complex128,
                )
            )
        },
    )
    assert uncacheable.materialization_cache_key(request.value_id) is None


def test_hdf5_file_pool_reuses_and_closes_read_only_handles(tmp_path):
    path = tmp_path / "pooled.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("value", data=np.arange(4))

    pool = HDF5FilePool()
    with pool:
        with pool.file(path) as handle:
            assert tuple(handle["value"][...]) == (0, 1, 2, 3)
        with pool.file(path) as handle:
            assert handle["value"].shape == (4,)
        assert pool.statistics.file_opens == 1
        assert pool.statistics.file_reuses == 1
        assert pool.statistics.open_files == 1

    assert pool.statistics.open_files == 0


def test_cpu_propagation_prefetch_stages_and_consumes_exact_hdf5_values(tmp_path):
    physical, materializers = _hdf5_prefetch_materializers(tmp_path)
    execution_plan = _large_cpu_execution_plan(physical)
    plan = plan_propagation_prefetch(
        materializers,
        batch_size=1,
        execution_plan=execution_plan,
        available_device_headroom_bytes=1024**3,
    )

    assert plan.enabled
    assert plan.batch_count == 2
    assert plan.maximum_batch_requests > 0
    assert plan.staged_tensor_bytes > 0
    assert plan.host_staging_bytes == plan.device_staging_bytes
    assert plan.device_staging_bytes == 2 * plan.staged_tensor_bytes

    with PropagationPrefetchSession(plan, device="cpu") as session:
        for materializer in materializers:
            assert session.submit((materializer,))
            assert session.resolve()
            eligible = materializer.prefetchable_hdf5_propagation_values()
            assert set(materializer.pending_prefetched_values) == set(eligible)
            for value_id in eligible:
                value = materializer.materialize(value_id)
                assert value.tensor.device.type == "cpu"
            assert materializer.pending_prefetched_values == ()
        statistics = session.statistics

    assert statistics.submitted_batches == 2
    assert statistics.completed_batches == 2
    assert statistics.staged_tensors == plan.eligible_requests
    assert statistics.staged_bytes > 0
    assert statistics.wait_seconds >= 0
    assert statistics.allocation_fallbacks == 0


def test_cpu_prefetch_uses_shared_dense_hdf5_batches(tmp_path, monkeypatch):
    nv = 2
    fixed_source_time = 1
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "prefetch_dense_batch.h5"
    raw = np.arange(8 * 4 * 4 * nv * nv, dtype=np.float64).reshape(
        8, 4, 4, nv, nv
    )
    with h5py.File(filepath, "w") as handle:
        handle.create_dataset(
            f"perambulators/src_t_{fixed_source_time}/data",
            data=(raw + 1j * raw / 11).astype(np.complex128),
            chunks=(1, 4, 4, nv, nv),
        )

    handler = HDF5TensorSourceHandler()
    reads = []
    original_read = handler._read_selection

    def counted_read(selection, *, indexers, value_id):
        reads.append((value_id, indexers))
        return original_read(selection, indexers=indexers, value_id=value_id)

    monkeypatch.setattr(handler, "_read_selection", counted_read)
    materializers, _ = _fixed_source_batch_materializers(
        physical,
        filepath,
        fixed_source_time=fixed_source_time,
        other_times=(2, 3, 4, 5),
        reverse=False,
        handler=handler,
    )
    execution_plan = _large_cpu_execution_plan(physical)
    plan = plan_propagation_prefetch(
        materializers,
        batch_size=2,
        execution_plan=execution_plan,
        available_device_headroom_bytes=4 * 1024**3,
    )
    assert plan.enabled

    eligible = materializers[0].prefetchable_hdf5_propagation_values()
    with PropagationPrefetchSession(plan, device="cpu") as session:
        assert session.submit(materializers[:2])
        assert session.resolve()

    assert len(reads) == len(eligible)
    assert all(indexers is not None for _, indexers in reads)
    assert all(indexers[0] == slice(2, 4) for _, indexers in reads)
    assert all(
        set(materializer.pending_prefetched_values) == set(eligible)
        for materializer in materializers[:2]
    )


def test_propagation_prefetch_requires_multiple_batches_and_exact_headroom(tmp_path):
    physical, materializers = _hdf5_prefetch_materializers(tmp_path)
    execution_plan = _large_cpu_execution_plan(physical)

    one_batch = plan_propagation_prefetch(
        materializers,
        batch_size=len(materializers),
        execution_plan=execution_plan,
        available_device_headroom_bytes=1024**3,
    )
    assert not one_batch.enabled
    assert one_batch.reason == "one resident batch is sufficient"

    candidate = plan_propagation_prefetch(
        materializers,
        batch_size=1,
        execution_plan=execution_plan,
        available_device_headroom_bytes=1024**3,
    )
    insufficient = plan_propagation_prefetch(
        materializers,
        batch_size=1,
        execution_plan=execution_plan,
        available_device_headroom_bytes=candidate.device_staging_bytes - 1,
    )
    assert not insufficient.enabled
    assert "headroom" in insufficient.reason


def test_generalized_hdf5_propagation_remains_outside_prefetch_boundary(tmp_path):
    physical = _physical_plan({"C3": _three_point()}, mode_size=2)
    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(tmp_path / "not_loaded_during_planning.h5"),
        dataset="tau",
    )
    binding = _binding_plan(
        physical,
        operator_times={3: (8, 5, 1)},
        propagation_sources=sources,
    )
    materializer = PrimitiveMaterializer(binding)
    generalized = {
        value_id
        for value_id, request in binding.materializations.items()
        if isinstance(request, PropagationMaterializationRequest)
        and request.insertion_times
    }

    assert generalized
    assert generalized.isdisjoint(
        materializer.prefetchable_hdf5_propagation_values()
    )


def test_prefetch_allocation_failure_preserves_synchronous_materialization(
    tmp_path,
    monkeypatch,
):
    physical, materializers = _hdf5_prefetch_materializers(tmp_path)
    execution_plan = _large_cpu_execution_plan(physical)
    plan = plan_propagation_prefetch(
        materializers,
        batch_size=1,
        execution_plan=execution_plan,
        available_device_headroom_bytes=1024**3,
    )
    materializer = materializers[0]
    eligible = materializer.prefetchable_hdf5_propagation_values()

    def fail_prefetch(value_id):
        raise MemoryError(f"synthetic allocation failure for {value_id}")

    monkeypatch.setattr(
        materializer,
        "materialize_hdf5_propagation_for_prefetch",
        fail_prefetch,
    )
    with PropagationPrefetchSession(plan, device="cpu") as session:
        assert session.submit((materializer,))
        assert not session.resolve()
        assert not session.submit((materializers[1],))
        statistics = session.statistics

    assert statistics.allocation_fallbacks == 1
    assert statistics.completed_batches == 0
    # The optional pipeline has not consumed or changed the synchronous source.
    assert materializer.materialize(eligible[0]).tensor.device.type == "cpu"


def test_prefetch_invalid_hdf5_data_propagates_instead_of_falling_back(tmp_path):
    physical, materializers = _hdf5_prefetch_materializers(tmp_path)
    execution_plan = _large_cpu_execution_plan(physical)
    plan = plan_propagation_prefetch(
        materializers,
        batch_size=1,
        execution_plan=execution_plan,
        available_device_headroom_bytes=1024**3,
    )
    materializer = materializers[0]
    request = materializer.binding_plan.materialization(
        materializer.prefetchable_hdf5_propagation_values()[0]
    )
    assert isinstance(request, PropagationMaterializationRequest)
    with h5py.File(request.source.location, "w") as handle:
        handle.create_dataset("tau", data=np.zeros((1,), dtype=np.complex128))

    with PropagationPrefetchSession(plan, device="cpu") as session:
        assert session.submit((materializer,))
        with pytest.raises(ValueError, match="shape"):
            session.resolve()

    assert session.statistics.allocation_fallbacks == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_propagation_prefetch_uses_device_resident_staging(tmp_path):
    physical, materializers = _hdf5_prefetch_materializers(tmp_path)
    execution_plan = resolve_runtime_device(
        physical,
        "cuda:0",
        hardware=detect_hardware_profile(),
    )
    headroom = execution_plan.selected_budget.headroom_after_known_plan_bytes
    assert headroom is not None
    plan = plan_propagation_prefetch(
        materializers,
        batch_size=1,
        execution_plan=execution_plan,
        available_device_headroom_bytes=headroom,
    )
    assert plan.enabled
    assert plan.host_staging_bytes == (
        plan.staged_tensor_bytes + plan.maximum_tensor_bytes
    )
    assert plan.device_staging_bytes == 2 * plan.staged_tensor_bytes

    materializer = materializers[0]
    with PropagationPrefetchSession(plan, device="cuda:0") as session:
        assert session.submit((materializer,))
        assert session.resolve()
        for value_id in materializer.prefetchable_hdf5_propagation_values():
            value = materializer.materialize(value_id)
            assert value.tensor.device.type == "cuda"
    torch.cuda.synchronize()


def test_spin_materialization_obeys_planned_complex64_dtype_and_buffer_contract():
    physical = _physical_plan(
        {"C00": _two_point()},
        mode_size=3,
        scalar_type="complex64",
        element_bytes=8,
    )
    binding = _binding_plan(physical, operator_times={2: (7, 1)})
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, SpinMaterializationRequest)
    )

    value = PrimitiveMaterializer(binding).materialize(request.value_id)

    assert value.value_id == request.value_id
    assert value.resource_id == request.resource_id
    assert value.buffer_id == request.output_buffer_id
    assert value.tensor.shape == torch.Size(request.expected_shape)
    assert value.tensor.dtype == torch.complex64
    assert value.tensor.numel() * value.tensor.element_size() == physical.binding(
        request.value_id
    ).size_bytes


@pytest.mark.parametrize("spin_basis", ["DR", "DP"])
@pytest.mark.parametrize(
    "scalar_type, element_bytes", [("complex64", 8), ("complex128", 16)]
)
def test_runtime_spin_tensor_edits_do_not_change_shared_constants(
    spin_basis, scalar_type, element_bytes
):
    physical = _physical_plan(
        {"C00": _two_point()},
        scalar_type=scalar_type,
        element_bytes=element_bytes,
    )
    binding = build_resource_binding_plan(
        physical,
        operator_times={2: (7, 1)},
        propagation_sources=_propagation_sources(physical),
        spin_basis=spin_basis,
    )
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, SpinMaterializationRequest)
    )
    expected = get_spin_matrices(spin_basis)[request.tensor_kind].copy()
    materializer = PrimitiveMaterializer(binding)
    value = materializer.materialize(request.value_id)
    original_tensor = value.tensor.clone()

    value.tensor.add_(7.0)

    np.testing.assert_array_equal(
        get_spin_matrices(spin_basis)[request.tensor_kind], expected
    )
    np.testing.assert_array_equal(np.asarray(request.matrix), expected)
    assert not torch.equal(value.tensor, original_tensor)
    torch.testing.assert_close(
        PrimitiveMaterializer(binding).materialize(request.value_id).tensor,
        original_tensor,
        rtol=0,
        atol=0,
    )


def test_compiled_kernel_provider_reuses_existing_stage3_kernel_isa():
    physical = _physical_plan({"C00": _two_point()}, mode_size=3)
    binding = _binding_plan(physical, operator_times={2: (9, 2)})
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, KernelMaterializationRequest)
    )

    nv = request.expected_shape[0]
    raw = torch.arange(1 * 2 * 3 * nv, dtype=torch.float64).reshape(1, 2, 3, nv)
    V = raw.to(torch.complex128) + 0.25j

    seen = []

    def provider_factory(actual_request):
        seen.append(actual_request.value_id)
        fields = {"u": V, "c": V, "s": V, "d": V}
        return DistillationTensorProvider(fields, fields, {})

    provider = CompiledKernelMaterializationProvider(provider_factory)
    materializer = PrimitiveMaterializer(
        binding,
        kernel_providers={request.provider_id: provider},
    )

    value = materializer.materialize(request.value_id)

    assert seen == [request.value_id]
    assert value.tensor.shape == torch.Size(request.expected_shape)
    assert value.tensor.dtype == torch.complex128


def test_kernels_and_only_kernels_are_built_with_the_kernel_building_block_limit(
    monkeypatch,
):
    building = []

    @contextmanager
    def recorded_kernel_building():
        building.append(True)
        try:
            yield
        finally:
            building.pop()

    monkeypatch.setattr(runtime_materialization, "kernel_building", recorded_kernel_building)
    physical = _physical_plan({"C00": _two_point()}, mode_size=2)
    seen = []

    class Provider(_BatchingKernelProvider):
        def materialize(self, request):
            seen.append(("kernel", bool(building)))
            return super().materialize(request)

        def materialize_batch_with_limit(self, requests, *, maximum_staging_bytes):
            seen.append(("kernel batch", bool(building)))
            return super().materialize_batch_with_limit(
                requests, maximum_staging_bytes=maximum_staging_bytes
            )

    value_id, materializers = _kernel_batch_materializers(physical, Provider())
    propagation_id = next(
        request.value_id
        for request in materializers[0].binding_plan.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
    )
    source_handler = materializers[0].source_handlers["memory"]
    read = source_handler.materialize

    def recorded_read(request):
        seen.append(("propagation", bool(building)))
        return read(request)

    monkeypatch.setattr(source_handler, "materialize", recorded_read)

    materializers[0].materialize(value_id)
    assert materialize_primitive_batch(materializers, value_id) is not None
    materializers[0].materialize(propagation_id)

    assert seen == [("kernel", True), ("kernel batch", True), ("propagation", False)]
    assert building == []


def test_kernel_provider_wrong_shape_is_rejected_before_buffer_write():
    physical = _physical_plan({"C00": _two_point()}, mode_size=3)
    binding = _binding_plan(physical, operator_times={2: (9, 2)})
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, KernelMaterializationRequest)
    )
    provider = _FunctionKernelProvider(
        lambda req: torch.zeros((*req.expected_shape, 1), dtype=torch.complex128)
    )

    with pytest.raises(ValueError, match="has shape"):
        PrimitiveMaterializer(
            binding,
            kernel_providers={request.provider_id: provider},
        ).materialize(request.value_id)


def test_kernel_provider_wrong_dtype_is_rejected_before_buffer_write():
    physical = _physical_plan({"C00": _two_point()}, mode_size=3)
    binding = _binding_plan(physical, operator_times={2: (9, 2)})
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, KernelMaterializationRequest)
    )
    provider = _FunctionKernelProvider(
        lambda req: torch.zeros(req.expected_shape, dtype=torch.complex64)
    )

    with pytest.raises(TypeError, match="has dtype"):
        PrimitiveMaterializer(
            binding,
            kernel_providers={request.provider_id: provider},
        ).materialize(request.value_id)


def test_in_memory_generalized_propagation_receives_exact_ordered_times():
    physical = _physical_plan({"C3": _three_point()}, mode_size=3)
    sources = _propagation_sources(physical, kind="memory")
    binding = _binding_plan(
        physical,
        operator_times={3: (15, 8, 2)},
        propagation_sources=sources,
    )
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
        and request.recipe.is_generalized
    )

    seen = []

    def load(actual_request):
        seen.append(
            (
                actual_request.source_time,
                actual_request.insertion_times,
                actual_request.sink_time,
            )
        )
        return torch.zeros(actual_request.expected_shape, dtype=torch.complex128)

    registry = {source.location: load for source in sources.values()}
    materializer = PrimitiveMaterializer(
        binding,
        source_handlers={"memory": InMemoryTensorSourceHandler(registry)},
    )
    value = materializer.materialize(request.value_id)

    assert seen == [(2, (8,), 15)]
    assert value.tensor.shape == torch.Size(request.expected_shape)
    assert value.tensor.dtype == torch.complex128


def test_hdf5_handler_supports_source_time_dataset_template_and_axis_permutation(tmp_path):
    nv = 3
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "ordinary.h5"

    raw = np.zeros((4, 4, 4, nv, nv), dtype=np.complex128)
    for t_sink in range(4):
        for sink_spin in range(4):
            for source_spin in range(4):
                for sink_mode in range(nv):
                    for source_mode in range(nv):
                        raw[t_sink, sink_spin, source_spin, sink_mode, source_mode] = (
                            10000 * t_sink
                            + 1000 * sink_spin
                            + 100 * source_spin
                            + 10 * sink_mode
                            + source_mode
                        )

    with h5py.File(filepath, "w") as handle:
        group = handle.create_group("perambulators")
        group.create_dataset("src_t_1", data=raw)

    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(filepath),
        dataset="perambulators/src_t_{source_time}",
        metadata={
            "axes": (
                "sink_time",
                "sink_spin",
                "source_spin",
                "sink_mode",
                "source_mode",
            )
        },
    )
    binding = _binding_plan(
        physical,
        operator_times={2: (3, 1)},
        propagation_sources=sources,
    )
    propagation_requests = [
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
    ]
    assert {
        (request.source_time, request.sink_time)
        for request in propagation_requests
    } == {(1, 3), (3, 1)}
    request = next(
        request
        for request in propagation_requests
        if request.source_time == 1 and request.sink_time == 3
    )

    value = PrimitiveMaterializer(binding).materialize(request.value_id)

    assert value.tensor.shape == (nv, 4, nv, 4)
    assert value.tensor[1, 2, 0, 3].item() == raw[3, 2, 3, 1, 0]


def test_hdf5_handler_indexes_generalized_insertion_time_axis(tmp_path):
    nv = 2
    physical = _physical_plan({"C3": _three_point()}, mode_size=nv)
    filepath = tmp_path / "generalized.h5"

    # Operator times are sink=2, insertion=1, source=0. Store all three time
    # coordinates densely, followed by the exact ConCorD propagation interface.
    raw = np.zeros((3, 2, 1, nv, 4, nv, 4), dtype=np.complex128)
    raw[2, 1, 0, 1, 2, 0, 3] = 42 + 5j
    with h5py.File(filepath, "w") as handle:
        handle.create_dataset("generalized", data=raw)

    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(filepath),
        dataset="generalized",
        metadata={
            "axes": (
                "sink_time",
                "insertion_time_0",
                "source_time",
                "sink_mode",
                "sink_spin",
                "source_mode",
                "source_spin",
            )
        },
    )
    binding = _binding_plan(
        physical,
        operator_times={3: (2, 1, 0)},
        propagation_sources=sources,
    )
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
        and request.recipe.is_generalized
    )

    value = PrimitiveMaterializer(binding).materialize(request.value_id)

    assert request.insertion_times == (1,)
    assert value.tensor[1, 2, 0, 3].item() == 42 + 5j


def test_hdf5_handler_reconstructs_fixed_source_reverse_after_canonicalization(tmp_path):
    nv = 3
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "fixed_source.h5"
    raw = np.arange(4 * 4 * 4 * nv * nv, dtype=np.float64).reshape(4, 4, 4, nv, nv)
    raw = raw + 1j * (raw + 0.5)

    with h5py.File(filepath, "w") as handle:
        handle.create_dataset("perambulators/src_t_1/data", data=raw)

    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(filepath),
        dataset="perambulators/src_t_{fixed_source_time}/data",
        metadata={
            "fixed_source_time": 1,
            "axes": (
                "sink_time",
                "sink_spin",
                "source_spin",
                "sink_mode",
                "source_mode",
            ),
        },
    )
    binding = _binding_plan(
        physical,
        operator_times={2: (3, 1)},
        propagation_sources=sources,
    )
    requests = [
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
    ]
    direct_request = next(request for request in requests if request.source_time == 1)
    reverse_request = next(request for request in requests if request.sink_time == 1)

    direct = PrimitiveMaterializer(binding).materialize(direct_request.value_id).tensor
    reverse = PrimitiveMaterializer(binding).materialize(reverse_request.value_id).tensor

    canonical = np.transpose(raw[3], (2, 0, 3, 1))
    gamma5 = np.diag(np.asarray([1, 1, -1, -1], dtype=np.complex128))
    matrix = canonical.reshape(4 * nv, 4 * nv)
    g5 = np.kron(np.eye(nv, dtype=np.complex128), gamma5)
    expected_reverse = (g5 @ matrix.conj().T @ g5).reshape(nv, 4, nv, 4)

    assert np.allclose(direct.numpy(), canonical)
    assert np.allclose(reverse.numpy(), expected_reverse)


def test_hdf5_handler_rejects_fixed_source_request_with_neither_endpoint(tmp_path):
    nv = 2
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "fixed_source_unavailable.h5"
    raw = np.zeros((4, 4, 4, nv, nv), dtype=np.complex128)
    with h5py.File(filepath, "w") as handle:
        handle.create_dataset("perambulators/src_t_1/data", data=raw)

    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(filepath),
        dataset="perambulators/src_t_{fixed_source_time}/data",
        metadata={
            "fixed_source_time": 1,
            "axes": (
                "sink_time",
                "sink_spin",
                "source_spin",
                "sink_mode",
                "source_mode",
            ),
        },
    )
    binding = _binding_plan(
        physical,
        operator_times={2: (3, 2)},
        propagation_sources=sources,
    )
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
    )

    with pytest.raises(ValueError, match="neither endpoint is fixed source time"):
        PrimitiveMaterializer(binding).materialize(request.value_id)


def test_hdf5_batch_reads_one_dense_slab_and_preserves_request_order(
    tmp_path,
    monkeypatch,
):
    nv = 2
    temporal_extent = 8
    fixed_source_time = 1
    requested_times = (4, 2, 4, 3)
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "fixed_source_batch.h5"
    raw = np.arange(
        temporal_extent * 4 * 4 * nv * nv,
        dtype=np.float64,
    ).reshape(temporal_extent, 4, 4, nv, nv)
    raw = raw + 1j * (raw / 7.0 + 0.25)
    with h5py.File(filepath, "w") as handle:
        handle.create_dataset(
            f"perambulators/src_t_{fixed_source_time}/data",
            data=raw,
            chunks=(1, 4, 4, nv, nv),
        )

    handler = HDF5TensorSourceHandler()
    reads = []
    original_read = handler._read_selection

    def counted_read(selection, *, indexers, value_id):
        reads.append(indexers)
        return original_read(selection, indexers=indexers, value_id=value_id)

    monkeypatch.setattr(handler, "_read_selection", counted_read)
    materializers, requests = _fixed_source_batch_materializers(
        physical,
        filepath,
        fixed_source_time=fixed_source_time,
        other_times=requested_times,
        reverse=False,
        handler=handler,
    )

    result = materialize_primitive_batch(materializers, requests[0].value_id)

    assert result is not None
    assert result.mode == "hdf5_dense_slab"
    expected = np.stack(
        [np.transpose(raw[time], (3, 1, 2, 0)) for time in requested_times]
    )
    assert np.array_equal(result.tensor.numpy(), expected)
    assert len(reads) == 1
    assert reads[0] is not None
    assert reads[0][0] == slice(2, 5)


def test_hdf5_batch_vectorizes_fixed_source_reverse_reconstruction(tmp_path):
    nv = 2
    fixed_source_time = 1
    requested_times = (4, 2, 3)
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "fixed_source_reverse_batch.h5"
    raw = np.arange(6 * 4 * 4 * nv * nv, dtype=np.float64).reshape(
        6, 4, 4, nv, nv
    )
    raw = raw + 1j * (raw + 0.5)
    with h5py.File(filepath, "w") as handle:
        handle.create_dataset(
            f"perambulators/src_t_{fixed_source_time}/data",
            data=raw,
        )

    handler = HDF5TensorSourceHandler()
    materializers, requests = _fixed_source_batch_materializers(
        physical,
        filepath,
        fixed_source_time=fixed_source_time,
        other_times=requested_times,
        reverse=True,
        handler=handler,
    )
    result = materialize_primitive_batch(materializers, requests[0].value_id)
    expected = torch.stack(
        tuple(HDF5TensorSourceHandler().materialize(request) for request in requests)
    )

    assert result is not None
    assert result.mode == "hdf5_dense_slab"
    torch.testing.assert_close(result.tensor, expected, rtol=0.0, atol=0.0)


def test_hdf5_batch_rejects_sparse_covering_slab(tmp_path):
    nv = 2
    fixed_source_time = 1
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "fixed_source_sparse_batch.h5"
    with h5py.File(filepath, "w") as handle:
        handle.create_dataset(
            f"perambulators/src_t_{fixed_source_time}/data",
            data=np.zeros((12, 4, 4, nv, nv), dtype=np.complex128),
        )

    handler = HDF5TensorSourceHandler()
    materializers, requests = _fixed_source_batch_materializers(
        physical,
        filepath,
        fixed_source_time=fixed_source_time,
        other_times=(2, 10),
        reverse=False,
        handler=handler,
    )

    assert materialize_primitive_batch(materializers, requests[0].value_id) is None


def test_custom_propagation_handler_retains_independent_fallback():
    class CountingHandler(PropagationSourceHandler):
        def __init__(self):
            self.calls = []

        def materialize(self, request):
            self.calls.append(request.sink_time)
            return torch.full(
                request.expected_shape,
                request.sink_time,
                dtype=torch.complex128,
            )

    physical = _physical_plan({"C00": _two_point()}, mode_size=2)
    handler = CountingHandler()
    materializers = []
    requests = []
    for sink_time in (2, 3):
        sources = _propagation_sources(physical, kind="custom")
        binding = _binding_plan(
            physical,
            operator_times={2: (sink_time, 1)},
            propagation_sources=sources,
        )
        request = next(
            request
            for request in binding.materializations.values()
            if isinstance(request, PropagationMaterializationRequest)
            and request.source_time == 1
        )
        materializers.append(
            PrimitiveMaterializer(
                binding,
                source_handlers={"custom": handler},
            )
        )
        requests.append(request)

    assert materialize_primitive_batch(materializers, requests[0].value_id) is None
    for materializer in materializers:
        materializer.materialize(requests[0].value_id)
    assert handler.calls == [2, 3]


def test_unknown_propagation_source_kind_is_rejected_without_fallback():
    physical = _physical_plan({"C00": _two_point()}, mode_size=3)
    sources = _propagation_sources(physical, kind="custom-unregistered")
    binding = _binding_plan(
        physical,
        operator_times={2: (5, 1)},
        propagation_sources=sources,
    )
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
    )

    with pytest.raises(ValueError, match="No propagation source handler"):
        PrimitiveMaterializer(binding).materialize(request.value_id)


def test_hdf5_handler_canonicalizes_quda_source_first_axes_and_validates_dimension_order(tmp_path):
    nv = 2
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "quda_source_first.h5"
    raw = np.zeros((4, 4, 4, nv, nv), dtype=np.complex128)
    for t_sink in range(4):
        for source_spin in range(4):
            for sink_spin in range(4):
                for source_mode in range(nv):
                    for sink_mode in range(nv):
                        raw[t_sink, source_spin, sink_spin, source_mode, sink_mode] = (
                            10000 * t_sink
                            + 1000 * source_spin
                            + 100 * sink_spin
                            + 10 * source_mode
                            + sink_mode
                        )
    with h5py.File(filepath, "w") as handle:
        dataset = handle.create_dataset("tau", data=raw)
        dataset.attrs["dimension_order"] = (
            "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
        )

    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(filepath),
        dataset="tau",
        metadata={
            "axes": (
                "sink_time",
                "source_spin",
                "sink_spin",
                "source_mode",
                "sink_mode",
            )
        },
    )
    binding = _binding_plan(
        physical,
        operator_times={2: (3, 1)},
        propagation_sources=sources,
    )
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
        and request.source_time == 1
        and request.sink_time == 3
    )

    value = PrimitiveMaterializer(binding).materialize(request.value_id)
    assert value.tensor.shape == (nv, 4, nv, 4)
    assert value.tensor[1, 2, 0, 3].item() == raw[3, 3, 2, 0, 1]


def test_hdf5_handler_rejects_dimension_order_mismatch(tmp_path):
    nv = 2
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "quda_mismatch.h5"
    raw = np.zeros((4, 4, 4, nv, nv), dtype=np.complex128)
    with h5py.File(filepath, "w") as handle:
        dataset = handle.create_dataset("tau", data=raw)
        dataset.attrs["dimension_order"] = (
            "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
        )
    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(filepath),
        dataset="tau",
        metadata={
            "axes": (
                "sink_time",
                "sink_spin",
                "source_spin",
                "sink_mode",
                "source_mode",
            )
        },
    )
    binding = _binding_plan(
        physical,
        operator_times={2: (3, 1)},
        propagation_sources=sources,
    )
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
        and request.source_time == 1
    )
    with pytest.raises(ValueError, match="dimension_order"):
        PrimitiveMaterializer(binding).materialize(request.value_id)


def test_hdf5_handler_converts_dp_qdagger_source_to_qbar(tmp_path):
    nv = 1
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "quda_qdagger.h5"
    raw = np.zeros((4, 4, 4, nv, nv), dtype=np.complex128)
    for source_spin in range(4):
        raw[3, source_spin, 0, 0, 0] = source_spin + 1
    with h5py.File(filepath, "w") as handle:
        dataset = handle.create_dataset("tau", data=raw)
        dataset.attrs["dimension_order"] = (
            "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
        )
    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(filepath),
        dataset="tau",
        metadata={
            "axes": (
                "sink_time",
                "source_spin",
                "sink_spin",
                "source_mode",
                "sink_mode",
            ),
            "spin_basis": "DP",
            "source_field": "qdagger",
        },
    )
    binding = _binding_plan(
        physical,
        operator_times={2: (3, 1)},
        propagation_sources=sources,
    )
    request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
        and request.source_time == 1
    )
    value = PrimitiveMaterializer(binding).materialize(request.value_id).tensor
    expected = np.asarray([1, 2, -3, -4], dtype=np.complex128)
    assert np.allclose(value[0, 0, 0, :].numpy(), expected)


def test_hdf5_handler_reverse_reconstruction_uses_configured_dp_gamma5(tmp_path):
    from concord_lqcd.core.dirac_basis import get_spin_matrices

    nv = 2
    physical = _physical_plan({"C00": _two_point()}, mode_size=nv)
    filepath = tmp_path / "fixed_source_dp.h5"
    raw = np.arange(4 * 4 * 4 * nv * nv, dtype=np.float64).reshape(4, 4, 4, nv, nv)
    raw = raw + 1j * (raw + 0.25)
    with h5py.File(filepath, "w") as handle:
        handle.create_dataset("perambulators/src_t_1/data", data=raw)
    sources = _propagation_sources(
        physical,
        kind="hdf5",
        location=str(filepath),
        dataset="perambulators/src_t_{fixed_source_time}/data",
        metadata={
            "fixed_source_time": 1,
            "spin_basis": "DP",
            "axes": (
                "sink_time",
                "sink_spin",
                "source_spin",
                "sink_mode",
                "source_mode",
            ),
        },
    )
    binding = _binding_plan(
        physical,
        operator_times={2: (3, 1)},
        propagation_sources=sources,
    )
    reverse_request = next(
        request
        for request in binding.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
        and request.sink_time == 1
    )
    reverse = PrimitiveMaterializer(binding).materialize(reverse_request.value_id).tensor

    canonical = np.transpose(raw[3], (2, 0, 3, 1))
    gamma5 = np.asarray(get_spin_matrices("DP")["gamma5"], dtype=np.complex128)
    matrix = canonical.reshape(4 * nv, 4 * nv)
    g5 = np.kron(np.eye(nv, dtype=np.complex128), gamma5)
    expected = (g5 @ matrix.conj().T @ g5).reshape(nv, 4, nv, 4)
    assert np.allclose(reverse.numpy(), expected)
