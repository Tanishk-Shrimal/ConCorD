"""Selected HDF5 mode reads and bounded reuse preserve numerical fields."""

from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime.resource_binding import KernelMaterializationRequest
from concord_lqcd.backend.runtime.resource_planning import HardwareBudgetPolicy
from concord_lqcd.job import JobDataError, JobRunError, compile_job_plan, run_job
from concord_lqcd.job._eigenvector_cache import RawEigenvectorCache
from concord_lqcd.job.data import HDF5DistillationKernelProvider, _build_job_configuration_runtime_resources
from concord_lqcd.job.schema import HDF5EigenvectorSpec
from tests.test_job_data_provider import (
    LX, LY, LZ, NV, T, _eigenvector_array, _job_spec, _write_eigenvectors,
    _write_identity_laph_gauge, _write_transformed_operator,
)
from concord_lqcd.job import JobSpec
from tests.test_job_runner import _spec, _write_data


def _provider(path, *, axes=None, dataset="evec", modes=NV):
    return HDF5DistillationKernelProvider(
        HDF5EigenvectorSpec(
            provider="hdf5_distillation", path=str(path), dataset=dataset,
            axes=axes or ("time", "z", "y", "x", "color", "mode"),
        ),
        configuration=0, spatial_extent=(LX, LY, LZ), temporal_extent=T,
        mode_size=modes, device="cpu",
    )


@pytest.mark.parametrize("encoding", ["native", "compound", "pair", "permuted", "time_template"])
@pytest.mark.parametrize("scalar_type", ["complex64", "complex128"])
def test_selected_modes_are_read_in_hdf5_and_match_full_read(tmp_path, monkeypatch, encoding, scalar_type):
    path = tmp_path / "vectors.h5"
    values = np.concatenate((_eigenvector_array(), 2 * _eigenvector_array()), axis=-1)
    axes = ("time", "z", "y", "x", "color", "mode")
    stored = values
    dataset = "evec"
    if encoding == "compound":
        stored = np.empty(values.shape, dtype=[("re", "f8"), ("im", "f8")])
        stored["re"], stored["im"] = values.real, values.imag
    elif encoding == "pair":
        stored = np.stack((values.real, values.imag), axis=2)
        axes = ("time", "z", "complex_pair", "y", "x", "color", "mode")
    elif encoding == "permuted":
        stored = values.transpose(5, 3, 0, 4, 1, 2)
        axes = ("mode", "x", "time", "color", "z", "y")
    elif encoding == "time_template":
        stored = values[3]
        axes = axes[1:]
        dataset = "t{time}"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset.format(time=3), data=stored)
    reads = []
    original = h5py.Dataset.__getitem__

    def read(dataset, selection):
        result = original(dataset, selection)
        reads.append((selection, result.shape))
        return result

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    provider = _provider(path, axes=axes, dataset=dataset)
    provider.configure_raw_eigenvector_cache(1024**2)
    actual = provider._load_eigenvectors(3, scalar_type)
    expected = torch.as_tensor(values[3, ..., :NV], dtype=getattr(torch, scalar_type)).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(provider._load_eigenvectors(3, scalar_type), expected, rtol=0, atol=0)
    assert len(reads) == 1
    assert reads[0][0][axes.index("mode")] == slice(0, NV)
    stats = provider.raw_eigenvector_cache_statistics
    assert stats.hits == 1 and stats.misses == 1
    assert stats.resident_bytes == actual.numel() * actual.element_size()


def test_raw_cache_lru_owns_storage_and_bounds_retained_bytes():
    cache = RawEigenvectorCache(64)
    slab = torch.arange(128, dtype=torch.float64)
    cache.put((1,), slab[:4])
    assert cache.get((1,)).untyped_storage().nbytes() == 32
    slab[:4] = -1
    assert cache.get((1,)).tolist() == [0, 1, 2, 3]
    cache.put((2,), slab[4:8])
    cache.get((1,))
    cache.put((3,), slab[8:12])
    assert cache.get((2,)) is None
    assert cache.statistics.evictions == 1
    assert cache.statistics.resident_bytes == cache.statistics.peak_resident_bytes == 64
    cache.put((4,), slab)
    assert cache.get((4,)) is None
    cache.clear()
    assert cache.statistics.resident_bytes == 0


@pytest.mark.parametrize("bad", [-1, True, 1.5, None])
def test_invalid_raw_cache_budgets_are_rejected(bad):
    with pytest.raises(ValueError):
        RawEigenvectorCache(bad)
    with pytest.raises(ValueError):
        HardwareBudgetPolicy(raw_eigenvector_cache_limit_bytes=bad)


def test_raw_cache_allocation_failure_retains_uncached_execution(monkeypatch):
    cache = RawEigenvectorCache(1024)
    values = torch.ones(4)

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("injected cache allocation failure")

    monkeypatch.setattr(torch.Tensor, "clone", fail)
    cache.put((1,), values)
    assert cache.statistics.allocation_failures == 1
    assert cache.statistics.resident_bytes == cache.maximum_bytes == 0
    assert values.tolist() == [1] * 4


def test_cache_keys_separate_time_dtype_modes_and_dataset(tmp_path):
    path = tmp_path / "vectors.h5"
    values = _write_eigenvectors(path)
    provider = _provider(path)
    provider.configure_raw_eigenvector_cache(1024**2)
    for time in (1, 3):
        for scalar_type in ("complex64", "complex128"):
            actual = provider._load_eigenvectors(time, scalar_type)
            expected = torch.tensor(values[time], dtype=getattr(torch, scalar_type)).unsqueeze(0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    provider.mode_size = 1
    assert provider._load_eigenvectors(1, "complex128").shape[-1] == 1
    assert provider.raw_eigenvector_cache_statistics.misses == 5


def test_changed_input_fails_instead_of_serving_cached_data(tmp_path):
    path = tmp_path / "vectors.h5"
    _write_eigenvectors(path)
    provider = _provider(path)
    provider.configure_raw_eigenvector_cache(1024**2)
    provider._load_eigenvectors(1, "complex128")
    with h5py.File(path, "r+") as handle:
        handle["evec"][1] *= 2
    with pytest.raises(JobDataError, match="changed during this configuration"):
        provider._load_eigenvectors(1, "complex128")
    assert provider.raw_eigenvector_cache_statistics.resident_bytes == 0


def test_subclass_keeps_independent_loads(tmp_path, monkeypatch):
    path = tmp_path / "vectors.h5"
    _write_eigenvectors(path)

    class CustomProvider(HDF5DistillationKernelProvider):
        pass

    base = _provider(path)
    provider = CustomProvider(base.spec, configuration=0, spatial_extent=(LX, LY, LZ), mode_size=NV, device="cpu")
    provider.configure_raw_eigenvector_cache(1024**2)
    provider._load_eigenvectors(1, "complex128")
    provider._load_eigenvectors(1, "complex128")
    assert provider.raw_eigenvector_cache_statistics.resident_bytes == 0
    assert provider.raw_eigenvector_cache_statistics.hits == 0


def test_batch_cache_preserves_duplicates_order_and_short_batches(tmp_path, monkeypatch):
    spec = _job_spec(tmp_path, momentum=(1, 0, 0))
    _write_eigenvectors(tmp_path / "evec_0.h5")
    plan = compile_job_plan(spec)
    resources = _build_job_configuration_runtime_resources(spec, plan, configuration=0, time_contexts=spec.time_contexts)
    provider = resources[0].kernel_providers["distillation"]
    request = next(request for request in resources[0].bind(plan).materializations.values() if isinstance(request, KernelMaterializationRequest))
    requests = tuple(replace(request, operator_times=(time,)) for time in (3, 1, 3))
    layout = provider._kernel_batch_layout(requests)
    assert layout is not None
    uncached = provider._load_eigenvector_batch(requests, layout)
    provider.configure_raw_eigenvector_cache(1024**2)
    first = provider._load_eigenvector_batch(requests, layout)

    def fail(*args, **kwargs):
        raise AssertionError("cache hit must not read HDF5")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", fail)
    second = provider._load_eigenvector_batch(requests, layout)
    short = provider._load_eigenvector_batch(requests[:1], provider._kernel_batch_layout(requests[:1]))
    torch.testing.assert_close(first, uncached, rtol=0, atol=0)
    torch.testing.assert_close(second, uncached, rtol=0, atol=0)
    torch.testing.assert_close(short, uncached[:1], rtol=0, atol=0)


def test_runner_reserves_cache_budget_and_clears_on_configuration_exit(tmp_path, monkeypatch):
    spec = _spec(tmp_path, time_batch_size=1, momentum=(1, 0, 0))
    _write_data(tmp_path)
    plan = compile_job_plan(spec)
    cleared = []
    original = HDF5DistillationKernelProvider.clear_raw_eigenvector_cache

    def clear(provider):
        original(provider)
        cleared.append(provider.raw_eigenvector_cache_statistics)

    monkeypatch.setattr(HDF5DistillationKernelProvider, "clear_raw_eigenvector_cache", clear)
    summary = run_job(spec, plan, budget_policy=HardwareBudgetPolicy(raw_eigenvector_cache_limit_bytes=4096))
    assert summary.all_complete
    assert len(cleared) == len(spec.configurations)
    assert all(value.resident_bytes == 0 for value in cleared)
    assert all(0 < decision.raw_eigenvector_cache_limit_bytes <= 4096 for decision in summary.configuration_decisions)


@pytest.mark.parametrize("transformation", [
    "derivative(u('y','a','s2'), '+x')",
    "transport(u('y','a','s2'), '+x', '-y')",
    "shift(u('y','a','s2'), '+x')",
])
def test_transformed_kernels_do_not_modify_cached_raw_fields(tmp_path, transformation):
    spec = _job_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["data"]["gauge_links"] = {"provider": "laph_iomap", "path": str(tmp_path / "gauge")}
    spec = JobSpec.model_validate(payload)
    _write_transformed_operator(spec.operators.file, transformation)
    _write_identity_laph_gauge(tmp_path / "gauge")
    raw = _write_eigenvectors(tmp_path / "evec_0.h5")
    plan = compile_job_plan(spec)
    resource = _build_job_configuration_runtime_resources(spec, plan, configuration=0, time_contexts=spec.time_contexts)[0]
    provider = resource.kernel_providers["distillation"]
    requests = [request for request in resource.bind(plan).materializations.values() if isinstance(request, KernelMaterializationRequest)]
    expected = [provider.materialize(request).clone() for request in requests]
    provider.configure_raw_eigenvector_cache(1024**2)
    for request, value in zip(requests, expected, strict=True):
        torch.testing.assert_close(provider.materialize(request), value, rtol=0, atol=0)
        torch.testing.assert_close(provider.materialize(request), value, rtol=0, atol=0)
        time = request.operator_times[0]
        torch.testing.assert_close(provider._load_eigenvectors(time, "complex128"), torch.tensor(raw[time]).unsqueeze(0), rtol=0, atol=0)


def test_configuration_exception_releases_raw_cache(tmp_path, monkeypatch):
    spec = _spec(tmp_path, time_batch_size=1, momentum=(1, 0, 0))
    _write_data(tmp_path)
    plan = compile_job_plan(spec)
    released = []
    retained = []
    clear = HDF5DistillationKernelProvider.clear_raw_eigenvector_cache
    load = HDF5DistillationKernelProvider._load_eigenvectors

    def fail(provider, *args):
        load(provider, *args)
        raise RuntimeError("injected failure after raw field admission")

    def record(provider):
        retained.append(provider.raw_eigenvector_cache_statistics.resident_bytes)
        clear(provider)
        released.append(provider.raw_eigenvector_cache_statistics.resident_bytes)

    monkeypatch.setattr(HDF5DistillationKernelProvider, "_load_eigenvectors", fail)
    monkeypatch.setattr(HDF5DistillationKernelProvider, "clear_raw_eigenvector_cache", record)
    with pytest.raises(JobRunError, match="injected failure"):
        run_job(spec, plan)
    assert released == [0]
    assert retained[0] > 0


def test_dataset_identity_does_not_alias_in_same_file(tmp_path):
    path = tmp_path / "vectors.h5"
    values = _eigenvector_array()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("evec", data=values)
        handle.create_dataset("other", data=3 * values)
    provider = _provider(path)
    provider.configure_raw_eigenvector_cache(1024**2)
    first = provider._load_eigenvectors(1, "complex128")
    provider.spec = provider.spec.model_copy(update={"dataset": "other"})
    other = provider._load_eigenvectors(1, "complex128")
    torch.testing.assert_close(other, 3 * first, rtol=0, atol=0)
    assert provider.raw_eigenvector_cache_statistics.misses == 2


def test_configuration_and_time_file_templates_are_isolated(tmp_path):
    values = _eigenvector_array()
    for configuration in (0, 1):
        with h5py.File(tmp_path / f"cfg{configuration}_t1.h5", "w") as handle:
            handle.create_dataset("evec", data=(configuration + 1) * values[1])
    provider = _provider(tmp_path / "cfg{configuration}_t{time}.h5", axes=("z", "y", "x", "color", "mode"))
    provider.configure_raw_eigenvector_cache(1024**2)
    first = provider._load_eigenvectors(1, "complex128")
    provider.configuration = 1
    second = provider._load_eigenvectors(1, "complex128")
    torch.testing.assert_close(second, 2 * first, rtol=0, atol=0)
    assert provider.raw_eigenvector_cache_statistics.misses == 2


def test_insufficient_stored_modes_still_fail_before_cache_admission(tmp_path):
    path = tmp_path / "vectors.h5"
    _write_eigenvectors(path)
    provider = _provider(path, modes=NV + 1)
    provider.configure_raw_eigenvector_cache(1024**2)
    with pytest.raises(JobDataError, match="compiled plan requires"):
        provider._load_eigenvectors(1, "complex128")
    assert provider.raw_eigenvector_cache_statistics.resident_bytes == 0


def test_runner_zero_tiny_and_enabled_cache_have_identical_outputs(tmp_path):
    spec = _spec(tmp_path, time_batch_size=1, momentum=(1, 0, 0))
    _write_data(tmp_path)
    plan = compile_job_plan(spec)
    expected = None
    for maximum_bytes in (0, 1, 4096):
        payload = spec.model_dump(mode="python")
        payload["output"]["file"] = tmp_path / f"cache_{maximum_bytes}.h5"
        current = JobSpec.model_validate(payload)
        summary = run_job(current, plan, budget_policy=HardwareBudgetPolicy(raw_eigenvector_cache_limit_bytes=maximum_bytes))
        assert summary.all_complete
        with h5py.File(current.output.file, "r") as handle:
            arrays = {name: group["values"][...] for name, group in handle["correlators"].items()}
        if expected is None:
            expected = arrays
        else:
            assert all(np.array_equal(expected[name], value) for name, value in arrays.items())
        if maximum_bytes < 4096:
            assert all(decision.raw_eigenvector_cache_limit_bytes == 0 for decision in summary.configuration_decisions)
