"""Independent numerical and admission tests for optional epsilon-kernel caching."""

from dataclasses import replace
import itertools

import h5py
import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime.resource_binding import KernelMaterializationRequest
from concord_lqcd.compiler.distillation_lowering import (
    KernelRecipe, ProjectedField, ProjectionContribution, identify_kernel_resource,
)
from concord_lqcd.compiler.physics_ir import ColorTensorRecord, GaugeTensorRecord, OperationKind, OperationRecord, QuarkRecord
from concord_lqcd.job.data import HDF5DistillationKernelProvider, _canonical_baryon_cache_transform
from concord_lqcd.job.kernel_store import HDF5KernelStore
from concord_lqcd.job.schema import HDF5EigenvectorSpec


def _recipe(momentum=(0, 0, 0), permutation=(0, 1, 2), anti=False, flavors=("u", "u", "d")):
    colors = ("a", "b", "c")
    projected = tuple(
        ProjectedField(QuarkRecord(flavor, anti, "X", colors[index], f"s{index}", operator_id=0),
                       ProjectionContribution.EIGENVECTOR_DAGGER if anti else ProjectionContribution.EIGENVECTOR)
        for index, flavor in enumerate(flavors)
    )
    return KernelRecipe("cache-test", ("X",), {"X": momentum}, projected,
                        (ColorTensorRecord("epsilon", tuple(colors[index] for index in permutation), 0),),
                        open_ports=("mode_0", "mode_1", "mode_2"))


def _request(recipe, modes, scalar_type="complex128", time=0):
    return KernelMaterializationRequest(
        0, "test", "test", identify_kernel_resource(recipe), (), "test",
        (modes,) * 3, scalar_type, "distillation", recipe, (0,), (time,),
    )


def _inputs(tmp_path, modes=5, extent=4, times=4, scalar_type="complex128"):
    generator = np.random.default_rng(619)
    vectors = []
    for time in range(times):
        raw = generator.normal(size=(3 * extent, modes)) + 1j * generator.normal(size=(3 * extent, modes))
        orthonormal, _ = np.linalg.qr(raw)
        vectors.append(orthonormal.reshape(1, 1, extent, 3, modes))
    values = np.asarray(vectors, dtype=scalar_type)
    path = tmp_path / "vectors.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("evec", data=values)
    return path, values


def _provider(path, values, store=None, device="cpu"):
    spec = HDF5EigenvectorSpec(provider="hdf5_distillation", path=str(path), dataset="evec",
                              axes=("time", "z", "y", "x", "color", "mode"))
    return HDF5DistillationKernelProvider(spec, configuration=0,
                                         spatial_extent=(values.shape[3], 1, 1),
                                         temporal_extent=values.shape[0], mode_size=values.shape[-1],
                                         device=device, kernel_cache=store)


def _reference(values, momentum, permutation, anti):
    fields = values.conj() if anti else values
    extent, modes = fields.shape[2], fields.shape[-1]
    output = np.zeros((modes,) * 3, dtype=np.complex128)
    sign = (-1) ** sum(permutation[first] > permutation[second]
                       for first in range(3) for second in range(first + 1, 3))
    for coordinate in range(extent):
        matrix = fields[0, 0, coordinate].astype(np.complex128)
        phase = np.exp(2j * np.pi * momentum[0] * coordinate / extent)
        for first, second, third in itertools.product(range(modes), repeat=3):
            output[first, second, third] += sign * phase * (
                matrix[0, first] * matrix[1, second] * matrix[2, third]
                + matrix[1, first] * matrix[2, second] * matrix[0, third]
                + matrix[2, first] * matrix[0, second] * matrix[1, third]
                - matrix[0, first] * matrix[2, second] * matrix[1, third]
                - matrix[2, first] * matrix[1, second] * matrix[0, third]
                - matrix[1, first] * matrix[0, second] * matrix[2, third]
            )
    return output


@pytest.mark.parametrize("scalar_type", ["complex64", "complex128"])
@pytest.mark.parametrize("packing", ["none", "antisym3"])
def test_cache_symmetries_match_independent_determinants(tmp_path, scalar_type, packing):
    path, values = _inputs(tmp_path, scalar_type=scalar_type)
    cases = [(momentum, permutation, anti, _reference(values[0], momentum, permutation, anti))
             for momentum in ((0, 0, 0), (1, 0, 0), (-1, 0, 0))
             for permutation in itertools.permutations(range(3)) for anti in (False, True)]
    store = HDF5KernelStore(tmp_path / "cache", configuration=0, packing=packing)
    provider = _provider(path, values, store)
    for momentum, permutation, anti, reference in cases:
        query = _request(_recipe(momentum, permutation, anti), 5, scalar_type)
        scale = float(np.max(np.abs(reference)))
        assert scale > 0.01
        tolerance = 1e-5 if scalar_type == "complex64" else 2e-13
        for actual in (provider.materialize(query).numpy(), provider.materialize(query).numpy()):
            assert float(np.max(np.abs(actual - reference))) < tolerance * scale
    assert store.statistics.writes == 3
    for entry in store.directory.glob("**/*.h5"):
        with h5py.File(entry, "r") as handle:
            assert handle["values"].attrs["storage_packing"] == packing


def test_shared_basis_reuse_does_not_erase_global_flavor_identity(tmp_path):
    path, values = _inputs(tmp_path)
    store = HDF5KernelStore(tmp_path / "cache", configuration=0)
    provider = _provider(path, values, store)
    first = _request(_recipe(), 5)
    second = _request(_recipe(flavors=("u", "u", "u")), 5)
    assert first.resource_id != second.resource_id
    assert provider._kernel_cache_entry(first)[0] == provider._kernel_cache_entry(second)[0]
    expected = provider.materialize(first)
    torch.testing.assert_close(provider.materialize(second), expected, rtol=0, atol=0)
    assert store.statistics.writes == 1

    class CustomProvider(HDF5DistillationKernelProvider):
        pass

    custom = CustomProvider(provider.spec, configuration=0, spatial_extent=(4, 1, 1),
                            temporal_extent=4, mode_size=5, device="cpu", kernel_cache=store)
    assert custom._baryon_cache_transform(first) is None
    assert custom._kernel_cache_entry(first)[0] != custom._kernel_cache_entry(second)[0]
    assert custom.materialize_batch_with_limit((first, second), maximum_staging_bytes=10**8) is None


def test_canonical_cache_namespace_does_not_collide_with_raw_recipes(tmp_path):
    path, values = _inputs(tmp_path)
    store = HDF5KernelStore(tmp_path / "cache", configuration=0)
    provider = _provider(path, values, store)

    class CustomProvider(HDF5DistillationKernelProvider):
        pass

    custom = CustomProvider(
        provider.spec, configuration=0, spatial_extent=(4, 1, 1),
        temporal_extent=4, mode_size=5, device="cpu", kernel_cache=store,
    )
    query = _request(_recipe(flavors=("__concord_shared_laph__",) * 3), 5)
    canonical = provider._baryon_cache_transform(query)
    assert canonical is not None and canonical.canonical_resource_id == query.resource_id
    assert provider._kernel_cache_entry(query)[0] != custom._kernel_cache_entry(query)[0]


@pytest.mark.parametrize("operation", list(OperationKind))
def test_transformed_fields_do_not_enter_canonical_cache(operation):
    base = _recipe()
    fields = list(base.projected_fields)
    fields[0] = replace(fields[0], operations=(OperationRecord(operation, {}),))
    assert _canonical_baryon_cache_transform(replace(base, projected_fields=tuple(fields))) is None


def test_incompatible_kernel_recipes_retain_independent_identity():
    base = _recipe()
    changed_fields = list(base.projected_fields)
    changed_fields[0] = replace(changed_fields[0], contribution=ProjectionContribution.EIGENVECTOR_DAGGER)
    for incompatible in (
        replace(base, position_labels=("X", "Y")),
        replace(base, open_ports=("mode_2", "mode_1", "mode_0")),
        replace(base, projected_fields=tuple(changed_fields)),
        replace(base, gauge_tensors=(GaugeTensorRecord("X", "Y", "a", "d"),)),
        replace(base, color_tensors=(ColorTensorRecord("epsilon", ("a", "b", "d")),)),
    ):
        assert _canonical_baryon_cache_transform(incompatible) is None


@pytest.mark.parametrize("packing", ["none", "antisym3"])
@pytest.mark.parametrize("cached_count", [0, 1, 2, 3, 4])
def test_cache_batch_builds_only_missing_times_and_respects_admission(tmp_path, monkeypatch, packing, cached_count):
    path, values = _inputs(tmp_path, modes=8)
    store = HDF5KernelStore(tmp_path / "cache", configuration=0, packing=packing)
    provider = _provider(path, values, store)
    queries = tuple(_request(_recipe(), 8, time=time) for time in range(4))
    expected = torch.stack(tuple(_provider(path, values).materialize(query) for query in queries))
    for query in queries[:cached_count]:
        provider.materialize(query)
    read_times = []
    original_load = provider._load_eigenvector_batch

    def load(requests, layout):
        read_times.extend(layout[0])
        return original_load(requests, layout)

    monkeypatch.setattr(provider, "_load_eigenvector_batch", load)
    full_estimate = provider.batch_staging_bytes(queries)
    assert full_estimate is not None
    lower, upper = 0, full_estimate
    while lower < upper:
        candidate = (lower + upper) // 2
        if provider.batch_staging_bytes_with_limit(queries, maximum_staging_bytes=candidate) is None:
            lower = candidate + 1
        else:
            upper = candidate
    assert provider.materialize_batch_with_limit(queries, maximum_staging_bytes=lower - 1) is None
    assert not read_times
    actual = provider.materialize_batch_with_limit(queries, maximum_staging_bytes=lower)
    assert actual is not None
    torch.testing.assert_close(actual.tensor, expected, rtol=1e-12, atol=1e-12)
    assert read_times == list(range(cached_count, 4))
    assert store.statistics.writes == 4
    assert store.statistics.misses == 4
    if cached_count == 4:
        assert actual.mode == "hdf5_kernel_cache_batch"


def test_corrupt_cached_batch_returns_independent_fallback(tmp_path):
    path, values = _inputs(tmp_path)
    store = HDF5KernelStore(tmp_path / "cache", configuration=0, packing="antisym3")
    provider = _provider(path, values, store)
    queries = tuple(_request(_recipe(), 5, time=time) for time in range(2))
    expected = tuple(provider.materialize(query) for query in queries)
    key = provider._kernel_cache_entry(queries[1])[0]
    with h5py.File(store.entry_path(key), "r+") as handle:
        handle["values"][0] += 1
    with pytest.warns(RuntimeWarning, match="checksum mismatch"):
        assert provider.materialize_batch_with_limit(queries, maximum_staging_bytes=10**8) is None
    with pytest.warns(RuntimeWarning, match="checksum mismatch"):
        rebuilt = provider.materialize(queries[1])
    torch.testing.assert_close(rebuilt, expected[1], rtol=1e-13, atol=1e-13)


def test_packing_workspace_is_reserved_in_the_evaluation_plan(tmp_path):
    path, values = _inputs(tmp_path, modes=16, extent=8)
    queries = tuple(_request(_recipe(), 16, time=time) for time in range(2))
    packed = _provider(path, values, HDF5KernelStore(tmp_path / "packed", configuration=0, packing="antisym3"))
    dense = _provider(path, values, HDF5KernelStore(tmp_path / "dense", configuration=0))
    assert (packed.batch_staging_bytes(queries) - dense.batch_staging_bytes(queries)
            == packed._kernel_cache_packing_workspace_bytes(queries[0]))
    minimum = dense._kernel_batch_evaluation_plan(queries, maximum_staging_bytes=10**9)
    assert minimum is not None
    result = packed.materialize_batch_with_limit(queries, maximum_staging_bytes=packed.batch_staging_bytes(queries))
    assert result is not None
    assert result.tensor.shape == (2, 16, 16, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("packing", ["none", "antisym3"])
def test_cuda_cache_batches_remain_on_the_requested_device(tmp_path, packing):
    path, values = _inputs(tmp_path, modes=4)
    queries = tuple(_request(_recipe(anti=True), 4, time=time) for time in range(3))
    expected = torch.stack(tuple(_provider(path, values).materialize(query) for query in queries))
    provider = _provider(path, values, HDF5KernelStore(tmp_path / "cache", configuration=0, packing=packing), "cuda")
    provider.materialize(queries[0])
    result = provider.materialize_batch_with_limit(queries, maximum_staging_bytes=10**8)
    assert result is not None and result.tensor.device.type == "cuda"
    torch.testing.assert_close(result.tensor.cpu(), expected, rtol=1e-12, atol=1e-12)
    result = provider.materialize_batch_with_limit(queries, maximum_staging_bytes=10**8)
    assert result is not None and result.tensor.device.type == "cuda"
    torch.testing.assert_close(result.tensor.cpu(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("packing", ["none", "antisym3"])
def test_full_baryon_job_cold_warm_and_uncached_outputs_agree(tmp_path, packing):
    from concord_lqcd.job import JobSpec, run_job, write_job_plan

    operator_file = tmp_path / "operators.py"
    operator_file.write_text(
        "from concord_lqcd.core import Correlator, Operator, u, d, epsilon, gamma5\n"
        "def build():\n"
        "    mixed = Operator(epsilon('a','b','c') * u('x','a','s1') * "
        "gamma5('s1','s2') * d('x','b','s2') * u('x','c','s3'))\n"
        "    identical = Operator(epsilon('a','b','c') * u('x','a','s1') * "
        "gamma5('s1','s2') * u('x','b','s2') * u('x','c','s3'))\n"
        "    return {'mixed': Correlator(mixed, mixed.dagger()), "
        "'identical': Correlator(identical, identical.dagger())}\n"
    )
    path, values = _inputs(tmp_path, modes=4)
    generator = np.random.default_rng(938)
    for flavor in ("u", "d"):
        with h5py.File(tmp_path / f"tau_{flavor}.h5", "w") as handle:
            for source in (0, 2):
                shape = (4, 4, 4, 4, 4)
                handle.create_dataset(f"src_{source}", data=(generator.normal(size=shape)
                                      + 1j * generator.normal(size=shape)) / 4)
    payload = {
        "schema_version": 1, "job": {"name": "epsilon-cache"},
        "operators": {"file": str(operator_file), "factory": "build"},
        "plan": {"file": str(tmp_path / "plan.concord")},
        "lattice": {"spatial_extent": [4, 1, 1], "temporal_extent": 4},
        "compile": {"mode_size": 4, "scalar_type": "complex128"},
        "run": {"device": "cpu", "time_batch_size": 2, "configurations": {"values": [0]},
                "times": {"generator": {"type": "two_point", "sources": {"values": [0, 2]},
                                         "sinks": {"values": [0, 1, 2, 3]}}}},
        "data": {"eigenvectors": {"provider": "hdf5_distillation", "path": str(path), "dataset": "evec",
                                   "axes": ["time", "z", "y", "x", "color", "mode"]},
                 "propagation": {"provider": "hdf5_perambulator", "ordinary": {
                     "path": str(tmp_path / "tau_{flavor}.h5"), "dataset": "src_{source_time}",
                     "axes": ["sink_time", "sink_spin", "source_spin", "sink_mode", "source_mode"]}}},
        "output": {"file": str(tmp_path / "uncached.h5")},
    }
    spec = JobSpec.model_validate(payload)
    plan = write_job_plan(spec)
    assert run_job(spec, plan).all_complete
    payload["data"]["kernel_cache"] = {"directory": str(tmp_path / "cache"), "packing": packing}
    payload["output"]["file"] = str(tmp_path / "cold.h5")
    cold = run_job(JobSpec.model_validate(payload), plan)
    assert cold.all_complete and cold.kernel_cache_writes == 4
    payload["output"]["file"] = str(tmp_path / "warm.h5")
    warm = run_job(JobSpec.model_validate(payload), plan)
    assert warm.all_complete and warm.kernel_cache_writes == 0 and warm.kernel_cache_misses == 0
    with h5py.File(tmp_path / "uncached.h5", "r") as reference:
        for name in ("cold.h5", "warm.h5"):
            with h5py.File(tmp_path / name, "r") as actual:
                for channel in reference["correlators"]:
                    np.testing.assert_allclose(actual["correlators"][channel]["values"][...],
                                               reference["correlators"][channel]["values"][...], rtol=1e-13, atol=1e-13)
