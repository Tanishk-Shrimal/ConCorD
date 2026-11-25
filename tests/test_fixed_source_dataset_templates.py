"""Fixed-source storage coordinates must agree in templates and axis selection."""

import h5py
import numpy as np
import pytest

from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.backend.runtime.materialization import (
    HDF5TensorSourceHandler,
    PrimitiveMaterializer,
    materialize_primitive_batch,
)
from concord_lqcd.backend.runtime.resource_binding import (
    ExternalTensorSource,
    PropagationMaterializationRequest,
    prepare_resource_binding_template,
)
from concord_lqcd.core import Correlator, Operator, c, cbar, reset_registry, u, ubar


@pytest.fixture(scope="module")
def compilation():
    reset_registry()
    sink = Operator(cbar("y", "a", "sink_spin") * u("y", "a", "sink_spin"))
    source = Operator(ubar("x", "b", "source_spin") * c("x", "b", "source_spin"))
    return compile_correlators(
        {"meson": Correlator(sink, source)}, RuntimeCompilationConfig(mode_size=2)
    )


def _stored_values(source_time, sink_time):
    generator = np.random.default_rng(1700 + 10 * source_time + sink_time)
    shape = (2, 4, 2, 4)
    return generator.normal(size=shape) + 1j * generator.normal(size=shape)


def _expected(values, *, reverse, basis):
    if not reverse:
        return values
    # Specify the matrices independently of the runtime's Dirac-basis registry.
    gamma5 = (
        np.diag([1, 1, -1, -1])
        if basis == "DR"
        else np.array(
            [[0, 0, -1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0]]
        )
    )
    mode_spin_gamma5 = np.kron(np.eye(2), gamma5)
    return (
        mode_spin_gamma5 @ values.reshape(8, 8).conj().T @ mode_spin_gamma5
    ).reshape(2, 4, 2, 4)


def _requests(compilation, path, dataset, metadata, *, other_times, reverse):
    sources = {
        resource_id: ExternalTensorSource(
            source_id=resource_id,
            kind="hdf5",
            location=str(path),
            dataset=dataset,
            metadata={"fixed_source_time": 1, **metadata},
        )
        for resource_id in compilation.required_propagation_resources
    }
    template = prepare_resource_binding_template(
        compilation.physical_plan, propagation_sources=sources
    )
    handler = HDF5TensorSourceHandler()
    materializers = []
    requests = []
    for other_time in other_times:
        times = (1, other_time) if reverse else (other_time, 1)
        binding = template.bind({2: times})
        request = next(
            request
            for request in binding.materializations.values()
            if isinstance(request, PropagationMaterializationRequest)
            and (request.source_time, request.sink_time)
            == ((other_time, 1) if reverse else (1, other_time))
        )
        requests.append(request)
        materializers.append(
            PrimitiveMaterializer(binding, source_handlers={"hdf5": handler})
        )
    return tuple(materializers), tuple(requests)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("basis", ["DR", "DP"])
@pytest.mark.parametrize(
    "dataset",
    [
        "src_{source_time}/sink_{sink_time}",
        "fixed_{fixed_source_time}/sink_{sink_time}",
        "src_{source_time}/fixed_{fixed_source_time}/sink_{sink_time}",
    ],
)
def test_per_time_datasets_use_stored_endpoints(
    compilation, tmp_path, reverse, basis, dataset
):
    path = tmp_path / "per_time.h5"
    other_times = (3, 2, 3)
    with h5py.File(path, "w") as handle:
        for time in set(other_times):
            name = dataset.format(source_time=1, fixed_source_time=1, sink_time=time)
            handle.create_dataset(name, data=_stored_values(1, time))
        if reverse:
            # Wrong logical-time formatting must not succeed by reading decoys.
            for time in set(other_times):
                name = dataset.format(source_time=time, fixed_source_time=1, sink_time=1)
                if name not in handle:
                    handle.create_dataset(name, data=_stored_values(time, 1))

    materializers, requests = _requests(
        compilation, path, dataset, {"spin_basis": basis},
        other_times=other_times, reverse=reverse,
    )
    # Different stored datasets retain independent materialization.
    assert materialize_primitive_batch(materializers, requests[0].value_id) is None
    for materializer, request, time in zip(materializers, requests, other_times, strict=True):
        actual = materializer.materialize(request.value_id).tensor.numpy()
        expected = _expected(_stored_values(1, time), reverse=reverse, basis=basis)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        assert (request.source_time, request.sink_time) == (
            (time, 1) if reverse else (1, time)
        )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("basis", ["DR", "DP"])
@pytest.mark.parametrize(
    "dataset",
    [
        "src_{source_time}/tau",
        "fixed_{fixed_source_time}/tau",
        "src_{source_time}/fixed_{fixed_source_time}/tau",
    ],
)
def test_fixed_source_time_axis_and_dense_batch_use_stored_endpoints(
    compilation, tmp_path, reverse, basis, dataset
):
    path = tmp_path / "time_axes.h5"
    canonical = np.stack([
        np.stack([_stored_values(source, sink) for sink in range(4)])
        for source in range(4)
    ])
    pairs = np.stack([canonical.real, canonical.imag], axis=-1)
    stored = pairs.transpose(0, 3, 1, 5, 4, 2, 6)
    axes = (
        "source_time", "sink_spin", "sink_time", "source_spin",
        "source_mode", "sink_mode", "complex_pair",
    )
    with h5py.File(path, "w") as handle:
        for source in range(4):
            name = dataset.format(source_time=source, fixed_source_time=1)
            if name not in handle:
                handle.create_dataset(name, data=stored)

    other_times = (3, 2, 3)
    materializers, requests = _requests(
        compilation, path, dataset, {"spin_basis": basis, "axes": axes},
        other_times=other_times, reverse=reverse,
    )
    batch = materialize_primitive_batch(materializers, requests[0].value_id)
    assert batch is not None
    assert batch.mode == "hdf5_dense_slab"
    expected = np.stack([
        _expected(canonical[1, time], reverse=reverse, basis=basis)
        for time in other_times
    ])
    np.testing.assert_allclose(batch.tensor.numpy(), expected, rtol=1e-13, atol=1e-13)
    for materializer, request, expected_value in zip(materializers, requests, expected, strict=True):
        actual = materializer.materialize(request.value_id).tensor.numpy()
        np.testing.assert_allclose(actual, expected_value, rtol=1e-13, atol=1e-13)
