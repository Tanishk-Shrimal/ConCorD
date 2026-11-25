"""Focused tests for V1 job-level HDF5 data/provider translation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import struct

import h5py
import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime.materialization import materialize_primitive_batch
from concord_lqcd.backend.runtime.numerical_executor import GenericNumericalExecutor
from concord_lqcd.backend.runtime.hardware import HardwareDeviceProfile, HardwareProfile
from concord_lqcd.backend.runtime.resource_planning import (
    HardwareBudgetPolicy,
    MemorySafetyMargin,
    estimate_execution_memory,
    plan_execution_resources,
)
from concord_lqcd.backend.runtime.tiling import plan_tiled_execution
from concord_lqcd.backend.runtime.resource_binding import KernelMaterializationRequest
from concord_lqcd.core.dirac_basis import get_spin_matrices
from concord_lqcd.compiler.distillation_lowering import ProjectionContribution
from concord_lqcd.job import (
    JobDataError,
    JobSpec,
    build_job_runtime_resources,
    compile_job_plan,
)
from concord_lqcd.job.runner import _run_fingerprint
from concord_lqcd.job.data import _build_job_configuration_runtime_resources


T = 8
LX = 4
LY = 1
LZ = 1
NV = 2


def _write_operator(path: Path, *, momentum=(0, 0, 0)) -> None:
    path.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, c, cbar, u, ubar, gamma5",
                "",
                "def build():",
                f"    p = {tuple(momentum)!r}",
                "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * u('y','a','s2'), momentum=p)",
                "    source = Operator(ubar('x','b','s3') * gamma5('s3','s4') * c('x','b','s4'), momentum=tuple(-v for v in p))",
                "    return {'D': Correlator(sink, source)}",
            ]
        ),
        encoding="utf-8",
    )


def _write_transformed_operator(path: Path, transformation: str) -> None:
    path.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, derivative, shift, transport, u, ubar, gamma5",
                "",
                "def build():",
                f"    transformed = {transformation}",
                "    sink = Operator(ubar('y','a','s1') * gamma5('s1','s2') * transformed)",
                "    return {'D': Correlator(sink, sink.dagger())}",
            ]
        ),
        encoding="utf-8",
    )


def _eigenvector_array() -> np.ndarray:
    count = T * LZ * LY * LX * 3 * NV
    real = np.arange(count, dtype=np.float64).reshape(T, LZ, LY, LX, 3, NV)
    imag = (real % 7) / 11.0
    return real / 23.0 + 1j * imag


def _write_eigenvectors(path: Path) -> np.ndarray:
    values = _eigenvector_array()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("evec", data=values)
    return values


def _write_identity_laph_gauge(path: Path) -> None:
    values = np.zeros((3, T, LZ, LY, LX, 3, 3), dtype=np.complex128)
    values[...] = np.eye(3, dtype=np.complex128)
    bytes_per_site = 9 * values.dtype.itemsize
    record = bytearray(struct.pack("<i", 3))
    for direction in range(3):
        record.extend(struct.pack("<6I", 4, LX, LY, LZ, T, bytes_per_site))
        record.extend(values[direction].tobytes(order="C"))

    header = b"<SmearedGaugeField/>"
    prefix = b"L" + b"Laph--SmearedGaugeField4D".ljust(32, b" ")
    data = bytearray(struct.pack("<Q", 0))
    data.extend(b"N")
    data.extend(struct.pack("<i", len(header)))
    data.extend(header)
    record_position = len(data)
    data.extend(struct.pack("<Q", len(record)))
    data.extend(record)
    map_start = len(data)
    data.extend(struct.pack("<QQ", 1, 4))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<Q", record_position))
    data.extend(b"E")
    struct.pack_into("<Q", data, 0, map_start)
    path.write_bytes(prefix + data)


def _write_perambulators(tmp_path: Path, configuration: int) -> None:
    for flavor_index, flavor in enumerate(("u", "c")):
        path = tmp_path / f"peram_{flavor}_{configuration}.h5"
        with h5py.File(path, "w") as handle:
            for source_time in (1, 3):
                data = np.empty((T, 4, 4, NV, NV), dtype=np.complex128)
                base = 100 * (flavor_index + 1) + 10 * source_time
                for sink_time in range(T):
                    raw = np.arange(4 * 4 * NV * NV, dtype=np.float64).reshape(4, 4, NV, NV)
                    data[sink_time] = (
                        base + sink_time + raw / 37.0
                        + 1j * (raw % 5) / 29.0
                    )
                handle.create_dataset(f"tau_{source_time}", data=data)


def _job_spec(
    tmp_path: Path,
    *,
    momentum=(0, 0, 0),
    ordinary_files=None,
) -> JobSpec:
    operator_file = tmp_path / "operators.py"
    _write_operator(operator_file, momentum=momentum)
    ordinary = {
        "path": str(tmp_path / "peram_{flavor}_{configuration}.h5"),
        "dataset": "/tau_{source_time}",
        "axes": [
            "sink_time",
            "sink_spin",
            "source_spin",
            "sink_mode",
            "source_mode",
        ],
    }
    if ordinary_files is not None:
        ordinary.pop("path")
        ordinary["files"] = ordinary_files

    return JobSpec.model_validate(
        {
            "schema_version": 1,
            "job": {"name": "job_data_test"},
            "operators": {"file": str(operator_file), "factory": "build"},
            "plan": {"file": str(tmp_path / "job.concord")},
            "lattice": {
                "spatial_extent": [LX, LY, LZ],
                "temporal_extent": T,
            },
            "compile": {
                "mode_size": NV,
                "scalar_type": "complex128",
            },
            "run": {
                "device": "cpu",
                "configurations": {"values": [0, 4]},
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
                    "ordinary": ordinary,
                },
            },
            "output": {"file": str(tmp_path / "result.h5")},
        }
    )


def test_job_runtime_resources_bind_exact_ids_and_execute_with_hdf5(tmp_path):
    spec = _job_spec(tmp_path)
    _write_eigenvectors(tmp_path / "evec_0.h5")
    _write_perambulators(tmp_path, 0)
    plan = compile_job_plan(spec)

    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context="2pt_src_1_snk_3",
    )

    assert set(resources.propagation_sources) == set(plan.required_propagation_resources)
    for resource_id, recipe in plan.required_propagation_resources.items():
        source = resources.propagation_sources[resource_id]
        assert recipe.is_generalized is False
        assert len(recipe.segment_flavors) == 1
        flavor = recipe.segment_flavors[0]
        assert source.kind == "hdf5"
        assert source.location == str(tmp_path / f"peram_{flavor}_0.h5")
        assert source.dataset == "/tau_{source_time}"
        assert tuple(source.metadata["axes"]) == tuple(spec.data.propagation.ordinary.axes)

    binding = resources.bind(plan)
    assert set(binding.bindings) == {
        step.operation.resource_id
        for step in plan.physical_plan.steps
        if hasattr(step.operation, "resource_id")
    }

    result = GenericNumericalExecutor(
        resources.materializer(plan),
        device=resources.storage_device,
        compute_device=resources.device,
        tiled_plan=resources.execution_plan.tiled_plan,
    ).execute()
    assert set(result.outputs) == {"D"}
    assert result.output("D").dtype == torch.complex128
    assert torch.isfinite(result.output("D")).all()


def test_configuration_resources_share_invariant_runtime_state(tmp_path):
    spec = _job_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["run"]["times"]["generator"]["sinks"] = {"values": [3, 4]}
    spec = JobSpec.model_validate(payload)
    plan = compile_job_plan(spec)

    resources = _build_job_configuration_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_contexts=spec.time_contexts,
    )

    assert len(resources) == 2
    assert resources[0].kernel_providers["distillation"] is resources[1].kernel_providers[
        "distillation"
    ]
    assert resources[0].source_handlers["hdf5"] is resources[1].source_handlers["hdf5"]
    assert resources[0].binding_plan is not None
    assert resources[1].binding_plan is not None
    assert resources[0].binding_plan.bindings is resources[1].binding_plan.bindings
    assert resources[0].bind(plan) is resources[0].binding_plan
    assert resources[1].bind(plan) is resources[1].binding_plan


@pytest.mark.parametrize('spatially_tiled', (False, True))
def test_hdf5_kernel_batch_reads_one_dense_time_slab_and_matches_independent(
    tmp_path,
    monkeypatch,
    spatially_tiled,
):
    spec = _job_spec(tmp_path, momentum=(1, 0, 0))
    payload = spec.model_dump(mode="python")
    payload["run"]["times"]["generator"]["sinks"] = {"values": [2, 3, 4]}
    payload["data"]["kernel_cache"] = {
        "directory": str(tmp_path / "kernel-cache")
    }
    spec = JobSpec.model_validate(payload)
    _write_eigenvectors(tmp_path / "evec_0.h5")
    plan = compile_job_plan(spec)
    resources = _build_job_configuration_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_contexts=spec.time_contexts,
    )
    materializers = tuple(
        resource.materializer(plan).with_kernel_batch_staging_limit(10 * 1024**2)
        for resource in resources
    )
    bindings = tuple(materializer.binding_plan for materializer in materializers)
    value_id = next(
        value_id
        for value_id, request in bindings[0].materializations.items()
        if isinstance(request, KernelMaterializationRequest)
        and len(
            {
                binding.materialization(value_id).operator_times
                for binding in bindings
            }
        )
        > 1
    )
    provider = resources[0].kernel_providers["distillation"]
    requests = tuple(binding.materialization(value_id) for binding in bindings)
    if spatially_tiled:
        full_bytes = provider.batch_staging_bytes(requests)
        assert full_bytes is not None
        full_workspace = provider._kernel_batch_workspace_bytes(
            requests[0], batch_size=len(requests), spatial_sites=LX * LY * LZ,
        )
        minimum_workspace = provider._kernel_batch_workspace_bytes(
            requests[0], batch_size=len(requests), spatial_sites=1,
        )
        tile_limit = full_bytes - full_workspace + minimum_workspace
        materializers = tuple(
            materializer.with_kernel_batch_staging_limit(tile_limit)
            for materializer in materializers
        )
    import concord_lqcd.job.data as job_data

    tile_extents = []
    original_evaluate = job_data.evaluate_kernel_program
    momentum = next(momentum for momentum in requests[0].recipe.momenta.values()
                    if momentum != (0, 0, 0))

    def record_tile(program, prepared_provider, context, **keywords):
        field = prepared_provider.eigenvector(None, context=context)
        phase = prepared_provider.phase(momentum, context)
        assert phase is not None
        assert field.shape[1] == phase.shape[0]
        tile_extents.append(field.shape[1])
        return original_evaluate(program, prepared_provider, context, **keywords)

    monkeypatch.setattr(job_data, "evaluate_kernel_program", record_tile)
    open_count = 0
    original_open = provider._open_file

    @contextmanager
    def counted_open(path):
        nonlocal open_count
        open_count += 1
        with original_open(path) as handle:
            yield handle

    monkeypatch.setattr(provider, "_open_file", counted_open)

    result = materialize_primitive_batch(materializers, value_id)

    assert result is not None
    assert result.mode == (
        "hdf5_distillation_dense_slab_spatial_tiled"
        if spatially_tiled else "hdf5_distillation_dense_slab"
    )
    assert tile_extents == ([1] * LX if spatially_tiled else [LX])
    assert open_count == 1
    assert provider.kernel_cache_statistics is not None
    assert provider.kernel_cache_statistics.misses == len(requests)
    assert provider.kernel_cache_statistics.writes == len(requests)
    expected = torch.stack(tuple(provider.materialize(request) for request in requests))
    torch.testing.assert_close(result.tensor, expected, rtol=1e-12, atol=1e-12)
    assert provider.kernel_cache_statistics.hits == len(requests)
    reference_values = []
    for request in requests:
        time = request.operator_times[0]
        vectors = _eigenvector_array()[time].reshape(-1, 3, NV)
        selected = [vectors.conj() if field.record.is_anti else vectors
                    for field in request.recipe.projected_fields]
        phase = np.exp(2j * np.pi * momentum[0] * np.arange(LX) / LX)
        reference_values.append(np.einsum('xai,xaj,x->ij', *selected, phase, optimize=False))
    reference = np.stack(reference_values)
    assert np.max(np.abs(reference)) > 1
    np.testing.assert_allclose(result.tensor.numpy(), reference, rtol=1e-12, atol=1e-12)


def test_hdf5_kernel_batch_strictly_falls_back_for_transformed_fields(tmp_path):
    spec = _job_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["run"]["times"]["generator"]["sinks"] = {"values": [2, 3]}
    spec = JobSpec.model_validate(payload)
    _write_transformed_operator(
        spec.operators.file,
        "shift(u('y','a','s2'), '+x')",
    )
    _write_eigenvectors(tmp_path / "evec_0.h5")
    plan = compile_job_plan(spec)
    resources = _build_job_configuration_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_contexts=spec.time_contexts,
    )
    materializers = tuple(
        resource.materializer(plan).with_kernel_batch_staging_limit(10 * 1024**2)
        for resource in resources
    )
    bindings = tuple(materializer.binding_plan for materializer in materializers)
    value_id = next(
        value_id
        for value_id, request in bindings[0].materializations.items()
        if isinstance(request, KernelMaterializationRequest)
        and any(field.operations for field in request.recipe.projected_fields)
        and len(
            {
                binding.materialization(value_id).operator_times
                for binding in bindings
            }
        )
        > 1
    )

    assert materialize_primitive_batch(materializers, value_id) is None


def test_host_staged_job_resources_materialize_complete_primitives_on_cpu(tmp_path):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    estimate = estimate_execution_memory(plan.physical_plan)
    full_operations = plan_tiled_execution(
        plan.physical_plan,
        device_memory_limit_bytes=10**18,
    )
    device_limit = full_operations.full_peak_device_bytes - 1
    host = HardwareDeviceProfile(
        device="cpu",
        kind="cpu",
        name="Synthetic CPU",
        total_memory_bytes=4 * estimate.known_working_set_bytes,
        available_memory_bytes=2 * estimate.known_working_set_bytes,
        logical_processors=8,
    )
    accelerator = HardwareDeviceProfile(
        device="cuda:0",
        kind="cuda",
        name="Synthetic GPU",
        total_memory_bytes=2 * device_limit,
        available_memory_bytes=device_limit,
        compute_capability=(9, 0),
        multiprocessor_count=32,
    )
    hardware = HardwareProfile(
        host=host,
        accelerators=(accelerator,),
        system="TestOS",
        machine="test-machine",
    )
    no_margin = MemorySafetyMargin(fraction=0.0, minimum_bytes=0)
    deployment = plan_execution_resources(
        plan.physical_plan,
        hardware,
        requested_device="cuda:0",
        policy=HardwareBudgetPolicy(
            cpu_margin=no_margin,
            cuda_margin=no_margin,
        ),
    )

    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context="2pt_src_1_snk_3",
        device="cuda:0",
        execution_plan=deployment,
    )

    assert resources.device == torch.device("cuda:0")
    assert resources.storage_device == torch.device("cpu")
    assert resources.execution_plan.execution_mode == "host_tiled"
    assert resources.execution_plan.tiled_plan is not None
    assert resources.execution_plan.tiled_plan.uses_index_tiling
    assert resources.kernel_providers["distillation"].device == torch.device("cpu")


def test_job_runtime_resources_preserve_fixed_source_metadata(tmp_path):
    spec = _job_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    ordinary = payload["data"]["propagation"]["ordinary"]
    ordinary["dataset"] = "/perambulators/src_t_{fixed_source_time}/data"
    ordinary["fixed_source_time"] = 1
    spec = JobSpec.model_validate(payload)

    plan = compile_job_plan(spec)
    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context="2pt_src_1_snk_3",
    )

    for source in resources.propagation_sources.values():
        assert source.dataset == "/perambulators/src_t_{fixed_source_time}/data"
        assert source.metadata["fixed_source_time"] == 1


def test_job_runtime_resources_bind_spin_matrices_in_plan_basis(tmp_path):
    spec = _job_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["compile"]["spin_basis"] = "DP"
    spec = JobSpec.model_validate(payload)
    plan = compile_job_plan(spec)
    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context="2pt_src_1_snk_3",
    )

    binding = resources.bind(plan)
    gamma5_bindings = [
        item
        for item in binding.bindings.values()
        if getattr(item, "tensor", None) is not None
        and item.tensor.kind == "gamma5"
    ]
    assert gamma5_bindings
    expected = np.asarray(get_spin_matrices("DP")["gamma5"])
    actual = np.asarray(gamma5_bindings[0].matrix)
    assert np.allclose(actual, expected)
    assert not np.allclose(actual, get_spin_matrices("DR")["gamma5"])


def test_hdf5_eigenvectors_and_nonzero_momentum_phase_match_semantic_axes(tmp_path):
    spec = _job_spec(tmp_path, momentum=(1, 0, 0))
    raw = _write_eigenvectors(tmp_path / "evec_0.h5")
    _write_perambulators(tmp_path, 0)
    plan = compile_job_plan(spec)
    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context=spec.time_contexts[0],
    )
    binding = resources.bind(plan)

    target_request = None
    for request in binding.materializations.values():
        if isinstance(request, KernelMaterializationRequest):
            momenta = tuple(request.recipe.momenta.values())
            if (1, 0, 0) in momenta:
                target_request = request
                break
    assert target_request is not None

    actual = resources.kernel_providers["distillation"].materialize(target_request)
    assert actual.shape == (NV, NV)

    time = target_request.operator_times[0]
    values = raw[time]  # [z, y, x, color, mode]
    projected = []
    for field in target_request.recipe.projected_fields:
        if field.contribution == ProjectionContribution.EIGENVECTOR_DAGGER:
            projected.append(values.conj())
        else:
            projected.append(values)
    assert len(projected) == 2

    x = np.arange(LX, dtype=np.float64)
    phase_x = np.exp(1j * 2.0 * np.pi * x / LX)
    phase = phase_x.reshape(1, 1, LX)
    expected = np.einsum(
        "zyxci,zyxcj,zyx->ij",
        projected[0],
        projected[1],
        phase,
    )
    assert torch.allclose(
        actual.cpu(),
        torch.as_tensor(expected, dtype=torch.complex128),
        atol=1e-11,
        rtol=1e-11,
    )


def test_ordinary_file_catalog_has_no_flavor_fallback(tmp_path):
    spec = _job_spec(
        tmp_path,
        ordinary_files={
            "c": str(tmp_path / "charm.h5"),
            "l": str(tmp_path / "light.h5"),
        },
    )
    plan = compile_job_plan(spec)

    with pytest.raises(JobDataError, match="exact flavor 'u'"):
        build_job_runtime_resources(
            spec,
            plan,
            configuration=0,
            time_context=spec.time_contexts[0],
        )


def test_job_runtime_resources_reject_unselected_configuration_and_context(tmp_path):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)

    with pytest.raises(JobDataError, match="not selected"):
        build_job_runtime_resources(
            spec,
            plan,
            configuration=2,
            time_context=spec.time_contexts[0],
        )

    with pytest.raises(JobDataError, match="unknown job time-context"):
        build_job_runtime_resources(
            spec,
            plan,
            configuration=0,
            time_context="not-a-context",
        )


def test_builtin_eigenvector_provider_requires_time_selection(tmp_path):
    spec = _job_spec(tmp_path, momentum=(1, 0, 0))
    payload = spec.model_dump(mode="python")
    payload["data"]["eigenvectors"]["axes"] = ["z", "y", "x", "color", "mode"]
    invalid_for_builtin = JobSpec.model_validate(payload)
    _write_eigenvectors(tmp_path / "evec_0.h5")
    plan = compile_job_plan(invalid_for_builtin)
    resources = build_job_runtime_resources(
        invalid_for_builtin,
        plan,
        configuration=0,
        time_context=invalid_for_builtin.time_contexts[0],
    )
    binding = resources.bind(plan)
    request = next(
        req for req in binding.materializations.values()
        if isinstance(req, KernelMaterializationRequest)
    )

    with pytest.raises(JobDataError, match="select Euclidean time"):
        resources.kernel_providers["distillation"].materialize(request)


def _first_transformed_request(resources, plan) -> KernelMaterializationRequest:
    return next(
        request
        for request in resources.bind(plan).materializations.values()
        if isinstance(request, KernelMaterializationRequest)
        and any(field.operations for field in request.recipe.projected_fields)
    )


@pytest.mark.parametrize(
    "transformation",
    (
        "derivative(u('y','a','s2'), '+x')",
        "transport(u('y','a','s2'), '+x')",
    ),
)
def test_gauge_covariant_job_fields_require_explicit_gauge_links(
    tmp_path,
    transformation,
):
    spec = _job_spec(tmp_path)
    _write_transformed_operator(spec.operators.file, transformation)
    _write_eigenvectors(tmp_path / "evec_0.h5")
    plan = compile_job_plan(spec)
    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context=spec.time_contexts[0],
    )
    request = _first_transformed_request(resources, plan)

    with pytest.raises(JobDataError, match="data.gauge_links"):
        resources.kernel_providers["distillation"].materialize(request)


def test_pure_shift_job_field_executes_without_gauge_links(tmp_path):
    spec = _job_spec(tmp_path)
    _write_transformed_operator(
        spec.operators.file,
        "shift(u('y','a','s2'), '+x', '-y')",
    )
    _write_eigenvectors(tmp_path / "evec_0.h5")
    plan = compile_job_plan(spec)
    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context=spec.time_contexts[0],
    )
    request = _first_transformed_request(resources, plan)

    value = resources.kernel_providers["distillation"].materialize(request)

    assert value.dtype == torch.complex128
    assert torch.isfinite(value).all()


def test_native_laph_gauge_derivative_matches_explicit_identity_link_formula(tmp_path):
    spec = _job_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["data"]["gauge_links"] = {
        "provider": "laph_iomap",
        "path": str(tmp_path / "smeared_gauge_{configuration}"),
    }
    spec = JobSpec.model_validate(payload)
    _write_transformed_operator(
        spec.operators.file,
        "derivative(u('y','a','s2'), '+x')",
    )
    raw = _write_eigenvectors(tmp_path / "evec_0.h5")
    _write_identity_laph_gauge(tmp_path / "smeared_gauge_0")
    plan = compile_job_plan(spec)
    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context=spec.time_contexts[0],
    )
    request = _first_transformed_request(resources, plan)
    provider = resources.kernel_providers["distillation"]
    projected = next(
        field for field in request.recipe.projected_fields if field.operations
    )
    operation = projected.operations[0]
    direction = str(operation.parameters["direction"])
    operation_payload = {
        "operations": [{"kind": operation.kind.value, "direction": direction}]
    }

    prepared = provider._tensor_provider_for_request(request)
    actual = prepared.eigenvector(
        projected.record.flavor,
        field_attributes=operation_payload,
    )
    time = request.operator_times[0]
    raw_field = torch.as_tensor(raw[time], dtype=torch.complex128).unsqueeze(0)
    sign = 1 if direction.startswith("+") else -1
    expected = 0.5 * sign * (
        torch.roll(raw_field, shifts=-1, dims=3)
        - torch.roll(raw_field, shifts=1, dims=3)
    )
    expected = expected.reshape(1, -1, 3, NV)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.isfinite(provider.materialize(request)).all()


def test_generalized_resolver_receives_exact_resource_recipe_and_configuration(tmp_path):
    operator_file = tmp_path / "operators.py"
    operator_file.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, c, cbar, s, sbar, u, ubar, gamma5, gammaX, point",
                "def build():",
                "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * s('y','a','s2'))",
                "    current = Operator(point(sbar('z','b','s3')) * gammaX('s3','s4') * point(u('z','b','s4')), momentum=(1,0,0))",
                "    source = Operator(ubar('x','c','s5') * gamma5('s5','s6') * c('x','c','s6'))",
                "    return {'C3': Correlator(sink, current, source)}",
            ]
        ),
        encoding="utf-8",
    )
    resolver_file = tmp_path / "resolver.py"
    resolver_file.write_text(
        "\n".join(
            [
                "from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource",
                "def resolve_generalized(*, resource_id, recipe, configuration):",
                "    assert recipe.resource_id == resource_id",
                "    assert recipe.is_generalized",
                "    return ExternalTensorSource(",
                "        source_id=f'cfg{configuration}:{resource_id}',",
                "        kind='hdf5',",
                "        location=f'/tmp/generalized_{configuration}.h5',",
                "        dataset='/tau_J_{source_time}_{insertion_time_0}_{sink_time}',",
                "    )",
            ]
        ),
        encoding="utf-8",
    )
    spec = JobSpec.model_validate(
        {
            "schema_version": 1,
            "job": {"name": "generalized_data"},
            "operators": {"file": str(operator_file), "factory": "build"},
            "plan": {"file": str(tmp_path / "three.concord")},
            "lattice": {"spatial_extent": [LX, LY, LZ], "temporal_extent": T},
            "compile": {"mode_size": NV},
            "run": {
                "device": "cpu",
                "configurations": {"values": [4]},
                "times": {
                    "generator": {
                        "type": "three_point",
                        "source": 1,
                        "sink": 5,
                        "insertions": {"values": [3]},
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
                    "generalized": {
                        "resolver": f"{resolver_file}:resolve_generalized",
                    },
                },
            },
            "output": {"file": str(tmp_path / "three.h5")},
        }
    )
    plan = compile_job_plan(spec)
    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=4,
        time_context=spec.time_contexts[0],
    )

    generalized_ids = {
        resource_id
        for resource_id, recipe in plan.required_propagation_resources.items()
        if recipe.is_generalized
    }
    assert generalized_ids
    for resource_id in generalized_ids:
        source = resources.propagation_sources[resource_id]
        assert source.source_id == f"cfg4:{resource_id}"
        assert source.location == "/tmp/generalized_4.h5"


def test_job_runtime_resources_preserve_dp_and_qdagger_conventions(tmp_path):
    spec = _job_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["compile"]["spin_basis"] = "DP"
    payload["data"]["propagation"]["ordinary"]["source_field"] = "qdagger"
    spec = JobSpec.model_validate(payload)
    plan = compile_job_plan(spec)

    resources = build_job_runtime_resources(
        spec,
        plan,
        configuration=0,
        time_context="2pt_src_1_snk_3",
    )
    assert all(
        source.metadata["spin_basis"] == "DP"
        and source.metadata["source_field"] == "qdagger"
        for source in resources.propagation_sources.values()
    )
    assert resources.bind(plan).spin_basis == "DP"


def test_persistent_kernel_cache_reuses_valid_entry_and_invalidates_changed_input(
    tmp_path,
    monkeypatch,
):
    spec = _job_spec(tmp_path, momentum=(1, 0, 0))
    payload = spec.model_dump(mode="python")
    cache_directory = tmp_path / "kernel-cache"
    payload["data"]["kernel_cache"] = {"directory": str(cache_directory)}
    spec = JobSpec.model_validate(payload)
    _write_eigenvectors(tmp_path / "evec_0.h5")
    plan = compile_job_plan(spec)
    uncached_payload = spec.model_dump(mode="python")
    uncached_payload["data"].pop("kernel_cache")
    uncached_spec = JobSpec.model_validate(uncached_payload)
    assert _run_fingerprint(uncached_spec, plan) == _run_fingerprint(spec, plan)
    packed_payload = spec.model_dump(mode="python")
    packed_payload["data"]["kernel_cache"]["packing"] = "antisym3"
    packed_spec = JobSpec.model_validate(packed_payload)
    assert _run_fingerprint(packed_spec, plan) != _run_fingerprint(spec, plan)
    packed_payload["data"]["kernel_cache"]["directory"] = str(tmp_path / "other-cache")
    assert _run_fingerprint(JobSpec.model_validate(packed_payload), plan) == _run_fingerprint(packed_spec, plan)

    def build_provider_and_request():
        resources = build_job_runtime_resources(
            spec,
            plan,
            configuration=0,
            time_context=spec.time_contexts[0],
        )
        binding = resources.bind(plan)
        request = next(
            request
            for request in binding.materializations.values()
            if isinstance(request, KernelMaterializationRequest)
            and (1, 0, 0) in tuple(request.recipe.momenta.values())
        )
        return resources.kernel_providers["distillation"], request

    provider, request = build_provider_and_request()
    reference = provider.materialize(request)
    assert provider.kernel_cache_statistics is not None
    assert provider.kernel_cache_statistics.hits == 0
    assert provider.kernel_cache_statistics.misses == 1
    assert provider.kernel_cache_statistics.writes == 1
    entry = provider._kernel_cache_entry(request)
    entry_path = provider.kernel_cache.entry_path(entry[0])
    assert entry_path.is_file()
    with h5py.File(entry_path, "r") as handle:
        values = handle["values"]
        assert tuple(values.shape) == tuple(request.expected_shape)
        assert values.attrs["scalar_type"] == request.scalar_type
        assert len(values.attrs["axes_json"]) > 2
        assert "recipe_signature" in handle.attrs["manifest_json"]

    reused_provider, reused_request = build_provider_and_request()
    monkeypatch.setattr(
        reused_provider._delegate,
        "materialize",
        lambda request: pytest.fail("valid persistent cache entry was not reused"),
    )
    reused = reused_provider.materialize(reused_request)
    torch.testing.assert_close(reused, reference, rtol=0.0, atol=0.0)
    assert reused_provider.kernel_cache_statistics is not None
    assert reused_provider.kernel_cache_statistics.hits == 1
    assert reused_provider.kernel_cache_statistics.misses == 0
    assert reused_provider.kernel_cache_statistics.writes == 0

    with h5py.File(tmp_path / "evec_0.h5", "r+") as handle:
        handle["evec"][...] *= 1.25
    changed_provider, changed_request = build_provider_and_request()
    original_materialize = changed_provider._delegate.materialize
    calls = 0

    def counted_materialize(request):
        nonlocal calls
        calls += 1
        return original_materialize(request)

    monkeypatch.setattr(
        changed_provider._delegate,
        "materialize",
        counted_materialize,
    )
    changed = changed_provider.materialize(changed_request)
    assert calls == 1
    assert not torch.allclose(changed, reference)
