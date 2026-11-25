"""Focused tests for portable V1 ``.concord`` plan serialization."""

from __future__ import annotations

from dataclasses import replace
from math import prod
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import torch

from concord_lqcd.backend.runtime.materialization import (
    KernelMaterializationProvider,
    PrimitiveMaterializer,
    PropagationSourceHandler,
)
from concord_lqcd.backend.runtime.numerical_executor import GenericNumericalExecutor
from concord_lqcd.backend.runtime.resource_binding import (
    ExternalTensorSource,
    build_resource_binding_plan,
)
from concord_lqcd.job import (
    JobSpec,
    EXECUTION_SCHEMA_VERSION,
    PLAN_FORMAT_VERSION,
    PlanSerializationError,
    StalePlanError,
    compile_job_plan,
    load_concord_plan,
    save_concord_plan,
    validate_plan_for_job,
    write_job_plan,
)
from concord_lqcd.job._prepared import _load_prepared_plan


# `concord run` loads through _load_prepared_plan, which checks only the snapshot.
LOADERS = pytest.mark.parametrize(
    "load", (load_concord_plan, _load_prepared_plan), ids=("load", "prepared")
)


def _write_two_point_operator(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, c, cbar, u, ubar, gamma5",
                "",
                "def build(momentum=(0,0,0)):",
                "    p = tuple(momentum)",
                "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * u('y','a','s2'), momentum=p)",
                "    source = Operator(ubar('x','b','s3') * gamma5('s3','s4') * c('x','b','s4'), momentum=tuple(-x for x in p))",
                "    return {'D': Correlator(sink, source)}",
            ]
        ),
        encoding="utf-8",
    )


def _write_three_point_operator(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from concord_lqcd.core import Correlator, Operator, c, cbar, s, sbar, u, ubar, gamma5, gammaX, point",
                "",
                "def build():",
                "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * s('y','a','s2'), momentum=(0,0,0))",
                "    current = Operator(point(sbar('z','b','s3')) * gammaX('s3','s4') * point(u('z','b','s4')), momentum=(1,0,0))",
                "    source = Operator(ubar('x','c','s5') * gamma5('s5','s6') * c('x','c','s6'), momentum=(-1,0,0))",
                "    return {'C3': Correlator(sink, current, source)}",
            ]
        ),
        encoding="utf-8",
    )


def _job_spec(tmp_path: Path, *, three_point: bool = False) -> JobSpec:
    operator_path = tmp_path / "operators.py"
    if three_point:
        _write_three_point_operator(operator_path)
        times = {
            "generator": {
                "type": "three_point",
                "source": 1,
                "sink": 5,
                "insertions": {"values": [3]},
            }
        }
        parameters = {}
    else:
        _write_two_point_operator(operator_path)
        times = {
            "generator": {
                "type": "two_point",
                "source": 1,
                "sinks": {"values": [3]},
            }
        }
        parameters = {"momentum": [0, 0, 0]}

    payload = {
        "schema_version": 1,
        "job": {"name": "portable_plan_test"},
        "operators": {
            "file": str(operator_path),
            "factory": "build",
            "parameters": parameters,
        },
        "plan": {"file": str(tmp_path / "portable.concord")},
        "lattice": {
            "spatial_extent": [4, 4, 4],
            "temporal_extent": 8,
        },
        "compile": {
            "mode_size": 2,
            "scalar_type": "complex128",
            "contraction": {"strategy": "greedy"},
            "scheduling": {"strategy": "memory_aware"},
            "physical": {"reuse_strategy": "best_fit"},
        },
        "run": {
            "device": "cpu",
            "configurations": {"values": [0]},
            "times": times,
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
                "generalized": (
                    {"resolver": "my_resolvers:resolve_generalized"}
                    if three_point
                    else None
                ),
            },
        },
        "output": {"file": str(tmp_path / "result.h5")},
    }
    if not three_point:
        payload["data"]["propagation"].pop("generalized")
    return JobSpec.model_validate(payload)


class _KernelProvider(KernelMaterializationProvider):
    def materialize(self, request):
        count = prod(request.expected_shape) if request.expected_shape else 1
        offset = sum(ord(ch) for ch in request.resource_id) % 13
        offset += sum(request.operator_times)
        data = torch.arange(count, dtype=torch.float64).reshape(request.expected_shape)
        return (data + offset + 1).to(torch.complex128)


class _PropagationHandler(PropagationSourceHandler):
    def materialize(self, request):
        count = prod(request.expected_shape) if request.expected_shape else 1
        offset = sum(ord(ch) for ch in request.resource_id) % 17
        offset += 10 * request.source_time + request.sink_time
        offset += sum(request.insertion_times)
        data = torch.arange(count, dtype=torch.float64).reshape(request.expected_shape)
        return (data / 19.0 + offset + 1).to(torch.complex128)


def _execute(plan, operator_times):
    sources = {
        resource_id: ExternalTensorSource(
            source_id=f"synthetic:{resource_id}",
            kind="synthetic",
            location=resource_id,
        )
        for resource_id in plan.required_propagation_resources
    }
    binding = build_resource_binding_plan(
        plan.physical_plan,
        operator_times=operator_times,
        propagation_sources=sources,
    )
    materializer = PrimitiveMaterializer(
        binding,
        kernel_providers={"distillation": _KernelProvider()},
        source_handlers={"synthetic": _PropagationHandler()},
    )
    return GenericNumericalExecutor(materializer, device="cpu").execute()


def test_two_point_plan_roundtrip_preserves_exact_executable_graph(tmp_path):
    spec = _job_spec(tmp_path)
    original = compile_job_plan(spec)
    path = save_concord_plan(original, spec.plan.file)
    loaded = load_concord_plan(path)

    assert loaded.manifest.plan_format_version == PLAN_FORMAT_VERSION
    assert loaded.manifest.plan_sha256 == original.manifest.plan_sha256
    assert loaded.correlator_ids == ("D",)
    assert loaded.resources == original.resources
    assert original.config.shared_orthonormal_laph_basis is True
    assert loaded.config.shared_orthonormal_laph_basis is True
    assert loaded.required_kernel_resource_ids == ()

    assert loaded.logical_dag.primitives == original.logical_dag.primitives
    assert loaded.logical_dag.contractions == original.logical_dag.contractions
    assert loaded.logical_dag.terms == original.logical_dag.terms
    assert loaded.logical_dag.correlators == original.logical_dag.correlators
    assert loaded.logical_dag.outputs == original.logical_dag.outputs

    assert loaded.static_analysis.values == original.static_analysis.values
    assert loaded.static_analysis.dependencies == original.static_analysis.dependencies
    assert loaded.static_analysis.topological_order == original.static_analysis.topological_order

    assert loaded.schedule.steps == original.schedule.steps
    assert loaded.schedule.lifetimes == original.schedule.lifetimes
    assert loaded.schedule.output_nodes == original.schedule.output_nodes

    assert loaded.physical_plan.buffers == original.physical_plan.buffers
    assert loaded.physical_plan.value_bindings == original.physical_plan.value_bindings
    assert loaded.physical_plan.steps == original.physical_plan.steps
    assert loaded.physical_plan.outputs == original.physical_plan.outputs
    loaded.validate()


def test_size_matched_plan_roundtrip_preserves_results(tmp_path):
    original_spec = _job_spec(tmp_path)
    original = compile_job_plan(original_spec)
    payload = original_spec.model_dump(mode="python")
    payload["compile"]["physical"]["reuse_strategy"] = "size_matched"
    payload["plan"]["file"] = str(tmp_path / "size_matched.concord")
    spec = JobSpec.model_validate(payload)
    plan = compile_job_plan(spec)
    loaded = load_concord_plan(save_concord_plan(plan, spec.plan.file))

    validate_plan_for_job(loaded, spec)
    assert loaded.physical_plan.reuse_strategy == "size_matched"
    assert all(
        loaded.physical_plan.buffer(binding.buffer_id).capacity_bytes == binding.size_bytes
        for binding in loaded.physical_plan.value_bindings.values()
    )
    original_result = _execute(original, {2: (3, 1)})
    loaded_result = _execute(loaded, {2: (3, 1)})
    for correlator_id in original.correlator_ids:
        torch.testing.assert_close(
            original_result.output(correlator_id),
            loaded_result.output(correlator_id),
            rtol=0,
            atol=0,
        )


def test_loaded_plan_binds_resources_and_executes_identically_without_recompilation(tmp_path):
    spec = _job_spec(tmp_path)
    original = compile_job_plan(spec)
    save_concord_plan(original, spec.plan.file)
    loaded = load_concord_plan(spec.plan.file)

    original_result = _execute(original, {2: (3, 1)})
    loaded_result = _execute(loaded, {2: (3, 1)})

    assert original_result.executed_steps == loaded_result.executed_steps
    assert set(original_result.outputs) == {"D"}
    assert set(loaded_result.outputs) == {"D"}
    assert torch.allclose(original_result.output("D"), loaded_result.output("D"))


def test_three_point_generalized_resource_recipe_survives_roundtrip(tmp_path):
    spec = _job_spec(tmp_path, three_point=True)
    original = compile_job_plan(spec)
    assert any(recipe.is_generalized for recipe in original.required_propagation_resources.values())

    save_concord_plan(original, spec.plan.file)
    loaded = load_concord_plan(spec.plan.file)

    assert set(loaded.required_propagation_resources) == set(original.required_propagation_resources)
    for resource_id, recipe in loaded.required_propagation_resources.items():
        assert recipe.resource_id == resource_id
        assert recipe == original.required_propagation_resources[resource_id]


def test_plan_archive_is_declarative_three_member_zip_and_not_pickle(tmp_path):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)

    with ZipFile(spec.plan.file, "r") as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "plan.json",
            "diagnostics.json",
        }
        plan_text = archive.read("plan.json").decode("utf-8")
        assert "pickle" not in plan_text.lower()
        assert "resource_id" in plan_text
        assert "PhysicalPlanStep" in plan_text


@LOADERS
def test_plan_json_integrity_hash_detects_modification(tmp_path, load):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)

    with ZipFile(spec.plan.file, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["plan.json"] = members["plan.json"] + b" "
    with ZipFile(spec.plan.file, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    with pytest.raises(PlanSerializationError, match="SHA-256"):
        load(spec.plan.file)


@LOADERS
def test_rehashed_plan_cannot_change_logical_wick_coefficients_without_changing_source(
    tmp_path, load
):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)

    with ZipFile(spec.plan.file, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    payload = json.loads(members["plan.json"])
    term = payload["logical"]["terms"]["$tuple"][0]["fields"]
    term["coefficient"] = {"$complex": [2.0, 3.0]}
    members["plan.json"] = json.dumps(payload).encode("utf-8")
    manifest = json.loads(members["manifest.json"])
    manifest["plan_sha256"] = hashlib.sha256(members["plan.json"]).hexdigest()
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with ZipFile(spec.plan.file, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    with pytest.raises(PlanSerializationError, match="compiler source"):
        load(spec.plan.file)


@LOADERS
@pytest.mark.parametrize("metadata", ("static_shape", "schedule_release", "physical_capacity"))
def test_rehashed_plan_rejects_inconsistent_static_or_release_metadata(tmp_path, metadata, load):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)
    with ZipFile(spec.plan.file, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    payload = json.loads(members["plan.json"])
    if metadata == "static_shape":
        descriptor = next(value["fields"] for _, value in payload["static"]["values"]["$map"]
                          if value["fields"]["shape"]["$tuple"])
        descriptor["shape"]["$tuple"][0] += 1
        descriptor["element_count"] = prod(descriptor["shape"]["$tuple"])
        descriptor["size_bytes"] = descriptor["element_count"] * descriptor["element_bytes"]
        message = "Static analysis"
    elif metadata == "schedule_release":
        step = next(value["fields"] for value in payload["schedule"]["steps"]["$tuple"]
                    if value["fields"]["released_values"]["$tuple"])
        step["released_values"] = {"$tuple": []}
        message = "releases disagree"
    else:
        step = payload["physical"]["steps"]["$tuple"][0]["fields"]
        step["active_capacity_before"] += 16
        message = "active capacity before"
    members["plan.json"] = json.dumps(payload).encode("utf-8")
    manifest = json.loads(members["manifest.json"])
    manifest["plan_sha256"] = hashlib.sha256(members["plan.json"]).hexdigest()
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with ZipFile(spec.plan.file, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(PlanSerializationError, match=message):
        load(spec.plan.file)


def test_plan_fingerprint_detects_operator_source_change(tmp_path):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    validate_plan_for_job(plan, spec)

    with spec.operators.file.open("a", encoding="utf-8") as handle:
        handle.write("\n# changed after planning\n")

    with pytest.raises(StalePlanError, match="stale"):
        validate_plan_for_job(plan, spec)


@pytest.mark.parametrize("semantics_version", (None, 1, 2, 3, 999))
def test_old_compiler_semantics_are_rejected_before_graph_decode(tmp_path, monkeypatch, semantics_version):
    import concord_lqcd.job.plan as plan_module

    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)
    with ZipFile(spec.plan.file, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    if semantics_version is None:
        del manifest["compiler_semantics_version"]
    else:
        manifest["compiler_semantics_version"] = semantics_version
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with ZipFile(spec.plan.file, "w", compression=ZIP_DEFLATED) as archive:
        for name, values in members.items():
            archive.writestr(name, values)

    def forbidden_decode(*arguments):
        pytest.fail("an incompatible plan reached graph decoding")

    monkeypatch.setattr(plan_module, "_decode_execution_payload", forbidden_decode)
    with pytest.raises(StalePlanError, match="[Rr]ecompile"):
        load_concord_plan(spec.plan.file)


@pytest.mark.parametrize("version", (1, 3))
def test_in_memory_plan_with_old_semantics_is_rejected(tmp_path, version):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    incompatible = replace(plan, manifest=replace(plan.manifest, compiler_semantics_version=version))
    with pytest.raises(StalePlanError, match="incompatible compiler semantics"):
        validate_plan_for_job(incompatible, spec)


@pytest.mark.parametrize("version", (True, 2.0, "2"))
def test_compiler_semantics_version_is_strictly_integral(tmp_path, version):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    incompatible = replace(plan, manifest=replace(plan.manifest, compiler_semantics_version=version))
    with pytest.raises(PlanSerializationError, match="semantics version must be an integer"):
        incompatible.validate()


def test_compiler_semantics_enters_job_fingerprint(tmp_path, monkeypatch):
    import concord_lqcd.job.plan as plan_module

    spec = _job_spec(tmp_path)
    original = plan_module.compute_job_fingerprint(spec)
    monkeypatch.setattr(plan_module, "COMPILER_SEMANTICS_VERSION", plan_module.COMPILER_SEMANTICS_VERSION + 1)
    changed = plan_module.compute_job_fingerprint(spec)
    assert changed.operator_source_sha256 == original.operator_source_sha256
    assert changed.compile_config_sha256 == original.compile_config_sha256
    assert changed.job_fingerprint != original.job_fingerprint


def test_write_job_plan_uses_job_plan_path_and_refuses_accidental_overwrite(tmp_path):
    spec = _job_spec(tmp_path)
    plan = write_job_plan(spec)
    assert spec.plan.file.exists()
    assert load_concord_plan(spec.plan.file).manifest.plan_sha256 == plan.manifest.plan_sha256

    with pytest.raises(FileExistsError):
        write_job_plan(spec)

    replacement = write_job_plan(spec, overwrite=True)
    assert load_concord_plan(spec.plan.file).manifest.plan_sha256 == replacement.manifest.plan_sha256


def test_precontract_execution_schema_requires_plan_regeneration(tmp_path):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)
    with ZipFile(spec.plan.file, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}

    payload = json.loads(members["plan.json"])
    payload["execution_schema_version"] = EXECUTION_SCHEMA_VERSION - 1
    members["plan.json"] = json.dumps(payload).encode("utf-8")
    manifest = json.loads(members["manifest.json"])
    manifest["plan_sha256"] = hashlib.sha256(members["plan.json"]).hexdigest()
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with ZipFile(spec.plan.file, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    with pytest.raises(PlanSerializationError, match="regenerate the plan"):
        load_concord_plan(spec.plan.file)


def test_plan_cannot_reuse_a_shared_basis_declaration_with_a_custom_adapter(tmp_path):
    spec = _job_spec(tmp_path)
    plan = compile_job_plan(spec)
    payload = spec.model_dump(mode="python")
    payload["data"]["eigenvectors"] = {
        "provider": "custom", "factory": "application:build_eigenvectors"
    }
    custom_spec = JobSpec.model_validate(payload)

    with pytest.raises(StalePlanError, match="stale"):
        validate_plan_for_job(plan, custom_spec)

    custom_plan = compile_job_plan(custom_spec)
    assert custom_plan.config.shared_orthonormal_laph_basis is False
    assert len(custom_plan.required_kernel_resource_ids) == 2
    loaded = load_concord_plan(save_concord_plan(custom_plan, custom_spec.plan.file))
    assert loaded.config.shared_orthonormal_laph_basis is False
    assert len(loaded.required_kernel_resource_ids) == 2
