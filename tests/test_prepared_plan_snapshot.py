"""Ownership and integrity gates for the private execution snapshot."""

from dataclasses import FrozenInstanceError, dataclass, fields, is_dataclass, replace
import hashlib
import os
from types import MappingProxyType

import numpy as np
import pytest
import torch

from concord_lqcd.compiler.distillation_lowering import KernelRecipe
from concord_lqcd.job import (
    PlanSerializationError,
    StalePlanError,
    ConcordPlan,
    compile_job_plan,
    load_concord_plan,
    save_concord_plan,
    validate_plan_for_job,
)
from concord_lqcd.job._prepared import (
    _PreparedPlanSnapshot,
    _SnapshotCopier,
    _UnsupportedPlanSnapshot,
    _try_prepare_plan_snapshot,
)
from concord_lqcd.job.plan import _canonical_json_bytes
from tests.test_plan_serialization import _execute, _job_spec


@pytest.fixture
def compiled(tmp_path):
    spec = _job_spec(tmp_path)
    spec = spec.model_copy(update={
        "operators": spec.operators.model_copy(update={
            "parameters": {"momentum": [1, 0, 0]},
        }),
    })
    spec.operators.file.write_text("from helper import build\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text(
        "from concord_lqcd.core import Correlator, Operator, c, cbar, u, ubar, gamma5\n"
        "def build(momentum=(1, 0, 0)):\n"
        "    sink = Operator(cbar('y','a','s1') * gamma5('s1','s2') * u('y','a','s2'), momentum=tuple(momentum))\n"
        "    source = Operator(ubar('x','b','s3') * gamma5('s3','s4') * c('x','b','s4'), momentum=tuple(-entry for entry in momentum))\n"
        "    return {'D': Correlator(sink, source)}\n",
        encoding="utf-8",
    )
    return spec, compile_job_plan(spec)


def _payload_bytes(plan):
    return _canonical_json_bytes(plan.execution_payload())


def _rehash(plan):
    manifest = replace(
        plan.manifest, plan_sha256=hashlib.sha256(_payload_bytes(plan)).hexdigest(),
    )
    return replace(plan, manifest=manifest)


def _replace_logical(plan, logical):
    analysis = replace(plan.static_analysis, source_dag=logical)
    schedule = replace(plan.schedule, source_analysis=analysis)
    return replace(plan, physical_plan=replace(plan.physical_plan, source_schedule=schedule))


def _mappings(value, seen=None):
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (dict, type(MappingProxyType({})))):
        yield value
        for key, item in value.items():
            yield from _mappings(key, seen)
            yield from _mappings(item, seen)
    elif is_dataclass(value):
        for record_field in fields(value):
            yield from _mappings(getattr(value, record_field.name), seen)
    elif isinstance(value, tuple):
        for item in value:
            yield from _mappings(item, seen)


def test_snapshot_keeps_execution_bytes_hash_order_and_source_identities(compiled):
    _, original = compiled
    expected = _payload_bytes(original)

    prepared = _try_prepare_plan_snapshot(original)

    assert prepared is not None
    owned = prepared.plan
    assert owned is not original
    assert owned.physical_plan is not original.physical_plan
    assert owned.manifest == original.manifest
    assert owned.manifest is not original.manifest
    assert _payload_bytes(owned) == expected
    assert hashlib.sha256(expected).hexdigest() == owned.manifest.plan_sha256
    assert tuple(owned.resources) == tuple(original.resources)
    assert owned.contraction_order.source_dag is owned.global_cse
    assert owned.intermediate_dag.source_plan is owned.contraction_order
    assert owned.logical_dag.source_dag is owned.intermediate_dag
    assert owned.static_analysis.source_dag is owned.logical_dag
    assert owned.schedule.source_analysis is owned.static_analysis
    assert owned.physical_plan.source_schedule is owned.schedule
    assert owned.diagnostics == {}
    assert original.diagnostics


def test_every_snapshot_mapping_is_owned_and_read_only(compiled):
    _, original = compiled
    owned = _PreparedPlanSnapshot(original).plan
    original_maps = {id(mapping) for mapping in _mappings(original)}
    owned_maps = tuple(_mappings(owned))

    assert len(owned_maps) > 10
    assert not original_maps.intersection(id(mapping) for mapping in owned_maps)
    for mapping in owned_maps:
        assert type(mapping) is type(MappingProxyType({}))
        with pytest.raises(TypeError):
            mapping["unexpected"] = None
    with pytest.raises(FrozenInstanceError):
        owned.manifest = original.manifest
    with pytest.raises(FrozenInstanceError):
        owned.physical_plan.steps[0].step_index = 500


@pytest.mark.parametrize("target", ("buffers", "outputs", "momenta", "dependencies"))
def test_original_nested_mutations_cannot_change_checked_instructions(compiled, target):
    _, original = compiled
    owned = _PreparedPlanSnapshot(original).plan
    expected = _payload_bytes(owned)

    if target == "buffers":
        original.physical_plan.buffers.clear()
    elif target == "outputs":
        original.logical_dag.outputs["D"] = "missing"
    elif target == "momenta":
        kernel = next(resource.payload for resource in original.resources.values()
                      if isinstance(resource.payload, KernelRecipe))
        position = kernel.position_labels[0]
        kernel.momenta[position] = (2, 0, 0)
    else:
        original.manifest.operator_dependencies["helper.py"] = "invalid"

    assert _payload_bytes(owned) == expected
    owned.validate()
    with pytest.raises(PlanSerializationError):
        original.validate()


def test_mapping_proxy_does_not_leave_an_alias_to_callers_backing_dict(compiled):
    _, original = compiled
    backing = dict(original.physical_plan.buffers)
    original = replace(original, physical_plan=replace(
        original.physical_plan, buffers=MappingProxyType(backing),
    ))
    owned = _PreparedPlanSnapshot(original).plan

    backing.clear()

    assert owned.physical_plan.buffers
    owned.validate()


def test_constructor_cannot_reintroduce_a_mutable_mapping(compiled, monkeypatch):
    _, original = compiled
    plan_type = type(original.physical_plan)
    original_init = plan_type.__init__

    def copy_to_mutable_mapping(record, *arguments, **keywords):
        original_init(record, *arguments, **keywords)
        object.__setattr__(record, "buffers", dict(record.buffers))

    monkeypatch.setattr(plan_type, "__init__", copy_to_mutable_mapping)

    assert _try_prepare_plan_snapshot(original) is None
    original.validate()


def test_copy_preserves_aliases_and_copies_nested_operation_parameters():
    from concord_lqcd.compiler.physics_ir import OperationKind, OperationRecord

    parameters = {"direction": "+x", "metadata": {"path": ("+x", "+y")}}
    operation = OperationRecord(OperationKind.DERIVATIVE, parameters)
    owned = _SnapshotCopier().copy((operation, operation, parameters))

    assert owned[0] is owned[1]
    assert owned[0].parameters is owned[2]
    parameters["direction"] = "+y"
    parameters["metadata"]["path"] = ("-x",)
    assert owned[0].parameters["direction"] == "+x"
    assert owned[0].parameters["metadata"]["path"] == ("+x", "+y")


def test_derived_indexes_are_rebuilt_from_owned_records(compiled):
    _, original = compiled
    logical = original.logical_dag
    logical.node(logical.terms[0].node_id)
    assert "_assembly_nodes" in logical.__dict__

    owned = _SnapshotCopier().copy(logical)

    assert "_assembly_nodes" not in owned.__dict__
    assert owned.node(owned.terms[0].node_id) is owned.terms[0]
    assert owned._assembly_nodes is not logical._assembly_nodes


def test_preparation_validates_exactly_the_owned_plan_once(compiled, monkeypatch):
    _, original = compiled
    original_validate = ConcordPlan.validate
    checked = []

    def record_validate(plan):
        checked.append(plan)
        return original_validate(plan)

    monkeypatch.setattr(ConcordPlan, "validate", record_validate)
    prepared = _try_prepare_plan_snapshot(original)

    assert prepared is not None
    assert len(checked) == 1
    assert checked[0] is prepared.plan
    assert checked[0] is not original


@pytest.mark.parametrize("damage", ("coefficient", "omitted_term", "capacity", "hash"))
def test_preparation_rejects_tampering_even_with_rehashed_metadata(compiled, damage):
    _, original = compiled
    if damage == "coefficient":
        logical = replace(original.logical_dag, terms=(replace(
            original.logical_dag.terms[0], coefficient=2 + 3j,
        ),) + original.logical_dag.terms[1:])
        damaged = _replace_logical(original, logical)
    elif damage == "omitted_term":
        damaged = _replace_logical(original, replace(original.logical_dag, terms=()))
    elif damage == "capacity":
        buffers = dict(original.physical_plan.buffers)
        buffer_id = next(iter(buffers))
        buffers[buffer_id] = replace(buffers[buffer_id], capacity_bytes=1)
        damaged = replace(original, physical_plan=replace(original.physical_plan, buffers=buffers))
    else:
        damaged = replace(original, manifest=replace(original.manifest, plan_sha256="0" * 64))
    if damage != "hash":
        damaged = _rehash(damaged)

    with pytest.raises(PlanSerializationError):
        _try_prepare_plan_snapshot(damaged)


def test_serializable_list_keeps_its_original_strict_path(compiled):
    _, original = compiled
    original = _rehash(replace(original, physical_plan=replace(
        original.physical_plan, steps=list(original.physical_plan.steps),
    )))
    expected = _payload_bytes(original)
    original.validate()

    assert _try_prepare_plan_snapshot(original) is None
    assert isinstance(original.physical_plan.steps, list)
    assert _payload_bytes(original) == expected
    original.validate()


def test_custom_mapping_keeps_its_original_strict_path(compiled):
    class CustomBuffers(dict):
        pass

    _, original = compiled
    original = replace(original, physical_plan=replace(
        original.physical_plan, buffers=CustomBuffers(original.physical_plan.buffers),
    ))

    assert _try_prepare_plan_snapshot(original) is None
    original.validate()


def test_custom_plan_type_keeps_its_original_strict_path(compiled):
    class CustomPlan(ConcordPlan):
        pass

    _, original = compiled
    custom = CustomPlan(
        original.manifest, original.config, original.physical_plan, original.diagnostics,
    )

    assert _try_prepare_plan_snapshot(custom) is None
    custom.validate()


@pytest.mark.parametrize("value", ([1, 2], {1, 2}, np.zeros(2), torch.zeros(2)))
def test_scientific_arrays_and_unknown_mutable_values_are_not_copied(value):
    with pytest.raises(_UnsupportedPlanSnapshot, match="unsupported"):
        _SnapshotCopier().copy(value)


def test_cyclic_metadata_is_not_prepared():
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(_UnsupportedPlanSnapshot, match="cyclic"):
        _SnapshotCopier().copy(cyclic)


def test_unknown_frozen_record_is_not_implicitly_trusted():
    @dataclass(frozen=True)
    class CustomRecord:
        data: dict

    with pytest.raises(_UnsupportedPlanSnapshot, match="CustomRecord"):
        _SnapshotCopier().copy(CustomRecord({"mutable": []}))


@pytest.mark.parametrize("value", (None, {}, object()))
def test_non_plan_arguments_are_errors(value):
    with pytest.raises(TypeError, match="ConcordPlan"):
        _try_prepare_plan_snapshot(value)


@pytest.mark.parametrize("change", ("source", "helper", "compile"))
def test_public_job_validation_still_detects_freshness_changes(compiled, change):
    spec, original = compiled
    owned = _PreparedPlanSnapshot(original).plan
    validate_plan_for_job(owned, spec)
    if change == "compile":
        spec = spec.model_copy(update={
            "compile": spec.compile.model_copy(update={"mode_size": 3}),
        })
    else:
        path = spec.operators.file if change == "source" else spec.operators.file.parent / "helper.py"
        previous_stat = path.stat()
        source = path.read_text(encoding="utf-8")
        changed = source.replace("helper", "absent") if change == "source" else source.replace("gamma5", "gammaX")
        assert len(changed) == len(source)
        path.write_text(changed, encoding="utf-8")
        os.utime(path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))

    with pytest.raises(StalePlanError):
        validate_plan_for_job(owned, spec)


def test_generalized_resource_snapshot_roundtrips_without_a_format_change(tmp_path):
    spec = _job_spec(tmp_path, three_point=True)
    original = compile_job_plan(spec)

    prepared = _try_prepare_plan_snapshot(original)

    assert prepared is not None
    assert any(recipe.is_generalized for recipe in prepared.plan.required_propagation_resources.values())
    assert _payload_bytes(prepared.plan) == _payload_bytes(original)
    saved = save_concord_plan(prepared.plan, spec.plan.file)
    restored = load_concord_plan(saved)
    assert restored.manifest.plan_sha256 == original.manifest.plan_sha256
    assert _payload_bytes(restored) == _payload_bytes(original)


def test_owned_instructions_execute_identically_for_different_time_contexts(compiled):
    _, original = compiled
    owned = _PreparedPlanSnapshot(original).plan
    results = []

    for operator_times in ({2: (3, 1)}, {2: (6, 2)}):
        reference = _execute(original, operator_times)
        actual = _execute(owned, operator_times)
        assert reference.executed_steps == actual.executed_steps
        for correlator_id in original.correlator_ids:
            expected = reference.output(correlator_id)
            assert torch.max(torch.abs(expected)).item() > 0
            torch.testing.assert_close(actual.output(correlator_id), expected, rtol=0, atol=0)
        results.append(actual.output("D"))

    assert not torch.equal(results[0], results[1])


@pytest.mark.parametrize("expression", (
    "derivative(u('y','a','s2'), '+x')",
    "transport(u('y','a','s2'), '+x', '-y')",
))
def test_displaced_kernel_operations_keep_their_order_and_payload(compiled, expression):
    spec, _ = compiled
    helper = spec.operators.file.parent / "helper.py"
    source = helper.read_text(encoding="utf-8")
    source = source.replace("gamma5\n", "gamma5, derivative, transport\n")
    source = source.replace("u('y','a','s2')", expression)
    helper.write_text(source, encoding="utf-8")
    original = compile_job_plan(spec)

    prepared = _try_prepare_plan_snapshot(original)

    assert prepared is not None
    assert _payload_bytes(prepared.plan) == _payload_bytes(original)
    operations = tuple(operation
        for resource in prepared.plan.resources.values()
        if isinstance(resource.payload, KernelRecipe)
        for field in resource.payload.projected_fields
        for operation in field.operations)
    assert operations
    for operation in operations:
        with pytest.raises(TypeError):
            operation.parameters["path"] = ("-z",)
