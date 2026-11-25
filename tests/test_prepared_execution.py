"""Static preparation never authorizes mutable bindings or runtime results."""

from copy import copy
from dataclasses import replace
import gc
from weakref import ref

import h5py
import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime.resource_binding import (
    build_resource_binding_plan,
    prepare_resource_binding_template,
)
from concord_lqcd.compiler._validation import _has_static_validation
from concord_lqcd.job import (
    JobRunError,
    PlanSerializationError,
    StalePlanError,
    ConcordPlan,
    compile_job_plan,
    load_concord_plan,
    run_job,
    save_concord_plan,
    validate_plan_for_job,
)
from concord_lqcd.job._prepared import _PreparedPlanSnapshot, _load_prepared_plan
from tests.test_job_runner import _spec, _write_data
from tests.test_plan_serialization import _execute
from tests.test_resource_binding import _sources_for


def _payload_counter(monkeypatch):
    original = ConcordPlan.execution_payload
    checked = []

    def encode(plan):
        checked.append(plan)
        return original(plan)

    monkeypatch.setattr(ConcordPlan, "execution_payload", encode)
    return checked


def _validation_counter(monkeypatch):
    original = ConcordPlan._validate
    checked = []

    def validate(plan, *, payload_proven):
        if not _has_static_validation(plan):  # checked plans return at once
            checked.append((plan, payload_proven))
        return original(plan, payload_proven=payload_proven)

    monkeypatch.setattr(ConcordPlan, "_validate", validate)
    return checked


def _static_nodes(plan):
    return (plan, plan.physical_plan, plan.schedule, plan.static_analysis, plan.logical_dag)


def test_only_owned_static_nodes_reuse_integrity_validation(tmp_path, monkeypatch):
    original = compile_job_plan(_spec(tmp_path))
    checked = _payload_counter(monkeypatch)
    owned = _PreparedPlanSnapshot(original).plan

    assert len(checked) == 1
    assert checked[0] is owned
    assert all(_has_static_validation(node) for node in _static_nodes(owned))
    assert not any(_has_static_validation(node) for node in _static_nodes(original))
    for _ in range(3):
        for node in _static_nodes(owned):
            node.validate()
    assert len(checked) == 1
    original.validate()
    original.validate()
    assert len(checked) == 3


def test_shallow_copy_cannot_inherit_static_evidence(tmp_path, monkeypatch):
    owned = _PreparedPlanSnapshot(compile_job_plan(_spec(tmp_path))).plan
    other = copy(owned)
    checked = _payload_counter(monkeypatch)

    assert not _has_static_validation(other)
    other.validate()
    assert len(checked) == 1
    assert checked[0] is other
    with pytest.raises(PlanSerializationError):
        replace(other, manifest=replace(other.manifest, plan_sha256="0" * 64)).validate()


def test_replaced_physical_records_do_not_inherit_static_evidence(tmp_path):
    owned = _PreparedPlanSnapshot(compile_job_plan(_spec(tmp_path))).plan
    physical = owned.physical_plan
    first = physical.steps[0]
    altered = replace(physical, steps=(replace(
        first, active_capacity_before=first.active_capacity_before + 16,
    ),) + physical.steps[1:])

    assert not _has_static_validation(altered)
    with pytest.raises(ValueError):
        altered.validate()


def test_archive_never_persists_a_validation_receipt(tmp_path):
    spec = _spec(tmp_path)
    owned = _PreparedPlanSnapshot(compile_job_plan(spec)).plan
    loaded = load_concord_plan(save_concord_plan(owned, spec.plan.file))

    assert not any(_has_static_validation(node) for node in _static_nodes(loaded))
    assert loaded.execution_payload() == owned.execution_payload()
    loaded.validate()


def test_receipts_do_not_keep_finished_plans_alive(tmp_path):
    owned = _PreparedPlanSnapshot(compile_job_plan(_spec(tmp_path))).plan
    reference = ref(owned)
    receipt = owned._concord_static_validation

    del owned
    gc.collect()

    assert reference() is None
    assert receipt.owner() is None


@pytest.mark.parametrize("prepared", (False, True))
def test_mutated_time_context_is_rechecked_after_binding_validation(tmp_path, prepared):
    plan = compile_job_plan(_spec(tmp_path))
    if prepared:
        plan = _PreparedPlanSnapshot(plan).plan
    binding = build_resource_binding_plan(
        plan.physical_plan,
        operator_times={2: (3, 1)},
        propagation_sources=_sources_for(plan.physical_plan),
    )
    binding.validate()
    binding.time_context.times_by_arity[2] = (4, 1)

    with pytest.raises(ValueError, match="binding/placement"):
        binding.validate()


def test_mutable_parent_is_rechecked_after_binding_validation(tmp_path):
    plan = compile_job_plan(_spec(tmp_path))
    binding = build_resource_binding_plan(
        plan.physical_plan, operator_times={2: (3, 1)},
        propagation_sources=_sources_for(plan.physical_plan),
    )
    binding.validate()
    plan.physical_plan.buffers.clear()

    with pytest.raises(ValueError):
        binding.validate()


def test_template_does_not_trust_a_later_changed_mutable_parent(tmp_path):
    plan = compile_job_plan(_spec(tmp_path))
    template = prepare_resource_binding_template(
        plan.physical_plan, propagation_sources=_sources_for(plan.physical_plan),
    )
    plan.physical_plan.buffers.clear()

    with pytest.raises(ValueError):
        template.bind({2: (3, 1)})


def test_numerical_outputs_are_checked_even_when_their_plan_is_prepared(tmp_path):
    owned = _PreparedPlanSnapshot(compile_job_plan(_spec(tmp_path))).plan
    result = _execute(owned, {2: (3, 1)})
    wrong = replace(result, outputs={"D": torch.ones(5, dtype=torch.complex128)})

    with pytest.raises(ValueError, match="shape"):
        wrong.validate()


def test_run_prepares_once_and_rebinds_distinct_configurations(tmp_path, monkeypatch):
    _write_data(tmp_path)
    spec = _spec(tmp_path)
    original = compile_job_plan(spec)
    checked = _payload_counter(monkeypatch)

    summary = run_job(spec, original)

    assert summary.executed_work_units == 4
    assert len(checked) == 1
    assert checked[0] is not original
    with h5py.File(spec.output.file, "r") as handle:
        assert np.all(handle["completed"][...])
        values = handle["correlators/0/values"][...]
        assert not np.array_equal(values[0], values[1])


def test_concord_run_checks_a_saved_plan_once(tmp_path, monkeypatch):
    # The CLI loads the plan, checks it against the job and passes it to run_job.
    _write_data(tmp_path)
    spec = _spec(tmp_path)
    save_concord_plan(compile_job_plan(spec), spec.plan.file)
    encoded = _payload_counter(monkeypatch)
    checked = _validation_counter(monkeypatch)

    plan = _load_prepared_plan(spec.plan.file)
    validate_plan_for_job(plan, spec)
    summary = run_job(spec, plan)

    assert summary.executed_work_units == 4
    # One full check; the payload was proven against the manifest while decoding.
    assert checked == [(plan, True)]
    assert encoded == []
    assert all(_has_static_validation(node) for node in _static_nodes(plan))


def test_run_job_checks_the_plan_it_loads_once(tmp_path, monkeypatch):
    _write_data(tmp_path)
    spec = _spec(tmp_path)
    save_concord_plan(compile_job_plan(spec), spec.plan.file)
    encoded = _payload_counter(monkeypatch)
    checked = _validation_counter(monkeypatch)

    assert run_job(spec).executed_work_units == 4
    assert len(checked) == 1
    assert checked[0][1] is True
    assert encoded == []


def test_unsupported_plan_stays_on_strict_execution_path(tmp_path, monkeypatch):
    class CustomBuffers(dict):
        pass

    _write_data(tmp_path)
    spec = _spec(tmp_path)
    original = compile_job_plan(spec)
    plan = replace(original, physical_plan=replace(
        original.physical_plan, buffers=CustomBuffers(original.physical_plan.buffers),
    ))
    checked = _payload_counter(monkeypatch)

    summary = run_job(spec, plan)

    assert summary.executed_work_units == 4
    assert len(checked) > 1
    assert all(node is plan for node in checked)
    assert not _has_static_validation(plan)


def test_source_change_between_configurations_still_fails(tmp_path, monkeypatch):
    import concord_lqcd.job.runner as runner

    _write_data(tmp_path)
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    build = runner._build_job_configuration_runtime_resources
    calls = []

    def build_and_edit(*arguments, **keywords):
        if calls:
            spec.operators.file.write_text(
                spec.operators.file.read_text() + "\nchanged_source = True\n",
            )
        calls.append(keywords["configuration"])
        return build(*arguments, **keywords)

    monkeypatch.setattr(runner, "_build_job_configuration_runtime_resources", build_and_edit)

    with pytest.raises((StalePlanError, JobRunError), match="stale"):
        run_job(spec, plan)
    assert calls == [0, 4]
    with h5py.File(spec.output.file, "r") as handle:
        assert np.all(handle["completed"][0])
        assert not np.any(handle["completed"][1])
