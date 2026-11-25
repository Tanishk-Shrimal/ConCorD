"""`concord run` decodes a saved plan straight into the snapshot's read-only form."""

import gc
import hashlib
import json
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from concord_lqcd.compiler._validation import _has_static_validation
from concord_lqcd.job import PlanSerializationError, load_concord_plan, save_concord_plan
from concord_lqcd.job import plan as plan_module
from concord_lqcd.job._prepared import _PreparedPlanSnapshot, _SnapshotCopier, _load_prepared_plan
from concord_lqcd.job.plan import PhysicalExecutionPlan, ConcordPlan
from tests.test_prepared_plan_snapshot import _mappings, _payload_bytes, compiled  # noqa: F401


def _copied_types(monkeypatch):
    copied = []
    copy = _SnapshotCopier.copy

    def recording(self, value):
        copied.append(type(value))
        return copy(self, value)

    monkeypatch.setattr(_SnapshotCopier, "copy", recording)
    return copied


def _assert_read_only_snapshot(plan, expected_bytes):
    assert _payload_bytes(plan) == expected_bytes
    assert all(_has_static_validation(node) for node in (
        plan, plan.logical_dag, plan.static_analysis, plan.schedule, plan.physical_plan,
    ))
    assert plan.diagnostics == {}
    assert plan.contraction_order.source_dag is plan.global_cse
    assert plan.intermediate_dag.source_plan is plan.contraction_order
    assert plan.logical_dag.source_dag is plan.intermediate_dag
    assert plan.static_analysis.source_dag is plan.logical_dag
    assert plan.schedule.source_analysis is plan.static_analysis
    assert plan.physical_plan.source_schedule is plan.schedule
    maps = tuple(_mappings(plan))
    assert len(maps) > 10
    assert all(type(mapping) is type(MappingProxyType({})) for mapping in maps)


def test_run_loader_decodes_into_the_read_only_form_without_copying(compiled, tmp_path, monkeypatch):
    _, original = compiled
    path = tmp_path / "plan.concord"
    save_concord_plan(original, path)
    copied = _copied_types(monkeypatch)

    plan = _load_prepared_plan(path)

    assert PhysicalExecutionPlan not in copied
    _assert_read_only_snapshot(plan, _payload_bytes(original))
    assert _payload_bytes(plan) == _payload_bytes(_PreparedPlanSnapshot(load_concord_plan(path)).plan)


def test_plan_outside_the_read_only_form_is_copied_as_before(compiled, tmp_path, monkeypatch):
    _, original = compiled
    path = tmp_path / "plan.concord"
    save_concord_plan(original, path)
    monkeypatch.setattr(plan_module, "_FROZEN_DATACLASS_TYPES", frozenset())
    copied = _copied_types(monkeypatch)

    plan = _load_prepared_plan(path)

    assert PhysicalExecutionPlan in copied
    _assert_read_only_snapshot(plan, _payload_bytes(original))


def _encodings(monkeypatch):
    encoded = []
    original = ConcordPlan.execution_payload

    def recording(plan):
        encoded.append(plan)
        return original(plan)

    monkeypatch.setattr(ConcordPlan, "execution_payload", recording)
    return encoded


def _rewrite_plan_json(path, edit=None, *, indent=None):
    """Rewrite plan.json (edited, or formatted differently) and give the manifest its new SHA-256."""
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    payload = json.loads(members["plan.json"])
    if edit is not None:
        edit(payload)
    raw = json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False).encode("utf-8")
    manifest = json.loads(members["manifest.json"])
    manifest["plan_sha256"] = hashlib.sha256(raw).hexdigest()
    members["plan.json"] = raw
    members["manifest.json"] = json.dumps(manifest, sort_keys=True).encode("utf-8")
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_run_loader_proves_the_payload_while_decoding(compiled, tmp_path, monkeypatch):
    _, original = compiled
    path = tmp_path / "plan.concord"
    save_concord_plan(original, path)
    encoded = _encodings(monkeypatch)

    plan = _load_prepared_plan(path)

    assert encoded == []
    _assert_read_only_snapshot(plan, _payload_bytes(original))


def test_record_that_changes_its_values_gets_the_full_hash_check(compiled, tmp_path, monkeypatch):
    _, original = compiled
    path = tmp_path / "plan.concord"
    save_concord_plan(original, path)

    def changed(record, kwargs):
        raise plan_module._UnsupportedOwnedDecode("record constructor changed owned metadata")

    monkeypatch.setattr(plan_module, "_require_owned_fields", changed)
    encoded = _encodings(monkeypatch)

    plan = _load_prepared_plan(path)

    assert len(encoded) == 1
    _assert_read_only_snapshot(plan, _payload_bytes(original))


def test_canonical_formatting_is_the_only_check_the_run_loader_skips(compiled, tmp_path):
    _, original = compiled
    path = tmp_path / "plan.concord"
    save_concord_plan(original, path)
    _rewrite_plan_json(path, indent=1)

    with pytest.raises(PlanSerializationError, match="does not match manifest plan_sha256"):
        load_concord_plan(path)
    plan = _load_prepared_plan(path)
    _assert_read_only_snapshot(plan, _payload_bytes(original))


def _bump_capacity(payload):
    payload["physical"]["steps"]["$tuple"][0]["fields"]["active_capacity_after"] += 1


def _drop_term(payload):
    del payload["logical"]["terms"]["$tuple"][0]


@pytest.mark.parametrize("load", (load_concord_plan, _load_prepared_plan), ids=("load", "prepared"))
@pytest.mark.parametrize("edit", (_bump_capacity, _drop_term), ids=("capacity", "omitted-term"))
def test_rehashed_edits_are_rejected_by_both_loaders(compiled, tmp_path, load, edit):
    _, original = compiled
    path = tmp_path / "plan.concord"
    save_concord_plan(original, path)
    _rewrite_plan_json(path, edit)

    with pytest.raises(PlanSerializationError):
        load(path)


@pytest.mark.parametrize("encoded", (
    {"$list": [1, 2]},
    {"$tuple": [{"$list": []}]},
    {"$map": [["key", {"$list": []}]]},
), ids=("list", "list-in-tuple", "list-in-map"))
def test_lists_leave_the_read_only_form(encoded):
    with pytest.raises(plan_module._UnsupportedOwnedDecode):
        plan_module._decode_owned_value(encoded)
    assert plan_module._decode_value(encoded) is not None


@pytest.mark.parametrize("encoded", (
    1.5, {"$complex": [1.0, -2.0]}, {"$enum": "OperationKind", "value": "shift"},
    {"$tuple": [1, "a", None, True]}, {"$map": [[{"$tuple": [1, 2]}, "x"]]},
), ids=("float", "complex", "enum", "tuple", "map"))
def test_read_only_decode_gives_the_same_values(encoded):
    owned = plan_module._decode_owned_value(encoded)
    ordinary = plan_module._decode_value(encoded)

    assert owned == (MappingProxyType(ordinary) if isinstance(ordinary, dict) else ordinary)
    if isinstance(ordinary, dict):
        assert type(owned) is type(MappingProxyType({}))


@pytest.mark.parametrize("encoded", (
    float("nan"), [1], {"$tuple": 1}, {"$map": [[1]]}, {"$map": [[1, 2], [1, 3]]},
    {"$map": [[{"$map": []}, 1]]}, {"unknown": 1}, {"$type": "NoSuchRecord", "fields": {}},
))
def test_read_only_decode_rejects_what_the_ordinary_decode_rejects(encoded):
    with pytest.raises(PlanSerializationError) as ordinary:
        plan_module._decode_value(encoded)
    with pytest.raises(PlanSerializationError) as owned:
        plan_module._decode_owned_value(encoded)
    assert str(owned.value) == str(ordinary.value)


@pytest.mark.parametrize("load", (load_concord_plan, _load_prepared_plan), ids=("load", "prepared"))
@pytest.mark.parametrize("enabled", (True, False), ids=("gc-on", "gc-off"))
def test_loading_pauses_the_cycle_collector_and_restores_it(compiled, tmp_path, monkeypatch, load, enabled):
    _, original = compiled
    path = tmp_path / "plan.concord"
    save_concord_plan(original, path)
    during = []
    decode = plan_module._decode_execution_payload

    def recording(payload, **kwargs):
        during.append(gc.isenabled())
        return decode(payload, **kwargs)

    monkeypatch.setattr(plan_module, "_decode_execution_payload", recording)
    was_enabled = gc.isenabled()
    (gc.enable if enabled else gc.disable)()
    try:
        load(path)
        assert during and not any(during)
        assert gc.isenabled() is enabled

        path.write_bytes(b"not a plan archive")
        with pytest.raises(PlanSerializationError):
            load(path)
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if was_enabled else gc.disable)()
