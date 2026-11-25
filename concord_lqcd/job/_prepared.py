"""Owned execution metadata with object-local static-validation evidence.

Preparation validates a private read-only snapshot instead of certifying a
caller-owned mutable graph. Diagnostic reports stay on the original plan:
they are not execution instructions and do not enter its execution hash.
No scientific tensors, bindings, hardware decisions or freshness checks are
cached here. Ordinary, caller-owned plans retain full strict validation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType

from concord_lqcd.compiler._validation import _has_static_validation, _record_static_validation

from .plan import (
    GlobalComputationDAG,
    GlobalContractionOrderPlan,
    GlobalIntermediateDAG,
    LogicalExecutionDAG,
    LogicalSchedule,
    PhysicalExecutionPlan,
    PlanManifest,
    PlanSerializationError,
    StaticExecutionAnalysis,
    ConcordPlan,
    _DATACLASS_TYPES,
    _ENUM_TYPES,
    _UnsupportedOwnedDecode,
    _gc_paused,
    _read_concord_plan,
)


_SNAPSHOT_RECORD_TYPES = frozenset((
    *_DATACLASS_TYPES,
    PlanManifest,
    GlobalComputationDAG,
    GlobalContractionOrderPlan,
    GlobalIntermediateDAG,
    LogicalExecutionDAG,
    StaticExecutionAnalysis,
    LogicalSchedule,
    PhysicalExecutionPlan,
))
_IMMUTABLE_VALUE_TYPES = frozenset((type(None), bool, int, float, complex, str))
_MAPPING_TYPES = frozenset((dict, type(MappingProxyType({}))))
_FIELD_NAMES = {
    record_type: tuple(item.name for item in fields(record_type))
    for record_type in _SNAPSHOT_RECORD_TYPES
}


class _UnsupportedPlanSnapshot(TypeError):
    """This graph must continue through ordinary strict validation."""


class _SnapshotCopier:
    def __init__(self) -> None:
        self._copies: dict[int, tuple[object, object]] = {}
        self._active: set[int] = set()

    def copy(self, value: object) -> object:
        value_type = type(value)
        if value_type in _IMMUTABLE_VALUE_TYPES or value_type in _ENUM_TYPES:
            return value
        identity = id(value)
        if identity in self._copies:
            return self._copies[identity][1]
        if identity in self._active:
            raise _UnsupportedPlanSnapshot("cyclic execution metadata cannot be prepared")
        self._active.add(identity)
        try:
            if value_type in _MAPPING_TYPES:
                owned = MappingProxyType({
                    self.copy(key): self.copy(item) for key, item in value.items()
                })
            elif value_type is tuple:
                owned = tuple(self.copy(item) for item in value)
            elif value_type in _SNAPSHOT_RECORD_TYPES:
                if not value_type.__dataclass_params__.frozen:
                    raise _UnsupportedPlanSnapshot(
                        f"mutable record {value_type.__name__} cannot be prepared"
                    )
                arguments = {
                    name: self.copy(getattr(value, name))
                    for name in _FIELD_NAMES[value_type]
                }
                owned = value_type(**arguments)
                if any(getattr(owned, name) is not item for name, item in arguments.items()):
                    raise _UnsupportedPlanSnapshot(
                        f"record constructor {value_type.__name__} changed owned metadata"
                    )
            else:
                raise _UnsupportedPlanSnapshot(
                    f"unsupported execution metadata type {value_type.__name__}"
                )
            self._copies[identity] = (value, owned)
            return owned
        finally:
            self._active.remove(identity)


@dataclass(frozen=True, slots=True, init=False)
class _PreparedPlanSnapshot:
    """An execution-only view validated after taking ownership of its graph.

    Static evidence never bypasses a runtime binding or numerical check.
    Reconstructing a snapshot always validates its own instructions. Copying
    only declared fields also excludes caller-owned derived property caches.
    """

    plan: ConcordPlan

    def __init__(self, plan: ConcordPlan) -> None:
        if not isinstance(plan, ConcordPlan):
            raise TypeError("plan must be ConcordPlan")
        if type(plan) is not ConcordPlan:
            raise _UnsupportedPlanSnapshot("custom plan types require strict validation")
        copier = _SnapshotCopier()
        owned = ConcordPlan(
            manifest=copier.copy(plan.manifest),
            config=copier.copy(plan.config),
            physical_plan=copier.copy(plan.physical_plan),
            diagnostics=MappingProxyType({}),
        )
        _validate_owned(owned)
        object.__setattr__(self, "plan", owned)

    @classmethod
    def _from_owned_decode(cls, plan: ConcordPlan) -> "_PreparedPlanSnapshot":
        """Take a plan decoded straight into the read-only form, without copying it.

        ``plan`` must come from ``_read_concord_plan(..., owned=True)``: a
        private graph that nobody else references, in the form the copier
        would produce, whose payload was proven against the manifest's SHA-256
        record by record while decoding. Only the small manifest is copied;
        every other check runs as usual.
        """

        owned = ConcordPlan(
            manifest=_SnapshotCopier().copy(plan.manifest),
            config=plan.config,
            physical_plan=plan.physical_plan,
            diagnostics=MappingProxyType({}),
        )
        _validate_owned(owned, payload_proven=True)
        snapshot = object.__new__(cls)
        object.__setattr__(snapshot, "plan", owned)
        return snapshot


def _validate_owned(owned: ConcordPlan, *, payload_proven: bool = False) -> None:
    if payload_proven:
        owned._validate(payload_proven=True)
    else:
        owned.validate()
    for node in (
        owned.logical_dag, owned.static_analysis, owned.schedule,
        owned.physical_plan, owned,
    ):
        _record_static_validation(node)


def _try_prepare_plan_snapshot(plan: ConcordPlan) -> _PreparedPlanSnapshot | None:
    """Return an owned snapshot, or None to require the ordinary strict path.

    Unsupported containers, custom classes and scientific arrays are not
    coerced into a different representation. Validation failures in supported
    graphs propagate; they are never treated as successful preparation.
    """

    try:
        return _PreparedPlanSnapshot(plan)
    except _UnsupportedPlanSnapshot:
        return None


def _is_prepared_plan(plan: object) -> bool:
    """True for the plan of a snapshot, whose checks need not run again."""

    return type(plan) is ConcordPlan and _has_static_validation(plan)


def _load_prepared_plan(path: str | Path) -> ConcordPlan:
    """Load a saved plan as a validated snapshot, checking it once.

    ``load_concord_plan`` validates the plan it decodes, and preparing a
    snapshot validates the copy again. The plan here is decoded straight into
    the snapshot's read-only form, private to this call, so it is neither
    copied nor checked twice: the same checks run once. A plan outside that
    form is decoded the ordinary way and copied, or validated as
    ``load_concord_plan`` does when it cannot be prepared.
    """

    with _gc_paused():
        try:
            decoded = _read_concord_plan(path, owned=True)
        except _UnsupportedOwnedDecode:
            decoded = None
        try:
            if decoded is not None:
                try:
                    return _PreparedPlanSnapshot._from_owned_decode(decoded).plan
                except _UnsupportedPlanSnapshot:
                    pass
            plan = _read_concord_plan(path)
            prepared = _try_prepare_plan_snapshot(plan)
            if prepared is None:
                plan.validate()
                return plan
        except PlanSerializationError:
            raise
        except Exception as exc:
            raise PlanSerializationError(
                f"decoded plan failed semantic validation: {exc}"
            ) from exc
    return prepared.plan
