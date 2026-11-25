"""Portable, versioned ``.concord`` compiled-plan serialization.

This module is an application-layer persistence boundary around the frozen
compiler/runtime pipeline.  It stores the complete compiled state from global
CSE onward, including selected contraction plans, partial-intermediate CSE,
logical operations, static descriptors, logical scheduling, and physical buffer
planning.  It deliberately does not serialize symbolic frontend Python objects
and never uses pickle.

Loading a plan reconstructs those frozen dataclasses directly from declarative
JSON.  It does not rerun Wick contraction, global/partial CSE, contraction-path
search, static analysis, scheduling, or physical planning.

Version 2 archive layout::

    calculation.concord
      manifest.json
      plan.json
      diagnostics.json
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from concord_lqcd.compiler._validation import _has_static_validation

from concord_lqcd.backend.planner.physical import (
    ContractOperation,
    CorrelatorSumOperation,
    MaterializePrimitiveOperation,
    PhysicalBufferSlot,
    PhysicalExecutionPlan,
    PhysicalInputUse,
    PhysicalOutputBinding,
    PhysicalPlanningStatistics,
    PhysicalPlanStep,
    PhysicalValueBinding,
    WickTermProductOperation,
)
from concord_lqcd.backend.runtime.pipeline import (
    RuntimeCompilation,
    RuntimeCompilationConfig,
    RuntimeExecutionStages,
    compile_correlators,
)
from concord_lqcd.backend.runtime.metrics import (
    CompilationCostReport,
    build_compilation_cost_report,
)
from concord_lqcd.compiler.contraction_graph import TensorLegBinding
from concord_lqcd.compiler.contraction_order import (
    ComponentContractionPlan,
    ContractionSizeModel,
    ContractionStep,
    GlobalContractionOrderPlan,
)
from concord_lqcd.compiler.distillation_lowering import (
    KernelRecipe,
    ProjectedField,
    ProjectionContribution,
    identify_kernel_resource,
)
from concord_lqcd.compiler.global_cse import (
    ComponentComputation,
    ComponentTensorUse,
    ComponentUse,
    GlobalCorrelatorUse,
    GlobalCSEStatistics,
    GlobalComputationDAG,
    GlobalTensorResource,
    GlobalWickTermUse,
)
from concord_lqcd.compiler.logical_execution import (
    LogicalContraction,
    LogicalCorrelatorSum,
    LogicalExecutionDAG,
    LogicalExecutionStatistics,
    LogicalPrimitiveEvaluation,
    LogicalValueUse,
    LogicalWickTermProduct,
)
from concord_lqcd.compiler.logical_scheduling import (
    LogicalSchedule,
    LogicalScheduleStatistics,
    LogicalScheduleStep,
    LogicalValueLifetime,
)
from concord_lqcd.compiler.partial_cse import (
    ComponentIntermediatePlan,
    GlobalIntermediateDAG,
    IntermediateComputation,
    IntermediateInputUse,
    IntermediateUse,
    PartialCSEStatistics,
    PrimitiveTensorValue,
)
from concord_lqcd.compiler.physics_ir import (
    ColorTensorRecord,
    GaugeTensorRecord,
    OperationKind,
    OperationRecord,
    QuarkRecord,
    Representation,
    SpinTensorRecord,
)
from concord_lqcd.compiler.propagation_lowering import (
    PropagationInsertionSpec,
    PropagationRecipe,
)
from concord_lqcd.compiler.static_analysis import (
    StaticDependencyRecord,
    StaticExecutionAnalysis,
    StaticExecutionStatistics,
    StaticNumericalModel,
    StaticValueDescriptor,
)

from .loader import _load_operator_snapshot
from .operator_sources import (
    OperatorSourceError, discover_operator_dependencies, validate_dependency_records,
)
from .schema import HDF5EigenvectorSpec, JobSpec
from .flavor_equivalence import validate_numerical_flavor_sources
from concord_lqcd.compiler.wick_aggregation import (
    CertifiedTermClass, CorrelatorAggregationProof, WickAggregationConfig,
    WickAggregationProof, validate_wick_aggregation,
)


PLAN_FORMAT_VERSION = 2
EXECUTION_SCHEMA_VERSION = 4
COMPILER_SEMANTICS_VERSION = 5
_MANIFEST_NAME = "manifest.json"
_PLAN_NAME = "plan.json"
_DIAGNOSTICS_NAME = "diagnostics.json"
_REQUIRED_ARCHIVE_MEMBERS = {_MANIFEST_NAME, _PLAN_NAME, _DIAGNOSTICS_NAME}


class PlanSerializationError(ValueError):
    """Raised when a ``.concord`` artifact is invalid or cannot be serialized."""


class StalePlanError(PlanSerializationError):
    """Raised when a compiled plan fingerprint no longer matches a job spec."""


def _concord_version() -> str:
    try:
        return package_version("concord-lqcd")
    except PackageNotFoundError:  # pragma: no cover - source-tree execution fallback
        # Kept equal to concord_lqcd.__version__ by tests/test_packaging.py; the job
        # layer does not import the package root.
        return "1.0.0"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise PlanSerializationError(f"cannot read operator source {path}: {exc}") from exc


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PlanSerializationError(f"value is not canonical JSON data: {exc}") from exc
    return text.encode("utf-8")


def _canonical_fingerprint_value(value: object, *, role: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanSerializationError(f"{role} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_canonical_fingerprint_value(item, role=role) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PlanSerializationError(f"{role} mapping keys must be strings")
            normalized[key] = _canonical_fingerprint_value(item, role=role)
        return {key: normalized[key] for key in sorted(normalized)}
    raise PlanSerializationError(
        f"{role} contains unsupported value type {type(value).__name__!r}"
    )


def _hash_canonical(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


@dataclass(frozen=True)
class PlanFingerprint:
    operator_source_sha256: str
    operator_dependencies: Mapping[str, str]
    operator_parameters_sha256: str
    compile_config_sha256: str
    job_fingerprint: str


def compute_job_fingerprint(
    spec: JobSpec,
    *,
    operator_source_sha256: str | None = None,
    operator_dependencies: Mapping[str, str] | None = None,
) -> PlanFingerprint:
    """Compute the compile-relevant V1 job fingerprint.

    Runtime data paths, configurations, time contexts, output paths, lattice
    extents, and device selection deliberately do not enter this fingerprint.
    The contract fingerprints symbolic source/factory/parameters, normalized
    compile choices, and the adapter's shared-LapH-basis declaration. The latter
    changes permitted algebra, unlike numerical file paths.

    Local Python helper imports are discovered without executing them. Planning
    supplies the captured dependency map, including helpers dynamically imported
    by the factory; saved-plan validation checks that recorded map as well.
    """

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    source_hash = operator_source_sha256 or _sha256_file(spec.operators.file)
    if operator_dependencies is None:
        dependencies = (
            discover_operator_dependencies(spec.operators.file)
            if spec.operators.file.exists() else {}
        )
    else:
        validate_dependency_records(operator_dependencies)
        dependencies = dict(operator_dependencies)
    parameters_payload = _canonical_fingerprint_value(
        dict(spec.operators.parameters),
        role="operators.parameters",
    )
    compile_choices = spec.compile.model_dump(mode="python")
    compile_choices["shared_orthonormal_laph_basis"] = (
        runtime_compilation_config_from_job(spec).shared_orthonormal_laph_basis
    )
    compile_payload = _canonical_fingerprint_value(
        compile_choices,
        role="compile configuration",
    )
    parameters_hash = _hash_canonical(parameters_payload)
    compile_hash = _hash_canonical(compile_payload)
    fingerprint_payload = {
        "schema_version": spec.schema_version,
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "operator_source_sha256": source_hash,
        "operator_dependencies": dependencies,
        "operator_factory": spec.operators.factory,
        "operator_parameters_sha256": parameters_hash,
        "compile_config_sha256": compile_hash,
    }
    return PlanFingerprint(
        operator_source_sha256=source_hash,
        operator_dependencies=dependencies,
        operator_parameters_sha256=parameters_hash,
        compile_config_sha256=compile_hash,
        job_fingerprint=_hash_canonical(fingerprint_payload),
    )


@dataclass(frozen=True)
class PlanManifest:
    plan_format_version: int
    job_schema_version: int
    concord_version: str
    compiler_semantics_version: int
    job_name: str
    created_utc: str
    operator_file: str
    operator_factory: str
    operator_source_sha256: str
    operator_dependencies: Mapping[str, str]
    operator_parameters_sha256: str
    compile_config_sha256: str
    job_fingerprint: str
    plan_sha256: str
    correlator_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.plan_format_version != PLAN_FORMAT_VERSION:
            raise PlanSerializationError(
                f"unsupported .concord plan format version {self.plan_format_version}; "
                f"expected {PLAN_FORMAT_VERSION}"
            )
        if self.job_schema_version != 1:
            raise PlanSerializationError(
                f"unsupported job schema version {self.job_schema_version}"
            )
        if type(self.compiler_semantics_version) is not int:
            raise PlanSerializationError("manifest compiler semantics version must be an integer")
        if self.compiler_semantics_version != COMPILER_SEMANTICS_VERSION:
            raise StalePlanError(
                "compiled plan uses incompatible compiler semantics "
                f"{self.compiler_semantics_version}; expected {COMPILER_SEMANTICS_VERSION}. "
                "Recompile the plan to apply the current source-adjoint, "
                "open-index, and disconnected-color contraction semantics"
            )
        try:
            validate_dependency_records(self.operator_dependencies)
        except OperatorSourceError as exc:
            raise PlanSerializationError(str(exc)) from exc
        for role, value in (
            ("concord_version", self.concord_version),
            ("job_name", self.job_name),
            ("created_utc", self.created_utc),
            ("operator_file", self.operator_file),
            ("operator_factory", self.operator_factory),
        ):
            if not value:
                raise PlanSerializationError(f"manifest field {role} must be non-empty")
        for role, value in (
            ("operator_source_sha256", self.operator_source_sha256),
            ("operator_parameters_sha256", self.operator_parameters_sha256),
            ("compile_config_sha256", self.compile_config_sha256),
            ("job_fingerprint", self.job_fingerprint),
            ("plan_sha256", self.plan_sha256),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise PlanSerializationError(
                    f"manifest field {role} must be a lowercase SHA-256 digest"
                )
        if not self.correlator_ids:
            raise PlanSerializationError("manifest must contain at least one correlator id")
        if any(not item for item in self.correlator_ids):
            raise PlanSerializationError("manifest correlator ids must be non-empty")
        if len(self.correlator_ids) != len(set(self.correlator_ids)):
            raise PlanSerializationError("manifest correlator ids must be unique")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "plan_format_version": self.plan_format_version,
            "job_schema_version": self.job_schema_version,
            "concord_version": self.concord_version,
            "compiler_semantics_version": self.compiler_semantics_version,
            "job_name": self.job_name,
            "created_utc": self.created_utc,
            "operator_file": self.operator_file,
            "operator_factory": self.operator_factory,
            "operator_source_sha256": self.operator_source_sha256,
            "operator_dependencies": dict(self.operator_dependencies),
            "operator_parameters_sha256": self.operator_parameters_sha256,
            "compile_config_sha256": self.compile_config_sha256,
            "job_fingerprint": self.job_fingerprint,
            "plan_sha256": self.plan_sha256,
            "correlator_ids": list(self.correlator_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PlanManifest":
        expected = {
            "plan_format_version",
            "job_schema_version",
            "concord_version",
            "compiler_semantics_version",
            "job_name",
            "created_utc",
            "operator_file",
            "operator_factory",
            "operator_source_sha256",
            "operator_dependencies",
            "operator_parameters_sha256",
            "compile_config_sha256",
            "job_fingerprint",
            "plan_sha256",
            "correlator_ids",
        }
        if set(payload) in (
            expected - {"compiler_semantics_version"},
            expected - {"compiler_semantics_version", "operator_dependencies"},
        ):
            raise StalePlanError(
                "compiled plan predates the transformed-field source-adjoint correction. "
                "Recompile the plan with the current compiler semantics"
            )
        if set(payload) == expected - {"operator_dependencies"}:
            raise StalePlanError(
                "compiled plan predates local operator-helper dependency tracking. "
                "Recompile the plan with the current plan format"
            )
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise PlanSerializationError(
                f"manifest fields are inconsistent; missing={missing}, extra={extra}"
            )
        if type(payload["plan_format_version"]) is not int or type(payload["job_schema_version"]) is not int:
            raise PlanSerializationError("manifest version fields must be integers")
        raw_correlators = payload["correlator_ids"]
        if not isinstance(raw_correlators, list) or any(
            not isinstance(item, str) for item in raw_correlators
        ):
            raise PlanSerializationError("manifest correlator_ids must be a list of strings")
        string_fields = (
            "concord_version",
            "job_name",
            "created_utc",
            "operator_file",
            "operator_factory",
            "operator_source_sha256",
            "operator_parameters_sha256",
            "compile_config_sha256",
            "job_fingerprint",
            "plan_sha256",
        )
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise PlanSerializationError("manifest textual fields must be strings")
        manifest = cls(
            plan_format_version=payload["plan_format_version"],
            job_schema_version=payload["job_schema_version"],
            concord_version=payload["concord_version"],
            compiler_semantics_version=payload["compiler_semantics_version"],
            job_name=payload["job_name"],
            created_utc=payload["created_utc"],
            operator_file=payload["operator_file"],
            operator_factory=payload["operator_factory"],
            operator_source_sha256=payload["operator_source_sha256"],
            operator_dependencies=payload["operator_dependencies"],
            operator_parameters_sha256=payload["operator_parameters_sha256"],
            compile_config_sha256=payload["compile_config_sha256"],
            job_fingerprint=payload["job_fingerprint"],
            plan_sha256=payload["plan_sha256"],
            correlator_ids=tuple(raw_correlators),
        )
        manifest.validate()
        return manifest


# Explicit whitelist used by the JSON object codec. Deserialization never
# imports a class named by archive content, so plan data cannot request arbitrary
# Python object construction.
_DATACLASS_TYPES = (
    RuntimeCompilationConfig,
    WickAggregationConfig,
    CertifiedTermClass,
    CorrelatorAggregationProof,
    WickAggregationProof,
    GlobalComputationDAG,
    GlobalTensorResource,
    ComponentTensorUse,
    ComponentComputation,
    ComponentUse,
    GlobalWickTermUse,
    GlobalCorrelatorUse,
    GlobalCSEStatistics,
    ContractionSizeModel,
    ContractionStep,
    ComponentContractionPlan,
    PrimitiveTensorValue,
    IntermediateInputUse,
    IntermediateComputation,
    IntermediateUse,
    ComponentIntermediatePlan,
    PartialCSEStatistics,
    TensorLegBinding,
    KernelRecipe,
    ProjectedField,
    QuarkRecord,
    OperationRecord,
    SpinTensorRecord,
    ColorTensorRecord,
    GaugeTensorRecord,
    PropagationInsertionSpec,
    PropagationRecipe,
    LogicalValueUse,
    LogicalPrimitiveEvaluation,
    LogicalContraction,
    LogicalWickTermProduct,
    LogicalCorrelatorSum,
    LogicalExecutionStatistics,
    StaticNumericalModel,
    StaticValueDescriptor,
    StaticDependencyRecord,
    StaticExecutionStatistics,
    LogicalValueLifetime,
    LogicalScheduleStep,
    LogicalScheduleStatistics,
    PhysicalBufferSlot,
    PhysicalValueBinding,
    PhysicalInputUse,
    MaterializePrimitiveOperation,
    ContractOperation,
    WickTermProductOperation,
    CorrelatorSumOperation,
    PhysicalPlanStep,
    PhysicalOutputBinding,
    PhysicalPlanningStatistics,
)
_DATACLASS_BY_TOKEN = {cls.__name__: cls for cls in _DATACLASS_TYPES}
_TOKEN_BY_DATACLASS = {cls: cls.__name__ for cls in _DATACLASS_TYPES}
# dataclasses.fields() rebuilds its tuple on every call, which dominated encoding
# and decoding the millions of records of a large plan.
_FIELD_NAMES_BY_DATACLASS = {
    cls: tuple(item.name for item in fields(cls)) for cls in _DATACLASS_TYPES
}
_FIELD_NAME_SET_BY_TOKEN = {
    cls.__name__: frozenset(names) for cls, names in _FIELD_NAMES_BY_DATACLASS.items()
}
_ENUM_TYPES = (ProjectionContribution, Representation, OperationKind)
_ENUM_BY_TOKEN = {cls.__name__: cls for cls in _ENUM_TYPES}
_TOKEN_BY_ENUM = {cls: cls.__name__ for cls in _ENUM_TYPES}
_COMPLEX_KEYS = frozenset(("$complex",))
_ENUM_KEYS = frozenset(("$enum", "value"))
_TUPLE_KEYS = frozenset(("$tuple",))
_LIST_KEYS = frozenset(("$list",))
_MAP_KEYS = frozenset(("$map",))
_DATACLASS_KEYS = frozenset(("$type", "fields"))


def _encode_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    field_names = _FIELD_NAMES_BY_DATACLASS.get(type(value))
    if field_names is not None:
        return {
            "$type": _TOKEN_BY_DATACLASS[type(value)],
            "fields": {
                name: _encode_value(getattr(value, name)) for name in field_names
            },
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanSerializationError("plan data contains a non-finite float")
        return value
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise PlanSerializationError("plan data contains a non-finite complex value")
        return {"$complex": [value.real, value.imag]}
    if isinstance(value, Enum):
        enum_type = type(value)
        try:
            token = _TOKEN_BY_ENUM[enum_type]
        except KeyError as exc:
            raise PlanSerializationError(
                f"unsupported enum type {enum_type.__name__!r} in plan data"
            ) from exc
        return {"$enum": token, "value": value.value}
    if is_dataclass(value) and not isinstance(value, type):
        # Registered plan records were encoded above.
        raise PlanSerializationError(
            f"unsupported dataclass {type(value).__name__!r} in plan data"
        )
    if isinstance(value, tuple):
        return {"$tuple": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return {"$list": [_encode_value(item) for item in value]}
    if isinstance(value, Mapping):
        return {
            "$map": [
                [_encode_value(key), _encode_value(item)]
                for key, item in value.items()
            ]
        }
    raise PlanSerializationError(
        f"unsupported plan value type {type(value).__name__!r}"
    )


def _decode_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanSerializationError("encoded plan contains a non-finite float")
        return value
    if not isinstance(value, dict):
        raise PlanSerializationError(
            f"invalid encoded plan value of type {type(value).__name__!r}"
        )
    keys = value.keys()
    if keys == _DATACLASS_KEYS:
        return _decode_dataclass(value)
    if keys == _TUPLE_KEYS:
        payload = value["$tuple"]
        if not isinstance(payload, list):
            raise PlanSerializationError("invalid encoded tuple")
        return tuple(_decode_value(item) for item in payload)
    if keys == _COMPLEX_KEYS:
        payload = value["$complex"]
        if not isinstance(payload, list) or len(payload) != 2:
            raise PlanSerializationError("invalid encoded complex value")
        try:
            real = float(payload[0])
            imag = float(payload[1])
        except (TypeError, ValueError) as exc:
            raise PlanSerializationError("invalid encoded complex components") from exc
        if not math.isfinite(real) or not math.isfinite(imag):
            raise PlanSerializationError("encoded plan contains a non-finite complex value")
        return complex(real, imag)
    if keys == _ENUM_KEYS:
        token = value["$enum"]
        if not isinstance(token, str) or token not in _ENUM_BY_TOKEN:
            raise PlanSerializationError(f"unknown encoded enum type {token!r}")
        try:
            return _ENUM_BY_TOKEN[token](value["value"])
        except ValueError as exc:
            raise PlanSerializationError(
                f"invalid value for encoded enum {token!r}: {value['value']!r}"
            ) from exc
    if keys == _LIST_KEYS:
        payload = value["$list"]
        if not isinstance(payload, list):
            raise PlanSerializationError("invalid encoded list")
        return [_decode_value(item) for item in payload]
    if keys == _MAP_KEYS:
        payload = value["$map"]
        if not isinstance(payload, list):
            raise PlanSerializationError("invalid encoded mapping")
        decoded: dict[object, object] = {}
        for entry in payload:
            if not isinstance(entry, list) or len(entry) != 2:
                raise PlanSerializationError("invalid encoded mapping entry")
            key = _decode_value(entry[0])
            try:
                if key in decoded:
                    raise PlanSerializationError("encoded mapping contains duplicate keys")
                decoded[key] = _decode_value(entry[1])
            except TypeError as exc:
                raise PlanSerializationError("encoded mapping contains unhashable key") from exc
        return decoded
    raise PlanSerializationError(f"unknown encoded plan object keys {sorted(value)}")


def _encoded_record_fields(value: dict) -> tuple[str, dict]:
    token = value["$type"]
    raw_fields = value["fields"]
    if not isinstance(token, str) or token not in _DATACLASS_BY_TOKEN:
        raise PlanSerializationError(f"unknown encoded dataclass type {token!r}")
    if not isinstance(raw_fields, dict):
        raise PlanSerializationError(f"encoded dataclass {token!r} has invalid fields")
    expected = _FIELD_NAME_SET_BY_TOKEN[token]
    if raw_fields.keys() != expected:
        missing = sorted(expected - set(raw_fields))
        extra = sorted(set(raw_fields) - expected)
        raise PlanSerializationError(
            f"encoded dataclass {token!r} fields disagree; "
            f"missing={missing}, extra={extra}"
        )
    return token, raw_fields


def _decode_dataclass(value: dict) -> object:
    token, raw_fields = _encoded_record_fields(value)
    kwargs = {name: _decode_value(raw_fields[name]) for name in raw_fields}
    try:
        return _DATACLASS_BY_TOKEN[token](**kwargs)
    except Exception as exc:
        raise PlanSerializationError(
            f"cannot reconstruct encoded dataclass {token!r}: {exc}"
        ) from exc


class _UnsupportedOwnedDecode(TypeError):
    """The plan holds values outside the read-only form; decode it the ordinary way."""


_FROZEN_DATACLASS_TYPES = frozenset(
    cls for cls in _DATACLASS_TYPES if cls.__dataclass_params__.frozen
)


def _decode_owned_value(value: object) -> object:
    """Decode like ``_decode_value``, straight into the read-only snapshot form.

    Mappings become read-only proxies, and every record must be frozen and
    hold exactly the decoded values it was given: the form that
    ``_prepared._SnapshotCopier`` makes when it copies a caller's plan. A plan
    with lists raises ``_UnsupportedOwnedDecode``, so its caller decodes it
    the ordinary way and keeps the strict path.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, dict):
        keys = value.keys()
        if keys == _DATACLASS_KEYS:
            return _decode_owned_dataclass(value)
        if keys == _TUPLE_KEYS:
            payload = value["$tuple"]
            if not isinstance(payload, list):
                raise PlanSerializationError("invalid encoded tuple")
            return tuple([_decode_owned_value(item) for item in payload])
        if keys == _MAP_KEYS:
            payload = value["$map"]
            if not isinstance(payload, list):
                raise PlanSerializationError("invalid encoded mapping")
            decoded: dict[object, object] = {}
            for entry in payload:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise PlanSerializationError("invalid encoded mapping entry")
                key = _decode_owned_value(entry[0])
                try:
                    if key in decoded:
                        raise PlanSerializationError("encoded mapping contains duplicate keys")
                    decoded[key] = _decode_owned_value(entry[1])
                except TypeError as exc:
                    if isinstance(exc, _UnsupportedOwnedDecode):
                        raise
                    raise PlanSerializationError("encoded mapping contains unhashable key") from exc
            return MappingProxyType(decoded)
        if keys == _LIST_KEYS:
            raise _UnsupportedOwnedDecode("lists are not part of the read-only plan form")
    # Floats, complex numbers, enums and every malformed value, exactly as before.
    return _decode_value(value)


def _decode_owned_dataclass(value: dict) -> object:
    token, raw_fields = _encoded_record_fields(value)
    record_type = _DATACLASS_BY_TOKEN[token]
    if record_type not in _FROZEN_DATACLASS_TYPES:
        raise _UnsupportedOwnedDecode(f"mutable record {token} is not part of the read-only plan form")
    kwargs = {name: _decode_owned_value(raw_fields[name]) for name in raw_fields}
    try:
        record = record_type(**kwargs)
    except Exception as exc:
        raise PlanSerializationError(
            f"cannot reconstruct encoded dataclass {token!r}: {exc}"
        ) from exc
    _require_owned_fields(record, kwargs)
    return record


def _require_owned_fields(record: object, kwargs: Mapping[str, object]) -> None:
    for name, item in kwargs.items():
        if getattr(record, name) is not item:
            raise _UnsupportedOwnedDecode(
                f"record constructor {type(record).__name__} changed owned metadata"
            )


def _build_record(record_type: type, **kwargs: object) -> object:
    return record_type(**kwargs)


def _build_owned_record(record_type: type, **kwargs: object) -> object:
    if not record_type.__dataclass_params__.frozen:
        raise _UnsupportedOwnedDecode(
            f"mutable record {record_type.__name__} is not part of the read-only plan form"
        )
    record = record_type(**kwargs)
    _require_owned_fields(record, kwargs)
    return record


@contextmanager
def _gc_paused():
    """Pause Python's cycle collector while a plan's records are built.

    Building millions of records makes the collector scan them again and
    again, which took about 40 % of loading a large plan. Reference counting
    still frees ordinary garbage at once; cycles are collected when the
    collector resumes.
    """

    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _execution_payload_from_parts(
    *,
    config: RuntimeCompilationConfig,
    global_cse: GlobalComputationDAG,
    contraction_order: GlobalContractionOrderPlan,
    intermediate_dag: GlobalIntermediateDAG,
    logical_dag: LogicalExecutionDAG,
    static_analysis: StaticExecutionAnalysis,
    schedule: LogicalSchedule,
    physical_plan: PhysicalExecutionPlan,
) -> dict[str, object]:
    return {
        "execution_schema_version": EXECUTION_SCHEMA_VERSION,
        "config": _encode_value(config),
        "global_cse": {
            "resources": _encode_value(global_cse.resources),
            "components": _encode_value(global_cse.components),
            "correlators": _encode_value(global_cse.correlators),
            "statistics": _encode_value(global_cse.statistics),
            "aggregation_proof": _encode_value(global_cse.aggregation_proof),
        },
        "contraction_order": {
            "size_model": _encode_value(contraction_order.size_model),
            "strategy": contraction_order.strategy,
            "memory_limit": contraction_order.memory_limit,
            "component_plans": _encode_value(contraction_order.component_plans),
        },
        "partial_cse": {
            "primitive_values": _encode_value(intermediate_dag.primitive_values),
            "intermediates": _encode_value(intermediate_dag.intermediates),
            "component_plans": _encode_value(intermediate_dag.component_plans),
            "statistics": _encode_value(intermediate_dag.statistics),
        },
        "logical": {
            "primitives": _encode_value(logical_dag.primitives),
            "contractions": _encode_value(logical_dag.contractions),
            "terms": _encode_value(logical_dag.terms),
            "correlators": _encode_value(logical_dag.correlators),
            "outputs": _encode_value(logical_dag.outputs),
            "statistics": _encode_value(logical_dag.statistics),
        },
        "static": {
            "numerical_model": _encode_value(static_analysis.numerical_model),
            "values": _encode_value(static_analysis.values),
            "dependencies": _encode_value(static_analysis.dependencies),
            "topological_order": _encode_value(static_analysis.topological_order),
            "roots": _encode_value(static_analysis.roots),
            "output_nodes": _encode_value(static_analysis.output_nodes),
            "dead_nodes": _encode_value(static_analysis.dead_nodes),
            "statistics": _encode_value(static_analysis.statistics),
        },
        "schedule": {
            "strategy": schedule.strategy,
            "include_dead": schedule.include_dead,
            "steps": _encode_value(schedule.steps),
            "lifetimes": _encode_value(schedule.lifetimes),
            "output_nodes": _encode_value(schedule.output_nodes),
            "statistics": _encode_value(schedule.statistics),
        },
        "physical": {
            "memory_space": physical_plan.memory_space,
            "reuse_strategy": physical_plan.reuse_strategy,
            "buffers": _encode_value(physical_plan.buffers),
            "value_bindings": _encode_value(physical_plan.value_bindings),
            "steps": _encode_value(physical_plan.steps),
            "outputs": _encode_value(physical_plan.outputs),
            "statistics": _encode_value(physical_plan.statistics),
        },
    }


def _execution_payload_from_compilation(compilation: RuntimeCompilation) -> dict[str, object]:
    return _execution_payload_from_parts(
        config=compilation.config,
        global_cse=compilation.global_cse,
        contraction_order=compilation.contraction_order,
        intermediate_dag=compilation.intermediate_dag,
        logical_dag=compilation.logical_dag,
        static_analysis=compilation.static_analysis,
        schedule=compilation.schedule,
        physical_plan=compilation.physical_plan,
    )


def _require_section(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise PlanSerializationError(f"plan section {name!r} must be an object")
    return value


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    role: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise PlanSerializationError(
            f"{role} fields disagree; missing={missing}, extra={extra}"
        )


def _decode_execution_payload(
    payload: Mapping[str, object],
    *,
    owned: bool = False,
) -> tuple[
    RuntimeCompilationConfig,
    GlobalComputationDAG,
    GlobalContractionOrderPlan,
    GlobalIntermediateDAG,
    LogicalExecutionDAG,
    StaticExecutionAnalysis,
    LogicalSchedule,
    PhysicalExecutionPlan,
]:
    """Decode the execution sections; ``owned`` decodes into the read-only form."""

    decode_value = _decode_owned_value if owned else _decode_value
    mapping_type = MappingProxyType if owned else dict
    build = _build_owned_record if owned else _build_record

    def decoded(value: object, expected_type: type, *, role: str):
        result = decode_value(value)
        if not isinstance(result, expected_type):
            raise PlanSerializationError(
                f"{role} decoded as {type(result).__name__}, expected {expected_type.__name__}"
            )
        return result

    def decoded_mapping(value: object, *, role: str) -> Mapping:
        result = decode_value(value)
        if not isinstance(result, mapping_type):
            raise PlanSerializationError(f"{role} must decode to a mapping")
        return result

    def decoded_tuple(value: object, *, role: str) -> tuple:
        result = decode_value(value)
        if not isinstance(result, tuple):
            raise PlanSerializationError(f"{role} must decode to a tuple")
        return result

    _require_exact_keys(
        payload,
        {
            "execution_schema_version",
            "config",
            "global_cse",
            "contraction_order",
            "partial_cse",
            "logical",
            "static",
            "schedule",
            "physical",
        },
        role="plan root",
    )
    if payload["execution_schema_version"] != EXECUTION_SCHEMA_VERSION:
        raise PlanSerializationError(
            f"unsupported execution schema version {payload['execution_schema_version']!r}; "
            "regenerate the plan from its job specification"
        )

    config = decoded(payload["config"], RuntimeCompilationConfig, role="plan config")

    global_payload = _require_section(payload, "global_cse")
    _require_exact_keys(
        global_payload,
        {"resources", "components", "correlators", "statistics", "aggregation_proof"},
        role="global_cse section",
    )
    global_cse = build(
        GlobalComputationDAG,
        resources=decoded_mapping(global_payload["resources"], role="global resources"),
        components=decoded_mapping(global_payload["components"], role="global components"),
        correlators=decoded_tuple(global_payload["correlators"], role="global correlators"),
        statistics=decoded(global_payload["statistics"], GlobalCSEStatistics, role="global CSE statistics"),
        aggregation_proof=decode_value(global_payload["aggregation_proof"]),
    )

    contraction_payload = _require_section(payload, "contraction_order")
    _require_exact_keys(
        contraction_payload,
        {"size_model", "strategy", "memory_limit", "component_plans"},
        role="contraction_order section",
    )
    if not isinstance(contraction_payload["strategy"], str):
        raise PlanSerializationError("contraction strategy must be a string")
    if contraction_payload["memory_limit"] is not None and type(contraction_payload["memory_limit"]) is not int:
        raise PlanSerializationError("contraction memory limit must be an integer or null")
    contraction_order = build(
        GlobalContractionOrderPlan,
        source_dag=global_cse,
        size_model=decoded(contraction_payload["size_model"], ContractionSizeModel, role="contraction size model"),
        strategy=contraction_payload["strategy"],
        memory_limit=contraction_payload["memory_limit"],
        component_plans=decoded_mapping(
            contraction_payload["component_plans"],
            role="component contraction plans",
        ),
    )

    partial_payload = _require_section(payload, "partial_cse")
    _require_exact_keys(
        partial_payload,
        {"primitive_values", "intermediates", "component_plans", "statistics"},
        role="partial_cse section",
    )
    intermediate_dag = build(
        GlobalIntermediateDAG,
        source_plan=contraction_order,
        primitive_values=decoded_mapping(partial_payload["primitive_values"], role="partial primitive values"),
        intermediates=decoded_mapping(partial_payload["intermediates"], role="partial intermediates"),
        component_plans=decoded_mapping(partial_payload["component_plans"], role="partial component plans"),
        statistics=decoded(partial_payload["statistics"], PartialCSEStatistics, role="partial CSE statistics"),
    )

    logical_payload = _require_section(payload, "logical")
    _require_exact_keys(
        logical_payload,
        {"primitives", "contractions", "terms", "correlators", "outputs", "statistics"},
        role="logical section",
    )
    logical_dag = build(
        LogicalExecutionDAG,
        source_dag=intermediate_dag,
        primitives=decoded_mapping(logical_payload["primitives"], role="logical primitives"),
        contractions=decoded_mapping(logical_payload["contractions"], role="logical contractions"),
        terms=decoded_tuple(logical_payload["terms"], role="logical terms"),
        correlators=decoded_tuple(logical_payload["correlators"], role="logical correlators"),
        outputs=decoded_mapping(logical_payload["outputs"], role="logical outputs"),
        statistics=decoded(logical_payload["statistics"], LogicalExecutionStatistics, role="logical statistics"),
    )

    static_payload = _require_section(payload, "static")
    _require_exact_keys(
        static_payload,
        {"numerical_model", "values", "dependencies", "topological_order", "roots", "output_nodes", "dead_nodes", "statistics"},
        role="static section",
    )
    static_analysis = build(
        StaticExecutionAnalysis,
        source_dag=logical_dag,
        numerical_model=decoded(static_payload["numerical_model"], StaticNumericalModel, role="static numerical model"),
        values=decoded_mapping(static_payload["values"], role="static values"),
        dependencies=decoded_mapping(static_payload["dependencies"], role="static dependencies"),
        topological_order=decoded_tuple(static_payload["topological_order"], role="static topological order"),
        roots=decoded_tuple(static_payload["roots"], role="static roots"),
        output_nodes=decoded_tuple(static_payload["output_nodes"], role="static output nodes"),
        dead_nodes=decoded_tuple(static_payload["dead_nodes"], role="static dead nodes"),
        statistics=decoded(static_payload["statistics"], StaticExecutionStatistics, role="static statistics"),
    )

    schedule_payload = _require_section(payload, "schedule")
    _require_exact_keys(
        schedule_payload,
        {"strategy", "include_dead", "steps", "lifetimes", "output_nodes", "statistics"},
        role="schedule section",
    )
    if not isinstance(schedule_payload["strategy"], str) or type(schedule_payload["include_dead"]) is not bool:
        raise PlanSerializationError("schedule strategy/include_dead have invalid types")
    schedule = build(
        LogicalSchedule,
        source_analysis=static_analysis,
        strategy=schedule_payload["strategy"],
        include_dead=schedule_payload["include_dead"],
        steps=decoded_tuple(schedule_payload["steps"], role="schedule steps"),
        lifetimes=decoded_mapping(schedule_payload["lifetimes"], role="schedule lifetimes"),
        output_nodes=decoded_tuple(schedule_payload["output_nodes"], role="schedule outputs"),
        statistics=decoded(schedule_payload["statistics"], LogicalScheduleStatistics, role="schedule statistics"),
    )

    physical_payload = _require_section(payload, "physical")
    _require_exact_keys(
        physical_payload,
        {"memory_space", "reuse_strategy", "buffers", "value_bindings", "steps", "outputs", "statistics"},
        role="physical section",
    )
    if not isinstance(physical_payload["memory_space"], str) or not isinstance(physical_payload["reuse_strategy"], str):
        raise PlanSerializationError("physical memory space/reuse strategy have invalid types")
    physical_plan = build(
        PhysicalExecutionPlan,
        source_schedule=schedule,
        memory_space=physical_payload["memory_space"],
        reuse_strategy=physical_payload["reuse_strategy"],
        buffers=decoded_mapping(physical_payload["buffers"], role="physical buffers"),
        value_bindings=decoded_mapping(physical_payload["value_bindings"], role="physical value bindings"),
        steps=decoded_tuple(physical_payload["steps"], role="physical steps"),
        outputs=decoded_mapping(physical_payload["outputs"], role="physical outputs"),
        statistics=decoded(physical_payload["statistics"], PhysicalPlanningStatistics, role="physical statistics"),
    )
    return (
        config,
        global_cse,
        contraction_order,
        intermediate_dag,
        logical_dag,
        static_analysis,
        schedule,
        physical_plan,
    )


def _diagnostics_from_compilation(compilation: RuntimeCompilation) -> dict[str, object]:
    resource_kinds = Counter(resource.kind for resource in compilation.global_cse.resources.values())
    return {
        "cost_report": asdict(compilation.cost_report),
        "global_cse": asdict(compilation.global_cse.statistics),
        "wick_aggregation": dict(compilation.wick_aggregation_diagnostics),
        "resources": {
            "total": len(compilation.global_cse.resources),
            "kernel": resource_kinds.get("kernel", 0),
            "spin": resource_kinds.get("spin", 0),
            "propagation": resource_kinds.get("propagation", 0),
        },
        "contraction_order": {
            "estimated_unique_flops": compilation.contraction_order.estimated_unique_flops,
            "largest_intermediate_elements": compilation.contraction_order.largest_intermediate_elements,
            "selected_strategies": list(
                compilation.contraction_order.selected_strategies
            ),
        },
        "partial_cse": asdict(compilation.intermediate_dag.statistics),
        "logical_execution": asdict(compilation.logical_dag.statistics),
        "static_execution": asdict(compilation.static_analysis.statistics),
        "schedule": asdict(compilation.schedule.statistics),
        "physical": asdict(compilation.physical_plan.statistics),
    }


@dataclass(frozen=True)
class ConcordPlan:
    """Loaded or freshly compiled portable execution plan."""

    manifest: PlanManifest
    config: RuntimeCompilationConfig
    physical_plan: PhysicalExecutionPlan
    diagnostics: Mapping[str, Any]

    @property
    def schedule(self) -> LogicalSchedule:
        return self.physical_plan.source_schedule

    @property
    def static_analysis(self) -> StaticExecutionAnalysis:
        return self.schedule.source_analysis

    @property
    def logical_dag(self) -> LogicalExecutionDAG:
        return self.static_analysis.source_dag

    @property
    def intermediate_dag(self) -> GlobalIntermediateDAG:
        return self.logical_dag.source_dag

    @property
    def contraction_order(self) -> GlobalContractionOrderPlan:
        return self.intermediate_dag.source_plan

    @property
    def global_cse(self) -> GlobalComputationDAG:
        return self.contraction_order.source_dag

    @property
    def resources(self) -> Mapping[str, GlobalTensorResource]:
        return self.global_cse.resources

    @property
    def correlator_ids(self) -> tuple[str, ...]:
        return tuple(self.manifest.correlator_ids)

    @property
    def cost_report(self) -> CompilationCostReport:
        """Reconstruct plan costs, retaining a valid compile-time term count."""

        input_wick_terms: int | None = None
        payload = self.diagnostics.get("cost_report")
        surviving_wick_terms = sum(
            len(correlator.terms) for correlator in self.global_cse.correlators
        )
        if isinstance(payload, Mapping):
            candidate = payload.get("input_wick_terms")
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and candidate >= surviving_wick_terms
            ):
                input_wick_terms = candidate
        return build_compilation_cost_report(
            self,
            input_wick_terms=input_wick_terms,
        )

    @property
    def required_propagation_resources(self) -> Mapping[str, PropagationRecipe]:
        ids = tuple(dict.fromkeys(
            step.operation.resource_id
            for step in self.physical_plan.steps
            if isinstance(step.operation, MaterializePrimitiveOperation)
            and step.operation.kind == "propagation"
        ))
        result: dict[str, PropagationRecipe] = {}
        for resource_id in ids:
            try:
                resource = self.resources[resource_id]
            except KeyError as exc:
                raise PlanSerializationError(
                    f"scheduled propagation resource {resource_id!r} is absent from plan resources"
                ) from exc
            if resource.kind != "propagation" or not isinstance(resource.payload, PropagationRecipe):
                raise PlanSerializationError(
                    f"scheduled propagation resource {resource_id!r} has invalid payload"
                )
            result[resource_id] = resource.payload
        return result

    @property
    def required_kernel_resource_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            step.operation.resource_id
            for step in self.physical_plan.steps
            if isinstance(step.operation, MaterializePrimitiveOperation)
            and step.operation.kind == "kernel"
        ))

    def execution_payload(self) -> dict[str, object]:
        return _execution_payload_from_parts(
            config=self.config,
            global_cse=self.global_cse,
            contraction_order=self.contraction_order,
            intermediate_dag=self.intermediate_dag,
            logical_dag=self.logical_dag,
            static_analysis=self.static_analysis,
            schedule=self.schedule,
            physical_plan=self.physical_plan,
        )

    def validate(self) -> None:
        self._validate(payload_proven=False)

    def _validate(self, *, payload_proven: bool) -> None:
        """Check the plan; ``payload_proven`` skips only re-encoding it for its hash.

        Re-encoding proves that the in-memory plan is exactly what the
        manifest's SHA-256 covers. ``_read_concord_plan(..., owned=True)``
        proves this while decoding: the SHA-256 of ``plan.json`` matched the
        manifest, and every record was checked to hold exactly the values
        decoded from it, so re-encoding would give back the same values. Only
        the canonical formatting of ``plan.json`` is then not rechecked.
        """

        if _has_static_validation(self):
            return
        self.manifest.validate()
        try:
            validate_wick_aggregation(
                self.global_cse, config=self.config.wick_aggregation,
                spin_basis=self.config.spin_basis,
                shared_orthonormal_laph_basis=self.config.shared_orthonormal_laph_basis,
                numerical_flavor_groups=self.config.numerical_flavor_groups,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise PlanSerializationError(f"invalid Wick aggregation certificate: {exc}") from exc
        if tuple(self.logical_dag.outputs) != self.manifest.correlator_ids:
            raise PlanSerializationError(
                "manifest correlator ids do not match logical execution outputs"
            )
        if self.contraction_order.source_dag is not self.global_cse:
            raise PlanSerializationError("contraction-order source identity is inconsistent")
        if self.intermediate_dag.source_plan is not self.contraction_order:
            raise PlanSerializationError("partial-CSE source identity is inconsistent")
        if self.logical_dag.source_dag is not self.intermediate_dag:
            raise PlanSerializationError("logical-DAG source identity is inconsistent")
        if self.static_analysis.source_dag is not self.logical_dag:
            raise PlanSerializationError("static-analysis source identity is inconsistent")
        if self.schedule.source_analysis is not self.static_analysis:
            raise PlanSerializationError("schedule source identity is inconsistent")
        if self.physical_plan.source_schedule is not self.schedule:
            raise PlanSerializationError("physical-plan source identity is inconsistent")

        expected_size_model = ContractionSizeModel(
            mode_size=self.config.mode_size,
            spin_size=self.config.spin_size,
        )
        if self.contraction_order.size_model != expected_size_model:
            raise PlanSerializationError("plan contraction size model disagrees with config")
        if self.contraction_order.strategy != self.config.contraction_strategy:
            raise PlanSerializationError("plan contraction strategy disagrees with config")
        if self.contraction_order.memory_limit != self.config.contraction_memory_limit:
            raise PlanSerializationError("plan contraction memory limit disagrees with config")

        for computation_id, component_plan in self.contraction_order.component_plans.items():
            try:
                component = self.global_cse.components[computation_id]
            except KeyError as exc:
                raise PlanSerializationError(
                    f"contraction plan references missing component {computation_id!r}"
                ) from exc
            try:
                component_plan.validate(component)
            except Exception as exc:
                raise PlanSerializationError(
                    f"contraction plan for {computation_id!r} is inconsistent: {exc}"
                ) from exc
        if set(self.contraction_order.component_plans) != set(self.global_cse.components):
            raise PlanSerializationError(
                "contraction-order component map does not match global CSE components"
            )

        for resource_id, resource in self.resources.items():
            if resource.resource_id != resource_id:
                raise PlanSerializationError("plan resource is stored under the wrong id")
            if resource.kind == "kernel":
                if not isinstance(resource.payload, KernelRecipe):
                    raise PlanSerializationError("kernel resource does not carry KernelRecipe")
                if identify_kernel_resource(resource.payload) != resource_id:
                    raise PlanSerializationError(
                        f"kernel resource {resource_id!r} disagrees with recipe identity"
                    )
            elif resource.kind == "spin":
                if not isinstance(resource.payload, SpinTensorRecord):
                    raise PlanSerializationError("spin resource does not carry SpinTensorRecord")
            elif resource.kind == "propagation":
                if not isinstance(resource.payload, PropagationRecipe):
                    raise PlanSerializationError(
                        "propagation resource does not carry PropagationRecipe"
                    )
                if resource.payload.resource_id != resource_id:
                    raise PlanSerializationError(
                        f"propagation resource {resource_id!r} disagrees with recipe identity"
                    )
            else:
                raise PlanSerializationError(
                    f"unsupported resource kind {resource.kind!r} in plan"
                )

        if self.static_analysis.numerical_model.scalar_type != self.config.scalar_type:
            raise PlanSerializationError("plan scalar type disagrees with static analysis")
        if self.static_analysis.numerical_model.element_bytes != self.config.element_bytes:
            raise PlanSerializationError("plan element byte size disagrees with static analysis")
        if self.static_analysis.numerical_model.mode_size != self.config.mode_size:
            raise PlanSerializationError("plan mode size disagrees with static analysis")
        if self.static_analysis.numerical_model.spin_size != self.config.spin_size:
            raise PlanSerializationError("plan spin size disagrees with static analysis")
        if self.schedule.strategy != self.config.schedule_strategy:
            raise PlanSerializationError("plan scheduling strategy disagrees with config")
        if self.schedule.include_dead != self.config.include_dead:
            raise PlanSerializationError("plan dead-node policy disagrees with config")
        if self.physical_plan.memory_space != self.config.memory_space:
            raise PlanSerializationError("plan memory space disagrees with config")
        if self.physical_plan.reuse_strategy != self.config.reuse_strategy:
            raise PlanSerializationError("plan buffer reuse policy disagrees with config")

        try:
            self.logical_dag.validate()
            self.schedule.validate()
            self.physical_plan.validate()
        except Exception as exc:
            raise PlanSerializationError(f"compiled execution plan is inconsistent: {exc}") from exc
        _ = self.required_propagation_resources

        if payload_proven:
            return
        actual_hash = _sha256_bytes(_canonical_json_bytes(self.execution_payload()))
        if actual_hash != self.manifest.plan_sha256:
            raise PlanSerializationError(
                "in-memory plan payload does not match manifest plan_sha256"
            )


def specialize_concord_plan(
    base: ConcordPlan,
    stages: RuntimeExecutionStages,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ConcordPlan:
    """Create an in-memory execution specialization of one portable plan.

    The job/operator/compile fingerprint and global mathematical DAG remain
    unchanged.  Only complete execution stages derived from that exact DAG may
    be substituted.  The resulting plan receives its own execution SHA so
    resumable output cannot silently mix different numerical schedules.
    """

    if not isinstance(base, ConcordPlan):
        raise TypeError("base must be a ConcordPlan")
    if not isinstance(stages, RuntimeExecutionStages):
        raise TypeError("stages must be RuntimeExecutionStages")
    base.validate()
    if stages.contraction_order.source_dag is not base.global_cse:
        raise PlanSerializationError(
            "specialized contraction order must derive from the base global DAG"
        )
    if stages.contraction_order.strategy != base.config.contraction_strategy:
        raise PlanSerializationError(
            "specialized execution strategy must retain the portable meta-strategy"
        )

    execution_payload = _execution_payload_from_parts(
        config=base.config,
        global_cse=base.global_cse,
        contraction_order=stages.contraction_order,
        intermediate_dag=stages.intermediate_dag,
        logical_dag=stages.logical_dag,
        static_analysis=stages.static_analysis,
        schedule=stages.schedule,
        physical_plan=stages.physical_plan,
    )
    plan_sha256 = _sha256_bytes(_canonical_json_bytes(execution_payload))
    merged_diagnostics = dict(base.diagnostics)
    if diagnostics is not None:
        merged_diagnostics.update(
            _canonical_fingerprint_value(dict(diagnostics), role="specialization diagnostics")
        )
    specialized = ConcordPlan(
        manifest=replace(base.manifest, plan_sha256=plan_sha256),
        config=base.config,
        physical_plan=stages.physical_plan,
        diagnostics=merged_diagnostics,
    )
    specialized.validate()
    return specialized


def runtime_compilation_config_from_job(spec: JobSpec) -> RuntimeCompilationConfig:
    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    compile_spec = spec.compile
    return RuntimeCompilationConfig(
        mode_size=compile_spec.mode_size,
        spin_size=compile_spec.spin_size,
        spin_basis=compile_spec.spin_basis,
        numerical_flavor_groups=compile_spec.numerical_flavor_groups,
        wick_aggregation=WickAggregationConfig(**compile_spec.wick_aggregation.model_dump()),
        shared_orthonormal_laph_basis=isinstance(
            spec.data.eigenvectors, HDF5EigenvectorSpec
        ),
        contraction_strategy=compile_spec.contraction.strategy,
        contraction_memory_limit=compile_spec.contraction.memory_limit,
        scalar_type=compile_spec.scalar_type,
        element_bytes=compile_spec.element_bytes,
        schedule_strategy=compile_spec.scheduling.strategy,
        include_dead=compile_spec.scheduling.include_dead,
        memory_space=compile_spec.physical.memory_space,
        reuse_strategy=compile_spec.physical.reuse_strategy,
        physical_memory_limit_bytes=compile_spec.physical.memory_limit_bytes,
    )


def compile_job_plan(spec: JobSpec) -> ConcordPlan:
    """Compile one validated job into a portable in-memory plan exactly once."""

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    validate_numerical_flavor_sources(spec)
    loaded = _load_operator_snapshot(spec)
    fingerprint = compute_job_fingerprint(
        spec, operator_source_sha256=loaded.source_sha256,
        operator_dependencies=loaded.dependencies,
    )
    compilation = compile_correlators(
        loaded.collection,
        runtime_compilation_config_from_job(spec),
    )
    try:
        current_dependencies = discover_operator_dependencies(spec.operators.file, loaded.dependencies)
    except OperatorSourceError as exc:
        raise PlanSerializationError(f"operator sources changed during compilation: {exc}") from exc
    current_fingerprint = compute_job_fingerprint(spec, operator_dependencies=current_dependencies)
    if current_fingerprint != fingerprint:
        raise PlanSerializationError(
            "operator source or helpers changed while the plan was being compiled; retry planning"
        )

    execution_payload = _execution_payload_from_compilation(compilation)
    plan_sha256 = _sha256_bytes(_canonical_json_bytes(execution_payload))
    manifest = PlanManifest(
        plan_format_version=PLAN_FORMAT_VERSION,
        job_schema_version=spec.schema_version,
        concord_version=_concord_version(),
        compiler_semantics_version=COMPILER_SEMANTICS_VERSION,
        job_name=spec.job.name,
        created_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        operator_file=str(spec.operators.file),
        operator_factory=spec.operators.factory,
        operator_source_sha256=fingerprint.operator_source_sha256,
        operator_dependencies=fingerprint.operator_dependencies,
        operator_parameters_sha256=fingerprint.operator_parameters_sha256,
        compile_config_sha256=fingerprint.compile_config_sha256,
        job_fingerprint=fingerprint.job_fingerprint,
        plan_sha256=plan_sha256,
        correlator_ids=tuple(compilation.correlator_ids),
    )
    plan = ConcordPlan(
        manifest=manifest,
        config=compilation.config,
        physical_plan=compilation.physical_plan,
        diagnostics=_diagnostics_from_compilation(compilation),
    )
    plan.validate()
    return plan


def validate_plan_for_job(plan: ConcordPlan, spec: JobSpec) -> None:
    """Reject a saved plan whose compile-relevant job fingerprint is stale."""

    if not isinstance(plan, ConcordPlan):
        raise TypeError("plan must be ConcordPlan")
    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    validate_numerical_flavor_sources(spec)
    plan.validate()
    if plan.config != runtime_compilation_config_from_job(spec):
        raise StalePlanError("compiled plan is stale: configuration disagrees with the job specification")
    try:
        current_dependencies = discover_operator_dependencies(
            spec.operators.file, plan.manifest.operator_dependencies,
        )
        fingerprint = compute_job_fingerprint(spec, operator_dependencies=current_dependencies)
    except (OperatorSourceError, OSError) as exc:
        raise StalePlanError(f"cannot validate operator source dependencies; recompile: {exc}") from exc
    if plan.manifest.job_schema_version != spec.schema_version:
        raise StalePlanError("compiled plan uses a different job schema version")
    if plan.manifest.job_fingerprint != fingerprint.job_fingerprint:
        raise StalePlanError(
            "compiled plan is stale relative to operator source/helpers/factory/parameters "
            "or compile configuration"
        )


def save_concord_plan(
    plan: ConcordPlan,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated ``.concord`` ZIP archive."""

    if not isinstance(plan, ConcordPlan):
        raise TypeError("plan must be ConcordPlan")
    plan.validate()
    target = Path(path).expanduser().resolve(strict=False)
    if target.suffix.lower() != ".concord":
        raise PlanSerializationError("compiled plan path must use .concord extension")
    if target.exists() and not overwrite:
        raise FileExistsError(f"compiled plan already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    plan_bytes = _canonical_json_bytes(plan.execution_payload())
    if _sha256_bytes(plan_bytes) != plan.manifest.plan_sha256:
        raise PlanSerializationError("plan payload changed after manifest construction")
    manifest_bytes = _canonical_json_bytes(plan.manifest.to_dict())
    diagnostics_bytes = _canonical_json_bytes(
        _canonical_fingerprint_value(dict(plan.diagnostics), role="plan diagnostics")
    )

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST_NAME, manifest_bytes)
            archive.writestr(_PLAN_NAME, plan_bytes)
            archive.writestr(_DIAGNOSTICS_NAME, diagnostics_bytes)
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
    return target


def write_job_plan(spec: JobSpec, *, overwrite: bool = False) -> ConcordPlan:
    """Compile and save a job using the ``plan.file`` path from ``job.yaml``."""

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    if spec.plan.file.exists() and not overwrite:
        raise FileExistsError(f"compiled plan already exists: {spec.plan.file}")
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file, overwrite=overwrite)
    return plan


def _read_json_member(archive: ZipFile, name: str) -> tuple[Mapping[str, object], bytes]:
    try:
        raw = archive.read(name)
    except KeyError as exc:
        raise PlanSerializationError(f"plan archive is missing {name}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanSerializationError(f"archive member {name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PlanSerializationError(f"archive member {name} must contain a JSON object")
    return payload, raw


def load_concord_plan(path: str | Path) -> ConcordPlan:
    """Load and validate a portable plan without rerunning compiler passes."""

    with _gc_paused():
        plan = _read_concord_plan(path)
        try:
            plan.validate()
        except PlanSerializationError:
            raise
        except Exception as exc:
            raise PlanSerializationError(
                f"decoded plan failed semantic validation: {exc}"
            ) from exc
    return plan


def _read_concord_plan(path: str | Path, *, owned: bool = False) -> ConcordPlan:
    """Read, hash-check and decode a plan archive; the caller validates the plan.

    ``owned`` decodes the execution sections straight into the read-only
    snapshot form and raises ``_UnsupportedOwnedDecode`` for a plan outside it.
    """

    source = Path(path).expanduser().resolve(strict=False)
    if not source.exists():
        raise PlanSerializationError(f"compiled plan does not exist: {source}")
    if not source.is_file():
        raise PlanSerializationError(f"compiled plan path is not a file: {source}")
    try:
        with ZipFile(source, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise PlanSerializationError("plan archive contains duplicate members")
            if set(names) != _REQUIRED_ARCHIVE_MEMBERS:
                missing = sorted(_REQUIRED_ARCHIVE_MEMBERS - set(names))
                extra = sorted(set(names) - _REQUIRED_ARCHIVE_MEMBERS)
                raise PlanSerializationError(
                    f"plan archive members disagree; missing={missing}, extra={extra}"
                )
            manifest_payload, _ = _read_json_member(archive, _MANIFEST_NAME)
            manifest = PlanManifest.from_dict(manifest_payload)
            execution_payload, plan_bytes = _read_json_member(archive, _PLAN_NAME)
            diagnostics_payload, _ = _read_json_member(archive, _DIAGNOSTICS_NAME)
    except BadZipFile as exc:
        raise PlanSerializationError(f"compiled plan is not a valid ZIP archive: {source}") from exc
    except OSError as exc:
        raise PlanSerializationError(f"cannot read compiled plan {source}: {exc}") from exc

    actual_sha = _sha256_bytes(plan_bytes)
    if actual_sha != manifest.plan_sha256:
        raise PlanSerializationError(
            "plan.json SHA-256 does not match manifest; archive is corrupted or modified"
        )

    try:
        (
            config,
            global_cse,
            contraction_order,
            intermediate_dag,
            logical_dag,
            static_analysis,
            schedule,
            physical_plan,
        ) = _decode_execution_payload(execution_payload, owned=owned)

        if contraction_order.source_dag is not global_cse:
            raise PlanSerializationError("decoded contraction-order source identity was lost")
        if intermediate_dag.source_plan is not contraction_order:
            raise PlanSerializationError("decoded partial-CSE source identity was lost")
        if logical_dag.source_dag is not intermediate_dag:
            raise PlanSerializationError("decoded logical-DAG source identity was lost")
        if static_analysis.source_dag is not logical_dag:
            raise PlanSerializationError("decoded static-analysis source identity was lost")
        if schedule.source_analysis is not static_analysis:
            raise PlanSerializationError("decoded schedule source identity was lost")
        if physical_plan.source_schedule is not schedule:
            raise PlanSerializationError("decoded physical-plan source identity was lost")

        plan = ConcordPlan(
            manifest=manifest,
            config=config,
            physical_plan=physical_plan,
            diagnostics=dict(diagnostics_payload),
        )
    except (PlanSerializationError, _UnsupportedOwnedDecode):
        raise
    except Exception as exc:
        raise PlanSerializationError(
            f"decoded plan failed semantic validation: {exc}"
        ) from exc
    return plan


__all__ = [
    "PLAN_FORMAT_VERSION",
    "EXECUTION_SCHEMA_VERSION",
    "COMPILER_SEMANTICS_VERSION",
    "PlanSerializationError",
    "StalePlanError",
    "PlanFingerprint",
    "PlanManifest",
    "ConcordPlan",
    "specialize_concord_plan",
    "compute_job_fingerprint",
    "runtime_compilation_config_from_job",
    "compile_job_plan",
    "validate_plan_for_job",
    "save_concord_plan",
    "write_job_plan",
    "load_concord_plan",
]
