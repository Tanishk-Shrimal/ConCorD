"""Versioned device-local calibration identities and selection records.

Calibration is an execution-policy input, never a physics identity.  Stable
hardware/software facts participate in the cache key; volatile free-memory
measurements do not.  Feasibility is re-evaluated from a fresh hardware
snapshot before execution, while a change in tensor graph, dtype, deployment
tiling, device, or software stack produces a cache miss.

This module owns no benchmark fixtures and contains no device-name or
hadron-type policy.  Callers provide candidate measurements from the exact
physical graph they intend to execute.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from math import isfinite
import os
from pathlib import Path
import platform
import statistics
import tempfile
from typing import Mapping, Sequence

import torch

from concord_lqcd.backend.planner.physical import PhysicalExecutionPlan

from .hardware import HardwareProfile
from .resource_planning import DeviceExecutionPlan


CALIBRATION_CACHE_SCHEMA_VERSION = 1
_DEFAULT_CACHE_NAME = "runtime-calibration-v1.json"


class CalibrationCacheError(ValueError):
    """Raised when a calibration record cannot be validated or persisted."""


def _concord_version() -> str:
    try:
        return package_version("concord-lqcd")
    except PackageNotFoundError:
        return "uninstalled"


def _canonical(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise CalibrationCacheError("calibration JSON cannot contain non-finite values")
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if is_dataclass(value):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _canonical(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            return {
                "$mapping": [
                    [_canonical(key), _canonical(item)]
                    for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
                ]
            }
        return {
            key: _canonical(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    raise CalibrationCacheError(
        f"unsupported calibration fingerprint value {type(value).__name__}"
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CalibrationCacheError(f"invalid canonical calibration value: {exc}") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def default_calibration_cache_path() -> Path:
    """Return the user-local cache path without creating directories."""

    configured = os.environ.get("XDG_CACHE_HOME")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return root / "concord-lqcd" / _DEFAULT_CACHE_NAME


def physical_plan_fingerprint(plan: PhysicalExecutionPlan) -> str:
    """Fingerprint execution-relevant physical state, excluding source objects."""

    if not isinstance(plan, PhysicalExecutionPlan):
        raise TypeError("plan must be a PhysicalExecutionPlan")
    plan.validate()
    payload = {
        "memory_space": plan.memory_space,
        "reuse_strategy": plan.reuse_strategy,
        "buffers": dict(plan.buffers),
        "value_bindings": dict(plan.value_bindings),
        "steps": plan.steps,
        "outputs": dict(plan.outputs),
        "statistics": plan.statistics,
    }
    return _sha256(payload)


def performance_environment_fingerprint(
    hardware: HardwareProfile,
    device_name: str,
) -> str:
    """Fingerprint stable device/software facts relevant to measured timing.

    Currently available RAM/VRAM and warnings are intentionally excluded: they
    are volatile feasibility inputs.  Total capacity and the configured
    PyTorch thread pools are included because they distinguish execution
    environments whose measured performance must not be shared blindly.
    """

    if not isinstance(hardware, HardwareProfile):
        raise TypeError("hardware must be a HardwareProfile")
    try:
        device = hardware.device(device_name)
    except KeyError as exc:
        raise CalibrationCacheError(
            f"cannot fingerprint undetected device {device_name!r}"
        ) from exc
    payload = {
        "schema_version": CALIBRATION_CACHE_SCHEMA_VERSION,
        "system": hardware.system,
        "machine": hardware.machine,
        "kernel_release": platform.release(),
        "host": {
            "name": hardware.host.name,
            "total_memory_bytes": hardware.host.total_memory_bytes,
            "logical_processors": hardware.host.logical_processors,
        },
        "device": {
            "kind": device.kind,
            "name": device.name,
            "total_memory_bytes": device.total_memory_bytes,
            "logical_processors": device.logical_processors,
            "compute_capability": device.compute_capability,
            "multiprocessor_count": device.multiprocessor_count,
        },
        "software": {
            "concord_lqcd": _concord_version(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "torch_num_threads": hardware.torch_num_threads,
            "torch_num_interop_threads": hardware.torch_num_interop_threads,
        },
    }
    return _sha256(payload)


def deployment_fingerprint(deployment: DeviceExecutionPlan) -> str:
    """Fingerprint a resident or exactly tiled runtime deployment."""

    if not isinstance(deployment, DeviceExecutionPlan):
        raise TypeError("deployment must be a DeviceExecutionPlan")
    tiled = deployment.tiled_plan
    payload = {
        "selected_device": deployment.selected_device,
        "storage_device": deployment.storage_device,
        "execution_mode": deployment.execution_mode,
        "tiling": None
        if tiled is None
        else {
            "device_memory_limit_bytes": tiled.device_memory_limit_bytes,
            "operations": dict(tiled.operations),
            "peak_device_bytes": tiled.peak_device_bytes,
            "full_peak_device_bytes": tiled.full_peak_device_bytes,
            "tiled_operation_count": tiled.tiled_operation_count,
            "total_contraction_calls": tiled.total_contraction_calls,
        },
    }
    return _sha256(payload)


@dataclass(frozen=True)
class CalibrationKey:
    """Exact applicability key for one candidate timing distribution."""

    workload_fingerprint: str
    candidate_fingerprint: str
    environment_fingerprint: str
    deployment_fingerprint: str
    scalar_type: str
    work_units_per_sample: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "workload_fingerprint",
            "candidate_fingerprint",
            "environment_fingerprint",
            "deployment_fingerprint",
            "scalar_type",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise CalibrationCacheError(f"{field_name} must be a non-empty string")
        if (
            not isinstance(self.work_units_per_sample, int)
            or isinstance(self.work_units_per_sample, bool)
            or self.work_units_per_sample <= 0
        ):
            raise CalibrationCacheError("work_units_per_sample must be positive")

    @property
    def digest(self) -> str:
        return _sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalibrationKey":
        expected = {item.name for item in fields(cls)}
        if set(payload) != expected:
            raise CalibrationCacheError("calibration key fields are incompatible")
        return cls(**dict(payload))


@dataclass(frozen=True)
class CalibrationRecord:
    """Measured seconds-per-work-unit samples for one exact key."""

    key: CalibrationKey
    candidate_name: str
    samples_seconds_per_work_unit: tuple[float, ...]
    created_utc: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples_seconds_per_work_unit",
            tuple(self.samples_seconds_per_work_unit),
        )
        if not isinstance(self.key, CalibrationKey):
            raise TypeError("calibration record requires a CalibrationKey")
        if not isinstance(self.candidate_name, str) or not self.candidate_name:
            raise CalibrationCacheError("candidate_name must be non-empty")
        if not self.samples_seconds_per_work_unit or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
            or value <= 0.0
            for value in self.samples_seconds_per_work_unit
        ):
            raise CalibrationCacheError(
                "calibration samples must contain positive finite seconds"
            )
        if not isinstance(self.created_utc, str) or not self.created_utc:
            raise CalibrationCacheError("created_utc must be non-empty")

    @classmethod
    def measured(
        cls,
        *,
        key: CalibrationKey,
        candidate_name: str,
        samples_seconds_per_work_unit: Sequence[float],
    ) -> "CalibrationRecord":
        return cls(
            key=key,
            candidate_name=candidate_name,
            samples_seconds_per_work_unit=tuple(
                float(value) for value in samples_seconds_per_work_unit
            ),
            created_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

    @property
    def median_seconds_per_work_unit(self) -> float:
        return float(statistics.median(self.samples_seconds_per_work_unit))

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "candidate_name": self.candidate_name,
            "samples_seconds_per_work_unit": list(
                self.samples_seconds_per_work_unit
            ),
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalibrationRecord":
        expected = {
            "key",
            "candidate_name",
            "samples_seconds_per_work_unit",
            "created_utc",
        }
        if set(payload) != expected:
            raise CalibrationCacheError("calibration record fields are incompatible")
        raw_key = payload["key"]
        raw_samples = payload["samples_seconds_per_work_unit"]
        if not isinstance(raw_key, Mapping) or not isinstance(raw_samples, list):
            raise CalibrationCacheError("calibration record contains invalid nested data")
        return cls(
            key=CalibrationKey.from_dict(raw_key),
            candidate_name=payload["candidate_name"],
            samples_seconds_per_work_unit=tuple(raw_samples),
            created_utc=payload["created_utc"],
        )


class CalibrationCache:
    """Small single-process JSON cache with atomic replacement writes."""

    def __init__(
        self,
        path: str | Path,
        *,
        records: Sequence[CalibrationRecord] = (),
        warnings: Sequence[str] = (),
    ) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self._records: dict[str, CalibrationRecord] = {}
        self.warnings = tuple(warnings)
        for record in records:
            self.put(record)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationCache":
        source = Path(path).expanduser().resolve(strict=False)
        if not source.exists():
            return cls(source)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise CalibrationCacheError("cache root must be a JSON object")
            if payload.get("schema_version") != CALIBRATION_CACHE_SCHEMA_VERSION:
                raise CalibrationCacheError("cache schema version is incompatible")
            raw_records = payload.get("records")
            if not isinstance(raw_records, list):
                raise CalibrationCacheError("cache records must be a JSON array")
            records = []
            for value in raw_records:
                if not isinstance(value, dict):
                    raise CalibrationCacheError("cache record must be a JSON object")
                records.append(CalibrationRecord.from_dict(value))
            return cls(source, records=records)
        except (OSError, UnicodeError, json.JSONDecodeError, CalibrationCacheError) as exc:
            return cls(
                source,
                warnings=(f"ignored invalid calibration cache {source}: {exc}",),
            )

    def get(self, key: CalibrationKey) -> CalibrationRecord | None:
        if not isinstance(key, CalibrationKey):
            raise TypeError("key must be a CalibrationKey")
        return self._records.get(key.digest)

    def put(self, record: CalibrationRecord) -> None:
        if not isinstance(record, CalibrationRecord):
            raise TypeError("record must be a CalibrationRecord")
        self._records[record.key.digest] = record

    @property
    def records(self) -> tuple[CalibrationRecord, ...]:
        return tuple(
            self._records[key]
            for key in sorted(self._records)
        )

    def save(self) -> Path:
        payload = {
            "schema_version": CALIBRATION_CACHE_SCHEMA_VERSION,
            "records": [record.to_dict() for record in self.records],
        }
        encoded = _canonical_json_bytes(payload) + b"\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
        except OSError as exc:
            raise CalibrationCacheError(
                f"cannot prepare calibration cache {self.path}: {exc}"
            ) from exc
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            finally:
                raise CalibrationCacheError(
                    f"cannot persist calibration cache {self.path}: {exc}"
                ) from exc
        return self.path


@dataclass(frozen=True)
class CalibrationCandidate:
    """One feasible candidate and its deterministic fallback rank."""

    name: str
    key: CalibrationKey
    fallback_rank: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise CalibrationCacheError("calibration candidate name must be non-empty")
        if not isinstance(self.key, CalibrationKey):
            raise TypeError("calibration candidate requires a CalibrationKey")
        if not isinstance(self.fallback_rank, tuple) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in self.fallback_rank
        ):
            raise CalibrationCacheError("fallback_rank must be a tuple of integers")


@dataclass(frozen=True)
class CalibrationSelection:
    """Selected candidate plus provenance for inspection and testing."""

    candidate: CalibrationCandidate
    source: str
    median_seconds_per_work_unit: float | None
    missing_candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in ("calibrated", "analytical_fallback"):
            raise CalibrationCacheError("unknown calibration-selection source")
        if self.source == "calibrated":
            if self.median_seconds_per_work_unit is None:
                raise CalibrationCacheError("calibrated selection requires a timing")
            if self.missing_candidates:
                raise CalibrationCacheError("calibrated selection cannot have missing records")
        elif self.median_seconds_per_work_unit is not None:
            raise CalibrationCacheError("analytical fallback cannot claim a timing")


def select_calibration_candidate(
    candidates: Sequence[CalibrationCandidate],
    cache: CalibrationCache,
) -> CalibrationSelection:
    """Select the fastest completely calibrated set or use a safe fallback.

    Partial timing coverage is never compared with analytical estimates.  If
    even one feasible candidate is unmeasured, selection falls back
    deterministically until the caller has calibrated the complete set.
    """

    values = tuple(candidates)
    if not values:
        raise CalibrationCacheError("at least one feasible candidate is required")
    if len({candidate.name for candidate in values}) != len(values):
        raise CalibrationCacheError("calibration candidate names must be unique")
    if not isinstance(cache, CalibrationCache):
        raise TypeError("cache must be a CalibrationCache")

    records = {
        candidate.name: (
            record
            if (record := cache.get(candidate.key)) is not None
            and record.candidate_name == candidate.name
            else None
        )
        for candidate in values
    }
    missing = tuple(name for name, record in records.items() if record is None)
    if not missing:
        selected = min(
            values,
            key=lambda candidate: (
                records[candidate.name].median_seconds_per_work_unit,
                candidate.name,
            ),
        )
        timing = records[selected.name].median_seconds_per_work_unit
        return CalibrationSelection(
            candidate=selected,
            source="calibrated",
            median_seconds_per_work_unit=timing,
        )

    selected = min(values, key=lambda candidate: (candidate.fallback_rank, candidate.name))
    return CalibrationSelection(
        candidate=selected,
        source="analytical_fallback",
        median_seconds_per_work_unit=None,
        missing_candidates=missing,
    )


__all__ = [
    "CALIBRATION_CACHE_SCHEMA_VERSION",
    "CalibrationCache",
    "CalibrationCacheError",
    "CalibrationCandidate",
    "CalibrationKey",
    "CalibrationRecord",
    "CalibrationSelection",
    "default_calibration_cache_path",
    "deployment_fingerprint",
    "performance_environment_fingerprint",
    "physical_plan_fingerprint",
    "select_calibration_candidate",
]
