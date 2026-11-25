"""Versioned, machine-readable records for completed ConCorD invocations.

The numerical result remains in the job HDF5 output.  A run record is a small
JSON sidecar containing execution provenance, exact top-level wall-time
attribution, and the runtime decisions needed by human-facing reports.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping


if TYPE_CHECKING:
    from concord_lqcd.backend.runtime.hardware import HardwareProfile

    from .runner import JobRunSummary


RUN_RECORD_FORMAT_VERSION = 3


class RunRecordError(ValueError):
    """Raised when a run record is invalid or cannot be read safely."""


def _non_negative_seconds(name: str, value: float) -> float:
    seconds = float(value)
    if not isfinite(seconds) or seconds < 0.0:
        raise RunRecordError(f"{name} must be a finite non-negative duration")
    return seconds


def _optional_positive_integer(name: str, value: int | None) -> int | None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
    ):
        raise RunRecordError(f"{name} must be a positive integer or None")
    return value


@dataclass(frozen=True)
class JobRunHardware:
    """Stable hardware identity and resource limits observed for one run."""

    host_name: str
    logical_cpus: int | None
    host_memory_bytes: int | None
    compute_device: str
    compute_name: str
    compute_memory_bytes: int | None
    torch_num_threads: int | None
    torch_num_interop_threads: int | None

    def __post_init__(self) -> None:
        for name in ("host_name", "compute_device", "compute_name"):
            if not getattr(self, name):
                raise RunRecordError(f"{name} cannot be empty")
        for name in (
            "logical_cpus",
            "host_memory_bytes",
            "compute_memory_bytes",
            "torch_num_threads",
            "torch_num_interop_threads",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_integer(name, getattr(self, name)),
            )

    @classmethod
    def from_profile(
        cls,
        profile: "HardwareProfile",
        *,
        selected_device: str,
    ) -> "JobRunHardware":
        if selected_device == "not-required":
            compute_name = "No numerical execution required"
            compute_memory_bytes = None
        else:
            try:
                compute = profile.device(selected_device)
            except KeyError:
                compute_name = selected_device
                compute_memory_bytes = None
            else:
                compute_name = compute.name
                compute_memory_bytes = compute.total_memory_bytes
        return cls(
            host_name=profile.host.name,
            logical_cpus=profile.host.logical_processors,
            host_memory_bytes=profile.host.total_memory_bytes,
            compute_device=selected_device,
            compute_name=compute_name,
            compute_memory_bytes=compute_memory_bytes,
            torch_num_threads=profile.torch_num_threads,
            torch_num_interop_threads=profile.torch_num_interop_threads,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_name": self.host_name,
            "logical_cpus": self.logical_cpus,
            "host_memory_bytes": self.host_memory_bytes,
            "compute_device": self.compute_device,
            "compute_name": self.compute_name,
            "compute_memory_bytes": self.compute_memory_bytes,
            "torch_num_threads": self.torch_num_threads,
            "torch_num_interop_threads": self.torch_num_interop_threads,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JobRunHardware":
        try:
            return cls(
                host_name=str(payload["host_name"]),
                logical_cpus=payload["logical_cpus"],
                host_memory_bytes=payload["host_memory_bytes"],
                compute_device=str(payload["compute_device"]),
                compute_name=str(payload["compute_name"]),
                compute_memory_bytes=payload["compute_memory_bytes"],
                torch_num_threads=payload["torch_num_threads"],
                torch_num_interop_threads=payload["torch_num_interop_threads"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunRecordError("invalid run hardware payload") from exc


@dataclass(frozen=True)
class JobRunTiming:
    """Exclusive top-level timing categories for one ``run_job`` invocation.

    The named host-observed categories never overlap. ``other_seconds`` closes
    the accounting identity, so all categories sum to
    ``execution_wall_seconds``. Detailed asynchronous I/O or device events,
    when collected separately, must not be added to these values because they
    may overlap the numerical-execution wall clock. In particular, deferred
    CUDA completion can be observed while result tensors are finalized.
    """

    execution_wall_seconds: float
    result_management_seconds: float
    runtime_planning_seconds: float
    data_preparation_seconds: float
    input_wait_seconds: float
    numerical_execution_seconds: float
    output_writing_seconds: float
    other_seconds: float

    def __post_init__(self) -> None:
        names = (
            "execution_wall_seconds",
            "result_management_seconds",
            "runtime_planning_seconds",
            "data_preparation_seconds",
            "input_wait_seconds",
            "numerical_execution_seconds",
            "output_writing_seconds",
            "other_seconds",
        )
        for name in names:
            object.__setattr__(self, name, _non_negative_seconds(name, getattr(self, name)))
        accounted = sum(seconds for _, seconds in self.stages)
        tolerance = max(1e-9, self.execution_wall_seconds * 1e-9)
        if abs(accounted - self.execution_wall_seconds) > tolerance:
            raise RunRecordError(
                "execution timing categories must sum to execution wall time"
            )

    @classmethod
    def measured(
        cls,
        *,
        execution_wall_seconds: float,
        result_management_seconds: float,
        runtime_planning_seconds: float,
        data_preparation_seconds: float,
        input_wait_seconds: float,
        numerical_execution_seconds: float,
        output_writing_seconds: float,
    ) -> "JobRunTiming":
        total = _non_negative_seconds(
            "execution_wall_seconds", execution_wall_seconds
        )
        measured = (
            _non_negative_seconds(
                "result_management_seconds", result_management_seconds
            ),
            _non_negative_seconds(
                "runtime_planning_seconds", runtime_planning_seconds
            ),
            _non_negative_seconds(
                "data_preparation_seconds", data_preparation_seconds
            ),
            _non_negative_seconds("input_wait_seconds", input_wait_seconds),
            _non_negative_seconds(
                "numerical_execution_seconds", numerical_execution_seconds
            ),
            _non_negative_seconds("output_writing_seconds", output_writing_seconds),
        )
        accounted = sum(measured)
        tolerance = max(1e-9, total * 1e-9)
        if accounted > total + tolerance:
            raise RunRecordError(
                "measured execution stages exceed the execution wall time"
            )
        other = max(0.0, total - accounted)
        return cls(
            execution_wall_seconds=total,
            result_management_seconds=measured[0],
            runtime_planning_seconds=measured[1],
            data_preparation_seconds=measured[2],
            input_wait_seconds=measured[3],
            numerical_execution_seconds=measured[4],
            output_writing_seconds=measured[5],
            other_seconds=other,
        )

    @property
    def stages(self) -> tuple[tuple[str, float], ...]:
        return (
            ("result_management", self.result_management_seconds),
            ("runtime_planning", self.runtime_planning_seconds),
            ("data_preparation", self.data_preparation_seconds),
            ("input_wait", self.input_wait_seconds),
            ("numerical_execution", self.numerical_execution_seconds),
            ("output_writing", self.output_writing_seconds),
            ("other", self.other_seconds),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "execution_wall_seconds": self.execution_wall_seconds,
            **{f"{name}_seconds": seconds for name, seconds in self.stages},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JobRunTiming":
        try:
            return cls(
                execution_wall_seconds=payload["execution_wall_seconds"],
                result_management_seconds=payload["result_management_seconds"],
                runtime_planning_seconds=payload["runtime_planning_seconds"],
                data_preparation_seconds=payload["data_preparation_seconds"],
                input_wait_seconds=payload["input_wait_seconds"],
                numerical_execution_seconds=payload[
                    "numerical_execution_seconds"
                ],
                output_writing_seconds=payload["output_writing_seconds"],
                other_seconds=payload["other_seconds"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunRecordError("invalid execution timing payload") from exc


@dataclass(frozen=True)
class JobRunWorker:
    """One worker process of a multi-process CPU run.

    ``cpus`` is empty when the worker was not pinned. ``max_rss_bytes`` counts
    pages shared with the parent; ``private_bytes`` counts only the worker's own
    pages at its end, so it shows how much of the shared plan was copied;
    ``peak_private_bytes`` is the highest of the worker's own memory during its
    run, which is what each additional worker costs the node.
    """

    index: int
    threads: int
    cpus: tuple[int, ...]
    work_units: int
    timing: JobRunTiming
    max_rss_bytes: int | None = None
    private_bytes: int | None = None
    peak_private_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpus", tuple(self.cpus))
        for name in ("index", "work_units"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RunRecordError(f"worker {name} must be a non-negative integer")
        if not isinstance(self.threads, int) or isinstance(self.threads, bool) or self.threads <= 0:
            raise RunRecordError("worker threads must be a positive integer")
        if any(not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in self.cpus):
            raise RunRecordError("worker CPUs must be non-negative integers")
        for name in ("max_rss_bytes", "private_bytes", "peak_private_bytes"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise RunRecordError(f"worker {name} must be a non-negative integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "threads": self.threads,
            "cpus": list(self.cpus),
            "work_units": self.work_units,
            "timing": self.timing.to_dict(),
            "max_rss_bytes": self.max_rss_bytes,
            "private_bytes": self.private_bytes,
            "peak_private_bytes": self.peak_private_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JobRunWorker":
        try:
            return cls(
                index=payload["index"],
                threads=payload["threads"],
                cpus=tuple(payload["cpus"]),
                work_units=payload["work_units"],
                timing=JobRunTiming.from_dict(payload["timing"]),
                max_rss_bytes=payload.get("max_rss_bytes"),
                private_bytes=payload.get("private_bytes"),
                peak_private_bytes=payload.get("peak_private_bytes"),
            )
        except RunRecordError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise RunRecordError("invalid worker payload") from exc


@dataclass(frozen=True)
class JobRunRecord:
    """Portable summary used to regenerate a post-run visualization."""

    job_name: str
    output_file: Path
    plan_sha256: str
    plan_action: str
    plan_preparation_seconds: float
    timing: JobRunTiming
    total_work_units: int
    executed_work_units: int
    skipped_work_units: int
    completed_work_units: int
    correlator_ids: tuple[str, ...]
    contraction_path: str
    contraction_selection_source: str
    time_batch_size: int
    time_batch_selection_source: str
    executed_batches: int
    selected_device: str
    storage_device: str
    execution_mode: str
    memory_verified: bool
    known_working_set_bytes: int
    peak_tile_bytes: int | None
    tiled_operations: int
    resident_pool_peak_bytes: int
    primitive_cache_hits: int
    primitive_cache_misses: int
    hdf5_file_opens: int
    hdf5_file_reuses: int
    propagation_prefetch_bytes: int
    propagation_prefetch_wait_seconds: float
    kernel_cache_enabled: bool = False
    kernel_cache_hits: int = 0
    kernel_cache_misses: int = 0
    kernel_cache_writes: int = 0
    kernel_cache_invalid_entries: int = 0
    kernel_cache_write_failures: int = 0
    hardware: JobRunHardware | None = None
    workers: tuple[JobRunWorker, ...] = ()
    worker_selection: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "workers", tuple(self.workers))
        if not self.job_name:
            raise RunRecordError("job_name cannot be empty")
        if not self.plan_sha256:
            raise RunRecordError("plan_sha256 cannot be empty")
        object.__setattr__(
            self,
            "plan_preparation_seconds",
            _non_negative_seconds(
                "plan_preparation_seconds", self.plan_preparation_seconds
            ),
        )
        object.__setattr__(
            self,
            "propagation_prefetch_wait_seconds",
            _non_negative_seconds(
                "propagation_prefetch_wait_seconds",
                self.propagation_prefetch_wait_seconds,
            ),
        )
        integer_fields = (
            "total_work_units",
            "executed_work_units",
            "skipped_work_units",
            "completed_work_units",
            "executed_batches",
            "known_working_set_bytes",
            "tiled_operations",
            "resident_pool_peak_bytes",
            "primitive_cache_hits",
            "primitive_cache_misses",
            "hdf5_file_opens",
            "hdf5_file_reuses",
            "propagation_prefetch_bytes",
            "kernel_cache_hits",
            "kernel_cache_misses",
            "kernel_cache_writes",
            "kernel_cache_invalid_entries",
            "kernel_cache_write_failures",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RunRecordError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.time_batch_size, int)
            or isinstance(self.time_batch_size, bool)
            or self.time_batch_size <= 0
        ):
            raise RunRecordError("time_batch_size must be positive")
        if not isinstance(self.memory_verified, bool):
            raise RunRecordError("memory_verified must be boolean")
        if not isinstance(self.kernel_cache_enabled, bool):
            raise RunRecordError("kernel_cache_enabled must be boolean")
        if self.peak_tile_bytes is not None and (
            not isinstance(self.peak_tile_bytes, int)
            or isinstance(self.peak_tile_bytes, bool)
            or self.peak_tile_bytes < 0
        ):
            raise RunRecordError("peak_tile_bytes must be non-negative or None")
        if self.completed_work_units > self.total_work_units:
            raise RunRecordError("completed work exceeds the declared work grid")

    @classmethod
    def from_summary(
        cls,
        summary: "JobRunSummary",
        *,
        job_name: str,
        plan_action: str,
        plan_preparation_seconds: float,
    ) -> "JobRunRecord":
        return cls(
            job_name=job_name,
            output_file=summary.output_file,
            plan_sha256=summary.plan_sha256,
            plan_action=plan_action,
            plan_preparation_seconds=plan_preparation_seconds,
            timing=summary.timing,
            total_work_units=summary.total_work_units,
            executed_work_units=summary.executed_work_units,
            skipped_work_units=summary.skipped_work_units,
            completed_work_units=summary.completed_work_units,
            correlator_ids=summary.correlator_ids,
            contraction_path=summary.contraction_path,
            contraction_selection_source=summary.contraction_selection_source,
            time_batch_size=summary.time_batch_size,
            time_batch_selection_source=summary.time_batch_selection_source,
            executed_batches=summary.executed_batches,
            selected_device=summary.selected_device,
            storage_device=summary.storage_device,
            execution_mode=summary.execution_mode,
            memory_verified=summary.memory_verified,
            known_working_set_bytes=summary.known_working_set_bytes,
            peak_tile_bytes=summary.peak_tile_bytes,
            tiled_operations=summary.tiled_operations,
            resident_pool_peak_bytes=summary.resident_pool_peak_bytes,
            primitive_cache_hits=summary.primitive_cache_hits,
            primitive_cache_misses=summary.primitive_cache_misses,
            hdf5_file_opens=summary.hdf5_file_opens,
            hdf5_file_reuses=summary.hdf5_file_reuses,
            propagation_prefetch_bytes=summary.propagation_prefetch_bytes,
            propagation_prefetch_wait_seconds=(
                summary.propagation_prefetch_wait_seconds
            ),
            kernel_cache_enabled=summary.kernel_cache_enabled,
            kernel_cache_hits=summary.kernel_cache_hits,
            kernel_cache_misses=summary.kernel_cache_misses,
            kernel_cache_writes=summary.kernel_cache_writes,
            kernel_cache_invalid_entries=summary.kernel_cache_invalid_entries,
            kernel_cache_write_failures=summary.kernel_cache_write_failures,
            hardware=(
                JobRunHardware.from_profile(
                    summary.hardware_profile,
                    selected_device=summary.selected_device,
                )
                if summary.hardware_profile is not None
                else None
            ),
            workers=summary.worker_reports,
            worker_selection=summary.worker_selection,
        )

    @property
    def all_complete(self) -> bool:
        return self.completed_work_units == self.total_work_units

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": RUN_RECORD_FORMAT_VERSION,
            "job_name": self.job_name,
            "output_file": str(self.output_file),
            "plan_sha256": self.plan_sha256,
            "plan_action": self.plan_action,
            "plan_preparation_seconds": self.plan_preparation_seconds,
            "timing": self.timing.to_dict(),
            "work": {
                "total": self.total_work_units,
                "executed": self.executed_work_units,
                "skipped": self.skipped_work_units,
                "completed": self.completed_work_units,
                "correlator_ids": list(self.correlator_ids),
            },
            "execution": {
                "contraction_path": self.contraction_path,
                "contraction_selection_source": self.contraction_selection_source,
                "time_batch_size": self.time_batch_size,
                "time_batch_selection_source": self.time_batch_selection_source,
                "executed_batches": self.executed_batches,
                "selected_device": self.selected_device,
                "storage_device": self.storage_device,
                "execution_mode": self.execution_mode,
                "workers": [worker.to_dict() for worker in self.workers],
                "worker_selection": self.worker_selection,
            },
            "resources": {
                "memory_verified": self.memory_verified,
                "known_working_set_bytes": self.known_working_set_bytes,
                "peak_tile_bytes": self.peak_tile_bytes,
                "tiled_operations": self.tiled_operations,
                "resident_pool_peak_bytes": self.resident_pool_peak_bytes,
                "primitive_cache_hits": self.primitive_cache_hits,
                "primitive_cache_misses": self.primitive_cache_misses,
                "hdf5_file_opens": self.hdf5_file_opens,
                "hdf5_file_reuses": self.hdf5_file_reuses,
                "propagation_prefetch_bytes": self.propagation_prefetch_bytes,
                "propagation_prefetch_wait_seconds": (
                    self.propagation_prefetch_wait_seconds
                ),
                "kernel_cache_enabled": self.kernel_cache_enabled,
                "kernel_cache_hits": self.kernel_cache_hits,
                "kernel_cache_misses": self.kernel_cache_misses,
                "kernel_cache_writes": self.kernel_cache_writes,
                "kernel_cache_invalid_entries": self.kernel_cache_invalid_entries,
                "kernel_cache_write_failures": self.kernel_cache_write_failures,
            },
            "hardware": self.hardware.to_dict() if self.hardware is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JobRunRecord":
        try:
            version = payload["format_version"]
            if version not in (1, 2, RUN_RECORD_FORMAT_VERSION):
                raise RunRecordError(
                    f"unsupported run-record format version {version!r}"
                )
            work = payload["work"]
            execution = payload["execution"]
            resources = payload["resources"]
            if not all(
                isinstance(item, Mapping)
                for item in (work, execution, resources, payload["timing"])
            ):
                raise TypeError
            return cls(
                job_name=str(payload["job_name"]),
                output_file=Path(payload["output_file"]),
                plan_sha256=str(payload["plan_sha256"]),
                plan_action=str(payload["plan_action"]),
                plan_preparation_seconds=payload["plan_preparation_seconds"],
                timing=JobRunTiming.from_dict(payload["timing"]),
                total_work_units=work["total"],
                executed_work_units=work["executed"],
                skipped_work_units=work["skipped"],
                completed_work_units=work["completed"],
                correlator_ids=tuple(str(value) for value in work["correlator_ids"]),
                contraction_path=str(execution["contraction_path"]),
                contraction_selection_source=str(
                    execution["contraction_selection_source"]
                ),
                time_batch_size=execution["time_batch_size"],
                time_batch_selection_source=str(
                    execution["time_batch_selection_source"]
                ),
                executed_batches=execution["executed_batches"],
                selected_device=str(execution["selected_device"]),
                storage_device=str(execution["storage_device"]),
                execution_mode=str(execution["execution_mode"]),
                memory_verified=resources["memory_verified"],
                known_working_set_bytes=resources["known_working_set_bytes"],
                peak_tile_bytes=resources["peak_tile_bytes"],
                tiled_operations=resources["tiled_operations"],
                resident_pool_peak_bytes=resources["resident_pool_peak_bytes"],
                primitive_cache_hits=resources["primitive_cache_hits"],
                primitive_cache_misses=resources["primitive_cache_misses"],
                hdf5_file_opens=resources["hdf5_file_opens"],
                hdf5_file_reuses=resources["hdf5_file_reuses"],
                propagation_prefetch_bytes=resources[
                    "propagation_prefetch_bytes"
                ],
                propagation_prefetch_wait_seconds=resources[
                    "propagation_prefetch_wait_seconds"
                ],
                kernel_cache_enabled=resources.get("kernel_cache_enabled", False),
                kernel_cache_hits=resources.get("kernel_cache_hits", 0),
                kernel_cache_misses=resources.get("kernel_cache_misses", 0),
                kernel_cache_writes=resources.get("kernel_cache_writes", 0),
                kernel_cache_invalid_entries=resources.get(
                    "kernel_cache_invalid_entries", 0
                ),
                kernel_cache_write_failures=resources.get(
                    "kernel_cache_write_failures", 0
                ),
                hardware=(
                    JobRunHardware.from_dict(payload["hardware"])
                    if version >= 2 and payload.get("hardware") is not None
                    else None
                ),
                workers=tuple(
                    JobRunWorker.from_dict(worker)
                    for worker in execution.get("workers", ())
                ),
                worker_selection=execution.get("worker_selection", ""),
            )
        except RunRecordError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise RunRecordError("invalid ConCorD run record") from exc


def write_job_run_record(
    record: JobRunRecord,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write ``record`` as canonical JSON."""

    if not isinstance(record, JobRunRecord):
        raise TypeError("record must be JobRunRecord")
    target = Path(path).expanduser().resolve(strict=False)
    if target.suffix.lower() != ".json":
        raise RunRecordError("run-record output must use the .json extension")
    if target.exists() and not overwrite:
        raise FileExistsError(f"run record already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                record.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_job_run_record(path: str | Path) -> JobRunRecord:
    """Load and validate one JSON run record."""

    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunRecordError(f"cannot read run record {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RunRecordError("run record root must be a JSON object")
    return JobRunRecord.from_dict(payload)


__all__ = [
    "RUN_RECORD_FORMAT_VERSION",
    "JobRunHardware",
    "JobRunRecord",
    "JobRunTiming",
    "JobRunWorker",
    "RunRecordError",
    "load_job_run_record",
    "write_job_run_record",
]
