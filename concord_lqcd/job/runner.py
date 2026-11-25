"""Production job-runner orchestration for compiled ConCorD plans.

This module owns the outer configuration/time work-unit loop.  It deliberately
adds no compiler or numerical semantics: every work unit is translated through
``build_job_runtime_resources`` and executed by the frozen physical plan. Safe
resident deployments may add one private leading time-context axis; size one
uses ``GenericNumericalExecutor`` unchanged.

Results are written incrementally to one HDF5 file.  A work unit is marked
complete only after all correlator outputs for that work unit have been written
and flushed.  Resume therefore re-executes an interrupted/incomplete work unit
rather than trusting partially written values.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, Iterator, Mapping, Sequence
import warnings

import h5py
import numpy as np
import torch

from concord_lqcd.backend.runtime.batching import (
    BatchPlanningError,
    estimate_batched_execution_memory,
    plan_time_batch_candidates,
)
from concord_lqcd.backend.runtime.calibration import CalibrationCacheError
from concord_lqcd.backend.runtime.materialization import (
    HDF5FilePool,
    PrimitiveCachePlan,
    PrimitiveTensorCache,
    plan_primitive_tensor_cache,
)
from concord_lqcd.backend.runtime.numerical_executor import GenericNumericalExecutor
from concord_lqcd.backend.runtime.hardware import HardwareProfile, detect_hardware_profile
from concord_lqcd.backend.runtime.heap import fix_block_limit, libc_function
from concord_lqcd.backend.runtime.prefetch import (
    PropagationPrefetchPlan,
    PropagationPrefetchSession,
    plan_propagation_prefetch,
)
from concord_lqcd.backend.runtime.residency import ResidentExecutionSession
from concord_lqcd.backend.runtime.resource_planning import (
    DeviceExecutionPlan,
    HardwareBudgetPolicy,
    ResourcePlanningError,
    estimate_execution_memory,
    revalidate_execution_resources,
    resolve_runtime_device,
)

from .adaptive import (
    AdaptiveCalibrationPolicy,
    AdaptivePlanningError,
    specialize_plan_for_hardware,
)
from .batching import (
    TimeBatchCalibrationPolicy,
    TimeBatchPlanningError,
    select_time_batch_size,
)
from .data import (
    JobDataError,
    _build_job_configuration_runtime_resources,
    build_job_runtime_resources,
)
from .kernel_store import KernelCacheStatistics
from .plan import ConcordPlan, validate_plan_for_job
from .flavor_equivalence import validate_numerical_flavor_sources
from ._prepared import _is_prepared_plan, _load_prepared_plan, _try_prepare_plan_snapshot
from ._raw_cache_planning import plan_raw_eigenvector_cache
from ._worker_split import WorkerSplit, auto_worker_split
from ._workers import (
    WorkerAssignment,
    WorkerFailure,
    WorkerStartError,
    node_shape,
    run_workers,
    split_contiguous,
    usable_cpus,
    worker_cpu_sets,
)
from .run_record import JobRunTiming, JobRunWorker
from .schema import ExpandedTimeContext, JobSpec


_RESULT_FORMAT_VERSION = 1
_UTF8 = h5py.string_dtype(encoding="utf-8")
_NUMPY_DTYPES = {
    "complex64": np.dtype(np.complex64),
    "complex128": np.dtype(np.complex128),
}


class JobRunError(ValueError):
    """Raised when a production job cannot be executed or resumed safely."""


@dataclass(frozen=True)
class JobWorkUnit:
    """One deterministic configuration/time execution point."""

    configuration_index: int
    time_index: int
    configuration: int
    time_context: ExpandedTimeContext

    @property
    def key(self) -> tuple[int, str]:
        return self.configuration, self.time_context.id


@dataclass(frozen=True)
class ConfigurationRuntimeDecision:
    """Inspectable runtime policy selected for one pending configuration."""

    configuration: int
    work_units: int
    batch_size: int
    execution_mode: str
    primitive_cache_plan: PrimitiveCachePlan | None
    propagation_prefetch_plan: PropagationPrefetchPlan | None
    kernel_batch_staging_limit_bytes: int | None
    raw_eigenvector_cache_limit_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.raw_eigenvector_cache_limit_bytes, int)
            or isinstance(self.raw_eigenvector_cache_limit_bytes, bool)
            or self.raw_eigenvector_cache_limit_bytes < 0
        ):
            raise ValueError("raw eigenvector cache limits must be non-negative integers")
        if self.work_units <= 0 or self.batch_size <= 0:
            raise ValueError("runtime decisions require positive work and batch sizes")
        if self.execution_mode == "resident":
            if self.primitive_cache_plan is None or self.propagation_prefetch_plan is None:
                raise ValueError("resident decisions require cache and prefetch plans")
            if (
                self.kernel_batch_staging_limit_bytes is not None
                and (
                    not isinstance(self.kernel_batch_staging_limit_bytes, int)
                    or isinstance(self.kernel_batch_staging_limit_bytes, bool)
                    or self.kernel_batch_staging_limit_bytes < 0
                )
            ):
                raise ValueError(
                    "kernel batch staging limits must be non-negative integers or None"
                )
        elif self.execution_mode == "host_tiled":
            if (
                self.primitive_cache_plan is not None
                or self.propagation_prefetch_plan is not None
                or self.kernel_batch_staging_limit_bytes is not None
            ):
                raise ValueError("host-tiled decisions cannot contain resident-only plans")
        else:
            raise ValueError(f"unsupported runtime decision mode {self.execution_mode!r}")


@dataclass(frozen=True)
class JobRunSummary:
    """Compact result of one runner invocation.

    Numerical correlator data live in ``output_file``; the runner intentionally
    does not retain all gauge configurations/time contexts in Python memory.
    It does retain the small immutable policy record for every configuration
    that required execution, so automatic choices remain inspectable.
    """

    output_file: Path
    plan_sha256: str
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
    hardware_profile: HardwareProfile | None
    configuration_decisions: tuple[ConfigurationRuntimeDecision, ...]
    memory_verified: bool
    known_working_set_bytes: int
    peak_tile_bytes: int | None
    tiled_operations: int
    resident_pool_allocations: int
    resident_pool_peak_bytes: int
    primitive_cache_hits: int
    primitive_cache_misses: int
    primitive_cache_allocation_failures: int
    primitive_cache_peak_bytes: int
    kernel_cache_enabled: bool
    kernel_cache_hits: int
    kernel_cache_misses: int
    kernel_cache_writes: int
    kernel_cache_invalid_entries: int
    kernel_cache_write_failures: int
    hdf5_file_opens: int
    hdf5_file_reuses: int
    propagation_prefetch_configurations: int
    propagation_prefetch_batches: int
    propagation_prefetch_tensors: int
    propagation_prefetch_bytes: int
    propagation_prefetch_wait_seconds: float
    propagation_prefetch_allocation_fallbacks: int
    # Workers that ran side by side; after a round of workers that measured
    # their memory, those started after it. worker_reports lists every worker.
    workers: int = 1
    threads_per_worker: int | None = None
    worker_reports: tuple[JobRunWorker, ...] = ()
    worker_selection: str = ""

    @property
    def all_complete(self) -> bool:
        return self.completed_work_units == self.total_work_units


def iter_job_work_units(spec: JobSpec) -> Iterator[JobWorkUnit]:
    """Yield work units in the frozen V1 order: configuration outer, time inner."""

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    for configuration_index, configuration in enumerate(spec.configurations):
        for time_index, time_context in enumerate(spec.time_contexts):
            yield JobWorkUnit(
                configuration_index=configuration_index,
                time_index=time_index,
                configuration=configuration,
                time_context=time_context,
            )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _kernel_cache_statistics(
    resources: Sequence[object],
) -> KernelCacheStatistics:
    totals = {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "invalid_entries": 0,
        "write_failures": 0,
    }
    seen: set[int] = set()
    for resource in resources:
        providers = getattr(resource, "kernel_providers", {})
        for provider in providers.values():
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            statistics = getattr(provider, "kernel_cache_statistics", None)
            if not isinstance(statistics, KernelCacheStatistics):
                continue
            for name in totals:
                totals[name] += getattr(statistics, name)
    return KernelCacheStatistics(**totals)


def _run_fingerprint(spec: JobSpec, plan: ConcordPlan) -> str:
    """Fingerprint runtime inputs whose change would make resume unsafe.

    Device and output policy/path are intentionally excluded.  The compiled
    algebra is represented by ``plan_sha256``.  Lattice geometry, selected work
    grid, and data catalog are included because changing any of them while
    retaining completed output slots would mix physically different runs.
    """

    runtime_data = spec.data.model_dump(mode="json")
    runtime_data.pop("kernel_cache", None)
    if spec.data.kernel_cache is not None and spec.data.kernel_cache.packing != "none":
        runtime_data["kernel_cache_packing"] = spec.data.kernel_cache.packing
    payload = {
        "result_format_version": _RESULT_FORMAT_VERSION,
        "plan_sha256": plan.manifest.plan_sha256,
        "lattice": spec.lattice.model_dump(mode="json"),
        "configurations": list(spec.configurations),
        "time_contexts": [
            {
                "id": context.id,
                "operator_times": {
                    str(arity): list(times)
                    for arity, times in sorted(context.operator_times.items())
                },
            }
            for context in spec.time_contexts
        ],
        "data": runtime_data,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _decode_strings(dataset: h5py.Dataset) -> tuple[str, ...]:
    values = dataset.asstr()[...]
    if np.ndim(values) == 0:
        return (str(values),)
    return tuple(str(value) for value in values.tolist())


def _operator_times_json(context: ExpandedTimeContext) -> str:
    return json.dumps(
        {
            str(arity): list(times)
            for arity, times in sorted(context.operator_times.items())
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class _JobResultStore:
    """Incremental single-process HDF5 result store for one ``JobSpec`` grid."""

    def __init__(self, spec: JobSpec, plan: ConcordPlan) -> None:
        self.spec = spec
        self.plan = plan
        self.path = spec.output.file.expanduser().resolve(strict=False)
        self.run_fingerprint = _run_fingerprint(spec, plan)
        self.handle: h5py.File | None = None
        self._correlator_groups: dict[str, h5py.Group] = {}

    def __enter__(self) -> "_JobResultStore":
        policy = self.spec.output.on_existing
        exists = self.path.exists()
        if exists and policy == "error":
            raise FileExistsError(f"job output already exists: {self.path}")
        if exists and policy == "overwrite":
            if not self.path.is_file():
                raise JobRunError(f"job output path is not a file: {self.path}")
            self.path.unlink()
            exists = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.handle = h5py.File(self.path, "r+" if exists else "w")
        except OSError as exc:
            raise JobRunError(f"cannot open job output {self.path}: {exc}") from exc

        try:
            if exists:
                self._validate_existing()
            else:
                self._initialize_new()
            self._bind_correlator_groups()
        except Exception:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    @property
    def h5(self) -> h5py.File:
        if self.handle is None:
            raise RuntimeError("result store is not open")
        return self.handle

    def _initialize_new(self) -> None:
        h5 = self.h5
        h5.attrs["concord_result_format_version"] = _RESULT_FORMAT_VERSION
        h5.attrs["plan_sha256"] = self.plan.manifest.plan_sha256
        h5.attrs["job_schema_version"] = self.spec.schema_version
        h5.attrs["job_name"] = self.spec.job.name
        h5.attrs["run_fingerprint"] = self.run_fingerprint

        metadata = h5.create_group("metadata")
        metadata.create_dataset(
            "plan_manifest_json",
            data=json.dumps(
                self.plan.manifest.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            dtype=_UTF8,
        )
        metadata.create_dataset(
            "runtime_spec_json",
            data=json.dumps(
                {
                    "lattice": self.spec.lattice.model_dump(mode="json"),
                    "run": self.spec.run.model_dump(mode="json"),
                    "data": self.spec.data.model_dump(mode="json"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            dtype=_UTF8,
        )

        work = h5.create_group("work")
        work.create_dataset(
            "configurations",
            data=np.asarray(self.spec.configurations, dtype=np.int64),
        )
        work.create_dataset(
            "time_context_ids",
            data=np.asarray([context.id for context in self.spec.time_contexts], dtype=object),
            dtype=_UTF8,
        )
        work.create_dataset(
            "operator_times_json",
            data=np.asarray(
                [_operator_times_json(context) for context in self.spec.time_contexts],
                dtype=object,
            ),
            dtype=_UTF8,
        )

        shape = (len(self.spec.configurations), len(self.spec.time_contexts))
        h5.create_dataset("completed", shape=shape, dtype=np.bool_, fillvalue=False)

        correlators = h5.create_group("correlators")
        for output_index, correlator_id in enumerate(self.plan.correlator_ids):
            output = self.plan.physical_plan.outputs[correlator_id]
            binding = self.plan.physical_plan.binding(output.value_id)
            try:
                dtype = _NUMPY_DTYPES[binding.scalar_type]
            except KeyError as exc:
                raise JobRunError(
                    f"result writer does not support scalar type {binding.scalar_type!r}"
                ) from exc

            group = correlators.create_group(str(output_index))
            group.attrs["correlator_id"] = correlator_id
            group.attrs["scalar_type"] = binding.scalar_type
            group.attrs["axis_labels_json"] = json.dumps(list(binding.axis_labels))
            group.attrs["axis_spaces_json"] = json.dumps(list(binding.axis_spaces))
            dataset_shape = shape + tuple(binding.shape)
            # One work unit per chunk keeps incremental writes and resume local.
            chunk_shape = (1, 1) + tuple(max(1, value) for value in binding.shape)
            group.create_dataset(
                "values",
                shape=dataset_shape,
                dtype=dtype,
                chunks=chunk_shape,
            )

        h5.flush()

    def _validate_existing(self) -> None:
        h5 = self.h5
        required_attrs = {
            "concord_result_format_version",
            "plan_sha256",
            "job_schema_version",
            "job_name",
            "run_fingerprint",
        }
        missing_attrs = required_attrs - set(h5.attrs)
        if missing_attrs:
            raise JobRunError(
                f"existing output is missing ConCorD metadata attributes {sorted(missing_attrs)}"
            )
        if int(h5.attrs["concord_result_format_version"]) != _RESULT_FORMAT_VERSION:
            raise JobRunError("existing output uses an incompatible ConCorD result format")
        if str(h5.attrs["plan_sha256"]) != self.plan.manifest.plan_sha256:
            raise JobRunError("existing output was produced by a different compiled plan")
        if str(h5.attrs["run_fingerprint"]) != self.run_fingerprint:
            raise JobRunError(
                "existing output runtime specification differs from this job; "
                "resume would mix configurations/times/data catalogs"
            )

        for name in ("work", "completed", "correlators"):
            if name not in h5:
                raise JobRunError(f"existing output is missing required object {name!r}")

        configurations = tuple(int(value) for value in h5["work/configurations"][...])
        if configurations != tuple(self.spec.configurations):
            raise JobRunError("existing output configuration grid differs from this job")
        context_ids = _decode_strings(h5["work/time_context_ids"])
        if context_ids != tuple(context.id for context in self.spec.time_contexts):
            raise JobRunError("existing output time-context grid differs from this job")
        operator_times = _decode_strings(h5["work/operator_times_json"])
        if operator_times != tuple(_operator_times_json(c) for c in self.spec.time_contexts):
            raise JobRunError("existing output operator-time assignments differ from this job")

        expected_work_shape = (len(self.spec.configurations), len(self.spec.time_contexts))
        if tuple(h5["completed"].shape) != expected_work_shape:
            raise JobRunError("existing output completion-mask shape is inconsistent")

        correlators = h5["correlators"]
        if set(correlators.keys()) != {str(i) for i in range(len(self.plan.correlator_ids))}:
            raise JobRunError("existing output correlator table is inconsistent")
        for output_index, correlator_id in enumerate(self.plan.correlator_ids):
            group = correlators[str(output_index)]
            if str(group.attrs.get("correlator_id", "")) != correlator_id:
                raise JobRunError("existing output correlator identifiers are inconsistent")
            if "values" not in group:
                raise JobRunError(f"existing output for {correlator_id!r} has no values dataset")
            output = self.plan.physical_plan.outputs[correlator_id]
            binding = self.plan.physical_plan.binding(output.value_id)
            expected_shape = expected_work_shape + tuple(binding.shape)
            if tuple(group["values"].shape) != expected_shape:
                raise JobRunError(
                    f"existing output shape for {correlator_id!r} is incompatible with the plan"
                )
            if np.dtype(group["values"].dtype) != _NUMPY_DTYPES[binding.scalar_type]:
                raise JobRunError(
                    f"existing output dtype for {correlator_id!r} is incompatible with the plan"
                )

    def _bind_correlator_groups(self) -> None:
        self._correlator_groups = {
            correlator_id: self.h5[f"correlators/{index}"]
            for index, correlator_id in enumerate(self.plan.correlator_ids)
        }

    def is_complete(self, work_unit: JobWorkUnit) -> bool:
        return bool(
            self.h5["completed"][
                work_unit.configuration_index,
                work_unit.time_index,
            ]
        )

    def write(self, work_unit: JobWorkUnit, outputs: Mapping[str, torch.Tensor]) -> None:
        if set(outputs) != set(self.plan.correlator_ids):
            raise JobRunError("numerical work-unit result does not contain exactly plan outputs")

        prefix = (work_unit.configuration_index, work_unit.time_index)
        for correlator_id in self.plan.correlator_ids:
            tensor = outputs[correlator_id]
            output = self.plan.physical_plan.outputs[correlator_id]
            binding = self.plan.physical_plan.binding(output.value_id)
            if tuple(tensor.shape) != tuple(binding.shape):
                raise JobRunError(
                    f"output {correlator_id!r} has shape {tuple(tensor.shape)}, "
                    f"expected {binding.shape}"
                )
            array = tensor.detach().cpu().numpy()
            expected_dtype = _NUMPY_DTYPES[binding.scalar_type]
            if np.dtype(array.dtype) != expected_dtype:
                raise JobRunError(
                    f"output {correlator_id!r} has dtype {array.dtype}, expected {expected_dtype}"
                )
            self._correlator_groups[correlator_id]["values"][prefix] = array

        # Persist all correlators before declaring the work unit complete.
        self.h5.flush()
        self.h5["completed"][prefix] = True
        self.h5.flush()

    def write_batch(
        self,
        work_units: Sequence[JobWorkUnit],
        outputs: Mapping[str, torch.Tensor],
    ) -> None:
        """Persist one leading-axis batch before marking any member complete."""

        units = tuple(work_units)
        if len(units) <= 1:
            raise JobRunError("batched result writing requires at least two work units")
        configuration_indices = {unit.configuration_index for unit in units}
        configurations = {unit.configuration for unit in units}
        if len(configuration_indices) != 1 or len(configurations) != 1:
            raise JobRunError(
                "one numerical batch may contain time contexts from only one configuration"
            )
        time_indices = tuple(unit.time_index for unit in units)
        if len(time_indices) != len(set(time_indices)):
            raise JobRunError("batched result contains duplicate time-context slots")
        if any(self.is_complete(unit) for unit in units):
            raise JobRunError("batched result attempted to overwrite a completed work unit")
        if set(outputs) != set(self.plan.correlator_ids):
            raise JobRunError("numerical batch result does not contain exactly plan outputs")

        configuration_index = units[0].configuration_index
        arrays: dict[str, np.ndarray] = {}
        for correlator_id in self.plan.correlator_ids:
            tensor = outputs[correlator_id]
            output = self.plan.physical_plan.outputs[correlator_id]
            binding = self.plan.physical_plan.binding(output.value_id)
            expected_shape = (len(units), *binding.shape)
            if tuple(tensor.shape) != expected_shape:
                raise JobRunError(
                    f"batched output {correlator_id!r} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            array = tensor.detach().cpu().numpy()
            expected_dtype = _NUMPY_DTYPES[binding.scalar_type]
            if np.dtype(array.dtype) != expected_dtype:
                raise JobRunError(
                    f"batched output {correlator_id!r} has dtype {array.dtype}, "
                    f"expected {expected_dtype}"
                )
            arrays[correlator_id] = array

        # Transfer each correlator to host once, then populate its individual
        # HDF5 work-unit chunks.  All correlator data are durable before the
        # first completion flag is set, so an interrupted run never marks unwritten work complete.
        for correlator_id, array in arrays.items():
            dataset = self._correlator_groups[correlator_id]["values"]
            for batch_index, time_index in enumerate(time_indices):
                dataset[configuration_index, time_index] = array[batch_index]
        self.h5.flush()
        for time_index in time_indices:
            self.h5["completed"][configuration_index, time_index] = True
        self.h5.flush()

    @property
    def completed_count(self) -> int:
        return int(np.count_nonzero(self.h5["completed"][...]))


@dataclass
class _ExecutionTally:
    """Times, counters and runtime choices of one process's execution pass."""

    runtime_planning_seconds: float = 0.0
    data_preparation_seconds: float = 0.0
    input_wait_seconds: float = 0.0
    numerical_execution_seconds: float = 0.0
    output_writing_seconds: float = 0.0
    executed: int = 0
    executed_batches: int = 0
    time_batch_size: int = 1
    time_batch_selection_source: str = "not-required"
    resident_pool_allocations: int = 0
    resident_pool_peak_bytes: int = 0
    primitive_cache_hits: int = 0
    primitive_cache_misses: int = 0
    primitive_cache_allocation_failures: int = 0
    primitive_cache_peak_bytes: int = 0
    kernel_cache_hits: int = 0
    kernel_cache_misses: int = 0
    kernel_cache_writes: int = 0
    kernel_cache_invalid_entries: int = 0
    kernel_cache_write_failures: int = 0
    hdf5_file_opens: int = 0
    hdf5_file_reuses: int = 0
    propagation_prefetch_configurations: int = 0
    propagation_prefetch_batches: int = 0
    propagation_prefetch_tensors: int = 0
    propagation_prefetch_bytes: int = 0
    propagation_prefetch_wait_seconds: float = 0.0
    propagation_prefetch_allocation_fallbacks: int = 0
    configuration_decisions: list[ConfigurationRuntimeDecision] = field(default_factory=list)
    execution_plan: DeviceExecutionPlan | None = None
    observed_hardware: HardwareProfile | None = None


def _execute_pending(
    spec: JobSpec,
    plan: ConcordPlan,
    pending_by_configuration: Sequence[Sequence[JobWorkUnit]],
    writer: "_JobResultStore | _ForwardingWriter",
    tally: _ExecutionTally,
    *,
    requested_device: str | torch.device,
    hardware: HardwareProfile | None,
    budget_policy: HardwareBudgetPolicy | None,
    time_batch_calibration_policy: TimeBatchCalibrationPolicy | None,
    calibration_cache_path: str | Path | None,
    write_buffers_first: bool = False,
) -> None:
    """Execute pending work units in order and hand every result to ``writer``.

    ``writer`` is the HDF5 result store, or inside a worker process a channel
    to the parent that owns the store. ``write_buffers_first``: write each
    configuration's whole buffer pool before its first work unit (see
    ``_run_pending_in_workers``).
    """

    execution_plan = tally.execution_plan
    if any(pending_by_configuration):
        stage_started = perf_counter()
        if tally.observed_hardware is None:
            tally.observed_hardware = detect_hardware_profile()
        selection_hardware = tally.observed_hardware
        execution_plan = resolve_runtime_device(
            plan.physical_plan,
            requested_device,
            hardware=selection_hardware,
            policy=budget_policy,
        )
        batch_selection = select_time_batch_size(
            spec,
            plan,
            execution_plan,
            hardware=selection_hardware,
            calibration_policy=time_batch_calibration_policy,
            cache_path=calibration_cache_path,
        )
        tally.time_batch_size = batch_selection.batch_size
        tally.time_batch_selection_source = batch_selection.source
        # Calibration allocates real tensors and may change volatile
        # allocator/free-memory state. Refresh before the first
        # production configuration; later refreshes occur only after
        # the previous configuration-owned session has been released.
        execution_plan = revalidate_execution_resources(
            plan.physical_plan,
            execution_plan,
            hardware=hardware,
        )
        tally.runtime_planning_seconds += perf_counter() - stage_started

    for pending in pending_by_configuration:
        if not pending:
            continue
        stage_started = perf_counter()
        if tally.executed_batches:
            execution_plan = revalidate_execution_resources(
                plan.physical_plan,
                execution_plan,
                hardware=hardware,
            )
        assert execution_plan is not None

        configuration_batch_size = min(tally.time_batch_size, len(pending))
        if configuration_batch_size > 1:
            try:
                plan_time_batch_candidates(
                    plan.physical_plan,
                    execution_plan,
                    work_units=len(pending),
                    requested_batch_size=configuration_batch_size,
                )
            except BatchPlanningError:
                if isinstance(spec.run.time_batch_size, int):
                    raise
                # Volatile free memory can fall after calibration. Only
                # the already-proven size-one path is used as fallback.
                configuration_batch_size = 1
                tally.time_batch_selection_source = "runtime_fallback"
        tally.runtime_planning_seconds += perf_counter() - stage_started

        file_pool = HDF5FilePool()
        with file_pool, ExitStack() as configuration_cleanup:
            if execution_plan.execution_mode == "resident":
                stage_started = perf_counter()
                resources = _build_job_configuration_runtime_resources(
                    spec,
                    plan,
                    configuration=pending[0].configuration,
                    time_contexts=tuple(
                        work_unit.time_context for work_unit in pending
                    ),
                    device=execution_plan.torch_device,
                    execution_plan=execution_plan,
                    hdf5_file_pool=file_pool,
                )
                uncached_materializers = tuple(
                    resource.materializer(plan) for resource in resources
                )
                tally.data_preparation_seconds += perf_counter() - stage_started
                stage_started = perf_counter()
                safe_limit = (
                    execution_plan.selected_budget.safe_working_set_limit_bytes
                )
                raw_cache_admission = plan_raw_eigenvector_cache(
                    resources,
                    plan.physical_plan,
                    execution_plan,
                    batch_size=configuration_batch_size,
                    automatic_batch=spec.run.time_batch_size == "auto",
                    element_bytes=plan.config.element_bytes,
                )
                if raw_cache_admission.warning is not None:
                    warnings.warn(
                        raw_cache_admission.warning,
                        UserWarning,
                        stacklevel=2,
                    )
                if raw_cache_admission.batch_size != configuration_batch_size:
                    configuration_batch_size = raw_cache_admission.batch_size
                    tally.time_batch_size = configuration_batch_size
                    tally.time_batch_selection_source = "cache_bounded"
                raw_cache_limit = raw_cache_admission.reserved_bytes
                batch_peak = estimate_batched_execution_memory(
                    plan.physical_plan,
                    configuration_batch_size,
                ).known_peak_bytes
                cache_budget = (
                    max(0, safe_limit - batch_peak - raw_cache_limit)
                    if safe_limit is not None
                    else 0
                )
                cache_plan = plan_primitive_tensor_cache(
                    uncached_materializers,
                    max_bytes=cache_budget,
                )
                tally.runtime_planning_seconds += perf_counter() - stage_started
                stage_started = perf_counter()
                tensor_cache = PrimitiveTensorCache(
                    cache_plan,
                    device=execution_plan.storage_torch_device,
                )
                cached_materializers = tuple(
                    resource.materializer(plan, tensor_cache=tensor_cache)
                    for resource in resources
                )
                tally.data_preparation_seconds += perf_counter() - stage_started
                stage_started = perf_counter()
                remaining_device_headroom = (
                    max(0, cache_budget - cache_plan.planned_bytes)
                    if safe_limit is not None
                    else None
                )
                prefetch_plan = plan_propagation_prefetch(
                    cached_materializers,
                    batch_size=configuration_batch_size,
                    execution_plan=execution_plan,
                    available_device_headroom_bytes=(
                        remaining_device_headroom
                    ),
                )
                measured_kernel_batch_headroom = (
                    max(
                        0,
                        remaining_device_headroom
                        - prefetch_plan.device_staging_bytes,
                    )
                    if remaining_device_headroom is not None
                    else None
                )
                for provider, limit in raw_cache_admission.provider_limits:
                    provider.configure_raw_eigenvector_cache(limit)
                    configuration_cleanup.callback(provider.clear_raw_eigenvector_cache)
                policy_kernel_batch_limit = (
                    execution_plan.policy.kernel_batch_staging_limit_for(
                        execution_plan.selected_budget.device
                    )
                )
                if measured_kernel_batch_headroom is None:
                    kernel_batch_staging_limit = None
                elif policy_kernel_batch_limit is None:
                    kernel_batch_staging_limit = (
                        measured_kernel_batch_headroom
                    )
                else:
                    kernel_batch_staging_limit = min(
                        measured_kernel_batch_headroom,
                        policy_kernel_batch_limit,
                    )
                tally.runtime_planning_seconds += perf_counter() - stage_started
                stage_started = perf_counter()
                materializers = tuple(
                    materializer.with_kernel_batch_staging_limit(
                        kernel_batch_staging_limit
                    )
                    for materializer in cached_materializers
                )
                if prefetch_plan.enabled:
                    tally.propagation_prefetch_configurations += 1
                tally.configuration_decisions.append(
                    ConfigurationRuntimeDecision(
                        configuration=pending[0].configuration,
                        work_units=len(pending),
                        batch_size=configuration_batch_size,
                        execution_mode=execution_plan.execution_mode,
                        primitive_cache_plan=cache_plan,
                        propagation_prefetch_plan=prefetch_plan,
                        kernel_batch_staging_limit_bytes=(
                            kernel_batch_staging_limit
                        ),
                        raw_eigenvector_cache_limit_bytes=raw_cache_limit,
                    )
                )
                tally.data_preparation_seconds += perf_counter() - stage_started

                stage_started = perf_counter()
                with ResidentExecutionSession(
                    plan.physical_plan,
                    device=execution_plan.storage_torch_device,
                    maximum_batch_size=configuration_batch_size,
                ) as session, PropagationPrefetchSession(
                    prefetch_plan,
                    device=execution_plan.storage_torch_device,
                ) as prefetch_session:
                    if write_buffers_first:
                        session.pool.write_all()
                    tally.data_preparation_seconds += perf_counter() - stage_started
                    session_statistics = session.statistics
                    tally.resident_pool_allocations += (
                        session_statistics.pool_allocations
                    )
                    tally.resident_pool_peak_bytes = max(
                        tally.resident_pool_peak_bytes,
                        session_statistics.pool_allocation_bytes,
                    )
                    cursor = 0
                    first_batch_size = min(
                        configuration_batch_size,
                        len(pending),
                    )
                    stage_started = perf_counter()
                    prefetch_session.submit(
                        materializers[:first_batch_size]
                    )
                    tally.data_preparation_seconds += perf_counter() - stage_started
                    while cursor < len(pending):
                        current_batch_size = min(
                            configuration_batch_size,
                            len(pending) - cursor,
                        )
                        batch = tuple(
                            pending[cursor : cursor + current_batch_size]
                        )
                        batch_materializers = materializers[
                            cursor : cursor + current_batch_size
                        ]
                        stage_started = perf_counter()
                        prefetch_session.resolve()
                        tally.input_wait_seconds += perf_counter() - stage_started

                        next_cursor = cursor + current_batch_size
                        if next_cursor < len(pending):
                            next_batch_size = min(
                                configuration_batch_size,
                                len(pending) - next_cursor,
                            )
                            stage_started = perf_counter()
                            prefetch_session.submit(
                                materializers[
                                    next_cursor : next_cursor + next_batch_size
                                ]
                            )
                            tally.data_preparation_seconds += (
                                perf_counter() - stage_started
                            )
                        stage_started = perf_counter()
                        result = session.execute(batch_materializers)
                        tally.numerical_execution_seconds += (
                            perf_counter() - stage_started
                        )
                        try:
                            unconsumed = tuple(
                                value_id
                                for materializer in batch_materializers
                                for value_id in (
                                    materializer.pending_prefetched_values
                                )
                            )
                            if unconsumed:
                                raise RuntimeError(
                                    "resident execution did not consume every "
                                    f"prefetched primitive: {unconsumed}"
                                )
                            stage_started = perf_counter()
                            if current_batch_size == 1:
                                writer.write(batch[0], result.outputs)
                            else:
                                writer.write_batch(batch, result.outputs)
                            tally.output_writing_seconds += (
                                perf_counter() - stage_started
                            )
                        finally:
                            # Result tensors are pool views. They become
                            # invalid as soon as the session is released.
                            del result
                            session.release()

                        tally.executed += current_batch_size
                        tally.executed_batches += 1
                        cursor = next_cursor

                    prefetch_statistics = prefetch_session.statistics

                tally.propagation_prefetch_batches += (
                    prefetch_statistics.completed_batches
                )
                tally.propagation_prefetch_tensors += (
                    prefetch_statistics.staged_tensors
                )
                tally.propagation_prefetch_bytes += prefetch_statistics.staged_bytes
                tally.propagation_prefetch_wait_seconds += (
                    prefetch_statistics.wait_seconds
                )
                tally.propagation_prefetch_allocation_fallbacks += (
                    prefetch_statistics.allocation_fallbacks
                )

                cache_statistics = tensor_cache.statistics
                tally.primitive_cache_hits += cache_statistics.hits
                tally.primitive_cache_misses += cache_statistics.misses
                tally.primitive_cache_allocation_failures += (
                    cache_statistics.allocation_failures
                )
                tally.primitive_cache_peak_bytes = max(
                    tally.primitive_cache_peak_bytes,
                    cache_statistics.peak_resident_bytes,
                )
                kernel_statistics = _kernel_cache_statistics(resources)
                tally.kernel_cache_hits += kernel_statistics.hits
                tally.kernel_cache_misses += kernel_statistics.misses
                tally.kernel_cache_writes += kernel_statistics.writes
                tally.kernel_cache_invalid_entries += (
                    kernel_statistics.invalid_entries
                )
                tally.kernel_cache_write_failures += (
                    kernel_statistics.write_failures
                )
                tensor_cache.clear()
                del materializers
                del cached_materializers
                del uncached_materializers
                del resources
                del tensor_cache
            else:
                # Host-tiled execution retains the proven sequential
                # executor. File handles are still configuration-local.
                tally.configuration_decisions.append(
                    ConfigurationRuntimeDecision(
                        configuration=pending[0].configuration,
                        work_units=len(pending),
                        batch_size=1,
                        execution_mode=execution_plan.execution_mode,
                        primitive_cache_plan=None,
                        propagation_prefetch_plan=None,
                        kernel_batch_staging_limit_bytes=None,
                    )
                )
                for work_unit in pending:
                    stage_started = perf_counter()
                    resources = build_job_runtime_resources(
                        spec,
                        plan,
                        configuration=work_unit.configuration,
                        time_context=work_unit.time_context,
                        device=execution_plan.torch_device,
                        execution_plan=execution_plan,
                        hdf5_file_pool=file_pool,
                    )
                    tally.data_preparation_seconds += perf_counter() - stage_started
                    stage_started = perf_counter()
                    result = GenericNumericalExecutor(
                        resources.materializer(plan),
                        device=resources.storage_device,
                        compute_device=resources.device,
                        tiled_plan=execution_plan.tiled_plan,
                    ).execute()
                    tally.numerical_execution_seconds += (
                        perf_counter() - stage_started
                    )
                    stage_started = perf_counter()
                    writer.write(work_unit, result.outputs)
                    tally.output_writing_seconds += perf_counter() - stage_started
                    kernel_statistics = _kernel_cache_statistics((resources,))
                    tally.kernel_cache_hits += kernel_statistics.hits
                    tally.kernel_cache_misses += kernel_statistics.misses
                    tally.kernel_cache_writes += kernel_statistics.writes
                    tally.kernel_cache_invalid_entries += (
                        kernel_statistics.invalid_entries
                    )
                    tally.kernel_cache_write_failures += (
                        kernel_statistics.write_failures
                    )
                    tally.executed += 1
                    tally.executed_batches += 1
                    del result
                    del resources

        file_statistics = file_pool.statistics
        tally.hdf5_file_opens += file_statistics.file_opens
        tally.hdf5_file_reuses += file_statistics.file_reuses

    tally.execution_plan = execution_plan


class _ForwardingWriter:
    """Result writer of a worker process: sends each finished result to the parent.

    After each result the worker gives the memory it freed back to the system
    (``_return_freed_memory``); a heap that keeps it would make every further
    work unit look larger to the other workers on the node.
    """

    def __init__(self, send: Callable[[object], None]) -> None:
        self._send = send

    def write(self, work_unit: JobWorkUnit, outputs: Mapping[str, torch.Tensor]) -> None:
        self._send(("unit", _unit_slot(work_unit), _host_arrays(outputs)))
        _return_freed_memory()

    def write_batch(
        self,
        work_units: Sequence[JobWorkUnit],
        outputs: Mapping[str, torch.Tensor],
    ) -> None:
        slots = tuple(_unit_slot(work_unit) for work_unit in work_units)
        self._send(("batch", slots, _host_arrays(outputs)))
        _return_freed_memory()


def _unit_slot(work_unit: JobWorkUnit) -> tuple[int, int]:
    return work_unit.configuration_index, work_unit.time_index


def _host_arrays(outputs: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    # Result tensors are pool views; sending pickles a copy before the pool is reused.
    return {key: tensor.detach().cpu().numpy() for key, tensor in outputs.items()}


def _deployment(execution_plan: DeviceExecutionPlan | None) -> dict[str, object]:
    if execution_plan is None:
        return {
            "selected_device": "not-required",
            "storage_device": "not-required",
            "execution_mode": "not-required",
            "memory_verified": False,
            "peak_tile_bytes": None,
            "tiled_operations": 0,
        }
    tiled_plan = execution_plan.tiled_plan
    return {
        "selected_device": execution_plan.selected_device,
        "storage_device": execution_plan.storage_device,
        "execution_mode": execution_plan.execution_mode,
        "memory_verified": execution_plan.memory_verified,
        "peak_tile_bytes": None if tiled_plan is None else tiled_plan.peak_device_bytes,
        "tiled_operations": 0 if tiled_plan is None else tiled_plan.tiled_operation_count,
    }


@dataclass(frozen=True)
class _WorkerOutcome:
    tally: _ExecutionTally
    deployment: dict[str, object]
    report: JobRunWorker


_SUMMED_FIELDS = (
    "executed", "executed_batches", "resident_pool_allocations", "primitive_cache_hits",
    "primitive_cache_misses", "primitive_cache_allocation_failures", "kernel_cache_hits",
    "kernel_cache_misses", "kernel_cache_writes", "kernel_cache_invalid_entries",
    "kernel_cache_write_failures", "hdf5_file_opens", "hdf5_file_reuses",
    "propagation_prefetch_configurations", "propagation_prefetch_batches",
    "propagation_prefetch_tensors", "propagation_prefetch_bytes",
    "propagation_prefetch_wait_seconds", "propagation_prefetch_allocation_fallbacks",
)
_PEAK_FIELDS = ("resident_pool_peak_bytes", "primitive_cache_peak_bytes")
# Workers run side by side, so their stage times are averaged, not added.
_MEAN_FIELDS = (
    "runtime_planning_seconds", "data_preparation_seconds", "input_wait_seconds",
    "numerical_execution_seconds", "output_writing_seconds",
)


def _merge_worker_tallies(tallies: Sequence[_ExecutionTally]) -> _ExecutionTally:
    merged = _ExecutionTally()
    for name in _SUMMED_FIELDS:
        setattr(merged, name, sum(getattr(tally, name) for tally in tallies))
    for name in _PEAK_FIELDS:
        setattr(merged, name, max(getattr(tally, name) for tally in tallies))
    for name in _MEAN_FIELDS:
        setattr(merged, name, sum(getattr(tally, name) for tally in tallies) / len(tallies))
    merged.time_batch_size = max(tally.time_batch_size for tally in tallies)
    merged.time_batch_selection_source = "+".join(
        sorted({tally.time_batch_selection_source for tally in tallies})
    )
    merged.configuration_decisions = [
        decision for tally in tallies for decision in tally.configuration_decisions
    ]
    return merged


def _merge_deployments(deployments: Sequence[dict[str, object]]) -> dict[str, object]:
    peaks = [item["peak_tile_bytes"] for item in deployments if item["peak_tile_bytes"] is not None]
    return {
        **{
            name: "+".join(sorted({str(item[name]) for item in deployments}))
            for name in ("selected_device", "storage_device", "execution_mode")
        },
        "memory_verified": all(item["memory_verified"] for item in deployments),
        "peak_tile_bytes": max(peaks) if peaks else None,
        "tiled_operations": max(item["tiled_operations"] for item in deployments),
    }


def _worker_hardware(profile: HardwareProfile, workers: int, threads: int) -> HardwareProfile:
    """The CPU share one worker may plan with: 1/workers of host memory, its own threads."""

    host = profile.host

    def share(value: int | None) -> int | None:
        return None if value is None else max(1, value // workers)

    return replace(
        profile,
        host=replace(
            host,
            total_memory_bytes=share(host.total_memory_bytes),
            available_memory_bytes=share(host.available_memory_bytes),
            logical_processors=threads,
        ),
        accelerators=(),
        torch_num_threads=threads,
    )


def _max_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _memory_usage() -> tuple[int, int, int] | None:
    """This process's resident bytes, its private bytes (pages no other
    process maps, such as the parent of a forked worker) and its bytes in
    swap; None off Linux."""

    try:
        lines = Path("/proc/self/smaps_rollup").read_text().splitlines()
    except OSError:
        return None
    kilobytes = {"Rss": 0, "Private_Clean": 0, "Private_Dirty": 0, "Swap": 0}
    for line in lines:
        name, _, value = line.partition(":")
        if name in kilobytes:
            kilobytes[name] += int(value.split()[0])
    return (
        kilobytes["Rss"] * 1024,
        (kilobytes["Private_Clean"] + kilobytes["Private_Dirty"]) * 1024,
        kilobytes["Swap"] * 1024,
    )


def _private_bytes() -> int | None:
    usage = _memory_usage()
    return None if usage is None else usage[1]


def _peak_private_bytes(start: tuple[int, int, int] | None) -> int | None:
    """Highest private memory of this process since ``start`` (``_memory_usage()``).

    New pages raise both the resident and the private size, so the rise of the
    resident high-water mark counts them, including ones freed again. A page
    copied from the parent only turns shared into private, so those are counted
    from the shared pages that left, less those that went to swap instead. One
    read at the end, no sampling.
    """

    end = _memory_usage()
    peak_resident = _max_rss_bytes()
    if start is None or end is None or peak_resident is None:
        return None
    (resident_start, private_start, swap_start), (resident_end, private_end, swap_end) = start, end
    shared_left = (resident_start - private_start) - (resident_end - private_end)
    copied = max(0, shared_left - max(0, swap_end - swap_start))
    return private_start + max(0, peak_resident - resident_start) + copied


def _return_freed_memory() -> None:
    """Give heap memory this process freed back to the system (glibc only).

    glibc keeps freed blocks for reuse; in a worker that finished a work unit
    they would stay counted against the node until the worker exits.
    """

    trim = libc_function("malloc_trim")
    if trim is not None:
        trim(0)


def _resume_spec(spec: JobSpec) -> JobSpec:
    return spec.model_copy(
        update={"output": spec.output.model_copy(update={"on_existing": "resume"})}
    )


def _worker_split(
    spec: JobSpec,
    plan: ConcordPlan,
    pending: int,
    *,
    requested_device: str | torch.device,
    hardware: HardwareProfile | None,
    budget_policy: HardwareBudgetPolicy | None,
    measured_peak_bytes: int | None = None,
    threads: int | None = None,
) -> WorkerSplit:
    """Resolve ``run.workers`` and ``run.threads_per_worker`` for ``pending`` work units.

    ``measured_peak_bytes``: the most own memory a worker reached in one work
    unit with its buffers written first. ``threads`` overrides
    ``run.threads_per_worker``.
    """

    run = spec.run
    if run.workers != "auto":
        threads = None if run.threads_per_worker == "auto" else run.threads_per_worker
        return WorkerSplit(run.workers, threads, False, "set in the job")
    # Detected after the plan is loaded, so the free memory excludes it.
    profile = detect_hardware_profile() if hardware is None else hardware
    device = str(requested_device)
    on_cpu = device == "cpu" or (device == "auto" and not profile.accelerators)
    cores, domain_cores = node_shape()
    return auto_worker_split(
        threads=run.threads_per_worker if threads is None else threads,
        pending=pending,
        steps=len(plan.physical_plan.steps),
        flops=plan.cost_report.scheduled_contraction_flops,
        peak_bytes=estimate_execution_memory(plan.physical_plan).known_working_set_bytes,
        available_bytes=profile.host.available_memory_bytes,
        margin=(budget_policy or HardwareBudgetPolicy()).cpu_margin,
        cores=cores,
        domain_cores=domain_cores,
        can_fork=on_cpu and sys.platform.startswith("linux"),
        measured_peak_bytes=measured_peak_bytes,
    )


def _execute_pending_in_workers(
    spec: JobSpec,
    plan: ConcordPlan,
    pending: Sequence[JobWorkUnit],
    split: WorkerSplit,
    *,
    requested_device: str | torch.device,
    hardware: HardwareProfile | None,
    budget_policy: HardwareBudgetPolicy | None,
    time_batch_calibration_policy: TimeBatchCalibrationPolicy | None,
    calibration_cache_path: str | Path | None,
    memory_shares: int | None = None,
    first_index: int = 0,
    write_buffers_first: bool = False,
) -> tuple[tuple[_WorkerOutcome, ...], int, float]:
    """Execute pending work units in forked CPU workers; the parent writes results.

    Each worker plans within 1/``memory_shares`` of host memory (default: one
    share per worker); reports are numbered from ``first_index``. With
    ``write_buffers_first`` each worker writes its buffer pool before its first
    work unit. Returns each worker's outcome, the completed count and the
    parent's result-management seconds.
    """

    if str(requested_device).split(":", 1)[0] == "cuda":
        raise JobRunError("run.workers > 1 runs on the CPU; use device 'cpu' or 'auto'")
    shares = split_contiguous(pending, split.workers)
    threads = split.threads or max(1, len(usable_cpus()) // len(shares))
    cpu_sets = worker_cpu_sets(len(shares), threads)
    profile = detect_hardware_profile() if hardware is None else hardware
    share_hardware = _worker_hardware(profile, max(len(shares), memory_shares or 0), threads)
    assignments = tuple(
        WorkerAssignment(index=first_index + index, threads=threads, cpus=cpus, items=items)
        for index, (items, cpus) in enumerate(zip(shares, cpu_sets))
    )

    def work(assignment: WorkerAssignment, send: Callable[[object], None]) -> _WorkerOutcome:
        started = perf_counter()
        # Kernel-building temporaries stay out of the heap (see heap.py), so
        # that workers with one and with several work units use the same memory.
        fix_block_limit()
        memory_at_start = _memory_usage()
        tally = _ExecutionTally(observed_hardware=share_hardware)
        pending_by_configuration: list[list[JobWorkUnit]] = [[] for _ in spec.configurations]
        for work_unit in assignment.items:
            pending_by_configuration[work_unit.configuration_index].append(work_unit)
        _execute_pending(
            spec,
            plan,
            pending_by_configuration,
            _ForwardingWriter(send),
            tally,
            requested_device=requested_device,
            hardware=share_hardware,
            budget_policy=budget_policy,
            time_batch_calibration_policy=time_batch_calibration_policy,
            calibration_cache_path=calibration_cache_path,
            write_buffers_first=write_buffers_first,
        )
        timing = JobRunTiming.measured(
            execution_wall_seconds=perf_counter() - started,
            result_management_seconds=0.0,
            runtime_planning_seconds=tally.runtime_planning_seconds,
            data_preparation_seconds=tally.data_preparation_seconds,
            input_wait_seconds=tally.input_wait_seconds,
            numerical_execution_seconds=tally.numerical_execution_seconds,
            output_writing_seconds=tally.output_writing_seconds,
        )
        report = JobRunWorker(
            index=assignment.index,
            threads=assignment.threads,
            cpus=assignment.cpus or (),
            work_units=len(assignment.items),
            timing=timing,
            max_rss_bytes=_max_rss_bytes(),
            private_bytes=_private_bytes(),
            peak_private_bytes=_peak_private_bytes(memory_at_start),
        )
        return _WorkerOutcome(
            tally=replace(tally, execution_plan=None, observed_hardware=None),
            deployment=_deployment(tally.execution_plan),
            report=report,
        )

    units_by_slot = {_unit_slot(work_unit): work_unit for work_unit in pending}
    resume_spec = _resume_spec(spec)
    result_management_seconds = 0.0
    with ExitStack() as cleanup:
        opened: list[_JobResultStore] = []

        def store() -> _JobResultStore:
            # Opened after the fork, so no worker inherits an open output file.
            nonlocal result_management_seconds
            if not opened:
                stage_started = perf_counter()
                opened.append(cleanup.enter_context(_JobResultStore(resume_spec, plan)))
                result_management_seconds += perf_counter() - stage_started
            return opened[0]

        def handle(index: int, message: object) -> None:
            kind, slots, arrays = message
            outputs = {key: torch.from_numpy(array) for key, array in arrays.items()}
            if kind == "unit":
                store().write(units_by_slot[slots], outputs)
            else:
                store().write_batch(tuple(units_by_slot[slot] for slot in slots), outputs)

        outcomes = run_workers(assignments, work, handle)
        stage_started = perf_counter()
        completed = store().completed_count
        result_management_seconds += perf_counter() - stage_started
    return outcomes, completed, result_management_seconds


def _run_pending_in_workers(
    spec: JobSpec,
    plan: ConcordPlan,
    pending: Sequence[JobWorkUnit],
    split: WorkerSplit,
    *,
    requested_device: str | torch.device,
    hardware: HardwareProfile | None,
    budget_policy: HardwareBudgetPolicy | None,
    time_batch_calibration_policy: TimeBatchCalibrationPolicy | None,
    calibration_cache_path: str | Path | None,
    one_thread: bool = False,
) -> tuple[_ExecutionTally, dict[str, object], tuple[JobRunWorker, ...], int, float, WorkerSplit]:
    """Run pending work units in workers, measuring first if ``split`` asks for it.

    With ``split.measure_with`` > 0, that many workers first run one work unit
    each, planning within the memory share of the full split, and write their
    whole buffer pool before it. The system backs the pool with memory only as
    it is first written, so in a worker's first unit the distillation kernels
    are built while most of the pool is not yet in memory; from the second unit
    on they are built on top of all of it. The largest private memory the
    measuring workers reach then decides how many workers run the remaining
    units (``one_thread``: keep one thread per worker), within the same share.
    Returns the merged tally, the merged deployment, one report per worker, the
    completed count, the parent's result-management seconds and the split of
    the workers started last.
    """

    options = {
        "requested_device": requested_device,
        "hardware": hardware,
        "budget_policy": budget_policy,
        "time_batch_calibration_policy": time_batch_calibration_policy,
        "calibration_cache_path": calibration_cache_path,
    }
    outcomes: list[_WorkerOutcome] = []
    result_management_seconds = 0.0
    memory_shares: int | None = None
    if split.measure_with and split.measure_with < len(pending):
        first, pending = pending[: split.measure_with], pending[split.measure_with :]
        memory_shares = split.workers
        measured, _, seconds = _execute_pending_in_workers(
            spec,
            plan,
            first,
            replace(split, workers=len(first)),
            memory_shares=memory_shares,
            write_buffers_first=True,
            **options,
        )
        outcomes += measured
        result_management_seconds += seconds
        first_workers = f"{len(first)} worker" + ("s" if len(first) > 1 else "")
        peaks = [outcome.report.peak_private_bytes for outcome in measured]
        if None in peaks:
            # Without a measurement keep the workers the rough allowance fitted.
            split = replace(
                split,
                workers=len(first),
                reason=f"{split.reason}; memory not measurable, {first_workers}",
                measure_with=0,
            )
        else:
            measured_split = _worker_split(
                spec,
                plan,
                len(pending),
                requested_device=requested_device,
                hardware=hardware,
                budget_policy=budget_policy,
                measured_peak_bytes=max(peaks),
                threads=1 if one_thread else None,
            )
            split = replace(
                measured_split,
                reason=f"{measured_split.reason}; measured first by {first_workers}, one work unit each",
            )
    else:
        split = replace(split, measure_with=0)
    rest, completed, seconds = _execute_pending_in_workers(
        spec,
        plan,
        pending,
        split,
        memory_shares=memory_shares,
        first_index=len(outcomes),
        **options,
    )
    outcomes += rest
    result_management_seconds += seconds
    split = replace(split, workers=len(rest), threads=rest[-1].report.threads)
    merged = _merge_worker_tallies([outcome.tally for outcome in outcomes])
    merged.observed_hardware = detect_hardware_profile() if hardware is None else hardware
    return (
        merged,
        _merge_deployments([outcome.deployment for outcome in outcomes]),
        tuple(outcome.report for outcome in outcomes),
        completed,
        result_management_seconds,
        split,
    )


def run_job(
    spec: JobSpec,
    plan: ConcordPlan | None = None,
    *,
    device: str | torch.device | None = None,
    hardware: HardwareProfile | None = None,
    budget_policy: HardwareBudgetPolicy | None = None,
    adaptive_calibration_policy: AdaptiveCalibrationPolicy | None = None,
    time_batch_calibration_policy: TimeBatchCalibrationPolicy | None = None,
    calibration_cache_path: str | Path | None = None,
) -> JobRunSummary:
    """Execute every selected job work unit and incrementally persist outputs.

    If ``plan`` is omitted, ``spec.plan.file`` must already exist and is loaded.
    Planning/replanning policy belongs to the higher-level CLI layer; this runner
    never silently recompiles a missing or stale plan.

    With more than one worker (``spec.run.workers``, chosen from the plan, the
    cores and the free memory when "auto") the plan is loaded once, the pending
    work units are split into contiguous shares and each share runs in its own
    forked CPU process with ``threads_per_worker`` threads; this process writes
    every result. When a rough allowance for the memory a worker needs beyond
    its plan's buffers does not fit all the workers, fewer workers first run
    one work unit each and measure it, and the rest of the units are split
    again from that measurement. The summary's stage times are then means over
    the workers, whose own times are in ``worker_reports``.
    """

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    previous_threads = torch.get_num_threads()
    if spec.run.workers != 1:
        # Workers may be forked, and a forked child cannot use OpenMP threads
        # once its parent has. One process gets its threads once that is known.
        torch.set_num_threads(1)
    elif spec.run.threads_per_worker != "auto":
        torch.set_num_threads(spec.run.threads_per_worker)
    try:
        return _run_job(
            spec,
            plan,
            default_threads=previous_threads,
            device=device,
            hardware=hardware,
            budget_policy=budget_policy,
            adaptive_calibration_policy=adaptive_calibration_policy,
            time_batch_calibration_policy=time_batch_calibration_policy,
            calibration_cache_path=calibration_cache_path,
        )
    finally:
        torch.set_num_threads(previous_threads)


def _run_job(
    spec: JobSpec,
    plan: ConcordPlan | None,
    *,
    default_threads: int,
    device: str | torch.device | None,
    hardware: HardwareProfile | None,
    budget_policy: HardwareBudgetPolicy | None,
    adaptive_calibration_policy: AdaptiveCalibrationPolicy | None,
    time_batch_calibration_policy: TimeBatchCalibrationPolicy | None,
    calibration_cache_path: str | Path | None,
) -> JobRunSummary:
    validate_numerical_flavor_sources(spec)
    run_started = perf_counter()
    result_management_seconds = 0.0
    runtime_planning_seconds = 0.0

    stage_started = perf_counter()
    if plan is None:
        plan = _load_prepared_plan(spec.plan.file)
    if not isinstance(plan, ConcordPlan):
        raise TypeError("plan must be ConcordPlan")

    requested_device = spec.run.device if device is None else device
    observed_hardware = hardware
    if not _is_prepared_plan(plan):
        prepared = _try_prepare_plan_snapshot(plan)
        if prepared is not None:
            plan = prepared.plan
    validate_plan_for_job(plan, spec)
    contraction_path = plan.contraction_order.strategy
    contraction_selection_source = "configured"
    if plan.config.contraction_strategy == "adaptive":
        if observed_hardware is None:
            observed_hardware = detect_hardware_profile()
        try:
            adaptive = specialize_plan_for_hardware(
                spec,
                plan,
                device=requested_device,
                hardware=observed_hardware,
                budget_policy=budget_policy,
                calibration_policy=adaptive_calibration_policy,
                cache_path=calibration_cache_path,
            )
        except (
            AdaptivePlanningError,
            CalibrationCacheError,
            MemoryError,
            ResourcePlanningError,
            RuntimeError,
        ) as exc:
            raise JobRunError(f"adaptive execution planning failed: {exc}") from exc
        plan = adaptive.plan
        prepared = _try_prepare_plan_snapshot(plan)
        if prepared is not None:
            plan = prepared.plan
        validate_plan_for_job(plan, spec)
        contraction_path = adaptive.candidate_strategy
        contraction_selection_source = adaptive.source
        # Execute on the compute device whose deployment was calibrated.  A
        # later resource revalidation may safely retile that same device, but
        # must not silently move the selected contraction path to a different
        # CPU/GPU performance environment.
        requested_device = adaptive.selected_device
    runtime_planning_seconds += perf_counter() - stage_started
    work_units = tuple(iter_job_work_units(spec))
    skipped = 0
    tally = _ExecutionTally(observed_hardware=observed_hardware)
    deployment: dict[str, object] | None = None
    worker_reports: tuple[JobRunWorker, ...] = ()
    threads_per_worker = (
        None if spec.run.threads_per_worker == "auto" else spec.run.threads_per_worker
    )
    execution_options = {
        "requested_device": requested_device,
        "hardware": hardware,
        "budget_policy": budget_policy,
        "time_batch_calibration_policy": time_batch_calibration_policy,
        "calibration_cache_path": calibration_cache_path,
    }

    try:
        result_store = _JobResultStore(spec, plan)
        stage_started = perf_counter()
        with result_store as store:
            result_management_seconds += perf_counter() - stage_started
            stage_started = perf_counter()
            pending_by_configuration: list[list[JobWorkUnit]] = [
                [] for _ in spec.configurations
            ]
            for work_unit in work_units:
                if store.is_complete(work_unit):
                    skipped += 1
                else:
                    pending_by_configuration[work_unit.configuration_index].append(
                        work_unit
                    )
            result_management_seconds += perf_counter() - stage_started
            pending = [unit for units in pending_by_configuration for unit in units]
            stage_started = perf_counter()
            split = _worker_split(
                spec,
                plan,
                len(pending),
                requested_device=requested_device,
                hardware=hardware,
                budget_policy=budget_policy,
            )
            runtime_planning_seconds += perf_counter() - stage_started
            in_workers = split.workers > 1 and bool(pending)
            if not in_workers:
                if spec.run.workers == "auto":
                    threads_per_worker = split.threads or default_threads
                    torch.set_num_threads(threads_per_worker)
                _execute_pending(
                    spec, plan, pending_by_configuration, store, tally, **execution_options
                )
                stage_started = perf_counter()
                completed = store.completed_count
                result_management_seconds += perf_counter() - stage_started
        if in_workers:
            # The output file is closed here, so the forked workers never inherit it.
            parent_planning_seconds = tally.runtime_planning_seconds
            try:
                outcome = _run_pending_in_workers(
                    spec, plan, pending, split, **execution_options
                )
            except WorkerStartError as exc:
                if not split.automatic or split.threads == 1:
                    raise
                # This process ran multi-threaded PyTorch work before the run;
                # workers with one thread each do not start OpenMP.
                warnings.warn(f"{exc}. Using one thread per worker instead.", RuntimeWarning)
                split = replace(
                    split,
                    threads=1,
                    reason=f"{split.reason}; one thread each, threaded workers did not start",
                )
                with _JobResultStore(_resume_spec(spec), plan) as store:
                    pending = [unit for unit in pending if not store.is_complete(unit)]
                if not pending:
                    raise
                outcome = _run_pending_in_workers(
                    spec, plan, pending, split, one_thread=True, **execution_options
                )
            tally, deployment, worker_reports, completed, worker_management_seconds, split = outcome
            tally.runtime_planning_seconds += parent_planning_seconds
            result_management_seconds += worker_management_seconds
            threads_per_worker = split.threads
    except JobDataError:
        raise
    except (FileExistsError, JobRunError):
        raise
    except WorkerStartError as exc:
        raise JobRunError(str(exc)) from exc
    except WorkerFailure as exc:
        if isinstance(exc.error, (JobDataError, FileExistsError, JobRunError)):
            error = exc.error
        else:
            error = JobRunError(f"job execution failed in {exc}")
        error.add_note(f"traceback of worker {exc.index}:\n{exc.details}")
        raise error from exc
    except (BatchPlanningError, CalibrationCacheError, TimeBatchPlanningError) as exc:
        raise JobRunError(f"time-batch planning failed: {exc}") from exc
    except ResourcePlanningError as exc:
        raise JobRunError(f"execution resource planning failed: {exc}") from exc
    except Exception as exc:
        raise JobRunError(f"job execution failed: {exc}") from exc

    if deployment is None:
        deployment = _deployment(tally.execution_plan)
    estimate = estimate_execution_memory(plan.physical_plan)
    timing = JobRunTiming.measured(
        execution_wall_seconds=perf_counter() - run_started,
        result_management_seconds=result_management_seconds,
        runtime_planning_seconds=runtime_planning_seconds + tally.runtime_planning_seconds,
        data_preparation_seconds=tally.data_preparation_seconds,
        input_wait_seconds=tally.input_wait_seconds,
        numerical_execution_seconds=tally.numerical_execution_seconds,
        output_writing_seconds=tally.output_writing_seconds,
    )

    return JobRunSummary(
        output_file=spec.output.file.expanduser().resolve(strict=False),
        plan_sha256=plan.manifest.plan_sha256,
        timing=timing,
        total_work_units=len(work_units),
        executed_work_units=tally.executed,
        skipped_work_units=skipped,
        completed_work_units=completed,
        correlator_ids=tuple(plan.correlator_ids),
        contraction_path=contraction_path,
        contraction_selection_source=contraction_selection_source,
        time_batch_size=tally.time_batch_size,
        time_batch_selection_source=tally.time_batch_selection_source,
        executed_batches=tally.executed_batches,
        selected_device=deployment["selected_device"],
        storage_device=deployment["storage_device"],
        execution_mode=deployment["execution_mode"],
        hardware_profile=tally.observed_hardware or observed_hardware,
        configuration_decisions=tuple(tally.configuration_decisions),
        memory_verified=deployment["memory_verified"],
        known_working_set_bytes=estimate.known_working_set_bytes,
        peak_tile_bytes=deployment["peak_tile_bytes"],
        tiled_operations=deployment["tiled_operations"],
        resident_pool_allocations=tally.resident_pool_allocations,
        resident_pool_peak_bytes=tally.resident_pool_peak_bytes,
        primitive_cache_hits=tally.primitive_cache_hits,
        primitive_cache_misses=tally.primitive_cache_misses,
        primitive_cache_allocation_failures=tally.primitive_cache_allocation_failures,
        primitive_cache_peak_bytes=tally.primitive_cache_peak_bytes,
        kernel_cache_enabled=spec.data.kernel_cache is not None,
        kernel_cache_hits=tally.kernel_cache_hits,
        kernel_cache_misses=tally.kernel_cache_misses,
        kernel_cache_writes=tally.kernel_cache_writes,
        kernel_cache_invalid_entries=tally.kernel_cache_invalid_entries,
        kernel_cache_write_failures=tally.kernel_cache_write_failures,
        hdf5_file_opens=tally.hdf5_file_opens,
        hdf5_file_reuses=tally.hdf5_file_reuses,
        propagation_prefetch_configurations=tally.propagation_prefetch_configurations,
        propagation_prefetch_batches=tally.propagation_prefetch_batches,
        propagation_prefetch_tensors=tally.propagation_prefetch_tensors,
        propagation_prefetch_bytes=tally.propagation_prefetch_bytes,
        propagation_prefetch_wait_seconds=tally.propagation_prefetch_wait_seconds,
        propagation_prefetch_allocation_fallbacks=(
            tally.propagation_prefetch_allocation_fallbacks
        ),
        workers=split.workers if worker_reports else 1,
        threads_per_worker=threads_per_worker,
        worker_reports=worker_reports,
        worker_selection=split.reason,
    )


__all__ = [
    "JobRunError",
    "JobWorkUnit",
    "ConfigurationRuntimeDecision",
    "JobRunSummary",
    "iter_job_work_units",
    "run_job",
]
