"""Primitive tensor materialization for the new ConCorD execution pipeline.

This module consumes the frozen :class:`ResourceBindingPlan` and turns one
scheduled primitive materialization request into one numerical ``torch.Tensor``.
It deliberately stops before contraction execution or physical buffer writes.

Layer boundary::

    ResourceBindingPlan
        -> primitive request
        -> provider/source handler
        -> validated tensor

The compiler/resource/value/buffer/source identities therefore remain distinct.
A returned tensor is associated with the planned logical value and destination
buffer, but this layer does not allocate or mutate the reusable physical buffer
pool. That belongs to the executor.

Kernel algebra reuses the existing Stage-3 ``emit_backend_kernel_program`` plus
``evaluate_kernel_program`` path through an optional provider adapter. External
propagation data is handled by source-kind-specific loaders. HDF5 support is
metadata driven and uses ``h5py``; it never falls back to flavor-based lookup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeAlias

import h5py
import numpy as np
import torch

from concord_lqcd.backend.runtime.heap import kernel_building
from concord_lqcd.backend.runtime.kernels import evaluate_kernel_program
from concord_lqcd.core.dirac_basis import get_spin_matrices
from concord_lqcd.compiler.distillation_lowering import (
    ProjectionContribution,
    emit_backend_kernel_program,
)

from .context import ExecutionContext
from .providers import TensorProvider
from .resource_binding import (
    KernelMaterializationRequest,
    MaterializationRequest,
    PropagationMaterializationRequest,
    ResourceBindingPlan,
    SpinMaterializationRequest,
)


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

_PROPAGATION_DATA_AXES = (
    "sink_mode",
    "sink_spin",
    "source_mode",
    "source_spin",
)
_HDF5_BATCH_AXIS = "__concord_hdf5_batch__"
_MAX_HDF5_SLAB_AMPLIFICATION = 2.0

_HDF5_DIMENSION_ORDER_ALIASES = {
    "sink_time": "sink_time",
    "t_sink": "sink_time",
    "source_time": "source_time",
    "t_source": "source_time",
    "t_src": "source_time",
    "sink_spin": "sink_spin",
    "spin_sink": "sink_spin",
    "spin_snk": "sink_spin",
    "snk_spin": "sink_spin",
    "source_spin": "source_spin",
    "spin_source": "source_spin",
    "spin_src": "source_spin",
    "src_spin": "source_spin",
    "sink_mode": "sink_mode",
    "snk_mode": "sink_mode",
    "sink_eigs": "sink_mode",
    "snk_eigs": "sink_mode",
    "eig_snk": "sink_mode",
    "source_mode": "source_mode",
    "src_mode": "source_mode",
    "source_eigs": "source_mode",
    "src_eigs": "source_mode",
    "eig_src": "source_mode",
    "complex_pair": "complex_pair",
    "configuration": "configuration",
}


def _torch_dtype(scalar_type: str) -> torch.dtype:
    try:
        return _TORCH_DTYPES[scalar_type]
    except KeyError as exc:
        raise ValueError(
            f"Primitive materialization does not know scalar type {scalar_type!r}. "
            f"Supported types are {tuple(_TORCH_DTYPES)}."
        ) from exc


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise ValueError(
        "The compiled distillation kernel adapter currently requires a complex "
        "scalar type."
    )


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except Exception as exc:  # pragma: no cover - backend-specific conversion failure
        raise TypeError("Primitive provider did not return tensor-convertible data.") from exc


_UNCACHEABLE = object()
MaterializationCacheKey: TypeAlias = tuple[object, ...]


def _freeze_cache_value(value: Any) -> object:
    """Return a collision-free immutable cache payload or ``_UNCACHEABLE``.

    Runtime extensions may attach arbitrary objects to source metadata or cache
    namespaces.  Values outside this deliberately small structural vocabulary
    bypass caching instead of relying on unstable ``repr`` or object identity.
    """

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if isinstance(value, Path):
        return ("path", str(value.expanduser().resolve(strict=False)))
    if isinstance(value, torch.device):
        return ("device", str(value))
    if isinstance(value, Mapping):
        frozen_items: list[tuple[object, object]] = []
        for key, item in value.items():
            frozen_key = _freeze_cache_value(key)
            frozen_item = _freeze_cache_value(item)
            if frozen_key is _UNCACHEABLE or frozen_item is _UNCACHEABLE:
                return _UNCACHEABLE
            frozen_items.append((frozen_key, frozen_item))
        try:
            return ("mapping", *sorted(frozen_items, key=repr))
        except TypeError:  # pragma: no cover - repr provides a total string key
            return _UNCACHEABLE
    if isinstance(value, (tuple, list)):
        frozen = tuple(_freeze_cache_value(item) for item in value)
        if any(item is _UNCACHEABLE for item in frozen):
            return _UNCACHEABLE
        return ("sequence", *frozen)
    if isinstance(value, (set, frozenset)):
        frozen = tuple(_freeze_cache_value(item) for item in value)
        if any(item is _UNCACHEABLE for item in frozen):
            return _UNCACHEABLE
        return ("set", *sorted(frozen, key=repr))
    return _UNCACHEABLE


@dataclass(frozen=True)
class HDF5FilePoolStatistics:
    file_opens: int
    file_reuses: int
    open_files: int


class HDF5FilePool:
    """Configuration-scoped pool of read-only HDF5 file handles.

    The lock deliberately covers the complete caller operation.  This matches
    h5py's serialized low-level API and prepares the same object for the later
    single-worker prefetch pipeline without allowing concurrent handle use.
    """

    def __init__(self) -> None:
        self._files: dict[str, h5py.File] = {}
        self._lock = RLock()
        self._closed = False
        self._file_opens = 0
        self._file_reuses = 0

    @staticmethod
    def _key(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve(strict=False))

    @contextmanager
    def file(self, path: str | Path) -> Iterator[h5py.File]:
        key = self._key(path)
        with self._lock:
            if self._closed:
                raise RuntimeError("HDF5 file pool is closed")
            handle = self._files.get(key)
            if handle is None or not bool(handle.id.valid):
                handle = h5py.File(key, "r")
                self._files[key] = handle
                self._file_opens += 1
            else:
                self._file_reuses += 1
            yield handle

    @property
    def statistics(self) -> HDF5FilePoolStatistics:
        with self._lock:
            return HDF5FilePoolStatistics(
                file_opens=self._file_opens,
                file_reuses=self._file_reuses,
                open_files=len(self._files),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for handle in self._files.values():
                handle.close()
            self._files.clear()
            self._closed = True

    def __enter__(self) -> "HDF5FilePool":
        if self._closed:
            raise RuntimeError("HDF5 file pool cannot be reopened")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class KernelMaterializationProvider(ABC):
    """Runtime provider for one bound spatial-kernel occurrence."""

    @abstractmethod
    def materialize(self, request: KernelMaterializationRequest) -> Any:
        """Return the numerical tensor for ``request`` without persistent caching."""
        ...

    def materialize_batch(
        self,
        requests: Sequence[KernelMaterializationRequest],
    ) -> "KernelBatchMaterialization | None":
        """Materialize a compatible leading-axis batch, when supported.

        Returning ``None`` preserves independent materialization. Providers opt
        in only when one batched evaluation has the same numerical meaning and
        request order as the individual calls.
        """

        return None

    def materialize_batch_with_limit(
        self,
        requests: Sequence[KernelMaterializationRequest],
        *,
        maximum_staging_bytes: int | None,
    ) -> "KernelBatchMaterialization | None":
        """Materialize a batch under an optional provider-staging limit.

        A provider that implements only :meth:`materialize_batch` has not
        established that it can honor a finite staging budget. Such providers
        retain the independent path whenever a limit is active. Providers that
        can plan bounded staging must override this method explicitly.
        """

        if maximum_staging_bytes is not None:
            return None
        return self.materialize_batch(requests)

    def batch_staging_bytes(
        self,
        requests: Sequence[KernelMaterializationRequest],
    ) -> int | None:
        """Return conservative provider-owned staging for a supported batch."""

        return None

    def batch_staging_bytes_with_limit(
        self,
        requests: Sequence[KernelMaterializationRequest],
        *,
        maximum_staging_bytes: int | None,
    ) -> int | None:
        """Plan provider staging under an optional memory limit."""

        del maximum_staging_bytes
        return self.batch_staging_bytes(requests)

    def cache_identity(self) -> object | None:
        """Return immutable provider state that makes equal requests reusable.

        Custom providers are uncacheable by default.  A provider opts in only
        when this identity completely describes all numerical state not already
        present in the materialization request.
        """

        return None


@dataclass(frozen=True)
class KernelBatchMaterialization:
    """One kernel-provider result with a private leading batch axis."""

    tensor: torch.Tensor
    mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("batched kernel materialization must be a torch tensor")
        if self.tensor.ndim < 1 or self.tensor.shape[0] <= 0:
            raise ValueError("batched kernel materialization must have a batch axis")
        if not self.mode:
            raise ValueError("batched kernel materialization mode must be non-empty")


class PropagationSourceHandler(ABC):
    """Load one propagation occurrence from an ``ExternalTensorSource`` kind."""

    @abstractmethod
    def materialize(self, request: PropagationMaterializationRequest) -> Any:
        ...

    def materialize_batch(
        self,
        requests: Sequence[PropagationMaterializationRequest],
    ) -> "PropagationBatchMaterialization | None":
        """Materialize an exact leading-axis batch when explicitly supported.

        Returning ``None`` preserves independent materialization. Custom source
        handlers therefore retain their established behavior unless they opt in
        with a provably equivalent batch implementation.
        """

        return None

    def cache_identity(self) -> object | None:
        """Return a stable implementation identity, or ``None`` to bypass cache."""

        return None


@dataclass(frozen=True)
class PropagationBatchMaterialization:
    """One source-handler result with a private leading batch axis."""

    tensor: torch.Tensor
    mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("batched propagation materialization must be a torch tensor")
        if self.tensor.ndim < 1 or self.tensor.shape[0] <= 0:
            raise ValueError("batched propagation materialization must have a batch axis")
        if not self.mode:
            raise ValueError("batched propagation materialization mode must be non-empty")


class CompiledKernelMaterializationProvider(KernelMaterializationProvider):
    """Adapter from a request-scoped ``TensorProvider`` to the Stage-3 kernel ISA.

    ``provider_factory`` receives the exact occurrence request and must return a
    ``TensorProvider`` configured so its spatial fields correspond to that
    occurrence. The current kernel ISA has one common time leg, so this adapter
    supports kernels whose operator placement resolves to one Euclidean time.
    More general multi-time spatial objects can use a custom
    ``KernelMaterializationProvider`` without changing the materializer.

    The adapter deliberately uses the existing ``emit_backend_kernel_program``
    and ``evaluate_kernel_program`` implementation instead of duplicating
    spatial/color contraction logic here.
    """

    def __init__(
        self,
        provider_factory: Callable[[KernelMaterializationRequest], TensorProvider],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.provider_factory = provider_factory
        self.device = device if isinstance(device, torch.device) else torch.device(device)

    def materialize(self, request: KernelMaterializationRequest) -> torch.Tensor:
        if any(
            field.contribution == ProjectionContribution.POINT
            for field in request.recipe.projected_fields
        ):
            raise NotImplementedError(
                "The compiled distillation kernel adapter cannot materialize a "
                "standalone POINT spatial contribution. POINT currents absorbed "
                "into generalized propagation are materialized as propagation data."
            )

        distinct_times = tuple(dict.fromkeys(request.operator_times))
        if len(distinct_times) > 1:
            raise NotImplementedError(
                "The compiled distillation kernel adapter currently requires one "
                "common Euclidean time for all operator slots in a spatial kernel."
            )
        occurrence_time = distinct_times[0] if distinct_times else 0

        provider = self.provider_factory(request)
        if not isinstance(provider, TensorProvider):
            raise TypeError("provider_factory must return a TensorProvider instance.")

        complex_dtype = _torch_dtype(request.scalar_type)
        real_dtype = _real_dtype_for_complex(complex_dtype)
        context = ExecutionContext.create(
            device=self.device,
            t_src=occurrence_time,
            t_snk_list=(occurrence_time,),
            dtype=real_dtype,
            complex_dtype=complex_dtype,
        )
        program = emit_backend_kernel_program(request.recipe)
        return evaluate_kernel_program(program, provider, context, is_sink=True)


class InMemoryTensorSourceHandler(PropagationSourceHandler):
    """Materialize propagation data from an application-owned in-memory registry.

    For ``ExternalTensorSource(kind="memory", location="key")``, ``location`` is
    used as the exact registry key. An entry may be a tensor/array-like value or
    a callable accepting the full ``PropagationMaterializationRequest``. The
    callable form is useful when one stored dataset serves several time slices.
    """

    def __init__(self, tensors: Mapping[str, Any]) -> None:
        self.tensors = dict(tensors)

    def materialize(self, request: PropagationMaterializationRequest) -> Any:
        key = request.source.location
        try:
            value = self.tensors[key]
        except KeyError as exc:
            raise KeyError(
                f"In-memory propagation source location {key!r} is not registered."
            ) from exc
        return value(request) if callable(value) else value


@dataclass(frozen=True)
class _HDF5PropagationSelection:
    location: str
    dataset_name: str
    axes: tuple[str, ...] | None
    indexers: tuple[int | slice, ...] | None
    remaining_axes: tuple[str, ...] | None
    expected_shape: tuple[int, ...]
    reverse: bool
    spin_basis: str
    source_field: str


class HDF5TensorSourceHandler(PropagationSourceHandler):
    """Metadata-driven HDF5 propagation loader.

    ``ExternalTensorSource.dataset`` names the HDF5 dataset and may contain
    ``str.format`` placeholders ``{source_time}``, ``{sink_time}``,
    ``{fixed_source_time}``, and ``{insertion_time_N}``.

    For datasets with explicit time/configuration axes, source metadata may
    provide::

        metadata={
            "axes": (
                "sink_time", "sink_spin", "source_spin",
                "sink_mode", "source_mode",
            ),
            "fixed_indices": {"configuration": 0},
        }

    Recognized numerical output axes are ``sink_mode``, ``sink_spin``,
    ``source_mode``, and ``source_spin``. They may occur in any storage order;
    the handler slices them to the planned dimensions and permutes to the exact
    ConCorD propagation interface. ``complex_pair`` may be used for an explicit
    real/imaginary axis of extent two. Other non-output axes must be selected by
    ``fixed_indices``. If ``axes`` is omitted, the selected dataset is treated as
    already containing exactly one materialized propagation value.

    This is intentionally storage-layout metadata, not compiler resource
    identity. Ordinary and generalized perambulators use the same handler; the
    request's ordered insertion times supply any generalized time coordinates.

    Fixed-source files may declare the source represented by the selected
    dataset with ``metadata["fixed_source_time"]``.  In that mode the handler
    supports the two orientations implied by a single fixed-source file:

    * ``source_time == fixed_source_time``: select the stored sink-time slice;
    * ``sink_time == fixed_source_time``: select the slice at ``source_time``
      and reconstruct the reverse tensor with ``gamma5 P^dagger gamma5``.

    Requests for which neither endpoint is the fixed source fail explicitly.
    Dataset placeholders and time axes refer to the stored orientation:
    ``source_time`` is the fixed source and ``sink_time`` is the varying
    endpoint. The materialization request retains its logical endpoint times.
    The reverse operation is applied only after complex decoding and
    canonicalization to ``(sink_mode, sink_spin, source_mode, source_spin)``.
    """

    def __init__(self, file_pool: HDF5FilePool | None = None) -> None:
        if file_pool is not None and not isinstance(file_pool, HDF5FilePool):
            raise TypeError("file_pool must be an HDF5FilePool or None")
        self.file_pool = file_pool

    def cache_identity(self) -> object | None:
        if type(self) is not HDF5TensorSourceHandler:
            return None
        return ("concord-hdf5-propagation", 1)

    @contextmanager
    def _open_file(self, path: str) -> Iterator[h5py.File]:
        if self.file_pool is None:
            with h5py.File(path, "r") as handle:
                yield handle
            return
        with self.file_pool.file(path) as handle:
            yield handle

    @staticmethod
    def _fixed_source_time(metadata: Mapping[str, Any]) -> int | None:
        value = metadata.get("fixed_source_time")
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("HDF5 fixed_source_time metadata must be an integer.")
        return value

    @staticmethod
    def _spin_basis(metadata: Mapping[str, Any]) -> str:
        basis = metadata.get("spin_basis", "DR")
        if basis not in ("DR", "DP"):
            raise ValueError("HDF5 propagation spin_basis must be 'DR' or 'DP'.")
        return str(basis)

    @staticmethod
    def _source_field(metadata: Mapping[str, Any]) -> str:
        source_field = metadata.get("source_field", "qbar")
        if source_field not in ("qbar", "qdagger"):
            raise ValueError("HDF5 propagation source_field must be 'qbar' or 'qdagger'.")
        return str(source_field)

    @staticmethod
    def _normalize_dimension_order(value: Any) -> tuple[str, ...]:
        if isinstance(value, np.ndarray):
            value = value.item() if value.ndim == 0 else value.tolist()
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            parts: list[Any] = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            raise ValueError(
                "HDF5 dimension_order attribute must be a comma-separated string "
                "or a sequence of axis labels."
            )

        normalized: list[str] = []
        for part in parts:
            if isinstance(part, (bytes, np.bytes_)):
                part = part.decode("utf-8")
            label = str(part).strip()
            if not label:
                raise ValueError("HDF5 dimension_order contains an empty axis label.")
            normalized.append(_HDF5_DIMENSION_ORDER_ALIASES.get(label, label))
        return tuple(normalized)

    @classmethod
    def _validate_dimension_order(
        cls,
        dataset: h5py.Dataset,
        axes: tuple[str, ...],
        *,
        dataset_name: str,
    ) -> None:
        if "dimension_order" not in dataset.attrs:
            return
        actual = cls._normalize_dimension_order(dataset.attrs["dimension_order"])
        if actual != axes:
            raise ValueError(
                f"HDF5 dimension_order for dataset {dataset_name!r} disagrees with "
                f"configured axes: attribute normalizes to {actual}, configured axes "
                f"are {axes}."
            )

    def _format_values(
        self,
        request: PropagationMaterializationRequest,
        *,
        fixed_source_time: int | None = None,
    ) -> dict[str, int]:
        source_time = request.source_time
        sink_time = request.sink_time
        if fixed_source_time is not None:
            # Templates and explicit time axes must select the same stored
            # orientation before reverse reconstruction changes the tensor.
            source_time = fixed_source_time
            sink_time = (
                request.sink_time
                if request.source_time == fixed_source_time
                else request.source_time
            )
        values = {
            "source_time": source_time,
            "sink_time": sink_time,
        }
        values.update({
            f"insertion_time_{idx}": value
            for idx, value in enumerate(request.insertion_times)
        })
        if fixed_source_time is not None:
            values["fixed_source_time"] = fixed_source_time
        return values

    def _dataset_name(
        self,
        request: PropagationMaterializationRequest,
        *,
        fixed_source_time: int | None = None,
    ) -> str:
        template = request.source.dataset
        if not template:
            raise ValueError(
                "HDF5 propagation sources require ExternalTensorSource.dataset."
            )
        try:
            return template.format(
                **self._format_values(request, fixed_source_time=fixed_source_time)
            )
        except KeyError as exc:
            raise ValueError(
                f"HDF5 dataset template references unavailable placeholder {exc.args[0]!r}."
            ) from exc

    @staticmethod
    def _decode_complex(
        array: np.ndarray,
        remaining_axes: list[str] | None,
    ) -> tuple[np.ndarray, list[str] | None]:
        arr = np.asarray(array)
        names = arr.dtype.names
        if names is not None:
            re_key = "re" if "re" in names else ("real" if "real" in names else None)
            im_key = "im" if "im" in names else ("imag" if "imag" in names else None)
            if re_key is None or im_key is None:
                if len(names) < 2:
                    raise ValueError(
                        "Compound HDF5 tensor dtype does not contain two real/imag fields."
                    )
                re_key, im_key = names[0], names[1]
            arr = np.asarray(arr[re_key]) + 1j * np.asarray(arr[im_key])

        if remaining_axes is not None and "complex_pair" in remaining_axes:
            pair_axis = remaining_axes.index("complex_pair")
            if arr.shape[pair_axis] != 2:
                raise ValueError("HDF5 complex_pair axis must have extent two.")
            arr = np.take(arr, 0, axis=pair_axis) + 1j * np.take(arr, 1, axis=pair_axis)
            remaining_axes = list(remaining_axes)
            remaining_axes.pop(pair_axis)

        return arr, remaining_axes

    def _resolve_selection(
        self,
        request: PropagationMaterializationRequest,
    ) -> _HDF5PropagationSelection:
        source = request.source
        metadata = dict(source.metadata)
        fixed_source_time = self._fixed_source_time(metadata)
        reverse = False
        if fixed_source_time is not None and request.source_time != fixed_source_time:
            if request.sink_time == fixed_source_time:
                reverse = True
            else:
                raise ValueError(
                    "Fixed-source HDF5 data cannot realize propagation request "
                    f"{request.sink_time} <- {request.source_time}: neither endpoint "
                    f"is fixed source time {fixed_source_time}."
                )

        dataset_name = self._dataset_name(
            request,
            fixed_source_time=fixed_source_time,
        )
        axes_payload = metadata.get("axes")
        fixed_indices = dict(metadata.get("fixed_indices", {}))
        time_values = self._format_values(
            request,
            fixed_source_time=fixed_source_time,
        )

        expected_shape = tuple(request.expected_shape)
        if len(expected_shape) != len(_PROPAGATION_DATA_AXES):
            raise ValueError(
                "HDF5 propagation materialization expects the four-axis ConCorD "
                "propagation interface."
            )
        data_sizes = dict(zip(_PROPAGATION_DATA_AXES, expected_shape))

        if axes_payload is None:
            return _HDF5PropagationSelection(
                location=source.location,
                dataset_name=dataset_name,
                axes=None,
                indexers=None,
                remaining_axes=None,
                expected_shape=expected_shape,
                reverse=reverse,
                spin_basis=self._spin_basis(metadata),
                source_field=self._source_field(metadata),
            )

        axes = tuple(str(axis) for axis in axes_payload)
        if len(axes) != len(set(axes)):
            raise ValueError("HDF5 axes metadata contains duplicate axis names.")

        indexers: list[int | slice] = []
        remaining_axes: list[str] = []
        for axis in axes:
            if axis in time_values:
                indexers.append(time_values[axis])
            elif axis in data_sizes:
                indexers.append(slice(0, data_sizes[axis]))
                remaining_axes.append(axis)
            elif axis == "complex_pair":
                indexers.append(slice(None))
                remaining_axes.append(axis)
            elif axis in fixed_indices:
                fixed = fixed_indices[axis]
                if not isinstance(fixed, int) or isinstance(fixed, bool):
                    raise ValueError(
                        f"Fixed HDF5 index for axis {axis!r} must be an integer."
                    )
                indexers.append(fixed)
            else:
                raise ValueError(
                    f"HDF5 axis {axis!r} is neither a propagation/time axis "
                    "nor selected by fixed_indices."
                )
        return _HDF5PropagationSelection(
            location=source.location,
            dataset_name=dataset_name,
            axes=axes,
            indexers=tuple(indexers),
            remaining_axes=tuple(remaining_axes),
            expected_shape=expected_shape,
            reverse=reverse,
            spin_basis=self._spin_basis(metadata),
            source_field=self._source_field(metadata),
        )

    @staticmethod
    def _indexer_identity(indexer: int | slice) -> tuple[object, ...]:
        if isinstance(indexer, int) and not isinstance(indexer, bool):
            return ("index", indexer)
        if not isinstance(indexer, slice):  # pragma: no cover - internal invariant
            raise TypeError("HDF5 indexers must contain integers or slices")
        return ("slice", indexer.start, indexer.stop, indexer.step)

    @staticmethod
    def _same_batch_structure(
        left: _HDF5PropagationSelection,
        right: _HDF5PropagationSelection,
    ) -> bool:
        return (
            left.location == right.location
            and left.dataset_name == right.dataset_name
            and left.axes == right.axes
            and left.remaining_axes == right.remaining_axes
            and left.expected_shape == right.expected_shape
            and left.reverse == right.reverse
            and left.spin_basis == right.spin_basis
            and left.source_field == right.source_field
        )

    def _read_selection(
        self,
        selection: _HDF5PropagationSelection,
        *,
        indexers: tuple[int | slice, ...] | None,
        value_id: str,
    ) -> np.ndarray:
        with self._open_file(selection.location) as handle:
            if selection.dataset_name not in handle:
                raise KeyError(
                    f"HDF5 dataset {selection.dataset_name!r} was not found in "
                    f"{selection.location!r}."
                )
            dataset = handle[selection.dataset_name]
            if selection.axes is not None:
                self._validate_dimension_order(
                    dataset,
                    selection.axes,
                    dataset_name=selection.dataset_name,
                )
                if len(selection.axes) != dataset.ndim:
                    raise ValueError(
                        f"HDF5 axes metadata has rank {len(selection.axes)}, but dataset "
                        f"{selection.dataset_name!r} has rank {dataset.ndim}."
                    )
            try:
                return np.asarray(dataset[...] if indexers is None else dataset[indexers])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"HDF5 indices derived for {value_id!r} are outside dataset "
                    f"{selection.dataset_name!r}."
                ) from exc

    @staticmethod
    def _reverse_fixed_source_array(
        array: np.ndarray,
        *,
        spin_basis: str,
    ) -> np.ndarray:
        batched = array.ndim == 5
        values = array if batched else array[np.newaxis, ...]
        if values.ndim != 5:
            raise ValueError(
                "Fixed-source reverse reconstruction requires a canonical "
                "four-axis propagation tensor."
            )
        batch, sink_modes, sink_spins, source_modes, source_spins = values.shape
        matrix = np.ascontiguousarray(values).reshape(
            batch,
            sink_modes * sink_spins,
            source_modes * source_spins,
        )
        gamma5 = np.asarray(
            get_spin_matrices(spin_basis)["gamma5"],
            dtype=np.complex128,
        )
        left = np.kron(np.eye(source_modes, dtype=np.complex128), gamma5)
        right = np.kron(np.eye(sink_modes, dtype=np.complex128), gamma5)
        reversed_values = (
            left[np.newaxis, ...]
            @ matrix.conj().transpose(0, 2, 1)
            @ right[np.newaxis, ...]
        ).reshape(
            batch,
            source_modes,
            source_spins,
            sink_modes,
            sink_spins,
        )
        return reversed_values if batched else reversed_values[0]

    def _canonical_tensor(
        self,
        array: np.ndarray,
        remaining_axes: tuple[str, ...] | None,
        selection: _HDF5PropagationSelection,
        *,
        batched: bool,
        scalar_type: str,
    ) -> torch.Tensor:
        decoded, decoded_axes = self._decode_complex(
            array,
            None if remaining_axes is None else list(remaining_axes),
        )
        expected_axes = (
            (_HDF5_BATCH_AXIS, *_PROPAGATION_DATA_AXES)
            if batched
            else _PROPAGATION_DATA_AXES
        )
        if decoded_axes is not None:
            if len(decoded_axes) != len(expected_axes) or set(decoded_axes) != set(
                expected_axes
            ):
                raise ValueError(
                    "After selecting HDF5 non-data axes, the remaining axes must be "
                    f"exactly {expected_axes}; got {tuple(decoded_axes)}."
                )
            permutation = tuple(decoded_axes.index(axis) for axis in expected_axes)
            if permutation != tuple(range(len(permutation))):
                decoded = np.transpose(decoded, permutation)

        expected_shape = (
            (decoded.shape[0], *selection.expected_shape)
            if batched
            else selection.expected_shape
        )
        if tuple(decoded.shape) != expected_shape:
            raise ValueError(
                f"HDF5 propagation selection produced shape {tuple(decoded.shape)}, "
                f"expected {expected_shape}."
            )

        if selection.source_field == "qdagger":
            gamma4 = np.asarray(
                get_spin_matrices(selection.spin_basis)["gammaT"],
                dtype=np.complex128,
            )
            decoded = np.einsum(
                "...iajc,cb->...iajb",
                decoded,
                gamma4,
                optimize=True,
            )

        if selection.reverse:
            decoded = self._reverse_fixed_source_array(
                decoded,
                spin_basis=selection.spin_basis,
            )

        return torch.as_tensor(
            np.ascontiguousarray(decoded),
            dtype=_torch_dtype(scalar_type),
        )

    def materialize_batch(
        self,
        requests: Sequence[PropagationMaterializationRequest],
    ) -> PropagationBatchMaterialization | None:
        values = tuple(requests)
        if len(values) <= 1:
            return None
        if any(
            not isinstance(request, PropagationMaterializationRequest)
            for request in values
        ):
            raise TypeError("HDF5 batched materialization requires propagation requests")

        first_request = values[0]
        if any(
            request.expected_shape != first_request.expected_shape
            or request.scalar_type != first_request.scalar_type
            for request in values[1:]
        ):
            return None

        selections = tuple(self._resolve_selection(request) for request in values)
        first = selections[0]
        if any(
            not self._same_batch_structure(first, selection)
            for selection in selections[1:]
        ):
            return None

        if first.indexers is None:
            if any(selection.indexers is not None for selection in selections[1:]):
                return None
            tensor = self.materialize(first_request)
            return PropagationBatchMaterialization(
                tensor=tensor.unsqueeze(0).expand(len(values), *tensor.shape),
                mode="hdf5_shared_selection",
            )

        concrete_indexers = tuple(selection.indexers for selection in selections)
        if any(indexers is None for indexers in concrete_indexers):
            return None
        indexers = tuple(
            selection_indexers
            for selection_indexers in concrete_indexers
            if selection_indexers is not None
        )
        differing_axes = tuple(
            axis_index
            for axis_index in range(len(first.indexers))
            if any(
                self._indexer_identity(candidate[axis_index])
                != self._indexer_identity(first.indexers[axis_index])
                for candidate in indexers[1:]
            )
        )
        if not differing_axes:
            tensor = self.materialize(first_request)
            return PropagationBatchMaterialization(
                tensor=tensor.unsqueeze(0).expand(len(values), *tensor.shape),
                mode="hdf5_shared_selection",
            )
        if len(differing_axes) != 1:
            return None

        varying_axis = differing_axes[0]
        requested_indices = tuple(candidate[varying_axis] for candidate in indexers)
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in requested_indices
        ):
            return None
        integer_indices = tuple(int(index) for index in requested_indices)
        unique_indices = tuple(sorted(set(integer_indices)))
        first_index = unique_indices[0]
        last_index = unique_indices[-1]
        span = last_index - first_index + 1
        if span / len(unique_indices) > _MAX_HDF5_SLAB_AMPLIFICATION:
            return None

        slab_indexers = list(first.indexers)
        slab_indexers[varying_axis] = slice(first_index, last_index + 1)
        slab_selection = tuple(slab_indexers)
        array = self._read_selection(
            first,
            indexers=slab_selection,
            value_id=first_request.value_id,
        )
        assert first.axes is not None
        slab_axes = tuple(
            _HDF5_BATCH_AXIS if axis_index == varying_axis else axis
            for axis_index, (axis, indexer) in enumerate(
                zip(first.axes, slab_selection, strict=True)
            )
            if isinstance(indexer, slice)
        )
        slab = self._canonical_tensor(
            array,
            slab_axes,
            first,
            batched=True,
            scalar_type=first_request.scalar_type,
        )
        offsets = tuple(index - first_index for index in integer_indices)
        if offsets == tuple(range(span)):
            batch = slab
        else:
            batch = slab.index_select(0, torch.tensor(offsets, dtype=torch.long))
        return PropagationBatchMaterialization(
            tensor=batch,
            mode="hdf5_dense_slab",
        )

    def materialize(self, request: PropagationMaterializationRequest) -> torch.Tensor:
        selection = self._resolve_selection(request)
        array = self._read_selection(
            selection,
            indexers=selection.indexers,
            value_id=request.value_id,
        )
        return self._canonical_tensor(
            array,
            selection.remaining_axes,
            selection,
            batched=False,
            scalar_type=request.scalar_type,
        )


@dataclass(frozen=True)
class PrimitiveCacheEntryPlan:
    key: MaterializationCacheKey
    size_bytes: int
    uses: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("primitive cache entry key must be non-empty")
        if self.size_bytes <= 0:
            raise ValueError("primitive cache entry size must be positive")
        if self.uses <= 1:
            raise ValueError("primitive cache entries require more than one use")


@dataclass(frozen=True)
class PrimitiveCachePlan:
    """Deterministic bounded admission plan for repeated primitive tensors."""

    max_bytes: int
    entries: tuple[PrimitiveCacheEntryPlan, ...]
    request_count: int
    cacheable_requests: int
    repeated_keys: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        for field_name in (
            "max_bytes",
            "request_count",
            "cacheable_requests",
            "repeated_keys",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        keys = tuple(entry.key for entry in self.entries)
        if len(keys) != len(set(keys)):
            raise ValueError("primitive cache plan contains duplicate keys")
        if self.planned_bytes > self.max_bytes:
            raise ValueError("primitive cache plan exceeds its byte budget")
        if len(self.entries) > self.repeated_keys:
            raise ValueError("primitive cache plan admits more keys than are reusable")

    @property
    def planned_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def reason(self) -> str:
        """Explain the deterministic cache-admission outcome."""

        if self.max_bytes == 0:
            return "disabled because no safe cache headroom remains"
        if self.cacheable_requests == 0:
            return "no provider supplied a stable cache identity"
        if self.repeated_keys == 0:
            return "no repeated exact primitive requests"
        if not self.entries:
            return "repeated primitive tensors exceed the cache byte budget"
        if len(self.entries) < self.repeated_keys:
            return "partially admitted under the exact cache byte budget"
        return "all repeated exact primitive requests fit the cache byte budget"


@dataclass(frozen=True)
class PrimitiveTensorCacheStatistics:
    lookups: int
    hits: int
    misses: int
    stores: int
    allocation_failures: int
    resident_entries: int
    resident_bytes: int
    peak_resident_bytes: int
    max_bytes: int


class PrimitiveTensorCache:
    """Own immutable repeated primitive tensors on one resident device."""

    def __init__(
        self,
        plan: PrimitiveCachePlan,
        *,
        device: str | torch.device,
    ) -> None:
        if not isinstance(plan, PrimitiveCachePlan):
            raise TypeError("primitive tensor cache requires a PrimitiveCachePlan")
        self.plan = plan
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self._admitted = {entry.key: entry for entry in plan.entries}
        self._entries: dict[MaterializationCacheKey, torch.Tensor] = {}
        self._lookups = 0
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._allocation_failures = 0
        self._resident_bytes = 0
        self._peak_resident_bytes = 0
        self._disabled = False

    def admits(self, key: MaterializationCacheKey | None) -> bool:
        return not self._disabled and key is not None and key in self._admitted

    def contains(self, key: MaterializationCacheKey | None) -> bool:
        """Return cache residency without changing lookup statistics."""

        return self.admits(key) and key in self._entries

    def lookup(self, key: MaterializationCacheKey) -> torch.Tensor | None:
        if not self.admits(key):
            return None
        self._lookups += 1
        value = self._entries.get(key)
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        return value

    @staticmethod
    def _is_allocation_failure(exc: BaseException) -> bool:
        return isinstance(exc, MemoryError) or (
            isinstance(exc, RuntimeError)
            and "out of memory" in str(exc).lower()
        )

    def store(
        self,
        key: MaterializationCacheKey,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not self.admits(key):
            return tensor
        existing = self._entries.get(key)
        if existing is not None:
            return existing
        entry = self._admitted[key]
        runtime_bytes = tensor.numel() * tensor.element_size()
        if runtime_bytes != entry.size_bytes:
            raise ValueError(
                "materialized primitive size disagrees with its cache admission plan"
            )
        try:
            stored = tensor.detach().to(device=self.device, copy=True)
        except (MemoryError, RuntimeError) as exc:
            if not self._is_allocation_failure(exc):
                raise
            # Caching is optional.  Drop every cache-owned allocation and
            # permanently bypass this configuration after an allocation race.
            self._allocation_failures += 1
            self.clear()
            self._disabled = True
            return tensor
        self._entries[key] = stored
        self._stores += 1
        self._resident_bytes += runtime_bytes
        self._peak_resident_bytes = max(
            self._peak_resident_bytes,
            self._resident_bytes,
        )
        if self._resident_bytes > self.plan.max_bytes:
            raise RuntimeError("primitive tensor cache exceeded its planned byte budget")
        return stored

    def clear(self) -> None:
        self._entries.clear()
        self._resident_bytes = 0

    @property
    def statistics(self) -> PrimitiveTensorCacheStatistics:
        return PrimitiveTensorCacheStatistics(
            lookups=self._lookups,
            hits=self._hits,
            misses=self._misses,
            stores=self._stores,
            allocation_failures=self._allocation_failures,
            resident_entries=len(self._entries),
            resident_bytes=self._resident_bytes,
            peak_resident_bytes=self._peak_resident_bytes,
            max_bytes=self.plan.max_bytes,
        )


@dataclass(frozen=True)
class MaterializedPrimitiveValue:
    """One validated primitive tensor before writing to its physical buffer."""

    value_id: str
    resource_id: str
    buffer_id: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class MaterializedPrimitiveBatch:
    """One validated primitive value with a private leading batch axis."""

    value_id: str
    resource_id: str
    buffer_id: str
    tensor: torch.Tensor
    mode: str


class PrimitiveMaterializer:
    """Dispatch and validate one primitive materialization request at a time.

    A configuration-scoped cache may be injected by the runtime session; it
    owns tensors separately from the physical buffer pool and admits only exact
    keys chosen by a bounded plan.  The resident runner may also install one
    batch of validated, asynchronously loaded HDF5 propagation tensors.  Those
    values are consumed exactly once and never change compiler/resource
    identity.
    """

    def __init__(
        self,
        binding_plan: ResourceBindingPlan,
        *,
        kernel_providers: Mapping[str, KernelMaterializationProvider] | None = None,
        source_handlers: Mapping[str, PropagationSourceHandler] | None = None,
        tensor_cache: PrimitiveTensorCache | None = None,
        kernel_batch_staging_limit_bytes: int | None = None,
    ) -> None:
        binding_plan.validate()
        self.binding_plan = binding_plan
        self.kernel_providers = dict(kernel_providers or {})
        handlers: dict[str, PropagationSourceHandler] = {
            "hdf5": HDF5TensorSourceHandler(),
        }
        handlers.update(source_handlers or {})
        self.source_handlers = handlers
        if tensor_cache is not None and not isinstance(
            tensor_cache,
            PrimitiveTensorCache,
        ):
            raise TypeError("tensor_cache must be a PrimitiveTensorCache or None")
        self.tensor_cache = tensor_cache
        if kernel_batch_staging_limit_bytes is not None and (
            not isinstance(kernel_batch_staging_limit_bytes, int)
            or isinstance(kernel_batch_staging_limit_bytes, bool)
            or kernel_batch_staging_limit_bytes < 0
        ):
            raise ValueError(
                "kernel_batch_staging_limit_bytes must be a non-negative "
                "integer or None"
            )
        self.kernel_batch_staging_limit_bytes = kernel_batch_staging_limit_bytes
        self._prefetched: dict[str, torch.Tensor] = {}

    def with_kernel_batch_staging_limit(
        self,
        maximum_staging_bytes: int | None,
    ) -> "PrimitiveMaterializer":
        """Reuse exact bindings and providers with one kernel staging limit."""

        return PrimitiveMaterializer(
            self.binding_plan,
            kernel_providers=self.kernel_providers,
            source_handlers=self.source_handlers,
            tensor_cache=self.tensor_cache,
            kernel_batch_staging_limit_bytes=maximum_staging_bytes,
        )

    def _validate_tensor(
        self,
        request: MaterializationRequest,
        tensor: torch.Tensor,
    ) -> None:
        expected_dtype = _torch_dtype(request.scalar_type)
        if tuple(tensor.shape) != tuple(request.expected_shape):
            raise ValueError(
                f"Materialized primitive {request.value_id!r} has shape "
                f"{tuple(tensor.shape)}, expected {tuple(request.expected_shape)}."
            )
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"Materialized primitive {request.value_id!r} has dtype {tensor.dtype}, "
                f"expected {expected_dtype} for scalar type {request.scalar_type!r}."
            )

        physical_binding = self.binding_plan.source_plan.binding(request.value_id)
        if physical_binding.buffer_id != request.output_buffer_id:
            raise ValueError("Materialization request disagrees with physical buffer binding.")

        expected_elements = prod(request.expected_shape) if request.expected_shape else 1
        if tensor.numel() != expected_elements:
            raise ValueError("Materialized primitive element count disagrees with its shape.")
        runtime_bytes = tensor.numel() * tensor.element_size()
        if runtime_bytes != physical_binding.size_bytes:
            raise ValueError(
                f"Runtime tensor for {request.value_id!r} occupies {runtime_bytes} bytes, "
                f"but static/physical planning expects {physical_binding.size_bytes}. "
                "The scalar_type/element_bytes runtime contract is inconsistent."
            )
        buffer = self.binding_plan.source_plan.buffer(request.output_buffer_id)
        if runtime_bytes > buffer.capacity_bytes:
            raise ValueError(
                f"Runtime tensor for {request.value_id!r} does not fit planned buffer "
                f"{buffer.buffer_id!r}."
            )

    def _kernel_provider(
        self,
        request: KernelMaterializationRequest,
    ) -> KernelMaterializationProvider:
        try:
            provider = self.kernel_providers[request.provider_id]
        except KeyError as exc:
            raise ValueError(
                f"No kernel materialization provider is registered for "
                f"provider id {request.provider_id!r}."
            ) from exc
        if not isinstance(provider, KernelMaterializationProvider):
            raise TypeError(
                f"Kernel provider {request.provider_id!r} must implement "
                "KernelMaterializationProvider."
            )
        return provider

    def _propagation_handler(
        self,
        request: PropagationMaterializationRequest,
    ) -> PropagationSourceHandler:
        try:
            handler = self.source_handlers[request.source.kind]
        except KeyError as exc:
            raise ValueError(
                f"No propagation source handler is registered for source kind "
                f"{request.source.kind!r}."
            ) from exc
        if not isinstance(handler, PropagationSourceHandler):
            raise TypeError(
                f"Propagation source handler for {request.source.kind!r} must "
                "implement PropagationSourceHandler."
            )
        return handler

    def materialization_cache_key(
        self,
        value_id: str,
    ) -> MaterializationCacheKey | None:
        """Return the exact numerical identity for a cacheable primitive.

        Logical value ids and buffer ids are deliberately absent: two scheduled
        occurrences may reuse one numerical tensor while retaining distinct
        compiler and physical identities.  Providers/handlers must explicitly
        opt in with a complete state identity; otherwise this returns ``None``.
        """

        try:
            request = self.binding_plan.materialization(value_id)
        except KeyError as exc:
            raise KeyError(
                f"No primitive materialization request exists for {value_id!r}."
            ) from exc

        common = (
            request.resource_id,
            request.expected_shape,
            request.scalar_type,
        )
        if isinstance(request, SpinMaterializationRequest):
            frozen_matrix = _freeze_cache_value(request.matrix)
            if frozen_matrix is _UNCACHEABLE:  # pragma: no cover - validated tuples
                return None
            return ("spin-v1", *common, request.tensor_kind, frozen_matrix)

        if isinstance(request, KernelMaterializationRequest):
            provider = self._kernel_provider(request)
            identity = provider.cache_identity()
            frozen_provider = _freeze_cache_value(identity)
            if identity is None or frozen_provider is _UNCACHEABLE:
                return None
            return (
                "kernel-v1",
                *common,
                frozen_provider,
                request.operator_times,
            )

        if isinstance(request, PropagationMaterializationRequest):
            handler = self._propagation_handler(request)
            identity = handler.cache_identity()
            frozen_handler = _freeze_cache_value(identity)
            frozen_source = _freeze_cache_value(
                (
                    request.source.source_id,
                    request.source.kind,
                    request.source.location,
                    request.source.dataset,
                    request.source.metadata,
                )
            )
            if (
                identity is None
                or frozen_handler is _UNCACHEABLE
                or frozen_source is _UNCACHEABLE
            ):
                return None
            return (
                "propagation-v1",
                *common,
                frozen_handler,
                frozen_source,
                request.source_time,
                request.insertion_times,
                request.sink_time,
            )

        return None  # pragma: no cover - binding validation closes the union

    def prefetchable_hdf5_propagation_values(self) -> tuple[str, ...]:
        """Return exact built-in HDF5 propagation values eligible for prefetch.

        Custom source handlers and generalized propagation insertions remain
        synchronous unless they explicitly gain a future validated prefetch
        contract. Values admitted to the exact primitive cache are also
        excluded: avoiding repeated I/O has priority over staging a second copy
        of the same resident tensor.
        """

        eligible: list[str] = []
        for value_id, request in self.binding_plan.materializations.items():
            if not isinstance(request, PropagationMaterializationRequest):
                continue
            if request.insertion_operator_ids or request.insertion_times:
                continue
            handler = self._propagation_handler(request)
            if type(handler) is not HDF5TensorSourceHandler:
                continue
            cache_key = self.materialization_cache_key(value_id)
            if self.tensor_cache is not None and self.tensor_cache.admits(cache_key):
                continue
            eligible.append(value_id)
        return tuple(eligible)

    def materialize_hdf5_propagation_for_prefetch(
        self,
        value_id: str,
    ) -> torch.Tensor:
        """Load and validate one eligible propagation tensor without installing it."""

        if value_id not in self.prefetchable_hdf5_propagation_values():
            raise ValueError(
                f"Primitive value {value_id!r} is not eligible for built-in HDF5 "
                "propagation prefetch."
            )
        request = self.binding_plan.materialization(value_id)
        assert isinstance(request, PropagationMaterializationRequest)
        handler = self._propagation_handler(request)
        tensor = _as_tensor(handler.materialize(request))
        self._validate_tensor(request, tensor)
        return tensor

    def install_prefetched_hdf5_propagation(
        self,
        value_id: str,
        tensor: torch.Tensor,
    ) -> None:
        """Install one validated tensor for single-use consumption by ``materialize``."""

        try:
            request = self.binding_plan.materialization(value_id)
        except KeyError as exc:
            raise KeyError(
                f"No primitive materialization request exists for {value_id!r}."
            ) from exc
        if not isinstance(request, PropagationMaterializationRequest):
            raise ValueError(
                f"Primitive value {value_id!r} is not a propagation request."
            )
        handler = self._propagation_handler(request)
        if type(handler) is not HDF5TensorSourceHandler:
            raise ValueError(
                f"Primitive value {value_id!r} does not use the built-in HDF5 "
                "propagation handler."
            )
        if value_id in self._prefetched:
            raise RuntimeError(
                f"Primitive value {value_id!r} already has an installed prefetch."
            )
        self._validate_tensor(request, tensor)
        self._prefetched[value_id] = tensor

    @property
    def pending_prefetched_values(self) -> tuple[str, ...]:
        """Value ids installed by the prefetch session but not yet consumed."""

        return tuple(self._prefetched)

    def materialize(self, value_id: str) -> MaterializedPrimitiveValue:
        try:
            request = self.binding_plan.materialization(value_id)
        except KeyError as exc:
            raise KeyError(
                f"No primitive materialization request exists for {value_id!r}."
            ) from exc

        cache_key = self.materialization_cache_key(value_id)
        if self.tensor_cache is not None and self.tensor_cache.admits(cache_key):
            assert cache_key is not None
            cached = self.tensor_cache.lookup(cache_key)
            if cached is not None:
                self._validate_tensor(request, cached)
                return MaterializedPrimitiveValue(
                    value_id=request.value_id,
                    resource_id=request.resource_id,
                    buffer_id=request.output_buffer_id,
                    tensor=cached,
                )

        prefetched = self._prefetched.pop(value_id, None)
        if prefetched is not None:
            self._validate_tensor(request, prefetched)
            if (
                self.tensor_cache is not None
                and cache_key is not None
                and self.tensor_cache.admits(cache_key)
            ):
                prefetched = self.tensor_cache.store(cache_key, prefetched)
            return MaterializedPrimitiveValue(
                value_id=request.value_id,
                resource_id=request.resource_id,
                buffer_id=request.output_buffer_id,
                tensor=prefetched,
            )

        if isinstance(request, KernelMaterializationRequest):
            provider = self._kernel_provider(request)
            with kernel_building():
                tensor = _as_tensor(provider.materialize(request))

        elif isinstance(request, SpinMaterializationRequest):
            tensor = torch.tensor(
                request.matrix,
                dtype=_torch_dtype(request.scalar_type),
            )

        elif isinstance(request, PropagationMaterializationRequest):
            handler = self._propagation_handler(request)
            tensor = _as_tensor(handler.materialize(request))

        else:  # pragma: no cover - ResourceBindingPlan validates the closed union
            raise TypeError(f"Unsupported materialization request type {type(request)!r}.")

        self._validate_tensor(request, tensor)
        if (
            self.tensor_cache is not None
            and cache_key is not None
            and self.tensor_cache.admits(cache_key)
        ):
            tensor = self.tensor_cache.store(cache_key, tensor)
        return MaterializedPrimitiveValue(
            value_id=request.value_id,
            resource_id=request.resource_id,
            buffer_id=request.output_buffer_id,
            tensor=tensor,
        )


def materialize_primitive_batch(
    materializers: Sequence[PrimitiveMaterializer],
    value_id: str,
) -> MaterializedPrimitiveBatch | None:
    """Use an opt-in provider batch or preserve independent materialization.

    Batching requires one physical primitive, one request type, and the exact
    same provider/handler instance across all work units. Exact caches cooperate
    only when every request is either absent or resident; mixed residency and
    installed-prefetch state retain the established independent path.
    """

    values = tuple(materializers)
    if len(values) <= 1:
        return None
    if any(not isinstance(item, PrimitiveMaterializer) for item in values):
        raise TypeError("batched primitive materialization requires materializers")

    try:
        requests = tuple(
            materializer.binding_plan.materialization(value_id)
            for materializer in values
        )
    except KeyError as exc:
        raise KeyError(
            f"No primitive materialization request exists for {value_id!r}."
        ) from exc

    first = requests[0]
    if any(type(request) is not type(first) for request in requests[1:]):
        return None
    if any(
        request.value_id != first.value_id
        or request.resource_id != first.resource_id
        or request.output_buffer_id != first.output_buffer_id
        or request.expected_shape != first.expected_shape
        or request.scalar_type != first.scalar_type
        for request in requests[1:]
    ):
        raise ValueError("batched materializers disagree on the physical primitive")
    if any(value_id in materializer._prefetched for materializer in values):
        return None
    cache_keys = tuple(
        materializer.materialization_cache_key(value_id)
        for materializer in values
    )
    any_cache_admitted = any(
        materializer.tensor_cache is not None
        and materializer.tensor_cache.admits(cache_key)
        for materializer, cache_key in zip(values, cache_keys, strict=True)
    )
    tensor_cache = _shared_admitted_primitive_cache(values, cache_keys)
    if any_cache_admitted and tensor_cache is None:
        return None
    if tensor_cache is not None:
        resident = tuple(tensor_cache.contains(key) for key in cache_keys)
        if any(resident) and not all(resident):
            return None
        if all(resident):
            cached = tuple(
                tensor_cache.lookup(key)
                for key in cache_keys
                if key is not None
            )
            if len(cached) != len(values) or any(tensor is None for tensor in cached):
                raise RuntimeError(
                    "primitive cache residency changed during batch materialization"
                )
            tensors = tuple(tensor for tensor in cached if tensor is not None)
            for materializer, request, tensor in zip(
                values,
                requests,
                tensors,
                strict=True,
            ):
                materializer._validate_tensor(request, tensor)
            return MaterializedPrimitiveBatch(
                value_id=first.value_id,
                resource_id=first.resource_id,
                buffer_id=first.output_buffer_id,
                tensor=torch.stack(tensors, dim=0),
                mode="primitive_cache_stack",
            )

    if isinstance(first, KernelMaterializationRequest):
        kernel_requests = tuple(
            request
            for request in requests
            if isinstance(request, KernelMaterializationRequest)
        )
        if len(kernel_requests) != len(requests):  # pragma: no cover - type guard
            return None
        if any(
            request.provider_id != first.provider_id
            or request.recipe != first.recipe
            or request.operator_ids != first.operator_ids
            for request in kernel_requests[1:]
        ):
            raise ValueError(
                "batched kernel materializers disagree on their provider recipe"
            )
        providers = tuple(
            materializer._kernel_provider(request)
            for materializer, request in zip(
                values,
                kernel_requests,
                strict=True,
            )
        )
        provider = providers[0]
        if any(candidate is not provider for candidate in providers[1:]):
            return None
        limits = {
            materializer.kernel_batch_staging_limit_bytes
            for materializer in values
        }
        if len(limits) != 1:
            raise ValueError(
                "batched kernel materializers disagree on their staging limit"
            )
        with kernel_building():
            result = provider.materialize_batch_with_limit(
                kernel_requests,
                maximum_staging_bytes=limits.pop(),
            )
    elif isinstance(first, PropagationMaterializationRequest):
        propagation_requests = tuple(
            request
            for request in requests
            if isinstance(request, PropagationMaterializationRequest)
        )
        if len(propagation_requests) != len(requests):  # pragma: no cover - type guard
            return None
        handlers = tuple(
            materializer._propagation_handler(request)
            for materializer, request in zip(
                values,
                propagation_requests,
                strict=True,
            )
        )
        handler = handlers[0]
        if any(candidate is not handler for candidate in handlers[1:]):
            return None
        result = handler.materialize_batch(propagation_requests)
    else:
        return None
    if result is None:
        return None
    expected_shape = (len(values), *first.expected_shape)
    if tuple(result.tensor.shape) != expected_shape:
        raise ValueError(
            f"Batched primitive provider returned shape {tuple(result.tensor.shape)}, "
            f"expected {expected_shape}."
        )
    for materializer, request, tensor in zip(
        values,
        requests,
        result.tensor.unbind(0),
        strict=True,
    ):
        materializer._validate_tensor(request, tensor)
    if tensor_cache is not None:
        for key, tensor in zip(cache_keys, result.tensor.unbind(0), strict=True):
            if key is None:  # pragma: no cover - shared cache rejects None keys
                raise RuntimeError("admitted primitive cache key unexpectedly absent")
            tensor_cache.store(key, tensor)
    return MaterializedPrimitiveBatch(
        value_id=first.value_id,
        resource_id=first.resource_id,
        buffer_id=first.output_buffer_id,
        tensor=result.tensor,
        mode=result.mode,
    )


def shared_materialization_cache_key(
    materializers: Sequence[PrimitiveMaterializer],
    value_id: str,
) -> MaterializationCacheKey | None:
    """Return a proven common numerical identity for one primitive value.

    ``None`` means that at least one provider did not opt into exact identity
    semantics, or that the concrete requests differ. Several uncacheable
    ``None`` keys never imply that values are equal.
    """

    values = tuple(materializers)
    if not values:
        raise ValueError("shared materialization identity requires materializers")
    if any(not isinstance(item, PrimitiveMaterializer) for item in values):
        raise TypeError(
            "shared materialization identity requires PrimitiveMaterializer entries"
        )

    first_key = values[0].materialization_cache_key(value_id)
    if first_key is None:
        return None
    if any(
        materializer.materialization_cache_key(value_id) != first_key
        for materializer in values[1:]
    ):
        return None
    return first_key


def _shared_admitted_primitive_cache(
    materializers: Sequence[PrimitiveMaterializer],
    cache_keys: Sequence[MaterializationCacheKey | None],
) -> PrimitiveTensorCache | None:
    values = tuple(materializers)
    keys = tuple(cache_keys)
    cache = values[0].tensor_cache
    if cache is None or any(value.tensor_cache is not cache for value in values[1:]):
        return None
    if not all(cache.admits(key) for key in keys):
        return None
    return cache


def plan_primitive_tensor_cache(
    materializers: Sequence[PrimitiveMaterializer],
    *,
    max_bytes: int,
) -> PrimitiveCachePlan:
    """Admit repeated exact primitive keys under a deterministic byte limit.

    The complete configuration request sequence is known before numerical
    execution.  Higher-reuse keys are admitted first, with smaller tensors as a
    deterministic packing tie-break.  Uncacheable custom providers simply do
    not appear in the candidate set.
    """

    values = tuple(materializers)
    if not values:
        raise ValueError("primitive cache planning requires at least one materializer")
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 0
    ):
        raise ValueError("primitive cache max_bytes must be a non-negative integer")
    if any(not isinstance(item, PrimitiveMaterializer) for item in values):
        raise TypeError("primitive cache planning requires PrimitiveMaterializer entries")
    source_plan = values[0].binding_plan.source_plan
    if any(item.binding_plan.source_plan is not source_plan for item in values[1:]):
        raise ValueError("primitive cache materializers must bind the same physical plan")

    counts: dict[MaterializationCacheKey, int] = {}
    sizes: dict[MaterializationCacheKey, int] = {}
    request_count = 0
    cacheable_requests = 0
    for materializer in values:
        for value_id in materializer.binding_plan.materializations:
            request_count += 1
            key = materializer.materialization_cache_key(value_id)
            if key is None:
                continue
            cacheable_requests += 1
            size = source_plan.binding(value_id).size_bytes
            if key in sizes and sizes[key] != size:
                raise ValueError(
                    "equal primitive cache keys have inconsistent planned sizes"
                )
            sizes[key] = size
            counts[key] = counts.get(key, 0) + 1

    candidates = [
        PrimitiveCacheEntryPlan(key=key, size_bytes=sizes[key], uses=uses)
        for key, uses in counts.items()
        if uses > 1
    ]
    candidates.sort(
        key=lambda entry: (
            -entry.uses,
            entry.size_bytes,
            repr(entry.key),
        )
    )
    admitted: list[PrimitiveCacheEntryPlan] = []
    used_bytes = 0
    for entry in candidates:
        if used_bytes + entry.size_bytes <= max_bytes:
            admitted.append(entry)
            used_bytes += entry.size_bytes

    return PrimitiveCachePlan(
        max_bytes=max_bytes,
        entries=tuple(admitted),
        request_count=request_count,
        cacheable_requests=cacheable_requests,
        repeated_keys=len(candidates),
    )


__all__ = [
    "MaterializationCacheKey",
    "HDF5FilePoolStatistics",
    "HDF5FilePool",
    "KernelMaterializationProvider",
    "KernelBatchMaterialization",
    "PropagationSourceHandler",
    "PropagationBatchMaterialization",
    "CompiledKernelMaterializationProvider",
    "InMemoryTensorSourceHandler",
    "HDF5TensorSourceHandler",
    "PrimitiveCacheEntryPlan",
    "PrimitiveCachePlan",
    "PrimitiveTensorCacheStatistics",
    "PrimitiveTensorCache",
    "MaterializedPrimitiveValue",
    "MaterializedPrimitiveBatch",
    "PrimitiveMaterializer",
    "materialize_primitive_batch",
    "shared_materialization_cache_key",
    "plan_primitive_tensor_cache",
]
