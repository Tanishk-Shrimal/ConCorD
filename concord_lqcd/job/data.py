"""User-facing data/provider translation for compiled ConCorD jobs.

This module translates one validated :class:`JobSpec`, one loaded/compiled
:class:`ConcordPlan`, one gauge configuration, and one concrete operator-time
context into the exact runtime resources consumed by the frozen resource-binding
and primitive-materialization layers.

The application-layer convenience does not weaken compiler identity:

    user storage catalog -> exact Stage-5 resource id -> ExternalTensorSource

Ordinary perambulators may be organized by flavor in ``job.yaml`` because the
exact compiled :class:`PropagationRecipe` is inspected before selecting the
configured file.  The resulting mapping is still keyed only by exact compiler
resource ids.  No flavor fallback is performed.

The built-in V1 eigenvector provider supports HDF5 datasets with semantic axes
``x``, ``y``, ``z``, ``color``, and ``mode`` plus either a ``time`` axis or a
``{time}`` path/dataset placeholder.  It normalizes one occurrence to
``[1, volume, color, mode]`` for the existing DistillationTensorProvider.
Momentum phases are generated from those semantic coordinates using the same
``exp(+i p.x)`` convention as the existing backend I/O helper.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import lru_cache, wraps
import hashlib
import importlib
import inspect
import math
import platform
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence

import h5py
import numpy as np
import opt_einsum as oe
import torch

from concord_lqcd.backend.runtime.context import ExecutionContext
from concord_lqcd.backend.runtime.kernels import evaluate_kernel_program
from concord_lqcd.backend.runtime.materialization import (
    CompiledKernelMaterializationProvider,
    HDF5FilePool,
    HDF5TensorSourceHandler,
    KernelBatchMaterialization,
    KernelMaterializationProvider,
    PrimitiveMaterializer,
    PrimitiveTensorCache,
    PropagationSourceHandler,
)
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider
from concord_lqcd.backend.runtime.hardware import HardwareProfile
from concord_lqcd.backend.runtime.resource_planning import (
    DeviceExecutionPlan,
    HardwareBudgetPolicy,
    estimate_execution_memory,
    resolve_runtime_device,
)
from concord_lqcd.backend.runtime.resource_binding import (
    ExternalTensorSource,
    KernelMaterializationRequest,
    KernelResourceBinding,
    OperatorTimeContext,
    PropagationResourceBinding,
    ResourceBindingPlan,
    build_resource_binding_plan,
    prepare_resource_binding_template,
)
from concord_lqcd.compiler.distillation_lowering import (
    KernelRecipe,
    ProjectionContribution,
    emit_backend_kernel_program,
    identify_kernel_resource,
    kernel_resource_signature,
)
from concord_lqcd.compiler.physics_ir import ColorTensorRecord, OperationKind, OperationRecord
from concord_lqcd.compiler.propagation_lowering import PropagationRecipe

from .laph_iomap import LaphGaugeIOMapError, LaphGaugeIOMapReader
from .openqcd_gauge import OpenQCDGaugeError, OpenQCDGaugeReader
from ._eigenvector_cache import RawEigenvectorCache, RawEigenvectorCacheStatistics
from .plan import ConcordPlan, validate_plan_for_job
from .operator_sources import OperatorSourceSession
from .kernel_store import (
    HDF5KernelStore,
    KernelCacheStatistics,
    KernelStoreError,
    _antisym3_workspace_bytes,
)
from .schema import (
    CustomEigenvectorSpec,
    ExpandedTimeContext,
    HDF5EigenvectorSpec,
    GaugeLinkSpec,
    JobSpec,
    OpenQCDGaugeSpec,
)


_KERNEL_PROVIDER_ID = "distillation"
_COMPLEX_DTYPES: dict[str, torch.dtype] = {
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
_COMPLEX_ELEMENT_BYTES = {
    "complex64": 8,
    "complex128": 16,
}
_REAL_DTYPES = {
    "complex64": torch.float32,
    "complex128": torch.float64,
}
_HDF5_EIGENVECTOR_BATCH_AXIS = "__concord_hdf5_eigenvector_batch__"
_MAX_HDF5_EIGENVECTOR_SLAB_AMPLIFICATION = 2.0
_KERNEL_CACHE_ALGORITHM_VERSION = 2


@dataclass(frozen=True)
class _BaryonCacheCanonicalization:
    canonical_resource_id: str
    canonical_recipe_signature: str
    color_sign: int
    needs_conj: bool


@dataclass(frozen=True)
class _KernelBatchCacheState:
    entries: tuple[tuple[str, str, tuple[str, ...]], ...]
    present: tuple[bool, ...]


def _canonical_baryon_cache_transform(
    recipe: KernelRecipe,
) -> _BaryonCacheCanonicalization | None:
    """Canonicalize an untransformed epsilon kernel under a declared common basis."""

    if (
        len(recipe.projected_fields) != 3
        or len(recipe.position_labels) != 1
        or recipe.gauge_tensors
        or len(recipe.color_tensors) != 1
        or recipe.color_tensors[0].kind != "epsilon"
        or recipe.open_ports != ("mode_0", "mode_1", "mode_2")
    ):
        return None
    position = recipe.position_labels[0]
    contributions = {projected.contribution for projected in recipe.projected_fields}
    if len(contributions) != 1:
        return None
    contribution = next(iter(contributions))
    if contribution not in (
        ProjectionContribution.EIGENVECTOR,
        ProjectionContribution.EIGENVECTOR_DAGGER,
    ):
        return None
    if any(
        projected.record.pos_label != position or projected.operations
        for projected in recipe.projected_fields
    ):
        return None
    colors = tuple(projected.record.color_label for projected in recipe.projected_fields)
    epsilon_indices = recipe.color_tensors[0].indices
    if len(set(colors)) != 3 or len(epsilon_indices) != 3 or set(epsilon_indices) != set(colors):
        return None
    permutation = tuple(colors.index(color) for color in epsilon_indices)
    color_sign = 1 if permutation in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1
    needs_conj = contribution is ProjectionContribution.EIGENVECTOR_DAGGER
    momentum = recipe.momenta.get(position, (0, 0, 0))
    canonical = replace(
        recipe,
        projected_fields=tuple(
            replace(
                projected,
                contribution=ProjectionContribution.EIGENVECTOR,
                record=replace(projected.record, flavor="__concord_shared_laph__"),
            )
            for projected in recipe.projected_fields
        ),
        color_tensors=(ColorTensorRecord("epsilon", colors, recipe.color_tensors[0].operator_id),),
        momenta={position: tuple(-component for component in momentum) if needs_conj else momentum},
    )
    return _BaryonCacheCanonicalization(
        identify_kernel_resource(canonical),
        kernel_resource_signature(canonical),
        color_sign,
        needs_conj,
    )


@lru_cache(maxsize=4096)
def _einsum_workspace_elements(
    equation: str,
    shapes: tuple[tuple[int, ...], ...],
) -> int:
    """Return the contraction path's largest intermediate shape size."""

    _, path_info = oe.contract_path(equation, *shapes, shapes=True)
    return int(path_info.largest_intermediate)


class JobDataError(ValueError):
    """Raised when job-level storage configuration cannot satisfy a compiled plan."""


def _torch_complex_dtype(scalar_type: str) -> torch.dtype:
    try:
        return _COMPLEX_DTYPES[scalar_type]
    except KeyError as exc:
        raise JobDataError(
            f"built-in job data providers require complex64/complex128, got {scalar_type!r}"
        ) from exc


def _format_template(template: str, values: Mapping[str, object], *, role: str) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise JobDataError(
            f"{role} requires unavailable placeholder {exc.args[0]!r}"
        ) from exc
    except (IndexError, ValueError) as exc:
        raise JobDataError(f"invalid template for {role}: {template!r}") from exc


class _PreserveMissing(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _partial_format_template(
    template: str,
    values: Mapping[str, object],
    *,
    role: str,
) -> str:
    try:
        return template.format_map(_PreserveMissing(values))
    except (IndexError, ValueError) as exc:
        raise JobDataError(f"invalid template for {role}: {template!r}") from exc


def _decode_hdf5_complex(
    array: np.ndarray,
    axes: list[str],
) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(array)
    names = arr.dtype.names
    if names is not None:
        re_key = "re" if "re" in names else ("real" if "real" in names else None)
        im_key = "im" if "im" in names else ("imag" if "imag" in names else None)
        if re_key is None or im_key is None:
            if len(names) < 2:
                raise JobDataError(
                    "compound eigenvector dtype must contain two real/imaginary fields"
                )
            re_key, im_key = names[0], names[1]
        arr = np.asarray(arr[re_key]) + 1j * np.asarray(arr[im_key])

    if "complex_pair" in axes:
        pair_axis = axes.index("complex_pair")
        if arr.shape[pair_axis] != 2:
            raise JobDataError("eigenvector complex_pair axis must have extent two")
        arr = np.take(arr, 0, axis=pair_axis) + 1j * np.take(arr, 1, axis=pair_axis)
        axes = list(axes)
        axes.pop(pair_axis)

    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128)
    return arr, axes


def _momentum_phase(
    momentum: tuple[int, int, int],
    spatial_extent: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return ``exp(+i 2pi p.x/L)`` in z-y-x flattening order.

    ``spatial_extent`` is interpreted as ``(Lx, Ly, Lz)``.  Eigenvectors are
    normalized below to z-y-x spatial order before flattening, so x is the
    fastest coordinate in the resulting volume index.
    """

    lx, ly, lz = spatial_extent
    px, py, pz = momentum
    x = torch.arange(lx, dtype=torch.float64, device=device)
    y = torch.arange(ly, dtype=torch.float64, device=device)
    z = torch.arange(lz, dtype=torch.float64, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
    arg = 2.0 * math.pi * (
        px * grid_x / lx + py * grid_y / ly + pz * grid_z / lz
    )
    phase = torch.exp(1j * arg.reshape(-1))
    return phase.to(dtype=dtype)


def _contains_time_placeholder(spec: HDF5EigenvectorSpec) -> bool:
    return "{time}" in spec.path or "{time}" in spec.dataset


def _operation_key(
    operations: Sequence[OperationRecord],
) -> tuple[tuple[object, ...], ...]:
    result: list[tuple[object, ...]] = []
    for operation in operations:
        if operation.kind is OperationKind.DERIVATIVE:
            result.append((operation.kind.value, operation.parameters["direction"]))
        else:
            result.append((operation.kind.value, tuple(operation.parameters["path"])))
    return tuple(result)


def _operation_payload_key(
    field_attributes: Mapping[str, Any] | None,
    derivatives: Sequence[Any] = (),
) -> tuple[tuple[object, ...], ...]:
    attributes = {} if field_attributes is None else dict(field_attributes)
    operations = attributes.get("operations")
    if operations is not None:
        result: list[tuple[object, ...]] = []
        for operation in operations:
            if not isinstance(operation, Mapping):
                raise JobDataError("compiled field operations must be mappings")
            kind = operation.get("kind")
            if kind == OperationKind.DERIVATIVE.value:
                result.append((kind, operation.get("direction")))
            elif kind in {
                OperationKind.TRANSPORT.value,
                OperationKind.SHIFT.value,
            }:
                result.append((kind, tuple(operation.get("path", ()))))
            else:
                raise JobDataError(f"unsupported compiled field operation {kind!r}")
        return tuple(result)

    legacy = attributes.get("derivatives", attributes.get("derivative", derivatives))
    result = []
    for derivative in legacy or ():
        if isinstance(derivative, tuple):
            axis, is_backward = derivative
            direction = f"{'-' if is_backward else '+'}{axis}"
        else:
            direction = str(derivative)
            if not direction.startswith(("+", "-")):
                direction = f"+{direction}"
        result.append((OperationKind.DERIVATIVE.value, direction))
    return tuple(result)


def _spatial_axis(direction: str) -> tuple[str, int, int]:
    if not isinstance(direction, str) or direction[:1] not in {"+", "-"}:
        raise JobDataError(f"invalid canonical lattice direction {direction!r}")
    axis_name = direction[1:]
    if axis_name == "t":
        raise NotImplementedError(
            "temporal derivative/transport/shift operations are not yet supported: "
            "they require multi-timeslice field materialization and explicit "
            "operator-time semantics"
        )
    try:
        axis = {"z": 1, "y": 2, "x": 3}[axis_name]
    except KeyError as exc:
        raise JobDataError(f"invalid canonical lattice direction {direction!r}") from exc
    sign = 1 if direction[0] == "+" else -1
    return axis_name, axis, sign


def _shift_one_hop(field: torch.Tensor, hop: str) -> torch.Tensor:
    _, axis, sign = _spatial_axis(hop)
    return torch.roll(field, shifts=-sign, dims=axis)


def _shift_path(field: torch.Tensor, path: Sequence[str]) -> torch.Tensor:
    current = field
    for hop in reversed(tuple(path)):
        current = _shift_one_hop(current, hop)
    return current


def _transport_one_hop(
    field: torch.Tensor,
    hop: str,
    gauge_link: torch.Tensor,
) -> torch.Tensor:
    _, axis, sign = _spatial_axis(hop)
    shifted_field = torch.roll(field, shifts=-sign, dims=axis)
    if sign > 0:
        link = gauge_link
    else:
        link = torch.roll(gauge_link, shifts=1, dims=axis).conj().transpose(-1, -2)
    return torch.matmul(link, shifted_field)


def _transport_path(
    field: torch.Tensor,
    path: Sequence[str],
    gauge_link: Callable[[str], torch.Tensor],
) -> torch.Tensor:
    current = field
    for hop in reversed(tuple(path)):
        axis_name, _, _ = _spatial_axis(hop)
        current = _transport_one_hop(current, hop, gauge_link(axis_name))
    return current


def _apply_laph_operations(
    field: torch.Tensor,
    operations: Sequence[OperationRecord],
    gauge_link: Callable[[str], torch.Tensor],
) -> torch.Tensor:
    if field.ndim != 6 or field.shape[-2] != 3:
        raise JobDataError(
            "LapH field transformations require [batch,z,y,x,color,mode] tensors"
        )
    current = field
    for operation in operations:
        if operation.kind is OperationKind.SHIFT:
            current = _shift_path(current, operation.parameters["path"])
        elif operation.kind is OperationKind.TRANSPORT:
            current = _transport_path(
                current,
                operation.parameters["path"],
                gauge_link,
            )
        elif operation.kind is OperationKind.DERIVATIVE:
            direction = str(operation.parameters["direction"])
            axis_name, _, sign = _spatial_axis(direction)
            link = gauge_link(axis_name)
            forward = _transport_one_hop(current, f"+{axis_name}", link)
            backward = _transport_one_hop(current, f"-{axis_name}", link)
            current = 0.5 * sign * (forward - backward)
        else:  # pragma: no cover - PhysicsIR validation owns this invariant
            raise JobDataError(f"unsupported field operation {operation.kind.value!r}")
    return current


class _PreparedDistillationTensorProvider(DistillationTensorProvider):
    """Request-local provider with exactly pre-transformed LapH fields."""

    def __init__(
        self,
        fields: Mapping[tuple[tuple[object, ...], ...], torch.Tensor],
        phases: Mapping[tuple[int, int, int], torch.Tensor],
    ) -> None:
        super().__init__(V_t_snk={}, V_t_src={}, momenta_phases=phases)
        self._prepared_fields = dict(fields)

    def eigenvector(
        self,
        flavor: str | None,
        is_anti: bool = False,
        derivatives: tuple[Any, ...] | list[Any] = (),
        is_sink: bool = True,
        context=None,
        field_attributes: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        del flavor, is_sink
        key = _operation_payload_key(field_attributes, derivatives)
        try:
            field = self._prepared_fields[key]
        except KeyError as exc:  # pragma: no cover - compiler/provider invariant
            raise JobDataError(
                f"compiled field transformation {key!r} was not prepared"
            ) from exc
        if is_anti:
            field = field.conj()
        if context is not None and field.shape[0] == 1:
            field = field.expand(len(context.t_snk_list), *field.shape[1:])
        return field


class HDF5DistillationKernelProvider(KernelMaterializationProvider):
    """Built-in V1 HDF5 LapH eigenvector provider.

    The storage dataset must expose semantic spatial axes ``x/y/z``, ``color``,
    and ``mode``.  A Euclidean time is selected either by a ``time`` axis or by
    ``{time}`` in the file/dataset template.  Native complex, compound real/imag,
    and an explicit ``complex_pair`` axis are supported.

    Every flavor uses that same dataset at a given time. Inputs must contain
    orthonormal LapH modes with perambulators in the same mode coordinates;
    this provider does not perform an additional orthonormality scan.

    Spatial derivative, transport, and shift operations are realized on the
    unflattened eigenvector field. Gauge-covariant operations require gauge links
    configured by ``gauge_spec``: a native Chroma-LapH or QUDA-LapH smeared-gauge IOMap
    or a file in the openQCD binary format. The links are used as stored.
    Temporal operations are rejected until the runtime supports multi-timeslice
    spatial kernels with explicit operator-time semantics.
    """

    def __init__(
        self,
        spec: HDF5EigenvectorSpec,
        *,
        configuration: int,
        spatial_extent: tuple[int, int, int],
        temporal_extent: int | None = None,
        mode_size: int,
        device: str | torch.device,
        gauge_spec: GaugeLinkSpec | None = None,
        file_pool: HDF5FilePool | None = None,
        kernel_cache: HDF5KernelStore | None = None,
    ) -> None:
        self.spec = spec
        self.configuration = configuration
        self.spatial_extent = tuple(spatial_extent)
        self.temporal_extent = temporal_extent
        self.mode_size = mode_size
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.gauge_spec = gauge_spec
        if kernel_cache is not None and not isinstance(kernel_cache, HDF5KernelStore):
            raise TypeError("kernel_cache must be an HDF5KernelStore or None")
        self.kernel_cache = kernel_cache
        self._kernel_cache_implementation: dict[str, object] | None = None
        self._kernel_cache_backend: dict[str, object] | None = None
        if file_pool is not None and not isinstance(file_pool, HDF5FilePool):
            raise TypeError("file_pool must be an HDF5FilePool or None")
        self.file_pool = file_pool
        self._raw_eigenvectors = RawEigenvectorCache()
        self._eigenvector_file_identities: dict[str, tuple[int, ...]] = {}
        self._gauge_reader: LaphGaugeIOMapReader | OpenQCDGaugeReader | None = None
        self._delegate = CompiledKernelMaterializationProvider(
            self._tensor_provider_for_request,
            device=self.device,
        )

    @property
    def raw_eigenvector_cache_statistics(self) -> RawEigenvectorCacheStatistics:
        return self._raw_eigenvectors.statistics

    def configure_raw_eigenvector_cache(self, maximum_bytes: int) -> None:
        self._raw_eigenvectors.configure(maximum_bytes)

    def clear_raw_eigenvector_cache(self) -> None:
        self._raw_eigenvectors.clear()

    def _raw_eigenvector_key(
        self, time: int, scalar_type: str
    ) -> tuple[object, ...] | None:
        if (
            type(self) is not HDF5DistillationKernelProvider
            or not self._raw_eigenvectors.maximum_bytes
        ):
            return None
        path = Path(_format_template(
            self.spec.path,
            {"configuration": self.configuration, "time": time},
            role="data.eigenvectors.path",
        )).expanduser().resolve(strict=False)
        try:
            status = path.stat()
        except OSError as exc:
            raise JobDataError(f"cannot inspect eigenvector HDF5 file {str(path)!r}: {exc}") from exc
        identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
        previous = self._eigenvector_file_identities.setdefault(str(path), identity)
        if previous != identity:
            self.clear_raw_eigenvector_cache()
            raise JobDataError(f"eigenvector input {str(path)!r} changed during this configuration")
        dataset = _format_template(
            self.spec.dataset,
            {"configuration": self.configuration, "time": time},
            role="data.eigenvectors.dataset",
        )
        return (
            str(path), identity, dataset, tuple(self.spec.axes), self.configuration,
            time, self.spatial_extent, self.mode_size, scalar_type, str(self.device),
        )

    @property
    def kernel_cache_statistics(self) -> KernelCacheStatistics | None:
        if self.kernel_cache is None:
            return None
        return self.kernel_cache.statistics

    def cache_identity(self) -> object | None:
        if type(self) is not HDF5DistillationKernelProvider:
            return None
        return (
            "concord-hdf5-distillation",
            3,
            self.spec.path,
            self.spec.dataset,
            tuple(self.spec.axes),
            self.configuration,
            self.spatial_extent,
            self.mode_size,
            str(self.device),
            None if self.gauge_spec is None else (self.gauge_spec.provider, self.gauge_spec.path),
            None if self.kernel_cache is None else self.kernel_cache.packing,
        )

    @contextmanager
    def _open_file(self, path: str) -> Iterator[h5py.File]:
        if self.file_pool is None:
            with h5py.File(path, "r") as handle:
                yield handle
            return
        with self.file_pool.file(path) as handle:
            yield handle

    def _eigenvector_axes(self) -> list[str]:
        axes = list(self.spec.axes)
        allowed = {"time", "x", "y", "z", "color", "mode", "complex_pair"}
        unsupported = set(axes) - allowed
        if unsupported:
            raise JobDataError(
                "built-in HDF5 eigenvector provider does not support axes "
                f"{sorted(unsupported)}"
            )
        required = {"x", "y", "z", "color", "mode"}
        missing = required - set(axes)
        if missing:
            raise JobDataError(
                "built-in HDF5 eigenvector provider requires semantic axes "
                f"x/y/z/color/mode; missing {sorted(missing)}"
            )
        if "time" not in axes and not _contains_time_placeholder(self.spec):
            raise JobDataError(
                "eigenvector storage must select Euclidean time through a 'time' axis "
                "or a {time} file/dataset placeholder"
            )
        return axes

    def _normalized_eigenvector_tensor(
        self,
        array: np.ndarray,
        axes: list[str],
        scalar_type: str,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        array, axes = _decode_hdf5_complex(array, axes)
        target_axes = (
            ((_HDF5_EIGENVECTOR_BATCH_AXIS,) if batch_size is not None else ())
            + ("z", "y", "x", "color", "mode")
        )
        if set(axes) != set(target_axes) or len(axes) != len(target_axes):
            raise JobDataError(
                "normalized eigenvector axes must be exactly "
                f"{target_axes}; got {tuple(axes)}"
            )

        expected_sizes = {
            "x": self.spatial_extent[0],
            "y": self.spatial_extent[1],
            "z": self.spatial_extent[2],
            "color": 3,
        }
        for axis, expected in expected_sizes.items():
            actual = array.shape[axes.index(axis)]
            if actual != expected:
                raise JobDataError(
                    f"eigenvector axis {axis!r} has extent {actual}, expected {expected}"
                )
        if batch_size is not None:
            actual_batch = array.shape[axes.index(_HDF5_EIGENVECTOR_BATCH_AXIS)]
            if actual_batch != batch_size:
                raise JobDataError(
                    f"eigenvector batch axis has extent {actual_batch}, "
                    f"expected {batch_size}"
                )
        mode_axis = axes.index("mode")
        if array.shape[mode_axis] < self.mode_size:
            raise JobDataError(
                f"eigenvector mode axis has extent {array.shape[mode_axis]}, "
                f"but compiled plan requires {self.mode_size} modes"
            )
        if array.shape[mode_axis] != self.mode_size:
            indexers = [slice(None)] * array.ndim
            indexers[mode_axis] = slice(0, self.mode_size)
            array = array[tuple(indexers)]

        permutation = tuple(axes.index(axis) for axis in target_axes)
        if permutation != tuple(range(len(permutation))):
            array = np.transpose(array, permutation)
        return torch.as_tensor(
            np.ascontiguousarray(array),
            dtype=_torch_complex_dtype(scalar_type),
            device=self.device,
        )

    @staticmethod
    def _operations_require_gauge(operations: Sequence[OperationRecord]) -> bool:
        return any(
            operation.kind is OperationKind.DERIVATIVE
            or (
                operation.kind is OperationKind.TRANSPORT
                and bool(operation.parameters["path"])
            )
            for operation in operations
        )

    def _resolved_gauge_path(self) -> str:
        if self.gauge_spec is None:
            raise JobDataError(
                "derivative and gauge-transported LapH fields require "
                "data.gauge_links with provider 'laph_iomap' or 'openqcd'"
            )
        return _format_template(
            self.gauge_spec.path,
            {"configuration": self.configuration},
            role="data.gauge_links.path",
        )

    def _gauge_reader_class(self) -> type[LaphGaugeIOMapReader] | type[OpenQCDGaugeReader]:
        if isinstance(self.gauge_spec, OpenQCDGaugeSpec):
            return OpenQCDGaugeReader
        return LaphGaugeIOMapReader

    def _reader(self) -> LaphGaugeIOMapReader | OpenQCDGaugeReader:
        if self._gauge_reader is not None:
            return self._gauge_reader
        path = self._resolved_gauge_path()
        reader_class = self._gauge_reader_class()
        description = (
            "openQCD gauge file"
            if reader_class is OpenQCDGaugeReader
            else "native LapH smeared-gauge file"
        )
        try:
            reader = reader_class(path)
        except (OSError, LaphGaugeIOMapError, OpenQCDGaugeError) as exc:
            raise JobDataError(f"cannot read {description} {path!r}: {exc}") from exc
        actual_extent = reader.metadata.lattice_extent
        expected_extent = (
            (*self.spatial_extent, self.temporal_extent)
            if self.temporal_extent is not None
            else None
        )
        if actual_extent[:3] != self.spatial_extent:
            raise JobDataError(
                "gauge-link spatial extent "
                f"{actual_extent[:3]} does not match job lattice {self.spatial_extent}"
            )
        if expected_extent is not None and actual_extent != expected_extent:
            raise JobDataError(
                f"gauge-link lattice extent {actual_extent} does not match "
                f"job lattice {expected_extent}"
            )
        self._gauge_reader = reader
        return reader

    def _load_gauge_link(
        self,
        time: int,
        direction: str,
        scalar_type: str,
    ) -> torch.Tensor:
        _spatial_axis(f"+{direction}")
        reader = self._reader()
        try:
            array = reader.read_timeslice(direction, time, copy=True)
        except (LaphGaugeIOMapError, OpenQCDGaugeError) as exc:
            raise JobDataError(str(exc)) from exc
        numpy_dtype = np.complex64 if scalar_type == "complex64" else np.complex128
        array = np.ascontiguousarray(array, dtype=numpy_dtype)
        return torch.as_tensor(
            array,
            dtype=_torch_complex_dtype(scalar_type),
            device=self.device,
        ).unsqueeze(0)

    def _load_eigenvectors(self, time: int, scalar_type: str) -> torch.Tensor:
        cache_key = self._raw_eigenvector_key(time, scalar_type)
        if cache_key is not None:
            cached = self._raw_eigenvectors.get(cache_key)
            if cached is not None:
                return cached
        path = _format_template(
            self.spec.path,
            {"configuration": self.configuration, "time": time},
            role="data.eigenvectors.path",
        )
        dataset_name = _format_template(
            self.spec.dataset,
            {"configuration": self.configuration, "time": time},
            role="data.eigenvectors.dataset",
        )
        axes = self._eigenvector_axes()

        try:
            with self._open_file(path) as handle:
                if dataset_name not in handle:
                    raise JobDataError(
                        f"eigenvector dataset {dataset_name!r} was not found in {path!r}"
                    )
                dataset = handle[dataset_name]
                if dataset.ndim != len(axes):
                    raise JobDataError(
                        f"eigenvector axes metadata has rank {len(axes)}, but dataset "
                        f"{dataset_name!r} has rank {dataset.ndim}"
                    )
                indexers: list[int | slice] = [slice(None)] * len(axes)
                if type(self) is HDF5DistillationKernelProvider:
                    indexers[axes.index("mode")] = slice(0, self.mode_size)
                if "time" in axes:
                    indexers[axes.index("time")] = time
                try:
                    array = np.asarray(dataset[tuple(indexers)])
                except (IndexError, ValueError) as exc:
                    raise JobDataError(
                        f"eigenvector time {time} is outside dataset {dataset_name!r}"
                    ) from exc
        except OSError as exc:
            raise JobDataError(f"cannot read eigenvector HDF5 file {path!r}: {exc}") from exc

        if "time" in axes:
            axes.pop(axes.index("time"))
        value = self._normalized_eigenvector_tensor(
            array,
            axes,
            scalar_type,
        ).unsqueeze(0)
        if cache_key is not None:
            self._raw_eigenvectors.put(cache_key, value)
        return value

    def _kernel_batch_layout(
        self,
        requests: Sequence[KernelMaterializationRequest],
    ) -> tuple[tuple[int, ...], tuple[int, ...], int, int] | None:
        values = tuple(requests)
        if not values:
            return None
        if any(
            not isinstance(request, KernelMaterializationRequest)
            for request in values
        ):
            raise TypeError(
                "HDF5 distillation batching requires kernel materialization requests"
            )

        first = values[0]
        if any(
            request.provider_id != first.provider_id
            or request.recipe != first.recipe
            or request.operator_ids != first.operator_ids
            or request.expected_shape != first.expected_shape
            or request.scalar_type != first.scalar_type
            for request in values[1:]
        ):
            return None
        if _contains_time_placeholder(self.spec) or "time" not in self.spec.axes:
            return None
        if first.recipe.gauge_tensors:
            return None
        if any(
            projected.contribution is ProjectionContribution.POINT
            or bool(projected.operations)
            for projected in first.recipe.projected_fields
        ):
            return None
        if not self._kernel_batch_supports_spatial_tiling(first):
            return None

        requested_times: list[int] = []
        for request in values:
            distinct_times = tuple(dict.fromkeys(request.operator_times))
            if len(distinct_times) > 1:
                return None
            time = distinct_times[0] if distinct_times else 0
            if time < 0:
                return None
            requested_times.append(time)
        times = tuple(requested_times)
        unique_times = tuple(sorted(set(times)))
        first_time = unique_times[0]
        span = unique_times[-1] - first_time + 1
        if span / len(unique_times) > _MAX_HDF5_EIGENVECTOR_SLAB_AMPLIFICATION:
            return None
        return times, unique_times, first_time, span

    @staticmethod
    def _kernel_batch_supports_spatial_tiling(
        request: KernelMaterializationRequest,
    ) -> bool:
        program = emit_backend_kernel_program(request.recipe)
        position_legs: set[str] = set()
        for operation in program.operations:
            legs = tuple(operation.params["legs"])
            if operation.kind == "eigenvector":
                if len(legs) != 4 or legs[0] != "T":
                    return False
                position_legs.add(legs[1])
            elif operation.kind == "phase":
                if len(legs) != 1:
                    return False
                position_legs.add(legs[0])
            elif operation.kind not in {"color_delta", "color_epsilon"}:
                return False
        output_legs = tuple(program.output_legs or program.mode_ports)
        return len(position_legs) == 1 and position_legs.isdisjoint(output_legs)

    def _kernel_batch_workspace_bytes(
        self,
        request: KernelMaterializationRequest,
        *,
        batch_size: int,
        spatial_sites: int,
    ) -> int:
        program = emit_backend_kernel_program(request.recipe)
        volume = math.prod(self.spatial_extent)
        if spatial_sites < 1 or spatial_sites > volume:
            raise ValueError("spatial_sites must lie within the lattice volume")
        shape_operands: list[tuple[int, ...]] = []
        input_legs: list[tuple[str, ...]] = []
        for operation in program.operations:
            legs = tuple(operation.params["legs"])
            if operation.kind == "eigenvector":
                shape = (batch_size, spatial_sites, 3, self.mode_size)
            elif operation.kind == "phase":
                if tuple(operation.params["momentum"]) == (0, 0, 0):
                    continue
                shape = (spatial_sites,)
            elif operation.kind == "color_delta":
                shape = (3, 3)
            elif operation.kind == "color_epsilon":
                shape = (3, 3, 3)
            else:  # pragma: no cover - eligibility closes the kernel ISA
                raise RuntimeError("unsupported operation in kernel batch estimate")
            shape_operands.append(shape)
            input_legs.append(legs)

        output_legs = ("T", *tuple(program.output_legs or program.mode_ports))
        all_legs = tuple(
            dict.fromkeys(
                leg for legs in (*input_legs, output_legs) for leg in legs
            )
        )
        symbols = {
            leg: oe.get_symbol(index) for index, leg in enumerate(all_legs)
        }
        equation = (
            ",".join(
                "".join(symbols[leg] for leg in legs) for legs in input_legs
            )
            + "->"
            + "".join(symbols[leg] for leg in output_legs)
        )
        return (
            _einsum_workspace_elements(equation, tuple(shape_operands))
            * _COMPLEX_ELEMENT_BYTES[request.scalar_type]
        )

    def _kernel_batch_evaluation_plan(
        self,
        requests: Sequence[KernelMaterializationRequest],
        *,
        maximum_staging_bytes: int | None,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int, int], int, int] | None:
        layout = self._kernel_batch_layout(requests)
        if layout is None:
            return None
        values = tuple(requests)
        first = values[0]
        times, _, first_time, span = layout
        element_bytes = _COMPLEX_ELEMENT_BYTES[first.scalar_type]
        volume = math.prod(self.spatial_extent)
        slice_bytes = volume * 3 * self.mode_size * element_bytes
        slab_bytes = span * slice_bytes
        offsets = tuple(time - first_time for time in times)
        gathered_bytes = (
            0 if offsets == tuple(range(span)) else len(times) * slice_bytes
        )
        output_bytes = len(times) * math.prod(first.expected_shape) * element_bytes
        fixed_bytes = (
            3 * slab_bytes + gathered_bytes + output_bytes
            + self._kernel_cache_write_workspace_bytes(first)
        )

        full_workspace = self._kernel_batch_workspace_bytes(
            first,
            batch_size=len(values),
            spatial_sites=volume,
        )
        full_bytes = fixed_bytes + full_workspace
        if maximum_staging_bytes is None or full_bytes <= maximum_staging_bytes:
            return layout, volume, full_bytes

        minimum_workspace = self._kernel_batch_workspace_bytes(
            first,
            batch_size=len(values),
            spatial_sites=1,
        )
        if fixed_bytes + minimum_workspace > maximum_staging_bytes:
            return None

        lower = 1
        upper = volume
        selected_sites = 1
        selected_workspace = minimum_workspace
        while lower <= upper:
            candidate = (lower + upper) // 2
            workspace = self._kernel_batch_workspace_bytes(
                first,
                batch_size=len(values),
                spatial_sites=candidate,
            )
            if fixed_bytes + workspace <= maximum_staging_bytes:
                selected_sites = candidate
                selected_workspace = workspace
                lower = candidate + 1
            else:
                upper = candidate - 1
        return layout, selected_sites, fixed_bytes + selected_workspace

    def _baryon_cache_transform(
        self, request: KernelMaterializationRequest
    ) -> _BaryonCacheCanonicalization | None:
        if type(self) is not HDF5DistillationKernelProvider:
            return None
        return _canonical_baryon_cache_transform(request.recipe)

    def _kernel_cache_packing_workspace_bytes(
        self, request: KernelMaterializationRequest
    ) -> int:
        if (
            self.kernel_cache is None
            or self.kernel_cache.packing != "antisym3"
            or self._baryon_cache_transform(request) is None
            or request.expected_shape != (self.mode_size,) * 3
            or self.mode_size < 3
        ):
            return 0
        return _antisym3_workspace_bytes(self.mode_size, np.dtype(request.scalar_type))

    def _kernel_cache_write_workspace_bytes(
        self, request: KernelMaterializationRequest
    ) -> int:
        if self.kernel_cache is None:
            return 0
        tensor_bytes = math.prod(request.expected_shape) * _COMPLEX_ELEMENT_BYTES[request.scalar_type]
        canonical = self._baryon_cache_transform(request)
        copies = 1 + int(self.device.type == "cuda")
        copies += int(
            canonical is not None and (canonical.needs_conj or canonical.color_sign == -1)
        )
        return copies * tensor_bytes + self._kernel_cache_packing_workspace_bytes(request)

    def _kernel_cache_load_workspace_bytes(
        self, request: KernelMaterializationRequest
    ) -> int:
        tensor_bytes = math.prod(request.expected_shape) * _COMPLEX_ELEMENT_BYTES[request.scalar_type]
        return (
            (1 + int(self.device.type == "cuda")) * tensor_bytes
            + self._kernel_cache_packing_workspace_bytes(request)
        )

    def _kernel_batch_cache_state(
        self, requests: Sequence[KernelMaterializationRequest]
    ) -> _KernelBatchCacheState | None:
        if self.kernel_cache is None:
            return None
        entries = tuple(self._kernel_cache_entry(request) for request in requests)
        return _KernelBatchCacheState(
            entries, tuple(self.kernel_cache.contains(entry[0]) for entry in entries)
        )

    def _kernel_batch_execution_plan(
        self,
        requests: Sequence[KernelMaterializationRequest],
        *,
        maximum_staging_bytes: int | None,
        cache_state: _KernelBatchCacheState | None = None,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int, int], int, int] | None:
        layout = self._kernel_batch_layout(requests)
        if layout is None:
            return None
        if self.kernel_cache is not None and type(self) is not HDF5DistillationKernelProvider:
            return None
        state = (
            cache_state if cache_state is not None
            else self._kernel_batch_cache_state(requests)
        )
        if state is None or not any(state.present):
            return self._kernel_batch_evaluation_plan(
                requests, maximum_staging_bytes=maximum_staging_bytes
            )
        values = tuple(requests)
        output_bytes = len(values) * math.prod(values[0].expected_shape) * _COMPLEX_ELEMENT_BYTES[values[0].scalar_type]
        loading_bytes = self._kernel_cache_load_workspace_bytes(values[0])
        if maximum_staging_bytes is not None and output_bytes + loading_bytes > maximum_staging_bytes:
            return None
        missing = tuple(
            request for request, present in zip(values, state.present, strict=True)
            if not present
        )
        if not missing:
            return layout, math.prod(self.spatial_extent), output_bytes + loading_bytes
        remaining = None if maximum_staging_bytes is None else maximum_staging_bytes - output_bytes
        evaluation = self._kernel_batch_evaluation_plan(
            missing, maximum_staging_bytes=remaining
        )
        if evaluation is None:
            return None
        return layout, evaluation[1], output_bytes + max(loading_bytes, evaluation[2])

    def batch_staging_bytes(
        self,
        requests: Sequence[KernelMaterializationRequest],
    ) -> int | None:
        plan = self._kernel_batch_execution_plan(
            requests,
            maximum_staging_bytes=None,
        )
        return None if plan is None else plan[2]

    def batch_staging_bytes_with_limit(
        self,
        requests: Sequence[KernelMaterializationRequest],
        *,
        maximum_staging_bytes: int | None,
    ) -> int | None:
        plan = self._kernel_batch_execution_plan(
            requests,
            maximum_staging_bytes=maximum_staging_bytes,
        )
        return None if plan is None else plan[2]

    def _load_eigenvector_batch(
        self,
        requests: Sequence[KernelMaterializationRequest],
        layout: tuple[tuple[int, ...], tuple[int, ...], int, int],
    ) -> torch.Tensor:
        values = tuple(requests)
        first = values[0]
        times, unique_times, first_time, span = layout
        cache_keys = {
            time: self._raw_eigenvector_key(time, first.scalar_type)
            for time in unique_times
        }
        cached = {
            time: self._raw_eigenvectors.get(key)
            for time, key in cache_keys.items() if key is not None
        }
        if len(cached) == len(unique_times) and all(value is not None for value in cached.values()):
            return torch.cat(tuple(cached[time] for time in times), dim=0)
        del cached
        last_time = first_time + span - 1
        path = _format_template(
            self.spec.path,
            {"configuration": self.configuration, "time": first_time},
            role="data.eigenvectors.path",
        )
        dataset_name = _format_template(
            self.spec.dataset,
            {"configuration": self.configuration, "time": first_time},
            role="data.eigenvectors.dataset",
        )
        axes = self._eigenvector_axes()
        time_axis = axes.index("time")
        indexers: list[int | slice] = [slice(None)] * len(axes)
        indexers[time_axis] = slice(first_time, last_time + 1)
        indexers[axes.index("mode")] = slice(0, self.mode_size)
        try:
            with self._open_file(path) as handle:
                if dataset_name not in handle:
                    raise JobDataError(
                        f"eigenvector dataset {dataset_name!r} was not found in {path!r}"
                    )
                dataset = handle[dataset_name]
                if dataset.ndim != len(axes):
                    raise JobDataError(
                        f"eigenvector axes metadata has rank {len(axes)}, but dataset "
                        f"{dataset_name!r} has rank {dataset.ndim}"
                    )
                if last_time >= dataset.shape[time_axis]:
                    raise JobDataError(
                        f"eigenvector time {last_time} is outside dataset "
                        f"{dataset_name!r}"
                    )
                array = np.asarray(dataset[tuple(indexers)])
        except OSError as exc:
            raise JobDataError(
                f"cannot read eigenvector HDF5 file {path!r}: {exc}"
            ) from exc

        normalized_axes = list(axes)
        normalized_axes[time_axis] = _HDF5_EIGENVECTOR_BATCH_AXIS
        slab = self._normalized_eigenvector_tensor(
            array,
            normalized_axes,
            first.scalar_type,
            batch_size=span,
        )
        for time, key in cache_keys.items():
            if key is not None:
                offset = time - first_time
                self._raw_eigenvectors.put(key, slab[offset : offset + 1])
        offsets = tuple(time - first_time for time in times)
        if len(unique_times) == 1 and len(times) > 1:
            return slab.expand(len(times), *slab.shape[1:])
        if offsets == tuple(range(span)):
            return slab
        gather = torch.tensor(offsets, dtype=torch.long, device=self.device)
        return slab.index_select(0, gather)

    def _evaluate_kernel_batch(
        self,
        requests: Sequence[KernelMaterializationRequest],
        eigenvectors: torch.Tensor,
        times: tuple[int, ...],
        *,
        spatial_tile_sites: int,
    ) -> torch.Tensor:
        first = tuple(requests)[0]
        dtype = _torch_complex_dtype(first.scalar_type)
        flattened = eigenvectors.reshape(
            len(times),
            -1,
            3,
            self.mode_size,
        )
        volume = flattened.shape[1]
        full_phases = {
            tuple(momentum): _momentum_phase(
                tuple(momentum),
                self.spatial_extent,
                dtype=dtype,
                device=self.device,
            )
            for momentum in dict.fromkeys(first.recipe.momenta.values())
            if tuple(momentum) != (0, 0, 0)
        }
        context = ExecutionContext.create(
            device=self.device,
            t_src=times[0],
            t_snk_list=times,
            dtype=_REAL_DTYPES[first.scalar_type],
            complex_dtype=dtype,
        )
        program = emit_backend_kernel_program(first.recipe)
        output_legs = list(program.output_legs or program.mode_ports)
        if "T" not in output_legs:
            output_legs.insert(0, "T")
        program.output_legs = output_legs

        result: torch.Tensor | None = None
        for start in range(0, volume, spatial_tile_sites):
            stop = min(start + spatial_tile_sites, volume)
            provider = _PreparedDistillationTensorProvider(
                fields={(): flattened[:, start:stop]},
                phases={
                    momentum: phase[start:stop]
                    for momentum, phase in full_phases.items()
                },
            )
            tile = evaluate_kernel_program(
                program,
                provider,
                context,
                is_sink=True,
            )
            if result is None:
                result = tile
            else:
                result.add_(tile)
        assert result is not None
        return result

    def materialize_batch_with_limit(
        self,
        requests: Sequence[KernelMaterializationRequest],
        *,
        maximum_staging_bytes: int | None,
    ) -> KernelBatchMaterialization | None:
        if maximum_staging_bytes is None:
            return None
        if self._kernel_batch_layout(requests) is None:
            return None
        cache_state = self._kernel_batch_cache_state(requests)
        plan = self._kernel_batch_execution_plan(
            requests,
            maximum_staging_bytes=maximum_staging_bytes,
            cache_state=cache_state,
        )
        if plan is None:
            return None
        if cache_state is not None and any(cache_state.present):
            values = tuple(requests)
            result = torch.empty(
                (len(values), *values[0].expected_shape),
                dtype=_torch_complex_dtype(values[0].scalar_type),
                device=self.device,
            )
            for index, present in enumerate(cache_state.present):
                if present:
                    loaded = self._load_kernel_cache_entry(
                        values[index], cache_state.entries[index]
                    )
                    if loaded is None:
                        return None
                    result[index].copy_(loaded)
                    del loaded
            missing_indices = tuple(
                index for index, present in enumerate(cache_state.present) if not present
            )
            if not missing_indices:
                return KernelBatchMaterialization(result, "hdf5_kernel_cache_batch")
            missing_requests = tuple(values[index] for index in missing_indices)
            missing_layout = self._kernel_batch_layout(missing_requests)
            assert missing_layout is not None
            eigenvectors = self._load_eigenvector_batch(missing_requests, missing_layout)
            tensor = self._evaluate_kernel_batch(
                missing_requests, eigenvectors, missing_layout[0],
                spatial_tile_sites=plan[1],
            )
            self.kernel_cache.record_misses(len(missing_requests))
            for offset, index in enumerate(missing_indices):
                self._store_kernel_cache_entry(
                    values[index], cache_state.entries[index], tensor[offset]
                )
                result[index].copy_(tensor[offset])
            return KernelBatchMaterialization(result, "hdf5_kernel_cache_mixed_batch")
        cache_entries = None if cache_state is None else cache_state.entries
        if cache_entries is not None:
            self.kernel_cache.record_misses(len(cache_entries))
        layout, spatial_tile_sites, _ = plan
        times = layout[0]
        eigenvectors = self._load_eigenvector_batch(requests, layout)
        tensor = self._evaluate_kernel_batch(
            requests,
            eigenvectors,
            times,
            spatial_tile_sites=spatial_tile_sites,
        )
        mode = (
            "hdf5_distillation_dense_slab"
            if spatial_tile_sites == math.prod(self.spatial_extent)
            else "hdf5_distillation_dense_slab_spatial_tiled"
        )
        result = KernelBatchMaterialization(tensor=tensor, mode=mode)
        if cache_entries is not None:
            for request, entry, value in zip(
                requests,
                cache_entries,
                tensor.unbind(0),
                strict=True,
            ):
                self._store_kernel_cache_entry(request, entry, value)
        return result

    def _prepare_eigenvector_field(
        self,
        raw_field: torch.Tensor,
        operations: Sequence[OperationRecord],
        *,
        time: int,
        scalar_type: str,
        request_gauge_links: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not operations:
            return raw_field.reshape(1, -1, 3, self.mode_size)
        if self._operations_require_gauge(operations) and self.gauge_spec is None:
            self._resolved_gauge_path()

        def gauge_link(direction: str) -> torch.Tensor:
            value = request_gauge_links.get(direction)
            if value is None:
                value = self._load_gauge_link(time, direction, scalar_type)
                request_gauge_links[direction] = value
            return value

        transformed = _apply_laph_operations(raw_field, operations, gauge_link)
        return transformed.reshape(1, -1, 3, self.mode_size)

    def _tensor_provider_for_request(self, request) -> DistillationTensorProvider:
        distinct_times = tuple(dict.fromkeys(request.operator_times))
        if len(distinct_times) > 1:
            raise NotImplementedError(
                "built-in HDF5 distillation kernels require one common Euclidean time"
            )
        time = distinct_times[0] if distinct_times else 0
        raw_eigenvectors = self._load_eigenvectors(time, request.scalar_type)
        dtype = _torch_complex_dtype(request.scalar_type)
        phases = {
            tuple(momentum): _momentum_phase(
                tuple(momentum),
                self.spatial_extent,
                dtype=dtype,
                device=self.device,
            )
            for momentum in dict.fromkeys(request.recipe.momenta.values())
            if tuple(momentum) != (0, 0, 0)
        }
        request_gauge_links: dict[str, torch.Tensor] = {}
        fields: dict[tuple[tuple[object, ...], ...], torch.Tensor] = {}
        for projected in request.recipe.projected_fields:
            operation_key = _operation_key(projected.operations)
            if operation_key in fields:
                continue
            fields[operation_key] = self._prepare_eigenvector_field(
                raw_eigenvectors,
                projected.operations,
                time=time,
                scalar_type=request.scalar_type,
                request_gauge_links=request_gauge_links,
            )
        return _PreparedDistillationTensorProvider(
            fields=fields,
            phases=phases,
        )

    def _kernel_cache_entry(
        self,
        request: KernelMaterializationRequest,
    ) -> tuple[str, str, tuple[str, ...]]:
        if self.kernel_cache is None:
            raise RuntimeError("persistent kernel cache is not configured")
        distinct_times = tuple(dict.fromkeys(request.operator_times))
        if len(distinct_times) > 1:
            raise NotImplementedError(
                "built-in HDF5 distillation kernels require one common Euclidean time"
            )
        time = distinct_times[0] if distinct_times else 0
        eigenvector_path = _format_template(
            self.spec.path,
            {"configuration": self.configuration, "time": time},
            role="data.eigenvectors.path",
        )
        dataset_name = _format_template(
            self.spec.dataset,
            {"configuration": self.configuration, "time": time},
            role="data.eigenvectors.dataset",
        )
        try:
            eigenvector_fingerprint = self.kernel_cache.source_fingerprint(
                eigenvector_path
            )
            gauge_fingerprint = None
            operations = tuple(
                operation
                for projected in request.recipe.projected_fields
                for operation in projected.operations
            )
            if self._operations_require_gauge(operations):
                gauge_path = self._resolved_gauge_path()
                gauge_fingerprint = self.kernel_cache.source_fingerprint(gauge_path)
        except (OSError, KernelStoreError) as exc:
            raise JobDataError(
                f"cannot fingerprint persistent kernel cache input: {exc}"
            ) from exc

        if len(request.recipe.open_ports) == len(request.expected_shape):
            axes = tuple(request.recipe.open_ports)
        else:
            axes = tuple(
                f"kernel_axis_{index}"
                for index in range(len(request.expected_shape))
            )

        implementation = self._kernel_cache_implementation_identity(
            include_gauge_reader=gauge_fingerprint is not None
        )
        canonical = self._baryon_cache_transform(request)
        manifest = {
            "cache_schema_version": HDF5KernelStore.format_version,
            "kernel_algorithm_version": _KERNEL_CACHE_ALGORITHM_VERSION,
            "provider": "hdf5_distillation",
            "configuration": self.configuration,
            "operator_times": list(request.operator_times),
            "resource_id": (
                request.resource_id if canonical is None else canonical.canonical_resource_id
            ),
            "recipe_signature": (
                kernel_resource_signature(request.recipe) if canonical is None
                else canonical.canonical_recipe_signature
            ),
            "recipe_canonicalization": None if canonical is None else "shared-basis-epsilon-v1",
            "storage_packing_policy": self.kernel_cache.packing,
            "shape": list(request.expected_shape),
            "axes": list(axes),
            "scalar_type": request.scalar_type,
            "spatial_extent": list(self.spatial_extent),
            "mode_size": self.mode_size,
            "eigenvectors": {
                **eigenvector_fingerprint,
                "dataset": dataset_name,
                "axes": list(self.spec.axes),
            },
            "gauge_links": gauge_fingerprint,
            "implementation": implementation,
            "numerical_backend": self._kernel_cache_backend_identity(),
        }
        manifest_json = self.kernel_cache.canonical_manifest(manifest)
        return self.kernel_cache.entry_key(manifest_json), manifest_json, axes

    def _kernel_cache_backend_identity(self) -> dict[str, object]:
        if self._kernel_cache_backend is not None:
            return dict(self._kernel_cache_backend)
        if self.device.type == "cuda" and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(self.device)
        else:
            device_name = platform.processor()
            cpu_info = Path("/proc/cpuinfo")
            if not device_name and cpu_info.is_file():
                for line in cpu_info.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines():
                    if line.lower().startswith("model name") and ":" in line:
                        device_name = line.split(":", 1)[1].strip()
                        break
            device_name = device_name or platform.machine()
        self._kernel_cache_backend = {
            "device_type": self.device.type,
            "device_name": device_name,
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "torch_config_sha256": hashlib.sha256(
                torch.__config__.show().encode("utf-8")
            ).hexdigest(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_version": torch.version.cuda,
            "opt_einsum_version": str(oe.__version__),
        }
        return dict(self._kernel_cache_backend)

    def _kernel_cache_implementation_identity(
        self,
        *,
        include_gauge_reader: bool,
    ) -> dict[str, object]:
        if self.kernel_cache is None:
            raise RuntimeError("persistent kernel cache is not configured")
        if self._kernel_cache_implementation is None:
            self._kernel_cache_implementation = {
                "provider": self.kernel_cache.source_fingerprint(__file__),
                "store": self.kernel_cache.source_fingerprint(
                    inspect.getsourcefile(HDF5KernelStore)
                ),
                "kernel_lowering": self.kernel_cache.source_fingerprint(
                    inspect.getsourcefile(emit_backend_kernel_program)
                ),
                "kernel_evaluator": self.kernel_cache.source_fingerprint(
                    inspect.getsourcefile(evaluate_kernel_program)
                ),
            }
        implementation = dict(self._kernel_cache_implementation)
        if include_gauge_reader and "gauge_reader" not in implementation:
            implementation["gauge_reader"] = self.kernel_cache.source_fingerprint(
                inspect.getsourcefile(self._gauge_reader_class())
            )
            self._kernel_cache_implementation["gauge_reader"] = implementation[
                "gauge_reader"
            ]
        return implementation

    def _load_kernel_cache_entry(
        self,
        request: KernelMaterializationRequest,
        entry: tuple[str, str, tuple[str, ...]],
    ) -> torch.Tensor | None:
        if self.kernel_cache is None:
            return None
        key, manifest_json, axes = entry
        numpy_dtype = (
            np.complex64
            if request.scalar_type == "complex64"
            else np.complex128
        )
        array = self.kernel_cache.load_array(
            key,
            manifest_json,
            expected_shape=tuple(request.expected_shape),
            expected_dtype=np.dtype(numpy_dtype),
            expected_axes=axes,
            expected_scalar_type=request.scalar_type,
        )
        if array is None:
            return None
        canonical = self._baryon_cache_transform(request)
        if canonical is not None:
            if canonical.needs_conj:
                np.conjugate(array, out=array)
            if canonical.color_sign == -1:
                np.negative(array, out=array)
        return torch.as_tensor(
            array,
            dtype=_torch_complex_dtype(request.scalar_type),
            device=self.device,
        )

    def _store_kernel_cache_entry(
        self,
        request: KernelMaterializationRequest,
        entry: tuple[str, str, tuple[str, ...]],
        tensor: torch.Tensor,
    ) -> None:
        if self.kernel_cache is None:
            return
        key, manifest_json, axes = entry
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        if tuple(array.shape) != tuple(request.expected_shape):
            raise ValueError(
                f"kernel cache write received shape {tuple(array.shape)}, "
                f"expected {tuple(request.expected_shape)}"
            )
        expected_dtype = (
            np.dtype(np.complex64)
            if request.scalar_type == "complex64"
            else np.dtype(np.complex128)
        )
        if array.dtype != expected_dtype:
            raise TypeError(
                f"kernel cache write received dtype {array.dtype}, expected {expected_dtype}"
            )
        canonical = self._baryon_cache_transform(request)
        if canonical is not None and (canonical.needs_conj or canonical.color_sign == -1):
            array = array.copy()
            if canonical.needs_conj:
                np.conjugate(array, out=array)
            if canonical.color_sign == -1:
                np.negative(array, out=array)
        self.kernel_cache.store_array(
            key,
            manifest_json,
            array,
            axes=axes,
            allow_antisym3=canonical is not None,
        )

    def materialize(self, request) -> torch.Tensor:
        if self.kernel_cache is None:
            return self._delegate.materialize(request)
        if not isinstance(request, KernelMaterializationRequest):
            raise TypeError("HDF5 distillation provider requires a kernel request")
        entry = self._kernel_cache_entry(request)
        cached = self._load_kernel_cache_entry(request, entry)
        if cached is not None:
            return cached
        tensor = self._delegate.materialize(request)
        self._store_kernel_cache_entry(request, entry, tensor)
        return tensor


def _load_reference_module(target: str) -> tuple[ModuleType, OperatorSourceSession | None]:
    """Load a module and retain any private local-source import session."""
    path = Path(target).expanduser()
    looks_like_file = path.suffix.lower() == ".py" or path.is_absolute() or target.startswith(".")
    if not looks_like_file:
        try:
            return importlib.import_module(target), None
        except Exception as exc:
            raise JobDataError(f"failed to import data resolver module {target!r}: {exc}") from exc

    path = path.resolve(strict=False)
    if not path.exists() or not path.is_file():
        raise JobDataError(f"data resolver file does not exist: {path}")
    try:
        session = OperatorSourceSession(path)
        with session:
            module = session.load_main()
    except Exception as exc:
        raise JobDataError(f"failed to import data resolver file {path}: {exc}") from exc
    return module, session


def _load_reference_callable(reference: str):
    target, attribute = reference.rsplit(":", 1)
    module, session = _load_reference_module(target)
    with session if session is not None else nullcontext():
        try:
            value = getattr(module, attribute)
        except AttributeError as exc:
            raise JobDataError(
                f"data resolver target {target!r} has no callable {attribute!r}"
            ) from exc
    if not callable(value):
        raise JobDataError(f"data resolver {reference!r} is not callable")
    if session is None:
        return value

    @wraps(value)
    def isolated_call(*args, **kwargs):
        # Late helper imports and dataclass decorators need their private module
        # registrations during execution, including on repeated calls.
        with session:
            return value(*args, **kwargs)

    return isolated_call


def _ordinary_propagation_source(
    spec: JobSpec,
    recipe: PropagationRecipe,
    *,
    resource_id: str,
    configuration: int,
) -> ExternalTensorSource:
    if recipe.is_generalized:
        raise JobDataError("ordinary source resolver received a generalized propagation recipe")
    if len(recipe.segment_flavors) != 1:
        raise JobDataError(
            f"ordinary propagation resource {resource_id!r} has segment flavors "
            f"{recipe.segment_flavors!r}; expected exactly one flavor"
        )
    flavor = recipe.segment_flavors[0]
    ordinary = spec.data.propagation.ordinary
    values = {"configuration": configuration, "flavor": flavor}

    if ordinary.path is not None:
        location = _format_template(
            ordinary.path,
            values,
            role="data.propagation.ordinary.path",
        )
    else:
        assert ordinary.files is not None
        try:
            template = ordinary.files[flavor]
        except KeyError as exc:
            raise JobDataError(
                f"no ordinary propagation file is configured for exact flavor {flavor!r}"
            ) from exc
        location = _format_template(
            template,
            values,
            role=f"data.propagation.ordinary.files[{flavor!r}]",
        )

    if "configuration" in ordinary.axes:
        raise JobDataError(
            "the built-in ordinary HDF5 catalog does not infer a configuration-axis "
            "row from a configuration id; select configuration in the file/dataset "
            "template or use a custom data provider"
        )

    dataset = _partial_format_template(
        ordinary.dataset,
        values,
        role="data.propagation.ordinary.dataset",
    )
    return ExternalTensorSource(
        source_id=f"cfg{configuration}:{resource_id}",
        kind="hdf5",
        location=location,
        dataset=dataset,
        metadata={
            "axes": tuple(ordinary.axes),
            "spin_basis": spec.compile.spin_basis,
            "source_field": ordinary.source_field,
            **(
                {"fixed_source_time": ordinary.fixed_source_time}
                if ordinary.fixed_source_time is not None
                else {}
            ),
        },
    )


def _generalized_propagation_source(
    spec: JobSpec,
    recipe: PropagationRecipe,
    *,
    resource_id: str,
    configuration: int,
) -> ExternalTensorSource:
    generalized = spec.data.propagation.generalized
    if generalized is None:
        raise JobDataError(
            f"compiled plan requires generalized propagation resource {resource_id!r}, "
            "but data.propagation.generalized is not configured"
        )
    resolver = _load_reference_callable(generalized.resolver)
    try:
        source = resolver(
            resource_id=resource_id,
            recipe=recipe,
            configuration=configuration,
        )
    except Exception as exc:
        raise JobDataError(
            f"generalized propagation resolver failed for {resource_id!r}: {exc}"
        ) from exc
    if not isinstance(source, ExternalTensorSource):
        raise JobDataError(
            "generalized propagation resolver must return ExternalTensorSource"
        )
    return source


def _resolve_time_context(
    declared_contexts: Mapping[str, ExpandedTimeContext],
    value: ExpandedTimeContext | str,
) -> ExpandedTimeContext:
    if isinstance(value, str):
        try:
            return declared_contexts[value]
        except KeyError as exc:
            raise JobDataError(f"unknown job time-context id {value!r}") from exc
    if not isinstance(value, ExpandedTimeContext):
        raise TypeError("time_context must be ExpandedTimeContext or context id string")
    expected = declared_contexts.get(value.id)
    if expected is None or expected != value:
        raise JobDataError(
            f"time context {value.id!r} is not one of the contexts declared by this job"
        )
    return value


@dataclass(frozen=True)
class JobRuntimeResources:
    """Concrete runtime resources for one configuration/time work unit."""

    configuration: int
    time_context: ExpandedTimeContext
    operator_times: OperatorTimeContext
    propagation_sources: Mapping[str, ExternalTensorSource]
    kernel_providers: Mapping[str, KernelMaterializationProvider]
    source_handlers: Mapping[str, PropagationSourceHandler] = field(default_factory=dict)
    kernel_provider_id: str = _KERNEL_PROVIDER_ID
    storage_device: torch.device = torch.device("cpu")
    device: torch.device = torch.device("cpu")
    execution_plan: DeviceExecutionPlan | None = None
    binding_plan: ResourceBindingPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "propagation_sources", dict(self.propagation_sources))
        object.__setattr__(self, "kernel_providers", dict(self.kernel_providers))
        object.__setattr__(self, "source_handlers", dict(self.source_handlers))
        if not self.kernel_provider_id:
            raise ValueError("kernel_provider_id must be non-empty")
        if self.binding_plan is not None:
            if not isinstance(self.binding_plan, ResourceBindingPlan):
                raise TypeError("binding_plan must be a ResourceBindingPlan or None")
            if self.binding_plan.time_context != self.operator_times:
                raise ValueError(
                    "prebound resource plan disagrees with operator-time context"
                )
        if self.execution_plan is not None:
            if self.execution_plan.torch_device != self.device:
                raise ValueError(
                    "runtime compute device disagrees with its execution resource plan"
                )
            if self.execution_plan.storage_torch_device != self.storage_device:
                raise ValueError(
                    "runtime storage device disagrees with its execution resource plan"
                )

    def bind(self, plan: ConcordPlan) -> ResourceBindingPlan:
        """Build the exact frozen resource-binding plan for this work unit."""
        if not isinstance(plan, ConcordPlan):
            raise TypeError("plan must be ConcordPlan")
        if self.binding_plan is not None:
            binding = self.binding_plan
            if binding.source_plan is not plan.physical_plan:
                raise ValueError("prebound resources belong to a different physical plan")
            if binding.spin_basis != plan.config.spin_basis:
                raise ValueError("prebound resources use a different spin basis")
            bound_sources = {
                resource_id: resource.source
                for resource_id, resource in binding.bindings.items()
                if isinstance(resource, PropagationResourceBinding)
            }
            if bound_sources != self.propagation_sources:
                raise ValueError(
                    "prebound resources disagree with propagation sources"
                )
            if any(
                resource.provider_id not in self.kernel_providers
                for resource in binding.bindings.values()
                if isinstance(resource, KernelResourceBinding)
            ):
                raise ValueError(
                    "prebound resources require an unavailable kernel provider"
                )
            binding.validate()
            return binding
        plan.validate()
        return build_resource_binding_plan(
            plan.physical_plan,
            operator_times=self.operator_times,
            propagation_sources=self.propagation_sources,
            kernel_provider_id=self.kernel_provider_id,
            spin_basis=plan.config.spin_basis,
        )

    def materializer(
        self,
        plan: ConcordPlan,
        *,
        tensor_cache: PrimitiveTensorCache | None = None,
    ) -> PrimitiveMaterializer:
        binding = self.bind(plan)
        return PrimitiveMaterializer(
            binding,
            kernel_providers=self.kernel_providers,
            source_handlers=self.source_handlers,
            tensor_cache=tensor_cache,
        )


def _build_job_configuration_runtime_resources(
    spec: JobSpec,
    plan: ConcordPlan,
    *,
    configuration: int,
    time_contexts: Sequence[ExpandedTimeContext | str],
    device: str | torch.device | None = None,
    hardware: HardwareProfile | None = None,
    budget_policy: HardwareBudgetPolicy | None = None,
    execution_plan: DeviceExecutionPlan | None = None,
    hdf5_file_pool: HDF5FilePool | None = None,
    hdf5_source_handler: HDF5TensorSourceHandler | None = None,
) -> tuple[JobRuntimeResources, ...]:
    """Build one configuration's work-unit resources with shared providers.

    This function performs no tensor loading.  HDF5 files are opened only when a
    primitive materialization request is executed later. Configuration-invariant
    providers, source handlers, and resource bindings are prepared exactly once.
    """

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    if not isinstance(plan, ConcordPlan):
        raise TypeError("plan must be ConcordPlan")
    from .flavor_equivalence import validate_numerical_flavor_sources

    validate_numerical_flavor_sources(spec, configuration=configuration)
    if not isinstance(configuration, int) or isinstance(configuration, bool):
        raise TypeError("configuration must be an integer")
    if configuration not in spec.configurations:
        raise JobDataError(
            f"configuration {configuration} is not selected by this job"
        )
    if isinstance(time_contexts, (str, bytes)) or not isinstance(
        time_contexts,
        Sequence,
    ):
        raise TypeError("time_contexts must be a sequence of contexts")

    validate_plan_for_job(plan, spec)
    declared_contexts = {context.id: context for context in spec.time_contexts}
    contexts = tuple(
        _resolve_time_context(declared_contexts, value) for value in time_contexts
    )
    requested_device = spec.run.device if device is None else device
    if execution_plan is None:
        execution_plan = resolve_runtime_device(
            plan.physical_plan,
            requested_device,
            hardware=hardware,
            policy=budget_policy,
        )
    else:
        if execution_plan.estimate != estimate_execution_memory(plan.physical_plan):
            raise ValueError("execution resource plan belongs to a different physical plan")
        if device is not None:
            requested = torch.device(device)
            if requested != execution_plan.torch_device:
                raise ValueError("device override disagrees with supplied execution plan")
    runtime_device = execution_plan.torch_device
    storage_device = execution_plan.storage_torch_device
    if hdf5_source_handler is not None:
        if not isinstance(hdf5_source_handler, HDF5TensorSourceHandler):
            raise TypeError(
                "hdf5_source_handler must be an HDF5TensorSourceHandler or None"
            )
        if (
            hdf5_file_pool is not None
            and hdf5_source_handler.file_pool is not hdf5_file_pool
        ):
            raise ValueError(
                "shared HDF5 source handler belongs to a different file pool"
            )

    eigenvector_spec = spec.data.eigenvectors
    if isinstance(eigenvector_spec, HDF5EigenvectorSpec):
        kernel_cache = (
            HDF5KernelStore(
                spec.data.kernel_cache.directory,
                configuration=configuration,
                packing=spec.data.kernel_cache.packing,
            )
            if spec.data.kernel_cache is not None
            else None
        )
        kernel_provider: KernelMaterializationProvider = HDF5DistillationKernelProvider(
            eigenvector_spec,
            configuration=configuration,
            spatial_extent=tuple(spec.lattice.spatial_extent),
            temporal_extent=spec.lattice.temporal_extent,
            mode_size=plan.config.mode_size,
            # Host-staged execution must materialize complete primitives in the
            # same memory domain as the physical buffer pool. Contraction tiles
            # are copied to ``runtime_device`` later by the numerical executor.
            device=storage_device,
            gauge_spec=spec.data.gauge_links,
            file_pool=hdf5_file_pool,
            kernel_cache=kernel_cache,
        )
    elif isinstance(eigenvector_spec, CustomEigenvectorSpec):
        raise JobDataError(
            "custom eigenvector factories are not yet wired into the V1 built-in job "
            "data provider; use the low-level runtime provider API for this job"
        )
    else:  # pragma: no cover - discriminated schema makes this unreachable
        raise TypeError(type(eigenvector_spec).__name__)

    propagation_sources: dict[str, ExternalTensorSource] = {}
    for resource_id, recipe in plan.required_propagation_resources.items():
        if recipe.is_generalized:
            source = _generalized_propagation_source(
                spec,
                recipe,
                resource_id=resource_id,
                configuration=configuration,
            )
        else:
            source = _ordinary_propagation_source(
                spec,
                recipe,
                resource_id=resource_id,
                configuration=configuration,
            )
        propagation_sources[resource_id] = source

    kernel_providers = {_KERNEL_PROVIDER_ID: kernel_provider}
    source_handlers = {
        "hdf5": hdf5_source_handler or HDF5TensorSourceHandler(hdf5_file_pool)
    }
    binding_template = prepare_resource_binding_template(
        plan.physical_plan,
        propagation_sources=propagation_sources,
        kernel_provider_id=_KERNEL_PROVIDER_ID,
        spin_basis=plan.config.spin_basis,
    )
    operator_time_contexts = tuple(
        OperatorTimeContext(dict(context.operator_times)) for context in contexts
    )
    binding_plans = tuple(
        binding_template.bind(operator_times)
        for operator_times in operator_time_contexts
    )
    return tuple(
        JobRuntimeResources(
            configuration=configuration,
            time_context=context,
            operator_times=operator_times,
            propagation_sources=propagation_sources,
            kernel_providers=kernel_providers,
            source_handlers=source_handlers,
            kernel_provider_id=_KERNEL_PROVIDER_ID,
            storage_device=storage_device,
            device=runtime_device,
            execution_plan=execution_plan,
            binding_plan=binding_plan,
        )
        for context, operator_times, binding_plan in zip(
            contexts,
            operator_time_contexts,
            binding_plans,
            strict=True,
        )
    )


def build_job_runtime_resources(
    spec: JobSpec,
    plan: ConcordPlan,
    *,
    configuration: int,
    time_context: ExpandedTimeContext | str,
    device: str | torch.device | None = None,
    hardware: HardwareProfile | None = None,
    budget_policy: HardwareBudgetPolicy | None = None,
    execution_plan: DeviceExecutionPlan | None = None,
    hdf5_file_pool: HDF5FilePool | None = None,
    hdf5_source_handler: HDF5TensorSourceHandler | None = None,
) -> JobRuntimeResources:
    """Translate one job work unit into exact runtime providers/sources."""

    return _build_job_configuration_runtime_resources(
        spec,
        plan,
        configuration=configuration,
        time_contexts=(time_context,),
        device=device,
        hardware=hardware,
        budget_policy=budget_policy,
        execution_plan=execution_plan,
        hdf5_file_pool=hdf5_file_pool,
        hdf5_source_handler=hdf5_source_handler,
    )[0]


__all__ = [
    "JobDataError",
    "HDF5DistillationKernelProvider",
    "JobRuntimeResources",
    "build_job_runtime_resources",
]
