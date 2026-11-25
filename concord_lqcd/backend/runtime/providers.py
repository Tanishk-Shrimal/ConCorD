"""Pure mathematical tensor materialization provider interface."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from numbers import Integral
from typing import Dict, Mapping, Tuple, Any, Sequence
import numpy as np
import torch

from .context import ExecutionContext


def _complex_matrix(payload: dict[str, list[list[float]]]) -> np.ndarray:
    return np.asarray(payload["real"], dtype=np.float64) + 1j * np.asarray(payload["imag"], dtype=np.float64)


class TensorProvider(ABC):
    """Abstract interface for mathematical tensor resource materialization.

    DETERMINISM INVARIANT: Given identical provider state and identical
    arguments, TensorProvider must always return identical tensors.
    Must never perform file IO or allocate persistent caches.
    """

    @abstractmethod
    def eigenvector(
        self,
        flavor: str | None,
        is_anti: bool,
        field_attributes: Mapping[str, Any],
        is_sink: bool,
        context: ExecutionContext,
    ) -> torch.Tensor:
        """Return fields with axes [batch, site, color, mode] for this view."""
        ...

    @abstractmethod
    def phase(self, momentum: Tuple[int, int, int], context: ExecutionContext) -> torch.Tensor | None:
        """Return a rank-one phase over exactly the fields' ordered sites.

        Spatial tiling must select the same sites in fields and phases.
        ``None`` omits this factor as an identity; the provider owns its meaning.
        """
        ...

    @abstractmethod
    def gauge_link(self, direction: str, context: ExecutionContext) -> torch.Tensor:
        """Returns the spatial gauge link matrix."""
        ...

    @abstractmethod
    def color_delta(self, Nc: int, dtype: torch.dtype, context: ExecutionContext) -> torch.Tensor:
        """Returns the SU(3) color delta identity matrix."""
        ...

    @abstractmethod
    def color_epsilon(self, dtype: torch.dtype, context: ExecutionContext) -> torch.Tensor:
        """Returns the 3x3x3 SU(3) Levi-Civita color epsilon tensor."""
        ...

    @abstractmethod
    def spin_matrix(self, matrix_payload: dict[str, Any], dtype: torch.dtype, context: ExecutionContext) -> torch.Tensor:
        """Returns the Dirac spin matrix tensor."""
        ...


class DistillationTensorProvider(TensorProvider):
    """Pure mathematical tensor materialization provider for distillation data.

    Dynamic derivatives of flattened fields require ``spatial_extent`` or
    ``context.lat_shape`` in (Lx, Ly, Lz) order. Sites are flattened in z-y-x
    order, with x varying fastest. Unflattened derivative inputs may instead
    use [batch, z, y, x, color, mode].
    """

    def __init__(
        self,
        V_t_snk: Mapping[str, torch.Tensor],
        V_t_src: Mapping[str, torch.Tensor],
        momenta_phases: Mapping[Tuple[int, int, int], torch.Tensor],
        gauge_links: Mapping[str, torch.Tensor] | None = None,
        *,
        spatial_extent: Tuple[int, int, int] | None = None,
    ):
        self.V_t_snk = V_t_snk
        self.V_t_src = V_t_src
        self.momenta_phases = momenta_phases
        self.gauge_links = gauge_links or {}
        self.spatial_extent = (
            self._validate_spatial_extent(spatial_extent)
            if spatial_extent is not None else None
        )

    @staticmethod
    def _validate_spatial_extent(extent) -> Tuple[int, int, int]:
        if (
            not isinstance(extent, (tuple, list)) or len(extent) != 3
            or any(isinstance(value, bool) or not isinstance(value, Integral)
                   or value <= 0 for value in extent)
        ):
            raise ValueError("spatial extent must contain three positive integers (Lx, Ly, Lz)")
        return tuple(int(value) for value in extent)

    @staticmethod
    def _derivative_gauge_grid(link: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Normalize links to the field's batch/z/y/x/color/color axes."""
        if not isinstance(link, torch.Tensor):
            raise TypeError("derivative gauge links must be torch tensors")
        batch, lz, ly, lx, colors, _ = grid.shape
        if tuple(link.shape[-2:]) != (colors, colors):
            raise ValueError("derivative gauge links must match the field's color axes")
        sites = lz * ly * lx
        if link.ndim == 2:
            link = link.reshape(1, 1, 1, 1, colors, colors)
        elif link.ndim == 3 and link.shape[0] == sites:
            link = link.reshape(1, lz, ly, lx, colors, colors)
        elif link.ndim == 4 and link.shape[1] == sites:
            link = link.reshape(link.shape[0], lz, ly, lx, colors, colors)
        elif link.ndim == 5:
            link = link.unsqueeze(0)
        elif link.ndim != 6:
            raise ValueError("derivative gauge links require constant, site, or spatial grid axes")
        expected = (batch, lz, ly, lx, colors, colors)
        if any(actual not in (1, wanted)
               for actual, wanted in zip(link.shape, expected, strict=True)):
            raise ValueError("derivative gauge-link batch/spatial axes disagree with the field")
        dtype = (
            torch.promote_types(grid.dtype, link.dtype)
            if not grid.is_complex() and link.is_complex() else grid.dtype
        )
        return link.to(device=grid.device, dtype=dtype).expand(expected)

    def _apply_dynamic_derivatives(
        self,
        V: torch.Tensor,
        derivs: Sequence[str],
        context: ExecutionContext | None = None,
    ) -> torch.Tensor:
        """Applies symmetric lattice spatial covariant derivatives dynamically to an eigenvector tensor."""
        extent = self.spatial_extent
        if context is not None and context.lat_shape is not None:
            context_extent = self._validate_spatial_extent(context.lat_shape)
            if extent is not None and extent != context_extent:
                raise ValueError("provider spatial extent disagrees with context.lat_shape")
            extent = context_extent
        if V.ndim == 4:
            if extent is None:
                raise ValueError("dynamic derivatives of flattened fields require spatial extent")
            lx, ly, lz = extent
            if V.shape[1] != lx * ly * lz:
                raise ValueError("field site count disagrees with the spatial extent")
            current_V = V.reshape(V.shape[0], lz, ly, lx, *V.shape[-2:])
        elif V.ndim == 6:
            if extent is not None and tuple(reversed(extent)) != tuple(V.shape[1:4]):
                raise ValueError("field spatial grid disagrees with the spatial extent")
            current_V = V
        else:
            raise ValueError("dynamic derivatives require [batch,site,color,mode] or spatial grid fields")
        dir_to_axis = {"z": 1, "y": 2, "x": 3}

        for d in derivs:
            if not d:
                continue
            if isinstance(d, tuple) and len(d) == 2:
                axis_name, is_bwd = d
                sign_str = "-" if is_bwd else "+"
                axis_name = axis_name.lower()
            elif isinstance(d, str):
                sign_str = d[0] if d[0] in ("+", "-") else "+"
                axis_name = d[1:].lower() if d[0] in ("+", "-") else d.lower()
            else:
                raise ValueError(f"invalid spatial derivative direction {d!r}")

            if axis_name not in dir_to_axis:
                raise ValueError(f"unsupported spatial derivative direction {d!r}")

            axis = dir_to_axis[axis_name]

            v_fwd = torch.roll(current_V, shifts=-1, dims=axis)
            v_bwd = torch.roll(current_V, shifts=+1, dims=axis)

            if axis_name in self.gauge_links:
                U = self._derivative_gauge_grid(self.gauge_links[axis_name], current_V)
                v_fwd = torch.matmul(U, v_fwd.to(dtype=U.dtype))
                U_bwd = torch.roll(U, shifts=+1, dims=axis).conj().transpose(-1, -2)
                v_bwd = torch.matmul(U_bwd, v_bwd.to(dtype=U.dtype))
            else:
                warnings.warn(
                    f"Gauge link configuration for direction {axis_name!r} not provided to DistillationTensorProvider. "
                    f"Evaluating spatial derivative with unit gauge links (free field central difference).",
                    UserWarning,
                    stacklevel=2,
                )

            diff = 0.5 * (v_fwd - v_bwd)
            if sign_str == "-":
                diff = -diff

            current_V = diff

        return current_V.reshape(V.shape)

    def _lookup_flavor(
        self,
        flavor: str | None,
        V_fields: Mapping[str, torch.Tensor],
        derivatives: Any = (),
        context: ExecutionContext | None = None,
    ) -> torch.Tensor:
        if isinstance(derivatives, Mapping):
            derivs = derivatives.get("derivative", derivatives.get("derivatives", []))
        else:
            derivs = derivatives or []

        formatted_derivs = []
        for d in derivs:
            if isinstance(d, tuple):
                axis_name, is_bwd = d
                formatted_derivs.append(f"{'-' if is_bwd else '+'}{axis_name}")
            else:
                formatted_derivs.append(str(d))

        lookup_key = f"{flavor}_{'_'.join(formatted_derivs)}" if formatted_derivs and flavor else flavor

        if lookup_key is not None and lookup_key in V_fields:
            return V_fields[lookup_key]

        raw_V = None
        if flavor is not None and flavor in V_fields:
            raw_V = V_fields[flavor]
        elif None in V_fields:
            raw_V = V_fields[None]
        elif V_fields:
            raw_V = next(iter(V_fields.values()))

        if raw_V is not None and derivs:
            warnings.warn(
                f"Pre-computed derivative eigenvector file {lookup_key!r} not found in eigenvector fields. "
                f"Falling back to dynamic spatial derivative computation on raw {flavor!r} eigenvectors.",
                UserWarning,
                stacklevel=2,
            )
            return self._apply_dynamic_derivatives(raw_V, derivs, context)

        if raw_V is not None:
            return raw_V

        raise ValueError("No eigenvectors available in TensorProvider.")

    def eigenvector(
        self,
        flavor: str | None,
        is_anti: bool = False,
        derivatives: tuple[Any, ...] | list[Any] = (),
        is_sink: bool = True,
        context: ExecutionContext | None = None,
        field_attributes: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        if field_attributes and "derivatives" in field_attributes:
            derivatives = field_attributes["derivatives"]
        elif field_attributes and "derivative" in field_attributes:
            derivatives = field_attributes["derivative"]

        V_fields = self.V_t_snk if is_sink else self.V_t_src
        V = self._lookup_flavor(flavor, V_fields, derivatives, context)
        if is_anti:
            V = V.conj()

        if context is not None:
            if is_sink and V.shape[0] == 1 and len(context.t_snk_list) != 1:
                V = V.expand(len(context.t_snk_list), *V.shape[1:])
            elif not is_sink:
                V = V.expand(len(context.t_snk_list), *V.shape[1:])

        return V

    def phase(self, momentum: Tuple[int, int, int], context: ExecutionContext) -> torch.Tensor | None:
        return self.momenta_phases.get(momentum)

    def gauge_link(self, direction: str, context: ExecutionContext) -> torch.Tensor:
        return torch.eye(3, dtype=context.complex_dtype, device=context.device)

    def color_delta(self, Nc: int, dtype: torch.dtype, context: ExecutionContext) -> torch.Tensor:
        return torch.eye(Nc, dtype=dtype, device=context.device)

    def color_epsilon(self, dtype: torch.dtype, context: ExecutionContext) -> torch.Tensor:
        eps = np.zeros((3, 3, 3), dtype=np.complex128)
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
        eps[2, 1, 0] = eps[0, 2, 1] = eps[1, 0, 2] = -1
        return torch.tensor(eps, dtype=dtype, device=context.device)

    def spin_matrix(self, matrix_payload: dict[str, Any], dtype: torch.dtype, context: ExecutionContext) -> torch.Tensor:
        return torch.tensor(_complex_matrix(matrix_payload), dtype=dtype, device=context.device)
