"""Immutable runtime execution context for ConCorD backend execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple
import torch


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable payload carrying runtime execution parameters and PyTorch device/dtype settings."""

    device: torch.device
    t_src: int
    t_snk_list: Tuple[int, ...]
    batch_size: int = 12
    n_eigs_dict: Mapping[str, int] | None = None
    lat_shape: Tuple[int, int, int] | None = None
    dtype: torch.dtype = torch.float64
    complex_dtype: torch.dtype = torch.complex128

    @classmethod
    def create(
        cls,
        device: torch.device | str,
        t_src: int,
        t_snk_list: Sequence[int],
        batch_size: int = 12,
        n_eigs_dict: Mapping[str, int] | None = None,
        lat_shape: Tuple[int, int, int] | None = None,
        dtype: torch.dtype = torch.float64,
        complex_dtype: torch.dtype = torch.complex128,
    ) -> ExecutionContext:
        dev = device if isinstance(device, torch.device) else torch.device(device)
        return cls(
            device=dev,
            t_src=t_src,
            t_snk_list=tuple(t_snk_list),
            batch_size=batch_size,
            n_eigs_dict=dict(n_eigs_dict) if n_eigs_dict else None,
            lat_shape=lat_shape,
            dtype=dtype,
            complex_dtype=complex_dtype,
        )
