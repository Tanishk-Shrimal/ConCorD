"""Small NumPy/HDF5 inputs shared by independent correlator references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import opt_einsum as oe


PERAMBULATOR_ORDER = "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
EIGENVECTOR_AXES = ("time", "z", "y", "x", "color", "mode")


def baryon_elemental(eigenvectors: np.ndarray, mode_size: int) -> np.ndarray:
    modes = np.asarray(eigenvectors[..., :mode_size], dtype=np.complex128)
    if modes.ndim != 5 or modes.shape[3] != 3:
        raise ValueError("one LapH eigenvector timeslice must have shape (z,y,x,3,Nv)")
    flattened = modes.reshape(-1, 3, mode_size)
    epsilon = np.zeros((3, 3, 3), dtype=np.complex128)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1
    elemental = np.zeros((mode_size,) * 3, dtype=np.complex128)
    for start in range(0, len(flattened), 128):
        block = flattened[start:start + 128]
        elemental += oe.contract(
            "sai,sbj,sck,abc->ijk", block, block, block, epsilon, optimize="greedy"
        )
    return elemental


def hdf5_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_eigenvector_times(
    path: Path, dataset_name: str, times: tuple[int, ...], mode_size: int
) -> tuple[dict[int, np.ndarray], tuple[int, int, int, int]]:
    with h5py.File(path, "r") as handle:
        dataset = handle[dataset_name]
        axes = tuple(json.loads(hdf5_text(dataset.attrs.get("axes", "[]"))))
        if axes != EIGENVECTOR_AXES:
            raise ValueError(f"eigenvector axes {axes} != {EIGENVECTOR_AXES}")
        if dataset.ndim != 6 or dataset.shape[4] != 3 or dataset.shape[5] < mode_size:
            raise ValueError(f"invalid eigenvector shape {dataset.shape} for Nv={mode_size}")
        if not np.issubdtype(dataset.dtype, np.complexfloating):
            raise ValueError("eigenvectors must be complex")
        temporal_extent, z_extent, y_extent, x_extent, _, _ = dataset.shape
        if any(time < 0 or time >= temporal_extent for time in times):
            raise ValueError("requested time is outside the eigenvector lattice")
        slices = {time: np.asarray(dataset[time, ..., :mode_size]) for time in times}
    return slices, (temporal_extent, x_extent, y_extent, z_extent)


def read_propagators(
    paths: Mapping[str, Path], times: tuple[int, ...], mode_size: int, temporal_extent: int
) -> dict[tuple[str, int, int], np.ndarray]:
    gamma4 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    blocks: dict[tuple[str, int, int], np.ndarray] = {}
    for flavor, path in paths.items():
        with h5py.File(path, "r") as handle:
            for source_time in times:
                dataset = handle[f"perambulators/src_t_{source_time}/data"]
                if dataset.shape[:3] != (temporal_extent, 4, 4):
                    raise ValueError(f"invalid perambulator shape {dataset.shape}")
                if dataset.shape[3] < mode_size or dataset.shape[4] < mode_size:
                    raise ValueError(f"perambulator has fewer than {mode_size} modes")
                if hdf5_text(dataset.attrs.get("dimension_order", "")) != PERAMBULATOR_ORDER:
                    raise ValueError(f"invalid dimension_order in {dataset.name}")
                for sink_time in times:
                    raw = np.asarray(
                        dataset[sink_time, :, :, :mode_size, :mode_size],
                        dtype=np.complex128,
                    )
                    blocks[(flavor, sink_time, source_time)] = raw.transpose(3, 1, 2, 0) @ gamma4
    return blocks
