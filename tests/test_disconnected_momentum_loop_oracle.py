"""Synthetic end-to-end gate for the independent momentum-loop validator."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np

from tools.validate_disconnected_momentum_loop import validate


def _write_inputs(
    eigenvector_path: Path,
    perambulator_path: Path,
    *,
    generator: np.random.Generator,
) -> None:
    temporal_extent, spatial_x, mode_size = 2, 3, 3
    eigenvectors = np.empty(
        (temporal_extent, 1, 1, spatial_x, 3, mode_size),
        dtype=np.complex128,
    )
    for time in range(temporal_extent):
        values = generator.standard_normal((spatial_x * 3, mode_size))
        values = values + 1j * generator.standard_normal((spatial_x * 3, mode_size))
        orthonormal, _ = np.linalg.qr(values)
        eigenvectors[time, 0, 0] = orthonormal.reshape(spatial_x, 3, mode_size)

    with h5py.File(eigenvector_path, "w") as handle:
        dataset = handle.create_dataset("eigenvectors", data=eigenvectors)
        dataset.attrs["axes"] = json.dumps(
            ["time", "z", "y", "x", "color", "mode"]
        )

    with h5py.File(perambulator_path, "w") as handle:
        perambulators = handle.create_group("perambulators")
        for source_time in range(temporal_extent):
            values = generator.standard_normal(
                (temporal_extent, 4, 4, mode_size, mode_size)
            )
            values = values + 1j * generator.standard_normal(values.shape)
            dataset = perambulators.create_group(f"src_t_{source_time}").create_dataset(
                "data",
                data=values,
            )
            dataset.attrs["dimension_order"] = (
                "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
            )


def test_nonzero_momentum_loop_matches_independent_full_job_oracle(tmp_path):
    generator = np.random.default_rng(20260924)
    perambulator_path = tmp_path / "perambulators.h5"
    eigenvector_path = tmp_path / "eigenvectors.h5"
    output_dir = tmp_path / "validation"
    _write_inputs(
        eigenvector_path,
        perambulator_path,
        generator=generator,
    )

    result = validate(
        Namespace(
            perambulator=perambulator_path,
            eigenvectors=eigenvector_path,
            eigenvector_dataset="/eigenvectors",
            output_dir=output_dir,
            configuration=17,
            rtol=1e-12,
            atol=1e-12,
        )
    )

    assert result["passed"]
    assert result["executed_work_units"] == 2
    assert result["max_opposite_momentum_separation"] > 1.0
    assert result["max_difference_if_phase_sign_is_reversed"] > 1.0
