"""Independent Delta++ Wick and rest-frame spin-projection gates."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tools.validate_delta_correlator import (
    dp_gamma_matrices,
    project_delta,
    rest_spin_projector,
    validate,
)


TEMPORAL_EXTENT = 3
MODE_SIZE = 3


def _write_eigenvectors(path: Path, generator: np.random.Generator) -> None:
    eigenvectors = np.empty((TEMPORAL_EXTENT, 1, 1, 2, 3, MODE_SIZE), dtype=np.complex128)
    for time in range(TEMPORAL_EXTENT):
        values = generator.standard_normal((6, MODE_SIZE))
        values = values + 1j * generator.standard_normal(values.shape)
        orthonormal, _ = np.linalg.qr(values)
        eigenvectors[time, 0, 0] = orthonormal.reshape(2, 3, MODE_SIZE)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("eigenvectors", data=eigenvectors)
        dataset.attrs["axes"] = json.dumps(["time", "z", "y", "x", "color", "mode"])


def _write_perambulators(path: Path, generator: np.random.Generator) -> None:
    gamma4 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("perambulators")
        for source_time in range(TEMPORAL_EXTENT):
            values = generator.standard_normal(
                (TEMPORAL_EXTENT, MODE_SIZE, 4, MODE_SIZE, 4)
            )
            values = values + 1j * generator.standard_normal(values.shape)
            data = np.empty((TEMPORAL_EXTENT, 4, 4, MODE_SIZE, MODE_SIZE), dtype=np.complex128)
            for sink_time in range(TEMPORAL_EXTENT):
                data[sink_time] = (values[sink_time] @ gamma4).transpose(3, 1, 2, 0)
            dataset = group.create_group(f"src_t_{source_time}").create_dataset("data", data=data)
            dataset.attrs["dimension_order"] = (
                "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
            )


def test_rest_spin_projector_is_idempotent_and_removes_spin_half() -> None:
    spatial, temporal = dp_gamma_matrices()
    projector = rest_spin_projector(spatial)
    flattened = projector.transpose(0, 2, 1, 3).reshape(12, 12)
    np.testing.assert_allclose(flattened @ flattened, flattened, atol=1e-14)
    np.testing.assert_allclose(flattened.conj().T, flattened, atol=1e-14)
    np.testing.assert_allclose(np.trace(flattened), 8, atol=1e-14)
    for column in range(3):
        np.testing.assert_allclose(
            sum(spatial[row] @ projector[row, column] for row in range(3)),
            0,
            atol=1e-14,
        )

    generator = np.random.default_rng(20260929)
    middle = generator.standard_normal((4, 4))
    middle = middle + 1j * generator.standard_normal(middle.shape)
    pure_spin_half = np.asarray([
        [spatial[row] @ middle @ spatial[column] for column in range(3)]
        for row in range(3)
    ])[None, ...]
    projected, positive, negative = project_delta(pure_spin_half, spatial, temporal)
    np.testing.assert_allclose(projected, 0, atol=1e-13)
    np.testing.assert_allclose(positive, 0, atol=1e-13)
    np.testing.assert_allclose(negative, 0, atol=1e-13)

    random_matrix = generator.standard_normal((2, 3, 3, 4, 4))
    random_matrix = random_matrix + 1j * generator.standard_normal(random_matrix.shape)
    projected, positive, negative = project_delta(random_matrix, spatial, temporal)
    identity = np.eye(4, dtype=np.complex128)
    for parity, actual in (((identity + temporal) / 2, positive), ((identity - temporal) / 2, negative)):
        direct = np.asarray([
            sum(
                np.trace(parity @ matrix[row, column] @ projector[column, row])
                for row in range(3)
                for column in range(3)
            )
            for matrix in random_matrix
        ])
        np.testing.assert_allclose(actual, direct, atol=1e-13)
    for matrix in projected:
        for column in range(3):
            np.testing.assert_allclose(
                sum(spatial[row] @ matrix[row, column] for row in range(3)),
                0,
                atol=1e-13,
            )


def test_delta_matches_independent_six_permutation_full_job(tmp_path: Path) -> None:
    generator = np.random.default_rng(20260929)
    eigenvectors = tmp_path / "eigenvectors.h5"
    perambulator = tmp_path / "up.h5"
    output_dir = tmp_path / "validation"
    _write_eigenvectors(eigenvectors, generator)
    _write_perambulators(perambulator, generator)

    report = validate(
        perambulator=perambulator,
        eigenvectors=eigenvectors,
        output_dir=output_dir,
        mode_size=MODE_SIZE,
        source_time=0,
        sink_times=(1, 2),
        configuration=17,
        rtol=1e-11,
        atol=1e-11,
    )

    assert report["passed"]
    assert report["wick_terms_per_channel"] == 6
    assert report["correlator_channels"] == 9
    assert report["max_reference_magnitude"] > 1e-5
    assert report["max_positive_parity_magnitude"] > 1e-5
    assert report["max_negative_parity_magnitude"] > 1e-5
    assert report["max_per_time_scaled"] < 1e-11
    assert report["max_spin_projected_scaled"] < 1e-11
    assert report["max_parity_projected_scaled"] < 1e-11
    with h5py.File(output_dir / "independent_reference.h5", "r") as handle:
        unprojected = np.asarray(handle["unprojected"])
        projected = np.asarray(handle["spin_projected"])
    assert np.max(np.abs(unprojected - projected)) > 1e-5
