"""Independent two-pion Wick and isospin gates on tiny synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.validate_npipi_correlator import PION_PAIR_BASIS, validate


MODE_SIZE = 3
TEMPORAL_EXTENT = 2


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


def _write_perambulators(
    path: Path, generator: np.random.Generator, *, suppress_equal_time: bool
) -> None:
    gamma4 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("perambulators")
        for source_time in range(TEMPORAL_EXTENT):
            data = np.empty((TEMPORAL_EXTENT, 4, 4, MODE_SIZE, MODE_SIZE), dtype=np.complex128)
            for sink_time in range(TEMPORAL_EXTENT):
                canonical = generator.standard_normal((MODE_SIZE, 4, MODE_SIZE, 4))
                canonical = canonical + 1j * generator.standard_normal(canonical.shape)
                if sink_time == source_time and suppress_equal_time:
                    canonical.fill(0)
                data[sink_time] = (canonical @ gamma4).transpose(3, 1, 2, 0)
            dataset = group.create_group(f"src_t_{source_time}").create_dataset("data", data=data)
            dataset.attrs["dimension_order"] = (
                "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
            )


def test_isoscalar_pion_pair_has_cartesian_isospin_coefficients() -> None:
    weights = {name: weight for name, _, weight in PION_PAIR_BASIS}
    charged = 1 / np.sqrt(3)
    assert weights["plus_minus"] == weights["minus_plus"] == charged
    assert weights["neutral_uu_uu"] == weights["neutral_dd_dd"] == charged / 2
    assert weights["neutral_uu_dd"] == weights["neutral_dd_uu"] == -charged / 2


def test_isoscalar_pion_pair_is_su2_invariant() -> None:
    generator = np.random.default_rng(20260930)
    coordinates = generator.standard_normal(4)
    coordinates /= np.linalg.norm(coordinates)
    alpha = coordinates[0] + 1j * coordinates[1]
    beta = coordinates[2] + 1j * coordinates[3]
    transform = np.array([
        [alpha, beta],
        [-beta.conjugate(), alpha.conjugate()],
    ])

    def pair_value(bar_fields: np.ndarray, fields: np.ndarray) -> complex:
        bilinears = [np.outer(bar_fields[index], fields[index]) for index in range(2)]
        first_bilinear, second_bilinear = bilinears
        neutral_first = (first_bilinear[0, 0] - first_bilinear[1, 1]) / np.sqrt(2)
        neutral_second = (second_bilinear[0, 0] - second_bilinear[1, 1]) / np.sqrt(2)
        return (
            first_bilinear[1, 0] * second_bilinear[0, 1]
            + first_bilinear[0, 1] * second_bilinear[1, 0]
            + neutral_first * neutral_second
        ) / np.sqrt(3)

    bar_fields = generator.standard_normal((2, 2)) + 1j * generator.standard_normal((2, 2))
    fields = generator.standard_normal((2, 2)) + 1j * generator.standard_normal((2, 2))
    rotated_bar = bar_fields @ transform.conj().T
    rotated_fields = fields @ transform.T
    np.testing.assert_allclose(
        pair_value(bar_fields, fields),
        pair_value(rotated_bar, rotated_fields),
        rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize(
    ("channel", "suppress_equal_time"),
    [("single_charge", False), ("isoscalar", False), ("isoscalar", True)],
)
def test_npipi_matches_independent_wick_reference(
    tmp_path: Path, channel: str, suppress_equal_time: bool
) -> None:
    generator = np.random.default_rng(20260929)
    eigenvector_path = tmp_path / "eigenvectors.h5"
    up_path = tmp_path / "up.h5"
    down_path = tmp_path / "down.h5"
    _write_eigenvectors(eigenvector_path, generator)
    _write_perambulators(up_path, generator, suppress_equal_time=suppress_equal_time)
    _write_perambulators(down_path, generator, suppress_equal_time=suppress_equal_time)

    output_dir = tmp_path / "validation"
    report = validate(
        perambulator=up_path,
        down_perambulator=down_path,
        eigenvectors=eigenvector_path,
        output_dir=output_dir,
        mode_size=MODE_SIZE,
        source_time=0,
        sink_times=(1,),
        configuration=29,
        channel=channel,
        rtol=1e-10,
        atol=1e-10,
    )

    assert report["passed"]
    assert report["max_reference_magnitude"] > 1e-5
    assert report["max_per_time_scaled"] < 1e-10
    with h5py.File(output_dir / "independent_reference.h5", "r") as handle:
        np.testing.assert_allclose(
            handle["values"][:],
            handle["cross_time"][:] + handle["same_timeslice"][:],
            rtol=1e-10, atol=1e-10,
        )
        if suppress_equal_time:
            np.testing.assert_allclose(handle["same_timeslice"][:], 0, rtol=0, atol=1e-10)
        else:
            assert np.max(np.abs(handle["same_timeslice"][:])) > 1e-5
        if channel == "isoscalar":
            order = [tuple(pair) for pair in json.loads(handle.attrs["component_order"])]
            components = np.asarray(handle["component_values"])[0]
            weights = {name: weight for name, _, weight in PION_PAIR_BASIS}
            reconstructed = sum(
                (weights[sink] * weights[source] * component
                 for (sink, source), component in zip(order, components, strict=True)),
                np.zeros((4, 4), dtype=np.complex128),
            )
            np.testing.assert_allclose(reconstructed, handle["values"][0], rtol=1e-10, atol=1e-10)
            wrong_weights = {
                name: -weight if name.startswith("neutral_") else weight
                for name, weight in weights.items()
            }
            wrong_sign = sum(
                (wrong_weights[sink] * wrong_weights[source] * component
                 for (sink, source), component in zip(order, components, strict=True)),
                np.zeros((4, 4), dtype=np.complex128),
            )
            assert np.max(np.abs(reconstructed - wrong_sign)) > 1e-5
