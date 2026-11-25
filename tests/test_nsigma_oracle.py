"""End-to-end checks of the independent raw proton-sigma oracle."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.validate_nsigma_correlator import SIGMA_BASIS, validate


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
    path: Path,
    generator: np.random.Generator,
    *,
    suppress_equal_time: bool,
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


@pytest.mark.parametrize("suppress_equal_time", [False, True])
def test_nsigma_matches_independent_raw_wick_reference(
    tmp_path: Path, suppress_equal_time: bool
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
        configuration=23,
        rtol=1e-11,
        atol=1e-11,
    )

    assert report["passed"]
    assert not report["vacuum_subtracted"]
    assert report["max_reference_magnitude"] > 1e-5
    assert report["max_per_time_scaled"] < 1e-11
    with h5py.File(output_dir / "independent_reference.h5", "r") as handle:
        order = [tuple(pair) for pair in json.loads(handle.attrs["component_order"])]
        components = np.asarray(handle["component_values"])[0]
        weights = {name: weight for name, _, weight in SIGMA_BASIS}
        reconstructed = sum(
            (weights[sink] * weights[source] * component
             for (sink, source), component in zip(order, components, strict=True)),
            np.zeros((4, 4), dtype=np.complex128),
        )
        np.testing.assert_allclose(reconstructed, handle["values"][0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            handle["values"][:], handle["cross_time"][:] + handle["same_timeslice"][:],
            rtol=1e-12, atol=1e-12,
        )
        if suppress_equal_time:
            np.testing.assert_allclose(handle["same_timeslice"][:], 0, rtol=0, atol=1e-12)
            np.testing.assert_allclose(handle["vacuum_vacuum"][:], 0, rtol=0, atol=1e-12)
        else:
            assert np.max(np.abs(handle["vacuum_vacuum"][:])) > 1e-5
            assert any(
                sink != source and np.max(np.abs(component)) > 1e-5
                for (sink, source), component in zip(order, components, strict=True)
            )
