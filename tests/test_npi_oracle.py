"""End-to-end checks of the independent highest-weight N-pion oracle."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.validate_npi_correlator import ISOSPIN_HALF_BASIS, validate


MODE_SIZE = 3
TEMPORAL_EXTENT = 2


def test_npi_preserves_existing_operator_source(tmp_path: Path) -> None:
    operator_source = tmp_path / "operators.py"
    original = "existing operator source\n"
    operator_source.write_text(original, encoding="utf-8")

    with pytest.raises(FileExistsError, match="operators.py"):
        validate(
            perambulator=tmp_path / "missing_perambulator.h5",
            eigenvectors=tmp_path / "missing_eigenvectors.h5",
            output_dir=tmp_path,
            mode_size=MODE_SIZE,
            sink_times=(1,),
        )

    assert operator_source.read_text(encoding="utf-8") == original


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
    suppress_equal_time: bool = False,
    suppress_cross_time: bool = False,
) -> None:
    gamma4 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("perambulators")
        for source_time in range(TEMPORAL_EXTENT):
            data = np.empty((TEMPORAL_EXTENT, 4, 4, MODE_SIZE, MODE_SIZE), dtype=np.complex128)
            for sink_time in range(TEMPORAL_EXTENT):
                canonical = generator.standard_normal((MODE_SIZE, 4, MODE_SIZE, 4))
                canonical = canonical + 1j * generator.standard_normal(canonical.shape)
                if (sink_time == source_time and suppress_equal_time) or (
                    sink_time != source_time and suppress_cross_time
                ):
                    canonical.fill(0)
                data[sink_time] = (canonical @ gamma4).transpose(3, 1, 2, 0)
            dataset = group.create_group(f"src_t_{source_time}").create_dataset("data", data=data)
            dataset.attrs["dimension_order"] = (
                "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"
            )


@pytest.mark.parametrize("sector", ["full", "cross_time", "same_timeslice"])
def test_npi_matches_independent_wick_reference(tmp_path: Path, sector: str) -> None:
    generator = np.random.default_rng(20260928)
    eigenvector_path = tmp_path / "eigenvectors.h5"
    up_path = tmp_path / "up.h5"
    down_path = tmp_path / "down.h5"
    _write_eigenvectors(eigenvector_path, generator)
    _write_perambulators(up_path, generator)
    _write_perambulators(
        down_path,
        generator,
        suppress_equal_time=sector == "cross_time",
        suppress_cross_time=sector == "same_timeslice",
    )
    output_dir = tmp_path / "validation"

    report = validate(
        perambulator=up_path,
        down_perambulator=down_path,
        eigenvectors=eigenvector_path,
        output_dir=output_dir,
        mode_size=MODE_SIZE,
        source_time=0,
        sink_times=(1,),
        configuration=17,
        rtol=1e-11,
        atol=1e-11,
    )

    assert report["passed"]
    assert report["terms_per_time"] == 12
    assert report["same_timeslice_terms_per_time"] == 6
    assert report["max_reference_magnitude"] > 1e-5
    assert report["minimum_reference_magnitude"] > 1e-5
    assert report["max_per_time_scaled"] < 1e-11
    with h5py.File(output_dir / "independent_reference.h5", "r") as handle:
        total = np.asarray(handle["values"])
        cross_time = np.asarray(handle["cross_time"])
        same_timeslice = np.asarray(handle["same_timeslice"])
        np.testing.assert_allclose(total, cross_time + same_timeslice, rtol=1e-12, atol=1e-12)
        if sector == "cross_time":
            np.testing.assert_allclose(same_timeslice, 0, rtol=0, atol=1e-12)
        elif sector == "same_timeslice":
            np.testing.assert_allclose(cross_time, 0, rtol=0, atol=1e-12)


def test_isospin_half_matches_independent_flavor_expansion(tmp_path: Path) -> None:
    generator = np.random.default_rng(20260929)
    eigenvector_path = tmp_path / "eigenvectors.h5"
    up_path = tmp_path / "up.h5"
    down_path = tmp_path / "down.h5"
    _write_eigenvectors(eigenvector_path, generator)
    _write_perambulators(up_path, generator)
    _write_perambulators(down_path, generator)

    output_dir = tmp_path / "isospin_half"
    report = validate(
        perambulator=up_path,
        down_perambulator=down_path,
        eigenvectors=eigenvector_path,
        output_dir=output_dir,
        mode_size=MODE_SIZE,
        source_time=0,
        sink_times=(1,),
        configuration=19,
        channel="isospin_half",
        rtol=1e-11,
        atol=1e-11,
    )

    assert report["passed"]
    assert report["max_reference_magnitude"] > 1e-5
    assert report["max_per_time_scaled"] < 1e-11
    with h5py.File(output_dir / "independent_reference.h5", "r") as handle:
        order = [tuple(pair) for pair in json.loads(handle.attrs["component_order"])]
        components = np.asarray(handle["component_values"])[0]
        weights = {name: weight for name, _, _, weight in ISOSPIN_HALF_BASIS}
        reconstructed = sum(
            (weights[sink] * weights[source] * component
             for (sink, source), component in zip(order, components, strict=True)),
            np.zeros((4, 4), dtype=np.complex128),
        )
        np.testing.assert_allclose(reconstructed, handle["values"][0], rtol=1e-12, atol=1e-12)
        assert any(
            sink != source and np.max(np.abs(component)) > 1e-5
            for (sink, source), component in zip(order, components, strict=True)
        )
        wrong_weights = {name: -weight if name == "n_piplus" else weight for name, weight in weights.items()}
        wrong_sign = sum(
            (wrong_weights[sink] * wrong_weights[source] * component
             for (sink, source), component in zip(order, components, strict=True)),
            np.zeros((4, 4), dtype=np.complex128),
        )
        assert np.max(np.abs(reconstructed - wrong_sign)) > 1e-5
