"""Independent Delta+, Lambda, proton and Delta++ references and their flavor identities."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.validate_baryon_flavor_identities import (
    CONTROL_MINIMUM,
    correlator_pairs,
    operator_terms,
    two_point,
    validate,
)
from tools.validate_delta_correlator import evaluate_reference as delta_pp_reference


TEMPORAL_EXTENT = 3
MODE_SIZE = 4


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
            values = generator.standard_normal((TEMPORAL_EXTENT, MODE_SIZE, 4, MODE_SIZE, 4))
            values = values + 1j * generator.standard_normal(values.shape)
            data = np.empty((TEMPORAL_EXTENT, 4, 4, MODE_SIZE, MODE_SIZE), dtype=np.complex128)
            for sink_time in range(TEMPORAL_EXTENT):
                data[sink_time] = (values[sink_time] @ gamma4).transpose(3, 1, 2, 0)
            dataset = group.create_group(f"src_t_{source_time}").create_dataset("data", data=data)
            dataset.attrs["dimension_order"] = "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"


def _random_antisymmetric_elemental(generator: np.random.Generator) -> np.ndarray:
    vectors = generator.standard_normal((5, 3, MODE_SIZE)) + 1j * generator.standard_normal((5, 3, MODE_SIZE))
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1
    return np.einsum("sai,sbj,sck,abc->ijk", vectors, vectors, vectors, epsilon)


def test_reference_reproduces_delta_tool_and_flavor_identities() -> None:
    generator = np.random.default_rng(31415)
    sink = _random_antisymmetric_elemental(generator)
    source = _random_antisymmetric_elemental(generator)
    shape = (MODE_SIZE, 4, MODE_SIZE, 4)
    propagator = generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    other = generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    operators = operator_terms()
    same = {"u": propagator, "d": propagator, "s": propagator}
    mixed = {"u": propagator, "d": other, "s": propagator}

    delta_pp = delta_pp_reference(sink, source.conj(), propagator)
    tolerance = 1e-13 * np.max(np.abs(delta_pp))
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            generic = two_point(
                operators[f"delta_pp_{i}"], operators[f"delta_pp_{j}"], sink, source.conj(), same
            )
            np.testing.assert_allclose(generic, delta_pp[i - 1, j - 1], rtol=0, atol=tolerance)
            delta_p = two_point(operators[f"delta_p_{i}"], operators[f"delta_p_{j}"], sink, source.conj(), same)
            np.testing.assert_allclose(delta_p, generic, rtol=0, atol=tolerance)
            control = two_point(
                operators[f"delta_p_control_{i}"], operators[f"delta_p_control_{j}"], sink, source.conj(), same
            )
            assert np.max(np.abs(control - generic)) > 1e-3 * np.max(np.abs(generic))
            unbound = two_point(operators[f"delta_p_{i}"], operators[f"delta_p_{j}"], sink, source.conj(), mixed)
            assert np.max(np.abs(unbound - generic)) > 1e-3 * np.max(np.abs(generic))

    proton = two_point(operators["proton"], operators["proton"], sink, source.conj(), same)
    lam = two_point(operators["lambda"], operators["lambda"], sink, source.conj(), same)
    lam_control = two_point(operators["lambda_control"], operators["lambda_control"], sink, source.conj(), same)
    assert np.max(np.abs(proton)) > 1e-3
    np.testing.assert_allclose(lam, proton, rtol=0, atol=1e-13 * np.max(np.abs(proton)))
    assert np.max(np.abs(lam_control - proton)) > 1e-3 * np.max(np.abs(proton))
    assert len(correlator_pairs()) == 30


def test_flavor_identities_and_reference_full_job(tmp_path: Path) -> None:
    generator = np.random.default_rng(31415)
    eigenvectors = tmp_path / "eigenvectors.h5"
    perambulator = tmp_path / "light.h5"
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
    assert report["correlators"] == 30
    assert report["identities_checked"]
    identities = report["identities_concord"]
    assert identities["delta_p_vs_delta_pp"] < 1e-11
    assert identities["lambda_vs_proton"] < 1e-11
    assert identities["delta_p_control_vs_delta_pp"] > CONTROL_MINIMUM
    assert identities["lambda_control_vs_proton"] > CONTROL_MINIMUM
    assert report["reference_selfcheck_vs_delta_pp_tool"] < 1e-12
    assert report["minimum_reference_magnitude"] > 1e-5
    assert report["max_per_time_scaled"] < 1e-11
    assert set(report["max_per_time_scaled_by_family"]) == {
        "delta_pp", "delta_p", "delta_p_control", "proton", "lambda", "lambda_control"
    }
    with h5py.File(output_dir / "independent_reference.h5", "r") as handle:
        assert len(handle) == 30
        assert handle["proton"].shape == (2, 4, 4)


def test_distinct_flavors_match_reference_without_identities(tmp_path: Path) -> None:
    generator = np.random.default_rng(27182)
    eigenvectors = tmp_path / "eigenvectors.h5"
    paths = {flavor: tmp_path / f"{flavor}.h5" for flavor in "uds"}
    _write_eigenvectors(eigenvectors, generator)
    for path in paths.values():
        _write_perambulators(path, generator)
    output_dir = tmp_path / "validation"

    report = validate(
        perambulator=paths["u"],
        down_perambulator=paths["d"],
        strange_perambulator=paths["s"],
        eigenvectors=eigenvectors,
        output_dir=output_dir,
        mode_size=MODE_SIZE,
        source_time=0,
        sink_times=(2,),
        configuration=19,
        rtol=1e-11,
        atol=1e-11,
    )

    assert report["passed"]
    assert not report["identities_checked"]
    assert report["identities_concord"] is None
    assert report["max_per_time_scaled"] < 1e-11
    with h5py.File(output_dir / "independent_reference.h5", "r") as handle:
        delta_p = np.asarray(handle["delta_p_1_1"])
        delta_pp = np.asarray(handle["delta_pp_1_1"])
        lam = np.asarray(handle["lambda"])
        proton = np.asarray(handle["proton"])
    # With distinct u, d and s data the identities fail, so the comparison resolves every flavor.
    assert np.max(np.abs(delta_p - delta_pp)) > 1e-3 * np.max(np.abs(delta_pp))
    assert np.max(np.abs(lam - proton)) > 1e-3 * np.max(np.abs(proton))


def test_refuses_to_overwrite_operator_source(tmp_path: Path) -> None:
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
