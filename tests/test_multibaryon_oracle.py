"""Independent multi-baryon (NN 1S0, NN 3S1, triton) references against ConCorD."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools._independent_laph import baryon_elemental
from tools.validate_multibaryon_correlators import (
    baryon,
    correlator,
    momentum_elemental,
    operator_terms,
    pairing_count,
    pairings,
    term,
    validate,
)


TEMPORAL_EXTENT = 2
MODE_SIZE = 4
# (z, y, x): an extent of 3 along the NN momentum (0, 0, 1) gives complex phases; the lattice is not cubic.
SPATIAL_SHAPE = (3, 1, 2)


def _write_eigenvectors(path: Path, generator: np.random.Generator) -> None:
    eigenvectors = np.empty((TEMPORAL_EXTENT, *SPATIAL_SHAPE, 3, MODE_SIZE), dtype=np.complex128)
    components = int(np.prod(SPATIAL_SHAPE)) * 3
    for time in range(TEMPORAL_EXTENT):
        values = generator.standard_normal((components, MODE_SIZE))
        values = values + 1j * generator.standard_normal(values.shape)
        orthonormal, _ = np.linalg.qr(values)
        eigenvectors[time] = orthonormal.reshape(*SPATIAL_SHAPE, 3, MODE_SIZE)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("eigenvectors", data=eigenvectors)
        dataset.attrs["axes"] = json.dumps(["time", "z", "y", "x", "color", "mode"])


def _write_perambulators(path: Path, generator: np.random.Generator) -> None:
    gamma4 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("perambulators")
        for source_time in range(TEMPORAL_EXTENT):
            data = np.empty((TEMPORAL_EXTENT, 4, 4, MODE_SIZE, MODE_SIZE), dtype=np.complex128)
            for sink_time in range(TEMPORAL_EXTENT):
                canonical = generator.standard_normal((MODE_SIZE, 4, MODE_SIZE, 4))
                canonical = canonical + 1j * generator.standard_normal(canonical.shape)
                data[sink_time] = (canonical @ gamma4).transpose(3, 1, 2, 0)
            dataset = group.create_group(f"src_t_{source_time}").create_dataset("data", data=data)
            dataset.attrs["dimension_order"] = "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"


def _random_slice(generator: np.random.Generator) -> np.ndarray:
    shape = (*SPATIAL_SHAPE, 3, MODE_SIZE)
    return generator.standard_normal(shape) + 1j * generator.standard_normal(shape)


def test_pairing_counts_and_direct_pairing_sign() -> None:
    expected = {"proton": 2, "nn_1s0": 48, "nn_3s1": 144, "triton": 2880}
    for system, count in expected.items():
        operators = operator_terms(system)
        for sink in operators:
            for source in operators:
                assert pairing_count(operators[sink], operators[source]) == count
    signs = dict((tuple(permutation), sign) for permutation, sign in pairings("udu" "udd", "udu" "udd"))
    assert len(signs) == 36
    assert signs[tuple(range(6))] == 1
    assert signs[(3, 1, 2, 0, 4, 5)] == -1
    assert not list(pairings("udu", "udd"))


def test_momentum_elemental_matches_rest_elemental_and_explicit_phase() -> None:
    generator = np.random.default_rng(31415)
    vectors = _random_slice(generator)
    np.testing.assert_allclose(
        momentum_elemental(vectors, (0, 0, 0)), baryon_elemental(vectors, MODE_SIZE), rtol=1e-13, atol=1e-13
    )
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1
    momentum = (1, 0, -1)
    nz, ny, nx = SPATIAL_SHAPE
    explicit = sum(
        np.exp(2j * np.pi * (momentum[0] * x / nx + momentum[1] * y / ny + momentum[2] * z / nz))
        * np.einsum("ai,bj,ck,abc->ijk", vectors[z, y, x], vectors[z, y, x], vectors[z, y, x], epsilon)
        for z in range(nz) for y in range(ny) for x in range(nx)
    )
    np.testing.assert_allclose(momentum_elemental(vectors, momentum), explicit, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(momentum_elemental(vectors, (0, 0, 1)) - momentum_elemental(vectors, (0, 0, -1)))) > 1e-3


def test_reference_anticommutes_baryons() -> None:
    """[P N - N P] C gamma_5 / sqrt(2) at rest vanishes as an operator (C gamma_5 is antisymmetric)."""
    generator = np.random.default_rng(27182)
    rest = (0, 0, 0)
    sink = {rest: momentum_elemental(_random_slice(generator), rest)}
    source = {rest: momentum_elemental(_random_slice(generator), rest)}
    shape = (MODE_SIZE, 4, MODE_SIZE, 4)
    propagators = {
        flavor: generator.standard_normal(shape) + 1j * generator.standard_normal(shape) for flavor in "ud"
    }
    proton, neutron = baryon("Cg5", "udu"), baryon("Cg5", "udd")
    single = [term(1.0, [proton, neutron], [("Cg5", 0, 1)])]
    isoscalar_singlet = [
        term(1 / sqrt(2), [proton, neutron], [("Cg5", 0, 1)]),
        term(-1 / sqrt(2), [neutron, proton], [("Cg5", 0, 1)]),
    ]
    scale = abs(correlator(single, single, sink, source, propagators))
    assert scale > 1e-3
    assert abs(correlator(isoscalar_singlet, isoscalar_singlet, sink, source, propagators)) < 1e-13 * scale
    assert abs(correlator(isoscalar_singlet, single, sink, source, propagators)) < 1e-13 * scale


@pytest.mark.parametrize("system", ["proton", "nn_1s0", "nn_3s1"])
def test_multibaryon_matches_independent_reference_full_job(tmp_path: Path, system: str) -> None:
    generator = np.random.default_rng(31415)
    eigenvectors = tmp_path / "eigenvectors.h5"
    up, down = tmp_path / "up.h5", tmp_path / "down.h5"
    _write_eigenvectors(eigenvectors, generator)
    _write_perambulators(up, generator)
    _write_perambulators(down, generator)
    output_dir = tmp_path / "validation"

    report = validate(
        system=system,
        perambulator=up,
        down_perambulator=down,
        eigenvectors=eigenvectors,
        output_dir=output_dir,
        mode_size=MODE_SIZE,
        source_time=0,
        sink_times=(1,),
        configuration=17,
        momentum=(0, 0, 1),
        rtol=1e-11,
    )

    assert report["passed"]
    assert report["max_per_time_scaled"] < 1e-11
    assert report["reference_selfcheck_vs_delta_pp_tool"] < 1e-12
    assert report["minimum_reference_magnitude"] > 1e-5
    expected_correlators = 1 if system == "proton" else 4
    assert report["correlators"] == expected_correlators
    with h5py.File(output_dir / "independent_reference.h5", "r") as handle:
        assert len(handle) == expected_correlators
        if system == "proton":
            assert handle["proton_0_0"].shape == (1, 4, 4)
        else:
            rest = np.asarray(handle[f"{system}_0_0"])
            moving = np.asarray(handle[f"{system}_1_1"])
            assert rest.shape == (1,)
            assert abs(rest[0] - moving[0]) > 1e-3 * abs(rest[0])


def test_refuses_to_overwrite_operator_source(tmp_path: Path) -> None:
    operator_source = tmp_path / "operators.py"
    original = "existing operator source\n"
    operator_source.write_text(original, encoding="utf-8")

    with pytest.raises(FileExistsError, match="operators.py"):
        validate(
            system="nn_1s0",
            perambulator=tmp_path / "missing_perambulator.h5",
            eigenvectors=tmp_path / "missing_eigenvectors.h5",
            output_dir=tmp_path,
            mode_size=MODE_SIZE,
            sink_times=(1,),
        )

    assert operator_source.read_text(encoding="utf-8") == original
