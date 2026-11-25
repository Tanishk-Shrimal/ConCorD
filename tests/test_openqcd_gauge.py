"""Gauge links in the openQCD binary format: reader, plaquette check, and use in a job."""

from __future__ import annotations

import os
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from concord_lqcd.job import (
    JobSpec,
    OpenQCDGaugeError,
    OpenQCDGaugeReader,
    build_job_runtime_resources,
    compile_job_plan,
)
from tests.test_laph_gauge_iomap import _four_dimensional_record, _write_iomap


DIRECTIONS = ("x", "y", "z", "t")


def _random_links(extent, *, seed: int, unitary: bool = True) -> dict[str, np.ndarray]:
    """Links U(x, mu) per direction, each of shape [t, z, y, x, 3, 3]."""
    lx, ly, lz, lt = extent
    rng = np.random.default_rng(seed)
    links = {}
    for name in DIRECTIONS:
        shape = (lt, lz, ly, lx, 3, 3)
        matrices = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        if unitary:
            matrices, _ = np.linalg.qr(matrices)
        links[name] = matrices
    return links


def _write_openqcd(path: Path, links: dict[str, np.ndarray], plaquette: float) -> None:
    """Write the openQCD layout site by site, independently of the reader."""
    lt, lz, ly, lx = links["x"].shape[:4]
    data = bytearray(struct.pack("<4id", lt, lx, ly, lz, plaquette))
    for t in range(lt):
        for x in range(lx):
            for y in range(ly):
                for z in range(lz):
                    if (t + x + y + z) % 2 == 0:
                        continue
                    backward = {
                        "t": ((t - 1) % lt, z, y, x),
                        "x": (t, z, y, (x - 1) % lx),
                        "y": (t, z, (y - 1) % ly, x),
                        "z": (t, (z - 1) % lz, y, x),
                    }
                    for name in ("t", "x", "y", "z"):
                        for site in ((t, z, y, x), backward[name]):
                            data.extend(np.ascontiguousarray(links[name][site], dtype="<c16").tobytes())
    path.write_bytes(bytes(data))


def _direct_plaquette(links: dict[str, np.ndarray]) -> float:
    """sum Re tr U(p) / (6 V) on the full four-dimensional field."""
    axis = {"t": 0, "z": 1, "y": 2, "x": 3}
    total = 0.0
    for first, second in (("x", "y"), ("x", "z"), ("y", "z"), ("t", "x"), ("t", "y"), ("t", "z")):
        u_mu, u_nu = links[first], links[second]
        product = (
            u_mu
            @ np.roll(u_nu, -1, axis=axis[first])
            @ np.swapaxes(np.roll(u_mu, -1, axis=axis[second]), -1, -2).conj()
            @ np.swapaxes(u_nu, -1, -2).conj()
        )
        total += np.trace(product, axis1=-2, axis2=-1).real.sum()
    return total / (6 * links["x"][..., 0, 0].size)


EXTENT = (4, 2, 6, 4)  # (x, y, z, t)


def test_reads_every_direction_and_timeslice(tmp_path):
    links = _random_links(EXTENT, seed=1, unitary=False)
    path = tmp_path / "cfg"
    _write_openqcd(path, links, plaquette=1.25)

    reader = OpenQCDGaugeReader(path)

    assert reader.metadata.lattice_extent == EXTENT
    assert reader.metadata.header_plaquette == 1.25
    assert reader.metadata.directions == DIRECTIONS
    for index, name in enumerate(DIRECTIONS):
        for time in range(EXTENT[3]):
            np.testing.assert_array_equal(reader.read_timeslice(name, time), links[name][time])
            np.testing.assert_array_equal(reader.read_timeslice(index, time), links[name][time])


def test_average_plaquette_matches_direct_computation(tmp_path):
    links = _random_links(EXTENT, seed=2)
    expected = _direct_plaquette(links)
    path = tmp_path / "cfg"
    _write_openqcd(path, links, plaquette=expected)

    assert OpenQCDGaugeReader(path).average_plaquette() == pytest.approx(expected, rel=0, abs=1e-13)

    unit = {name: np.broadcast_to(np.eye(3), value.shape).copy() for name, value in links.items()}
    _write_openqcd(tmp_path / "unit", unit, plaquette=3.0)
    assert OpenQCDGaugeReader(tmp_path / "unit").average_plaquette() == pytest.approx(3.0, abs=1e-14)


def test_reads_the_same_links_as_the_laph_reader(tmp_path):
    links = _random_links(EXTENT, seed=3)
    _write_openqcd(tmp_path / "openqcd", links, plaquette=0.0)
    stacked = np.stack([links[name] for name in DIRECTIONS])  # (direction, t, z, y, x, 3, 3)
    _write_iomap(
        tmp_path / "laph",
        file_id="Laph--SmearedGaugeField4D",
        records={0: _four_dimensional_record(stacked, byte_order="<")},
        byte_order="<",
    )
    from concord_lqcd.job import LaphGaugeIOMapReader

    openqcd, laph = OpenQCDGaugeReader(tmp_path / "openqcd"), LaphGaugeIOMapReader(tmp_path / "laph")

    assert openqcd.metadata.lattice_extent == laph.metadata.lattice_extent
    for name in DIRECTIONS:
        for time in range(EXTENT[3]):
            np.testing.assert_array_equal(openqcd.read_timeslice(name, time), laph.read_timeslice(name, time))


def test_rejects_malformed_files_and_requests(tmp_path):
    with pytest.raises(OpenQCDGaugeError, match="does not exist"):
        OpenQCDGaugeReader(tmp_path / "missing")

    links = _random_links(EXTENT, seed=4, unitary=False)
    path = tmp_path / "cfg"
    _write_openqcd(path, links, plaquette=1.0)
    path.write_bytes(path.read_bytes()[:-16])
    with pytest.raises(OpenQCDGaugeError, match="bytes"):
        OpenQCDGaugeReader(path)

    odd = tmp_path / "odd"
    odd.write_bytes(struct.pack("<4id", 4, 3, 2, 2, 1.0))
    with pytest.raises(OpenQCDGaugeError, match="odd lattice extents"):
        OpenQCDGaugeReader(odd)

    short = tmp_path / "short"
    short.write_bytes(b"\x00" * 10)
    with pytest.raises(OpenQCDGaugeError, match="header"):
        OpenQCDGaugeReader(short)

    _write_openqcd(path, links, plaquette=1.0)
    reader = OpenQCDGaugeReader(path)
    with pytest.raises(OpenQCDGaugeError, match="outside the lattice range"):
        reader.read_timeslice("x", EXTENT[3])
    with pytest.raises(OpenQCDGaugeError, match="unavailable"):
        reader.read_timeslice("w", 0)
    with pytest.raises(OpenQCDGaugeError, match="outside the stored range"):
        reader.read_timeslice(4, 0)


@pytest.mark.skipif(
    not os.environ.get("CONCORD_OPENQCD_TEST_FILE"),
    reason="set CONCORD_OPENQCD_TEST_FILE to an openQCD configuration to check it",
)
def test_real_configuration_plaquette_matches_its_header():
    reader = OpenQCDGaugeReader(os.environ["CONCORD_OPENQCD_TEST_FILE"])
    assert reader.average_plaquette() == pytest.approx(reader.metadata.header_plaquette, rel=1e-12)


@pytest.mark.parametrize(
    "transformation",
    (
        "derivative(u('y','a','s2'), '+y')",
        "transport(u('y','a','s2'), '+z')",
    ),
)
def test_job_kernels_agree_for_openqcd_and_laph_gauge_files(tmp_path, monkeypatch, transformation):
    import tests.test_job_data_provider as jobs

    # The openQCD layout needs even extents; the shared helpers read these module constants.
    monkeypatch.setattr(jobs, "LY", 2)
    monkeypatch.setattr(jobs, "LZ", 2)
    extent = (jobs.LX, jobs.LY, jobs.LZ, jobs.T)
    links = _random_links(extent, seed=5)
    unit = {name: np.broadcast_to(np.eye(3), value.shape).copy() for name, value in links.items()}

    def kernel(provider: str, field: dict[str, np.ndarray], label: str) -> torch.Tensor:
        directory = tmp_path / label
        directory.mkdir()
        spec = jobs._job_spec(directory)
        payload = spec.model_dump(mode="python")
        payload["data"]["gauge_links"] = {
            "provider": provider,
            "path": str(directory / "gauge_{configuration}"),
        }
        spec = JobSpec.model_validate(payload)
        jobs._write_transformed_operator(spec.operators.file, transformation)
        jobs._write_eigenvectors(directory / "evec_0.h5")
        if provider == "openqcd":
            _write_openqcd(directory / "gauge_0", field, plaquette=0.0)
        else:
            stacked = np.stack([field[name] for name in DIRECTIONS])
            _write_iomap(
                directory / "gauge_0",
                file_id="Laph--SmearedGaugeField4D",
                records={0: _four_dimensional_record(stacked, byte_order="<")},
                byte_order="<",
            )
        plan = compile_job_plan(spec)
        resources = build_job_runtime_resources(
            spec, plan, configuration=0, time_context=spec.time_contexts[0]
        )
        request = jobs._first_transformed_request(resources, plan)
        return resources.kernel_providers["distillation"].materialize(request)

    from_openqcd = kernel("openqcd", links, "openqcd")
    from_laph = kernel("laph_iomap", links, "laph")
    with_unit_links = kernel("openqcd", unit, "unit")

    torch.testing.assert_close(from_openqcd, from_laph, rtol=0.0, atol=0.0)
    assert not torch.allclose(from_openqcd, with_unit_links)
