"""Random distillation inputs: format, orthonormal modes, reproducibility, and a full check on them."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from tools._independent_laph import read_eigenvector_times, read_propagators
from tools.make_random_laph_data import write
from tools.validate_multibaryon_correlators import validate


def test_random_data_has_the_tool_format_and_orthonormal_modes(tmp_path) -> None:
    paths = write(tmp_path / "data", spatial_extent=(3, 2, 2), temporal_extent=4, mode_size=5, flavors=("u", "s"))
    assert sorted(paths) == ["eigenvectors", "perambulator_s", "perambulator_u"]

    slices, lattice = read_eigenvector_times(paths["eigenvectors"], "/eigenvectors", (0, 1, 2, 3), 5)
    assert lattice == (4, 3, 2, 2)  # (T, X, Y, Z)
    for modes in slices.values():
        assert modes.shape == (2, 2, 3, 3, 5)  # (z, y, x, color, mode)
        flat = modes.reshape(-1, 5)
        np.testing.assert_allclose(flat.conj().T @ flat, np.eye(5), atol=1e-14)

    blocks = read_propagators({"u": paths["perambulator_u"], "s": paths["perambulator_s"]}, (0, 3), 5, 4)
    assert blocks[("u", 3, 0)].shape == (5, 4, 5, 4)
    assert not np.allclose(blocks[("u", 3, 0)], blocks[("s", 3, 0)])


def test_random_data_is_reproducible_and_never_overwritten(tmp_path) -> None:
    first = write(tmp_path / "a", temporal_extent=3, mode_size=4, seed=7)
    second = write(tmp_path / "b", temporal_extent=3, mode_size=4, seed=7)
    for name in first:
        with h5py.File(first[name], "r") as one, h5py.File(second[name], "r") as two:
            key = "eigenvectors" if name == "eigenvectors" else "perambulators/src_t_2/data"
            np.testing.assert_array_equal(one[key][()], two[key][()])
    with pytest.raises(FileExistsError):
        write(tmp_path / "a", temporal_extent=3, mode_size=4, seed=7)


def test_proton_check_passes_on_random_data(tmp_path) -> None:
    paths = write(tmp_path / "data", spatial_extent=(2, 2, 2), temporal_extent=4, mode_size=4)
    report = validate(
        perambulator=paths["perambulator_u"],
        down_perambulator=paths["perambulator_d"],
        eigenvectors=paths["eigenvectors"],
        output_dir=tmp_path / "proton",
        system="proton",
        mode_size=4,
        sink_times=(1, 3),
        configuration=1,
    )
    assert report["passed"]
    assert report["max_per_time_scaled"] < 1e-13
