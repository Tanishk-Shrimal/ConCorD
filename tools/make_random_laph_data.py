#!/usr/bin/env python3
"""Write random distillation inputs in the format that ConCorD jobs and the validation tools read.

The comparisons of ConCorD with an independent contraction (tools/validate_*.py) and the timings do not
depend on the values of the data, only on their shapes. This tool writes data of any shape so that the
checks can be repeated without lattice data:

  eigenvectors.h5        dataset /eigenvectors, axes (time, z, y, x, color, mode); in every timeslice the
                         modes are exactly orthonormal, as Laplacian eigenvectors are
  perambulator_<f>.h5    /perambulators/src_t_<t>/data for every source time t, axes
                         (t_sink, spin_src, spin_sink, src_eigs, snk_eigs), one file per flavor f

Example, a test run of the two-nucleon check on a 4^3 x 8 lattice with N_v = 8:

  python tools/make_random_laph_data.py --output-dir random_data
  python tools/validate_multibaryon_correlators.py --system nn_1s0 \\
      --perambulator random_data/perambulator_u.h5 --down-perambulator random_data/perambulator_d.h5 \\
      --eigenvectors random_data/eigenvectors.h5 --output-dir random_nn_1s0 \\
      --configuration 1 --mode-size 8 --sink-times 1 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


EIGENVECTOR_AXES = ["time", "z", "y", "x", "color", "mode"]
PERAMBULATOR_ORDER = "t_sink, spin_src, spin_sink, src_eigs, snk_eigs"


def random_eigenvectors(
    generator: np.random.Generator, temporal_extent: int, spatial_extent: Sequence[int], mode_size: int
) -> np.ndarray:
    """Complex (T, Z, Y, X, 3, N_v) array, orthonormal over space and color in every timeslice."""
    x_extent, y_extent, z_extent = spatial_extent
    rows = x_extent * y_extent * z_extent * 3
    if mode_size > rows:
        raise ValueError(f"N_v = {mode_size} exceeds the {rows} space-color components of a timeslice")
    out = np.empty((temporal_extent, z_extent, y_extent, x_extent, 3, mode_size), dtype=np.complex128)
    for time in range(temporal_extent):
        values = generator.standard_normal((rows, mode_size)) + 1j * generator.standard_normal((rows, mode_size))
        orthonormal, _ = np.linalg.qr(values)
        out[time] = orthonormal.reshape(z_extent, y_extent, x_extent, 3, mode_size)
    return out


def random_perambulators(
    generator: np.random.Generator, temporal_extent: int, mode_size: int
) -> dict[int, np.ndarray]:
    """Complex (T, 4, 4, N_v, N_v) array per source time."""
    shape = (temporal_extent, 4, 4, mode_size, mode_size)
    return {
        source_time: generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
        for source_time in range(temporal_extent)
    }


def write(
    output_dir: Path,
    *,
    spatial_extent: Sequence[int] = (4, 4, 4),
    temporal_extent: int = 8,
    mode_size: int = 8,
    flavors: Sequence[str] = ("u", "d"),
    seed: int = 1,
) -> dict[str, Path]:
    """Write the files and return their paths, keyed ``eigenvectors`` and ``perambulator_<f>``."""
    if temporal_extent < 2 or mode_size < 1 or len(spatial_extent) != 3 or min(spatial_extent) < 1:
        raise ValueError("choose T >= 2, N_v >= 1 and three positive spatial extents")
    if not flavors or len(set(flavors)) != len(flavors):
        raise ValueError("choose distinct flavor names")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"eigenvectors": output_dir / "eigenvectors.h5"}
    paths.update({f"perambulator_{flavor}": output_dir / f"perambulator_{flavor}.h5" for flavor in flavors})
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    # One independent stream per file, so adding a flavor does not change the others.
    streams = np.random.SeedSequence(seed).spawn(1 + len(flavors))
    with h5py.File(paths["eigenvectors"], "w") as handle:
        dataset = handle.create_dataset(
            "eigenvectors",
            data=random_eigenvectors(np.random.default_rng(streams[0]), temporal_extent, spatial_extent, mode_size),
        )
        dataset.attrs["axes"] = json.dumps(EIGENVECTOR_AXES)
        dataset.attrs["note"] = "random data, orthonormal modes in every timeslice"
    for stream, flavor in zip(streams[1:], flavors):
        blocks = random_perambulators(np.random.default_rng(stream), temporal_extent, mode_size)
        with h5py.File(paths[f"perambulator_{flavor}"], "w") as handle:
            group = handle.create_group("perambulators")
            for source_time, data in blocks.items():
                dataset = group.create_group(f"src_t_{source_time}").create_dataset("data", data=data)
                dataset.attrs["dimension_order"] = PERAMBULATOR_ORDER
            handle.attrs["note"] = f"random data, flavor {flavor}"
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-extent", type=int, nargs=3, default=(4, 4, 4), metavar=("X", "Y", "Z"))
    parser.add_argument("--temporal-extent", type=int, default=8)
    parser.add_argument("--mode-size", type=int, default=8, help="number of distillation modes N_v")
    parser.add_argument("--flavors", nargs="+", default=("u", "d"),
                        help="one perambulator file per flavor (default: u d)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    paths = write(
        args.output_dir,
        spatial_extent=tuple(args.spatial_extent),
        temporal_extent=args.temporal_extent,
        mode_size=args.mode_size,
        flavors=tuple(args.flavors),
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
