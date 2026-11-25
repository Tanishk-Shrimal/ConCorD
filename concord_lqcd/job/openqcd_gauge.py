"""Read gauge fields stored in the openQCD binary format.

openQCD, the code that generates the CLS ensembles, stores a gauge field as

* four 4-byte integers ``N0, N1, N2, N3``: the time extent and the x, y and z extents;
* one 8-byte float: the average plaquette, ``sum Re tr U(p)`` over all ``6 V`` plaquettes divided by
  ``6 V``;
* for every odd site (``x0 + x1 + x2 + x3`` odd), in lexicographic order with ``x0`` (time) slowest and
  ``x3`` (z) fastest, eight SU(3) matrices ``U(x,0), U(x-0,0), U(x,1), U(x-1,1), U(x,2), U(x-2,2),
  U(x,3), U(x-3,3)``, where ``U(x-mu, mu)`` is the link that ends at ``x``. Each matrix is 3 x 3 complex,
  double precision, row major. Everything is little endian.

Direction 0 is time; directions 1, 2, 3 are x, y, z. ConCorD uses the links as stored. Whether they are
smeared, and smeared in the same way as the links used for the Laplacian eigenvectors, is up to the user.
Only the timeslices that are requested are read from the file.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Literal

import numpy as np


_HEADER = struct.Struct("<4id")
_MATRIX_DTYPE = np.dtype("<c16")
_MATRIX_BYTES = 9 * _MATRIX_DTYPE.itemsize
_MATRICES_PER_ODD_SITE = 8
_DIRECTION_NAMES = ("x", "y", "z", "t")
# Position of the forward link U(x, mu) among the eight matrices of an odd site; the backward link
# U(x - mu, mu) follows it.
_FORWARD_SLOT = {"t": 0, "x": 2, "y": 4, "z": 6}


class OpenQCDGaugeError(ValueError):
    """Raised when an openQCD gauge file is malformed or a request is outside it."""


@dataclass(frozen=True)
class OpenQCDGaugeMetadata:
    """Validated header of one openQCD gauge file."""

    path: Path
    lattice_extent: tuple[int, int, int, int]  # (x, y, z, t), as in the job lattice
    header_plaquette: float
    directions: tuple[str, ...] = _DIRECTION_NAMES
    scalar_type: Literal["complex128"] = "complex128"
    byte_order: Literal["little"] = "little"


class OpenQCDGaugeReader:
    """Random-access reader for gauge links in the openQCD binary format."""

    def __init__(self, path: str | Path):
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise OpenQCDGaugeError(f"openQCD gauge file does not exist: {resolved}")
        size = resolved.stat().st_size
        if size < _HEADER.size:
            raise OpenQCDGaugeError(
                f"openQCD gauge file {resolved} is shorter than its {_HEADER.size}-byte header"
            )
        with resolved.open("rb") as handle:
            n0, n1, n2, n3, plaquette = _HEADER.unpack(handle.read(_HEADER.size))
        extents = (n0, n1, n2, n3)
        if any(extent <= 0 for extent in extents):
            raise OpenQCDGaugeError(
                f"openQCD gauge file {resolved} has invalid lattice extents (N0, N1, N2, N3) = {extents}"
            )
        if any(extent % 2 for extent in extents):
            raise OpenQCDGaugeError(
                f"openQCD gauge file {resolved} has odd lattice extents {extents}; the format stores "
                "links on odd sites and needs even extents"
            )
        volume = n0 * n1 * n2 * n3
        expected = _HEADER.size + (volume // 2) * _MATRICES_PER_ODD_SITE * _MATRIX_BYTES
        if size != expected:
            raise OpenQCDGaugeError(
                f"openQCD gauge file {resolved} has {size} bytes; a {n1}x{n2}x{n3}x{n0} field "
                f"(x, y, z, t) has {expected}"
            )
        if not math.isfinite(plaquette):
            raise OpenQCDGaugeError(f"openQCD gauge file {resolved} has a non-finite plaquette")
        self._metadata = OpenQCDGaugeMetadata(
            path=resolved,
            lattice_extent=(n1, n2, n3, n0),
            header_plaquette=float(plaquette),
        )
        self._odd_sites_per_timeslice = (n1 * n2 * n3) // 2
        x, y, z = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        # Odd sites of a timeslice in file order (x slowest, z fastest), for even and odd t.
        self._odd_sites = {
            time_parity: tuple(
                axis[(x + y + z + time_parity) % 2 == 1] for axis in (x, y, z)
            )
            for time_parity in (0, 1)
        }

    @property
    def metadata(self) -> OpenQCDGaugeMetadata:
        return self._metadata

    def _direction_name(self, direction: str | int) -> str:
        if isinstance(direction, bool):
            raise TypeError("gauge direction must be a name or integer index")
        if isinstance(direction, int):
            if direction < 0 or direction >= len(_DIRECTION_NAMES):
                raise OpenQCDGaugeError(
                    f"gauge direction index {direction} is outside the stored range"
                )
            return _DIRECTION_NAMES[direction]
        if isinstance(direction, str):
            name = direction.strip().lower()
            if name not in _DIRECTION_NAMES:
                raise OpenQCDGaugeError(
                    f"gauge direction {direction!r} is unavailable; file contains {_DIRECTION_NAMES}"
                )
            return name
        raise TypeError("gauge direction must be a name or integer index")

    def _timeslice_block(self, time: int) -> np.ndarray:
        """The eight matrices of every odd site of one timeslice, shape [site, 8, 3, 3]."""
        offset = _HEADER.size + time * self._odd_sites_per_timeslice * _MATRICES_PER_ODD_SITE * _MATRIX_BYTES
        return np.memmap(
            self._metadata.path,
            mode="r",
            dtype=_MATRIX_DTYPE,
            offset=offset,
            shape=(self._odd_sites_per_timeslice, _MATRICES_PER_ODD_SITE, 3, 3),
            order="C",
        )

    def read_timeslice(
        self,
        direction: str | int,
        time: int,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        """Read the links ``U(x, mu)`` of one direction at one Euclidean time.

        The returned shape is ``[z, y, x, color_row, color_column]``, as for the LapH smeared-gauge
        reader. The links are reordered from the odd-site storage, so the result is always a new array
        (``copy`` is accepted for interface compatibility).
        """
        del copy
        if not isinstance(time, int) or isinstance(time, bool):
            raise TypeError("gauge-link time must be an integer")
        lx, ly, lz, lt = self._metadata.lattice_extent
        if time < 0 or time >= lt:
            raise OpenQCDGaugeError(
                f"gauge-link time {time} is outside the lattice range [0, {lt})"
            )
        name = self._direction_name(direction)
        slot = _FORWARD_SLOT[name]
        links = np.empty((lz, ly, lx, 3, 3), dtype=np.complex128)

        # Forward links U(x, mu) of the odd sites of this timeslice.
        block = self._timeslice_block(time)
        x, y, z = self._odd_sites[time % 2]
        links[z, y, x] = block[:, slot]

        # The backward links U(o - mu, mu) stored at odd sites o give the links of the even sites.
        if name == "t":
            following = (time + 1) % lt
            x, y, z = self._odd_sites[following % 2]
            links[z, y, x] = self._timeslice_block(following)[:, slot + 1]
        else:
            shifted = [x, y, z]
            axis = "xyz".index(name)
            shifted[axis] = (shifted[axis] - 1) % (lx, ly, lz)[axis]
            links[shifted[2], shifted[1], shifted[0]] = block[:, slot + 1]
        return links

    def average_plaquette(self) -> float:
        """``sum Re tr U(p) / (6 V)`` over all plaquettes, from the stored links.

        This is the quantity stored in the header, so comparing the two checks that the file is read
        with the right site and direction order. It reads the whole file, one timeslice at a time.
        """
        lx, ly, lz, lt = self._metadata.lattice_extent
        spatial_axes = {"x": 2, "y": 1, "z": 0}  # array axis of each direction in [z, y, x, 3, 3]

        def shift(field: np.ndarray, direction: str) -> np.ndarray:
            # field(x + mu) at site x
            return np.roll(field, -1, axis=spatial_axes[direction])

        def plaquette_trace(u_mu, u_nu_at_x_plus_mu, u_mu_at_x_plus_nu, u_nu) -> float:
            product = np.einsum(
                "...ab,...bc,...dc,...ed->...ae",
                u_mu,
                u_nu_at_x_plus_mu,
                u_mu_at_x_plus_nu.conj(),
                u_nu.conj(),
                optimize=True,
            )
            return float(np.trace(product, axis1=-2, axis2=-1).real.sum())

        total = 0.0
        current = {name: self.read_timeslice(name, 0) for name in _DIRECTION_NAMES}
        for time in range(lt):
            following = (time + 1) % lt
            later = {name: self.read_timeslice(name, following) for name in "xyz"}
            for first, second in (("x", "y"), ("x", "z"), ("y", "z")):
                total += plaquette_trace(
                    current[first],
                    shift(current[second], first),
                    shift(current[first], second),
                    current[second],
                )
            for name in "xyz":
                total += plaquette_trace(
                    current["t"], later[name], shift(current["t"], name), current[name]
                )
            if following:
                current = {**later, "t": self.read_timeslice("t", following)}
        return total / (6 * lx * ly * lz * lt)


__all__ = [
    "OpenQCDGaugeError",
    "OpenQCDGaugeMetadata",
    "OpenQCDGaugeReader",
]
