"""Ordered-Grassmann reference for a local baryon and bilinear meson."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import opt_einsum as oe


FIELD_IS_ANTI = (False, False, False, True, False, True, False, True, True, True)


@dataclass(frozen=True)
class ReferenceTerm:
    sign: int
    pairing: tuple[tuple[int, int], ...]
    same_timeslice: bool
    value: np.ndarray


@dataclass(frozen=True)
class ReferenceResult:
    total: np.ndarray
    cross_time: np.ndarray
    same_timeslice: np.ndarray
    terms: tuple[ReferenceTerm, ...]


def _dp_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma4 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    gamma5 = np.zeros((4, 4), dtype=np.complex128)
    gamma5[0, 2] = gamma5[1, 3] = gamma5[2, 0] = gamma5[3, 1] = -1
    c_gamma5 = np.zeros((4, 4), dtype=np.complex128)
    c_gamma5[0, 1] = c_gamma5[2, 3] = -1
    c_gamma5[1, 0] = c_gamma5[3, 2] = 1
    return gamma4, gamma5, c_gamma5


def _pairings(
    remaining: tuple[int, ...],
    field_flavors: tuple[str, ...],
    field_is_anti: tuple[bool, ...] = FIELD_IS_ANTI,
):
    if not remaining:
        yield 1, ()
        return
    first = remaining[0]
    for offset, other in enumerate(remaining[1:], start=1):
        if field_flavors[first] != field_flavors[other]:
            continue
        if field_is_anti[first] == field_is_anti[other]:
            continue
        quark, antiquark = (other, first) if field_is_anti[first] else (first, other)
        pair_sign = (-1 if field_is_anti[first] else 1) * (-1 if (offset - 1) % 2 else 1)
        tail = remaining[1:offset] + remaining[offset + 1:]
        for tail_sign, tail_pairs in _pairings(tail, field_flavors, field_is_anti):
            yield pair_sign * tail_sign, ((quark, antiquark),) + tail_pairs


def evaluate_reference(
    sink_elemental: np.ndarray,
    source_elemental: np.ndarray,
    propagators: Mapping[tuple[str, int, int], np.ndarray],
    *,
    sink_time: int,
    source_time: int,
    sink_baryon_flavors: tuple[str, str, str] = ("u", "d", "u"),
    sink_meson_flavors: tuple[str, str] = ("d", "u"),
    source_baryon_flavors: tuple[str, str, str] = ("u", "d", "u"),
    source_meson_flavors: tuple[str, str] = ("d", "u"),
    meson_spin: str = "pseudoscalar",
) -> ReferenceResult:
    """Contract baryon-meson field order against its reversed source adjoint."""
    if sink_time == source_time:
        raise ValueError("the baryon-meson pilot requires distinct source and sink times")
    sink_elemental = np.asarray(sink_elemental, dtype=np.complex128)
    source_elemental = np.asarray(source_elemental, dtype=np.complex128)
    if sink_elemental.ndim != 3 or len(set(sink_elemental.shape)) != 1:
        raise ValueError("sink baryon elemental must have shape (Nv,Nv,Nv)")
    if source_elemental.shape != sink_elemental.shape:
        raise ValueError("source and sink baryon elementals must have the same shape")
    if meson_spin not in ("pseudoscalar", "scalar"):
        raise ValueError(f"unsupported meson spin structure {meson_spin!r}")

    mode_size = sink_elemental.shape[0]
    gamma4, gamma5, c_gamma5 = _dp_matrices()
    meson_gamma = gamma5 if meson_spin == "pseudoscalar" else np.eye(4, dtype=np.complex128)
    source_meson_gamma = gamma4 @ meson_gamma.conj().T @ gamma4
    mode = tuple(range(0, 20, 2))
    spin = tuple(range(1, 20, 2))
    source_output, source_alpha, source_beta = 20, 21, 22
    identity = np.eye(mode_size, dtype=np.complex128)
    common = [
        sink_elemental, [mode[0], mode[1], mode[2]],
        source_elemental, [mode[9], mode[8], mode[7]],
        c_gamma5, [spin[0], spin[1]],
        meson_gamma, [spin[3], spin[4]],
        identity, [mode[3], mode[4]],
        identity, [mode[5], mode[6]],
        source_meson_gamma, [spin[5], spin[6]],
        gamma4, [spin[7], source_output],
        gamma4, [spin[8], source_beta],
        gamma4, [spin[9], source_alpha],
        c_gamma5.conj().T, [source_beta, source_alpha],
    ]
    field_flavors = (
        *sink_baryon_flavors,
        *sink_meson_flavors,
        source_meson_flavors[1],
        source_meson_flavors[0],
        *reversed(source_baryon_flavors),
    )
    times = (sink_time,) * 5 + (source_time,) * 5
    total = np.zeros((4, 4), dtype=np.complex128)
    cross_time = np.zeros_like(total)
    same_timeslice = np.zeros_like(total)
    terms: list[ReferenceTerm] = []
    for sign, pairs in _pairings(tuple(range(10)), field_flavors):
        operands = list(common)
        for quark, antiquark in pairs:
            key = (field_flavors[quark], times[quark], times[antiquark])
            try:
                block = np.asarray(propagators[key], dtype=np.complex128)
            except KeyError as exc:
                raise ValueError(f"missing propagation block {key}") from exc
            if block.shape != (mode_size, 4, mode_size, 4):
                raise ValueError(f"propagation block {key} has shape {block.shape}")
            operands.extend((block, [mode[quark], spin[quark], mode[antiquark], spin[antiquark]]))
        value = sign * oe.contract(*operands, [spin[2], source_output], optimize="greedy")
        local = any(times[quark] == times[antiquark] for quark, antiquark in pairs)
        terms.append(ReferenceTerm(sign, pairs, local, value))
        total += value
        if local:
            same_timeslice += value
        else:
            cross_time += value
    return ReferenceResult(total, cross_time, same_timeslice, tuple(terms))
