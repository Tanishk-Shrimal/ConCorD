"""Ordered-Grassmann reference for a baryon and two pseudoscalar bilinears."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import opt_einsum as oe

from tools._independent_baryon_meson import ReferenceResult, ReferenceTerm, _dp_matrices, _pairings


FIELD_IS_ANTI = (
    False, False, False, True, False, True, False,
    True, False, True, False, True, True, True,
)


def evaluate_reference(
    sink_elemental: np.ndarray,
    source_elemental: np.ndarray,
    propagators: Mapping[tuple[str, int, int], np.ndarray],
    *,
    sink_time: int,
    source_time: int,
    sink_mesons: tuple[tuple[str, str], tuple[str, str]],
    source_mesons: tuple[tuple[str, str], tuple[str, str]],
) -> ReferenceResult:
    """Contract the explicit 14-field order of ``p pi_A pi_B`` and its adjoint."""
    if sink_time == source_time:
        raise ValueError("source and sink times must differ")
    sink_elemental = np.asarray(sink_elemental, dtype=np.complex128)
    source_elemental = np.asarray(source_elemental, dtype=np.complex128)
    if sink_elemental.ndim != 3 or len(set(sink_elemental.shape)) != 1:
        raise ValueError("sink baryon elemental must have shape (Nv,Nv,Nv)")
    if source_elemental.shape != sink_elemental.shape:
        raise ValueError("source and sink baryon elementals must have the same shape")

    mode_size = sink_elemental.shape[0]
    gamma4, gamma5, c_gamma5 = _dp_matrices()
    source_gamma = gamma4 @ gamma5.conj().T @ gamma4
    mode = tuple(range(0, 28, 2))
    spin = tuple(range(1, 28, 2))
    source_output, source_alpha, source_beta = 28, 29, 30
    identity = np.eye(mode_size, dtype=np.complex128)
    common = [
        sink_elemental, [mode[0], mode[1], mode[2]],
        source_elemental, [mode[13], mode[12], mode[11]],
        c_gamma5, [spin[0], spin[1]],
        gamma5, [spin[3], spin[4]],
        identity, [mode[3], mode[4]],
        gamma5, [spin[5], spin[6]],
        identity, [mode[5], mode[6]],
        source_gamma, [spin[7], spin[8]],
        identity, [mode[7], mode[8]],
        source_gamma, [spin[9], spin[10]],
        identity, [mode[9], mode[10]],
        gamma4, [spin[11], source_output],
        gamma4, [spin[12], source_beta],
        gamma4, [spin[13], source_alpha],
        c_gamma5.conj().T, [source_beta, source_alpha],
    ]
    field_flavors = (
        "u", "d", "u", *sink_mesons[0], *sink_mesons[1],
        source_mesons[1][1], source_mesons[1][0],
        source_mesons[0][1], source_mesons[0][0],
        "u", "d", "u",
    )
    times = (sink_time,) * 7 + (source_time,) * 7
    total = np.zeros((4, 4), dtype=np.complex128)
    cross_time = np.zeros_like(total)
    same_timeslice = np.zeros_like(total)
    terms: list[ReferenceTerm] = []
    for sign, pairs in _pairings(tuple(range(14)), field_flavors, FIELD_IS_ANTI):
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
