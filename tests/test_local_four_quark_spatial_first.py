"""Exact coordinate-space oracle for a local color-singlet four-quark correlator."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime import (
    CompiledKernelMaterializationProvider,
    PropagationSourceHandler,
    RuntimeCompilationConfig,
    RuntimeExecutionConfig,
    run_correlators,
)
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider
from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource
from concord_lqcd.core import Correlator, Operator, d, dbar, gamma_identity, u, ubar
from concord_lqcd.core.wick import wick_contract


def _local_operator(position: str) -> Operator:
    return Operator(
        dbar(position, "a", "s0")
        * gamma_identity("s0", "s1")
        * u(position, "a", "s1")
        * ubar(position, "b", "s2")
        * gamma_identity("s2", "s3")
        * d(position, "b", "s3"),
        momentum=(0, 0, 0),
    )


def _input_tensors() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    generator = np.random.default_rng(20260929)
    eigenvectors = np.empty((2, 2, 3, 2), dtype=np.complex128)
    for time in range(2):
        random_matrix = generator.normal(size=(6, 2)) + 1j * generator.normal(size=(6, 2))
        eigenvectors[time] = np.linalg.qr(random_matrix)[0].reshape(2, 3, 2)

    perambulators = {
        flavor: generator.normal(size=(2, 2, 2, 4, 2, 4))
        + 1j * generator.normal(size=(2, 2, 2, 4, 2, 4))
        for flavor in ("u", "d")
    }
    return eigenvectors, perambulators


class _PerambulatorSource(PropagationSourceHandler):
    def __init__(self, perambulators: dict[str, np.ndarray]):
        self.perambulators = perambulators

    def materialize(self, request):
        flavor = request.recipe.segment_flavors[0]
        return self.perambulators[flavor][request.sink_time, request.source_time]


def _source_resolver(resource_id, recipe):
    return ExternalTensorSource(
        source_id=resource_id,
        kind="local-four-quark-oracle",
        location=resource_id,
    )


def _mode_first(
    eigenvectors: np.ndarray | dict[str, np.ndarray],
    perambulators: dict[str, np.ndarray],
    correlator: Correlator | None = None,
    spin_basis: str = "DP",
) -> complex:
    def provider_factory(request):
        time = request.operator_times[0]
        fields = {
            flavor: torch.from_numpy(values[time : time + 1])
            for flavor, values in (
                eigenvectors.items()
                if isinstance(eigenvectors, dict)
                else (("u", eigenvectors), ("d", eigenvectors))
            )
        }
        return DistillationTensorProvider(fields, fields, {})

    if correlator is None:
        correlator = Correlator(_local_operator("x"), _local_operator("y"))
    terms = wick_contract(correlator)
    assert len(terms) == 4
    assert {term.topology for term in terms} == {"connected", "disconnected"}

    result = run_correlators(
        {"local_four_quark": correlator},
        RuntimeCompilationConfig(mode_size=2, spin_basis=spin_basis, scalar_type="complex128"),
        RuntimeExecutionConfig(
            operator_times={2: (1, 0)},
            propagation_source_resolver=_source_resolver,
            kernel_providers={
                "distillation": CompiledKernelMaterializationProvider(provider_factory)
            },
            source_handlers={"local-four-quark-oracle": _PerambulatorSource(perambulators)},
            device="cpu",
        ),
    )
    assert result.compilation.required_kernel_resource_ids
    return complex(result.output("local_four_quark").item())


def _ordered_fields():
    return (
        ("d", True, 1, "a", "s0"),
        ("u", False, 1, "a", "s1"),
        ("u", True, 1, "b", "s2"),
        ("d", False, 1, "b", "s3"),
        ("d", True, 0, "c", "s4"),
        ("u", False, 0, "c", "s5"),
        ("u", True, 0, "d", "s6"),
        ("d", False, 0, "d", "s7"),
    )


def _pairings(fields, remaining):
    if not remaining:
        yield 1, ()
        return

    first = remaining[0]
    for offset, other in enumerate(remaining[1:], start=1):
        if fields[first][0] != fields[other][0] or fields[first][1] == fields[other][1]:
            continue
        quark, antiquark = (other, first) if fields[first][1] else (first, other)
        sign = (-1 if fields[first][1] else 1) * (-1 if (offset - 1) % 2 else 1)
        tail = remaining[1:offset] + remaining[offset + 1 :]
        for tail_sign, tail_pairs in _pairings(fields, tail):
            yield sign * tail_sign, ((quark, antiquark),) + tail_pairs


def _smeared_block(
    eigenvectors: np.ndarray,
    perambulator: np.ndarray,
    quark_time: int,
    antiquark_time: int,
    quark_sites: slice,
    antiquark_sites: slice,
) -> np.ndarray:
    return np.einsum(
        "xci,iajb,ydj->xcaydb",
        eigenvectors[quark_time, quark_sites],
        perambulator[quark_time, antiquark_time],
        eigenvectors[antiquark_time, antiquark_sites].conj(),
        optimize=True,
    )


def _spatial_first(
    eigenvectors: np.ndarray | dict[str, np.ndarray],
    perambulators: dict[str, np.ndarray],
    tile_size: int,
    spin_matrix: np.ndarray | None = None,
) -> complex:
    fields = _ordered_fields()
    contractions = tuple(_pairings(fields, tuple(range(len(fields)))))
    assert len(contractions) == 4
    index_names = ("x", "y", "a", "b", "c", "d", *(f"s{index}" for index in range(8)))
    labels = {label: number for number, label in enumerate(index_names)}
    matrix = np.eye(4, dtype=np.complex128) if spin_matrix is None else spin_matrix
    result = 0j

    for sink_start, source_start in product(range(0, 2, tile_size), repeat=2):
        sites = {
            1: slice(sink_start, sink_start + tile_size),
            0: slice(source_start, source_start + tile_size),
        }
        for sign, pairs in contractions:
            operands = []
            for quark_index, antiquark_index in pairs:
                quark = fields[quark_index]
                antiquark = fields[antiquark_index]
                operands.extend(
                    (
                        _smeared_block(
                            eigenvectors[quark[0]] if isinstance(eigenvectors, dict) else eigenvectors,
                            perambulators[quark[0]],
                            quark[2],
                            antiquark[2],
                            sites[quark[2]],
                            sites[antiquark[2]],
                        ),
                        [
                            labels["x" if quark[2] == 1 else "y"],
                            labels[quark[3]],
                            labels[quark[4]],
                            labels["x" if antiquark[2] == 1 else "y"],
                            labels[antiquark[3]],
                            labels[antiquark[4]],
                        ],
                    )
                )
            for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
                operands.extend((matrix, [labels[f"s{left}"], labels[f"s{right}"]]))
            result += sign * np.einsum(*operands, [], optimize="greedy")

    return complex(result)


@pytest.mark.parametrize("contributions", ("all", "cross_time", "equal_time"))
def test_local_four_quark_mode_first_matches_tiled_spatial_first(contributions: str):
    eigenvectors, perambulators = _input_tensors()
    for perambulator in perambulators.values():
        if contributions == "cross_time":
            perambulator[0, 0] = 0
            perambulator[1, 1] = 0
        elif contributions == "equal_time":
            perambulator[0, 1] = 0
            perambulator[1, 0] = 0

    mode_first = _mode_first(eigenvectors, perambulators)
    spatial_first = _spatial_first(eigenvectors, perambulators, tile_size=2)
    tiled_spatial_first = _spatial_first(eigenvectors, perambulators, tile_size=1)

    assert abs(spatial_first) > 1e-6
    np.testing.assert_allclose(spatial_first, tiled_spatial_first, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mode_first, spatial_first, rtol=1e-12, atol=1e-12)


def test_local_four_quark_is_invariant_under_sitewise_color_rotation():
    eigenvectors, perambulators = _input_tensors()
    generator = np.random.default_rng(20260930)
    rotations = np.empty((2, 2, 3, 3), dtype=np.complex128)
    for time, site in product(range(2), repeat=2):
        random_matrix = generator.normal(size=(3, 3)) + 1j * generator.normal(size=(3, 3))
        rotations[time, site] = np.linalg.qr(random_matrix)[0]

    rotated = np.einsum("txcd,txdi->txci", rotations, eigenvectors, optimize=True)
    unrotated_result = _mode_first(eigenvectors, perambulators)
    rotated_result = _mode_first(rotated, perambulators)
    np.testing.assert_allclose(rotated_result, unrotated_result, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        _spatial_first(rotated, perambulators, tile_size=1),
        unrotated_result,
        rtol=1e-12,
        atol=1e-12,
    )
