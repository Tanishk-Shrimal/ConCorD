"""Independent numerical oracles for the local proton two-point function."""

from __future__ import annotations

import math

import torch

from concord_lqcd.backend.runtime import (
    KernelMaterializationProvider,
    PropagationSourceHandler,
    RuntimeCompilationConfig,
    RuntimeExecutionConfig,
    run_correlators,
)
from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource
from concord_lqcd.core import CorrMatrix, build_baryon_operator, reset_registry
from concord_lqcd.core.wick import wick_contract


_COLOR_AXIS = {"a": 0, "b": 1, "c": 2}


def _complex_random(generator: torch.Generator, *shape: int) -> torch.Tensor:
    real = torch.randn(*shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(*shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


class _PhysicalElementalProvider(KernelMaterializationProvider):
    def __init__(self, sink: torch.Tensor, source: torch.Tensor):
        self.sink = sink
        self.source = source

    def materialize(self, request):
        fields = request.recipe.projected_fields
        colors = tuple(field.record.color_label.split(":")[-1] for field in fields)
        assert set(colors) == set(_COLOR_AXIS)

        anti_flags = tuple(field.record.is_anti for field in fields)
        assert all(anti_flags) or not any(anti_flags)
        base = self.source if all(anti_flags) else self.sink
        axes = tuple(_COLOR_AXIS[color] for color in colors)
        return base.permute(axes).contiguous()


class _PhysicalPropagationProvider(PropagationSourceHandler):
    def __init__(self, up: torch.Tensor, down: torch.Tensor):
        self.up = up
        self.down = down

    def materialize(self, request):
        flavor = request.recipe.segment_flavors[0]
        assert flavor in {"u", "d"}
        return self.up if flavor == "u" else self.down


def _source_resolver(resource_id, recipe):
    assert recipe.resource_id == resource_id
    return ExternalTensorSource(
        source_id=f"local-proton-oracle:{resource_id}",
        kind="local-proton-oracle",
        location=resource_id,
    )


def _local_proton_correlator():
    reset_registry()
    proton = build_baryon_operator(("u", "d", "u"), "Cg5", momentum=(0, 0, 0))
    return CorrMatrix([proton])[0, 0]


def _run_local_proton(
    *,
    spin_basis: str,
    sink_elemental: torch.Tensor,
    source_elemental: torch.Tensor,
    up_propagator: torch.Tensor,
    down_propagator: torch.Tensor,
) -> torch.Tensor:
    correlator = _local_proton_correlator()
    return run_correlators(
        {"proton": correlator},
        RuntimeCompilationConfig(
            mode_size=sink_elemental.shape[0],
            spin_basis=spin_basis,
            scalar_type="complex128",
        ),
        RuntimeExecutionConfig(
            operator_times={2: (3, 1)},
            propagation_source_resolver=_source_resolver,
            kernel_providers={
                "distillation": _PhysicalElementalProvider(
                    sink_elemental,
                    source_elemental,
                )
            },
            source_handlers={
                "local-proton-oracle": _PhysicalPropagationProvider(
                    up_propagator,
                    down_propagator,
                )
            },
            device="cpu",
        ),
    ).output("proton")


def test_local_proton_matches_direct_minus_exchange_oracle_and_pplus_trace():
    mode_size = 2
    generator = torch.Generator().manual_seed(91723)
    sink_elemental = _complex_random(generator, mode_size, mode_size, mode_size)
    source_elemental = _complex_random(generator, mode_size, mode_size, mode_size)
    up_propagator = _complex_random(generator, mode_size, 4, mode_size, 4)
    down_propagator = _complex_random(generator, mode_size, 4, mode_size, 4)

    correlator = _local_proton_correlator()
    assert [term.sign for term in wick_contract(correlator)] == [-1, 1]

    actual = _run_local_proton(
        spin_basis="DP",
        sink_elemental=sink_elemental,
        source_elemental=source_elemental,
        up_propagator=up_propagator,
        down_propagator=down_propagator,
    )

    gamma4 = torch.diag(torch.tensor([1, 1, -1, -1], dtype=torch.complex128))
    cgamma5 = torch.tensor(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
        dtype=torch.complex128,
    )
    direct = torch.einsum(
        "ijk,lmn,ab,ialr,jbms,kdnq,rA,sB,qD,BA->dD",
        sink_elemental,
        source_elemental,
        cgamma5,
        up_propagator,
        down_propagator,
        up_propagator,
        gamma4,
        gamma4,
        gamma4,
        cgamma5.conj().T,
    )
    exchange = torch.einsum(
        "ijk,lmn,ab,ianq,jbms,kdlr,qD,sB,rA,BA->dD",
        sink_elemental,
        source_elemental,
        cgamma5,
        up_propagator,
        down_propagator,
        up_propagator,
        gamma4,
        gamma4,
        gamma4,
        cgamma5.conj().T,
    )
    expected = direct - exchange

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    pplus = (torch.eye(4, dtype=torch.complex128) + gamma4) / 2
    torch.testing.assert_close(
        torch.trace(pplus @ actual),
        actual[0, 0] + actual[1, 1],
        rtol=0,
        atol=0,
    )


def test_local_proton_is_covariant_between_dp_and_dr_spin_bases():
    mode_size = 2
    generator = torch.Generator().manual_seed(2026090817)
    sink_elemental = _complex_random(generator, mode_size, mode_size, mode_size)
    source_elemental = _complex_random(generator, mode_size, mode_size, mode_size)
    up_dp = _complex_random(generator, mode_size, 4, mode_size, 4)
    down_dp = _complex_random(generator, mode_size, 4, mode_size, 4)

    dp_to_dr = torch.tensor(
        [
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 1, 0, 1],
            [-1, 0, -1, 0],
        ],
        dtype=torch.complex128,
    ) / math.sqrt(2.0)
    dr_to_dp = dp_to_dr.conj().T
    identity = torch.eye(4, dtype=torch.complex128)
    torch.testing.assert_close(dr_to_dp @ dp_to_dr, identity, rtol=0, atol=5e-16)

    def rotate_propagator(propagator: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "aA,iAjB,Bb->iajb",
            dp_to_dr,
            propagator,
            dr_to_dp,
        )

    proton_dp = _run_local_proton(
        spin_basis="DP",
        sink_elemental=sink_elemental,
        source_elemental=source_elemental,
        up_propagator=up_dp,
        down_propagator=down_dp,
    )
    proton_dr = _run_local_proton(
        spin_basis="DR",
        sink_elemental=sink_elemental,
        source_elemental=source_elemental,
        up_propagator=rotate_propagator(up_dp),
        down_propagator=rotate_propagator(down_dp),
    )

    expected_dr = dp_to_dr @ proton_dp @ dr_to_dp
    torch.testing.assert_close(proton_dr, expected_dr, rtol=2e-12, atol=2e-12)

    gamma4_dp = torch.diag(torch.tensor([1, 1, -1, -1], dtype=torch.complex128))
    pplus_dp = (identity + gamma4_dp) / 2
    pplus_dr = dp_to_dr @ pplus_dp @ dr_to_dp
    torch.testing.assert_close(
        torch.trace(pplus_dr @ proton_dr),
        torch.trace(pplus_dp @ proton_dp),
        rtol=2e-12,
        atol=2e-12,
    )
