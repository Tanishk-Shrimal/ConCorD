"""Spectator-displaced proton used by the spatial LapH validation gate."""

from __future__ import annotations

from concord_lqcd.core import (
    Correlator,
    Operator,
    d,
    epsilon,
    reset_registry,
    spin_matrix,
    transport,
    u,
)


def build() -> dict[str, Correlator]:
    reset_registry()
    proton = Operator(
        epsilon("a", "b", "c")
        * spin_matrix("Cg5", "alpha", "beta")
        * u("x", "a", "alpha")
        * d("x", "b", "beta")
        * transport(u("x", "c", "delta"), "+x"),
        momenta={"x": (0, 0, 0)},
        name="proton_spectator_transport_plus_x",
    )
    return {
        "proton_spectator_transport_plus_x": Correlator(
            proton,
            proton.dagger(),
        )
    }
