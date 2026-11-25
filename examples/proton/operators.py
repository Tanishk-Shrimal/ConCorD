"""Positive-parity proton at rest, for examples/proton/job.yaml."""

from __future__ import annotations

from concord_lqcd.core import (
    Correlator,
    build_baryon_operator,
    reset_registry,
    spin_matrix,
)


def build() -> dict[str, Correlator]:
    reset_registry()
    base = build_baryon_operator(
        ("u", "d", "u"),
        diquark_gamma="Cg5",
        momentum=(0, 0, 0),
        name="proton_pplus",
    )
    proton = spin_matrix("Pplus", "projected_spin", "de") * base
    return {"proton_pplus": Correlator(proton, proton.dagger())}
