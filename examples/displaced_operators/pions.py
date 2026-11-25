"""Spatial nonlocal pion validation channels."""

from __future__ import annotations

from concord_lqcd.core import (
    Correlator,
    Operator,
    dbar,
    derivative,
    gamma5,
    reset_registry,
    transport,
    u,
)


def _pion(transformed_u) -> Operator:
    return Operator(
        dbar("x", "color", "dbar_spin")
        * gamma5("dbar_spin", "u_spin")
        * transformed_u
    )


def build() -> dict[str, Correlator]:
    reset_registry()
    transformed_fields = {
        "pion_derivative_x": derivative(u("x", "color", "u_spin"), "+x"),
        "pion_derivative_y": derivative(u("x", "color", "u_spin"), "+y"),
        "pion_derivative_z": derivative(u("x", "color", "u_spin"), "+z"),
        "pion_transport_plus_x": transport(u("x", "color", "u_spin"), "+x"),
        "pion_transport_plus_y": transport(u("x", "color", "u_spin"), "+y"),
        "pion_transport_plus_z": transport(u("x", "color", "u_spin"), "+z"),
        "pion_transport_plus_x_plus_y": transport(
            u("x", "color", "u_spin"), "+x", "+y"
        ),
    }
    correlators: dict[str, Correlator] = {}
    for name, transformed in transformed_fields.items():
        operator = _pion(transformed)
        correlators[name] = Correlator(operator, operator.dagger())
    return correlators
