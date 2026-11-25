"""Local pion (pi+) at zero momentum, for examples/pion/job.yaml."""

from __future__ import annotations

from concord_lqcd.core import Correlator, Operator, dbar, gamma5, u


def build() -> dict[str, Correlator]:
    """Build the single-channel pion two-point correlator at zero momentum."""
    # O_pion(x) = dbar(x, color, s1) * gamma5(s1, s2) * u(x, color, s2)
    pion = Operator(
        dbar("x", "color", "s1")
        * gamma5("s1", "s2")
        * u("x", "color", "s2")
    )
    return {"pion": Correlator(pion, pion.dagger())}
