"""Physics-formula and report tests for the visualization entry layer."""

from __future__ import annotations

from concord_lqcd.core import (
    Correlator,
    Operator,
    d,
    dbar,
    derivative,
    epsilon,
    gamma5,
    spin_matrix,
    transport,
    u,
)
from concord_lqcd.visualization import latex_correlator, latex_operator


def test_pion_operator_and_correlator_have_conventional_latex():
    pion = Operator(
        dbar("x", "a", "s1")
        * gamma5("s1", "s2")
        * u("x", "a", "s2"),
        momentum=(1, 0, -1),
    )
    correlator = Correlator(pion, pion.dagger())

    operator_latex = latex_operator(pion)
    correlator_latex = latex_correlator("pion", correlator)

    assert r"\sum_{\mathbf{x}}" in operator_latex
    assert r"e^{+2\pi i" in operator_latex
    assert r"\bar{d}_{a\alpha}" in operator_latex
    assert r"\gamma_{5}" in operator_latex
    assert r"u_{a\beta}" in operator_latex
    assert correlator_latex.startswith(r"C_{\mathrm{pion}}")
    assert r"\left\langle" in correlator_latex


def test_baryon_derivative_and_transport_are_rendered_as_physics_operations():
    proton = Operator(
        epsilon("a", "b", "c")
        * spin_matrix("Cg5", "s1", "s2")
        * u("x", "a", "s1")
        * derivative(d("x", "b", "s2"), "+y")
        * transport(u("x", "c", "s3"), "+x", "-z"),
        momentum=(0, 0, 0),
    )

    operator_latex = latex_operator(proton)

    assert r"\varepsilon_{abc}" in operator_latex
    assert r"C\,\gamma_{5}" in operator_latex
    assert r"\nabla_{+\hat{y}}" in operator_latex
    assert r"\mathcal{U}_{+\hat{x},-\hat{z}}" in operator_latex
