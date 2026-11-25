"""Unit test for dynamic spatial derivative calculation in DistillationTensorProvider."""

import pytest
import torch
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider
from concord_lqcd.backend.runtime.context import ExecutionContext


def test_dynamic_spatial_derivative_calculation():
    # Spatial grid axes: [batch, z, y, x, color, mode].
    V_raw = torch.randn(1, 4, 4, 4, 3, 2, dtype=torch.complex128)
    provider = DistillationTensorProvider(
        V_t_snk={"c": V_raw},
        V_t_src={"c": V_raw},
        momenta_phases={},
    )

    ctx = ExecutionContext.create("cpu", 0, [0])

    # Request derivative "+x" when only raw "c" eigenvector is supplied
    V_dx = provider.eigenvector(
        flavor="c",
        is_anti=False,
        field_attributes={"derivative": ["+x"]},
        is_sink=True,
        context=ctx,
    )

    # Expected central difference along x (spatial grid axis -3)
    v_fwd = torch.roll(V_raw, shifts=-1, dims=-3)
    v_bwd = torch.roll(V_raw, shifts=+1, dims=-3)
    expected_V_dx = 0.5 * (v_fwd - v_bwd)

    torch.testing.assert_close(V_dx, expected_V_dx)
