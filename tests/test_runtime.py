"""Unit tests for runtime components shared by the production pipeline."""

import torch

from concord_lqcd.backend.runtime import ExecutionContext
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider


def test_execution_context_creation():
    ctx = ExecutionContext.create(
        device="cpu",
        t_src=0,
        t_snk_list=[0, 1, 2],
        batch_size=12,
        lat_shape=(24, 24, 24),
    )
    assert isinstance(ctx.device, torch.device)
    assert ctx.t_src == 0
    assert ctx.t_snk_list == (0, 1, 2)
    assert ctx.dtype == torch.float64
    assert ctx.complex_dtype == torch.complex128


def test_distillation_provider_materializes_shared_tensor_primitives():
    eigenvectors = torch.ones((1, 10, 3, 4), dtype=torch.complex128)
    provider = DistillationTensorProvider(
        {"u": eigenvectors},
        {"u": eigenvectors},
        {},
    )
    ctx = ExecutionContext.create("cpu", 0, [0])

    eig = provider.eigenvector("u", False, {}, True, ctx)
    assert eig.shape == (1, 10, 3, 4)

    delta = provider.color_delta(3, torch.complex128, ctx)
    assert delta.shape == (3, 3)

    epsilon = provider.color_epsilon(torch.complex128, ctx)
    assert epsilon.shape == (3, 3, 3)
