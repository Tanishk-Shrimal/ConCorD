"""Deterministic exact-shape primitive resources for runtime calibration.

These tensors are measurement inputs only.  They may rank already-equivalent
execution policies and test their numerical agreement, but they never justify
symbolic identities or enter production correlator output.
"""

from __future__ import annotations

import hashlib
import json
from math import sqrt
from typing import Iterable

import torch

from concord_lqcd.backend.runtime.materialization import (
    KernelMaterializationProvider,
    PrimitiveMaterializer,
    PropagationSourceHandler,
)
from concord_lqcd.backend.runtime.resource_binding import (
    ExternalTensorSource,
    KernelMaterializationRequest,
    PropagationMaterializationRequest,
    build_resource_binding_plan,
)

from .plan import ConcordPlan
from .schema import ExpandedTimeContext


_COMPLEX_DTYPES = {
    "complex64": (torch.complex64, torch.float32),
    "complex128": (torch.complex128, torch.float64),
}


class SyntheticResourceError(ValueError):
    """Raised when deterministic calibration resources cannot be constructed."""


def _stable_seed(seed: int, payload: object) -> int:
    encoded = json.dumps(
        [seed, payload],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (2**63 - 1)


class DeterministicSyntheticTensorFactory:
    """Lazily cache deterministic CPU tensors by semantic request identity."""

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("synthetic tensor seed must be non-negative")
        self.seed = seed
        self._cache: dict[tuple[object, ...], torch.Tensor] = {}

    def tensor(
        self,
        *,
        kind: str,
        identity: object,
        shape: tuple[int, ...],
        scalar_type: str,
    ) -> torch.Tensor:
        try:
            complex_dtype, real_dtype = _COMPLEX_DTYPES[scalar_type]
        except KeyError as exc:
            raise SyntheticResourceError(
                "synthetic calibration supports complex64 and complex128 only"
            ) from exc
        key = (kind, identity, tuple(shape), scalar_type)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(self.seed, key))
        tensor_shape = tuple(shape) if shape else ()
        real = torch.randn(tensor_shape, dtype=real_dtype, generator=generator)
        imaginary = torch.randn(tensor_shape, dtype=real_dtype, generator=generator)
        normalization = sqrt(max(1, max(shape, default=1)))
        value = torch.complex(real, imaginary).to(complex_dtype) / normalization
        self._cache[key] = value
        return value


class SyntheticKernelProvider(KernelMaterializationProvider):
    def __init__(self, factory: DeterministicSyntheticTensorFactory) -> None:
        self.factory = factory

    def materialize(self, request: KernelMaterializationRequest) -> torch.Tensor:
        return self.factory.tensor(
            kind="kernel",
            identity=(
                request.resource_id,
                request.operator_ids,
                request.operator_times,
            ),
            shape=request.expected_shape,
            scalar_type=request.scalar_type,
        )


class SyntheticPropagationHandler(PropagationSourceHandler):
    def __init__(self, factory: DeterministicSyntheticTensorFactory) -> None:
        self.factory = factory

    def materialize(self, request: PropagationMaterializationRequest) -> torch.Tensor:
        return self.factory.tensor(
            kind="propagation",
            identity=(
                request.resource_id,
                request.source_time,
                request.insertion_times,
                request.sink_time,
            ),
            shape=request.expected_shape,
            scalar_type=request.scalar_type,
        )


def synthetic_propagation_sources(
    resource_ids: Iterable[str],
) -> dict[str, ExternalTensorSource]:
    return {
        resource_id: ExternalTensorSource(
            source_id=f"concord-synthetic:{resource_id}",
            kind="concord-synthetic",
            location=resource_id,
        )
        for resource_id in resource_ids
    }


def build_synthetic_materializer(
    plan: ConcordPlan,
    factory: DeterministicSyntheticTensorFactory,
    time_context: ExpandedTimeContext,
) -> PrimitiveMaterializer:
    if not isinstance(plan, ConcordPlan):
        raise TypeError("plan must be a ConcordPlan")
    if not isinstance(factory, DeterministicSyntheticTensorFactory):
        raise TypeError("factory must be DeterministicSyntheticTensorFactory")
    if not isinstance(time_context, ExpandedTimeContext):
        raise TypeError("time_context must be ExpandedTimeContext")
    binding = build_resource_binding_plan(
        plan.physical_plan,
        operator_times=time_context.operator_times,
        propagation_sources=synthetic_propagation_sources(
            plan.required_propagation_resources
        ),
        spin_basis=plan.config.spin_basis,
    )
    return PrimitiveMaterializer(
        binding,
        kernel_providers={"distillation": SyntheticKernelProvider(factory)},
        source_handlers={
            "concord-synthetic": SyntheticPropagationHandler(factory),
        },
    )


def prime_synthetic_materializer(materializer: PrimitiveMaterializer) -> None:
    if not isinstance(materializer, PrimitiveMaterializer):
        raise TypeError("materializer must be a PrimitiveMaterializer")
    for value_id in materializer.binding_plan.materializations:
        materializer.materialize(value_id)


__all__ = [
    "DeterministicSyntheticTensorFactory",
    "SyntheticKernelProvider",
    "SyntheticPropagationHandler",
    "SyntheticResourceError",
    "build_synthetic_materializer",
    "prime_synthetic_materializer",
    "synthetic_propagation_sources",
]
