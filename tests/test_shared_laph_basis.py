"""Independent overlap checks for the compiler's LapH basis contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime import (
    RuntimeCompilationConfig,
    RuntimeExecutionConfig,
    compile_correlators,
    execute_compilation,
)
from concord_lqcd.backend.runtime.materialization import (
    CompiledKernelMaterializationProvider,
    InMemoryTensorSourceHandler,
)
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider
from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource
from concord_lqcd.compiler.global_cse import lower_physics_ir_collection_to_global_cse
from concord_lqcd.core import Correlator, Operator, c, cbar, reset_registry, u, ubar


def _scalar_meson():
    reset_registry()
    sink = Operator(cbar("y", "a", "sink_spin") * u("y", "a", "sink_spin"))
    source = Operator(ubar("x", "b", "source_spin") * c("x", "b", "source_spin"))
    return Correlator(sink, source)


def _complex_random(generator, shape):
    return generator.normal(size=shape) + 1j * generator.normal(size=shape)


def _inputs(*, distinct_flavor_bases):
    generator = np.random.default_rng(8102026)
    rotation = np.array(
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.complex128
    )
    fields = {}
    overlaps = {}
    for time, antiquark_flavor, quark_flavor in ((1, "c", "u"), (0, "u", "c")):
        basis = np.linalg.qr(_complex_random(generator, (12, 3)))[0]
        other = basis @ rotation if distinct_flavor_bases else basis.copy()
        flavor_fields = {"u": basis, "c": other}
        for values in flavor_fields.values():
            np.testing.assert_allclose(
                values.conj().T @ values, np.eye(3), rtol=0, atol=1e-14
            )
        fields[time] = {
            flavor: torch.from_numpy(values.reshape(1, 4, 3, 3))
            for flavor, values in flavor_fields.items()
        }
        overlaps[time] = (
            flavor_fields[antiquark_flavor].conj().T @ flavor_fields[quark_flavor]
        )
    propagators = {
        flavor: _complex_random(generator, (3, 4, 3, 4)) for flavor in ("u", "c")
    }
    reference = -np.einsum(
        "ij,jskv,kl,lvis->",
        overlaps[1], propagators["u"], overlaps[0], propagators["c"],
    )
    identity_reference = -np.einsum(
        "iskv,kvis->", propagators["u"], propagators["c"]
    )
    return fields, propagators, reference, identity_reference


def _execute(fields, propagators, *, shared_basis):
    compilation = compile_correlators(
        {"meson": _scalar_meson()},
        RuntimeCompilationConfig(mode_size=3, shared_orthonormal_laph_basis=shared_basis),
    )

    def provider_factory(request):
        values = fields[request.operator_times[0]]
        return DistillationTensorProvider(values, values, {})

    def source_resolver(resource_id, recipe):
        return ExternalTensorSource(
            source_id=resource_id, kind="memory", location=recipe.segment_flavors[0]
        )

    execution = RuntimeExecutionConfig(
        operator_times={2: (1, 0)},
        propagation_source_resolver=source_resolver,
        kernel_providers={
            "distillation": CompiledKernelMaterializationProvider(provider_factory)
        },
        source_handlers={"memory": InMemoryTensorSourceHandler(propagators)},
    )
    actual = execute_compilation(compilation, execution).output("meson").item()
    return compilation, actual


def test_orthonormal_but_distinct_flavor_bases_keep_their_nonidentity_overlaps():
    fields, propagators, reference, identity_reference = _inputs(
        distinct_flavor_bases=True
    )
    assert abs(reference) > 0.1
    assert abs(reference - identity_reference) > 0.1

    compilation, actual = _execute(fields, propagators, shared_basis=False)

    assert len(compilation.required_kernel_resource_ids) == 2
    np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)


def test_declared_shared_basis_shortcut_matches_explicit_overlap_evaluation():
    fields, propagators, reference, identity_reference = _inputs(
        distinct_flavor_bases=False
    )
    explicit, explicit_result = _execute(fields, propagators, shared_basis=False)
    optimized, optimized_result = _execute(fields, propagators, shared_basis=True)

    assert len(explicit.required_kernel_resource_ids) == 2
    assert optimized.required_kernel_resource_ids == ()
    assert abs(reference) > 0.1
    np.testing.assert_allclose(reference, identity_reference, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(explicit_result, reference, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(optimized_result, reference, rtol=1e-13, atol=1e-13)


def test_common_but_nonorthonormal_inputs_keep_their_gram_matrices_by_default():
    fields, propagators, reference, identity_reference = _inputs(
        distinct_flavor_bases=False
    )
    scaled_fields = {
        time: {flavor: 2 * values for flavor, values in flavor_fields.items()}
        for time, flavor_fields in fields.items()
    }
    scaled_reference = 16 * reference
    assert abs(scaled_reference - identity_reference) > 0.1

    compilation, actual = _execute(scaled_fields, propagators, shared_basis=False)

    assert len(compilation.required_kernel_resource_ids) == 2
    np.testing.assert_allclose(actual, scaled_reference, rtol=1e-13, atol=1e-13)


def test_global_lowering_does_not_infer_shared_basis_from_mode_size():
    compilation = compile_correlators(_scalar_meson(), RuntimeCompilationConfig(mode_size=3))
    explicit = lower_physics_ir_collection_to_global_cse(
        compilation.physics_ir, mode_size=3
    )
    optimized = lower_physics_ir_collection_to_global_cse(
        compilation.physics_ir, mode_size=3, shared_orthonormal_laph_basis=True
    )

    assert sum(resource.kind == "kernel" for resource in explicit.resources.values()) == 2
    assert all(resource.kind != "kernel" for resource in optimized.resources.values())


def test_global_lowering_rejects_a_shared_basis_contract_without_mode_size():
    with pytest.raises(ValueError, match="requires mode_size"):
        lower_physics_ir_collection_to_global_cse({}, shared_orthonormal_laph_basis=True)


@pytest.mark.parametrize("declaration", (None, 1, "true"))
def test_global_lowering_rejects_nonboolean_basis_contracts(declaration):
    with pytest.raises(TypeError, match="must be a boolean"):
        lower_physics_ir_collection_to_global_cse(
            {}, mode_size=3, shared_orthonormal_laph_basis=declaration
        )
