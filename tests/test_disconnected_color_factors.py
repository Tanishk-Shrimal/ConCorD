"""Independent contractions of color factors without quark endpoints."""

from functools import reduce
from itertools import permutations
from operator import mul

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
from concord_lqcd.compiler.distillation_lowering import identify_operator_vertices
from concord_lqcd.compiler.prewick import lower_correlator_to_physics_ir
from concord_lqcd.core import (
    ColorDelta, ColorEpsilon, Correlator, Operator, Product,
    delta, epsilon, link, u, ubar, validate,
)


def _loop():
    return Operator(ubar("x", "field_color", "spin") * u("x", "field_color", "spin"))


def _independent_color_value(expression):
    """Evaluate small color networks with dense NumPy Einstein contraction."""
    levi_civita = np.zeros((3, 3, 3), dtype=np.int64)
    for ordering in permutations(range(3)):
        inversions = sum(ordering[i] > ordering[j] for i in range(3) for j in range(i + 1, 3))
        levi_civita[ordering] = (-1) ** inversions
    factors = expression.factors if isinstance(expression, Product) else (expression,)
    indices = tuple(dict.fromkeys(index for factor in factors for index in factor.indices))
    numbers = {index: number for number, index in enumerate(indices)}
    arguments = []
    for factor in factors:
        values = np.eye(3, dtype=np.int64) if isinstance(factor, ColorDelta) else levi_civita
        assert isinstance(factor, (ColorDelta, ColorEpsilon))
        arguments.extend((values, [numbers[index] for index in factor.indices]))
    arguments.append([])
    return np.einsum(*arguments, optimize=False).item()


def _execute(correlator, *, operator_times=None, shared_basis=False):
    vectors = torch.eye(3, 2, dtype=torch.complex128).reshape(1, 1, 3, 2)
    perambulator = torch.eye(8, dtype=torch.complex128).reshape(2, 4, 2, 4)

    def provider_factory(request):
        fields = {"u": vectors}
        return DistillationTensorProvider(fields, fields, {})

    def source_resolver(resource_id, recipe):
        return ExternalTensorSource(source_id=resource_id, kind="memory", location="u")

    compilation = compile_correlators(
        {"C": correlator},
        RuntimeCompilationConfig(mode_size=2, shared_orthonormal_laph_basis=shared_basis),
    )
    execution = RuntimeExecutionConfig(
        operator_times=operator_times or {1: (0,)},
        propagation_source_resolver=source_resolver,
        kernel_providers={
            "distillation": CompiledKernelMaterializationProvider(provider_factory),
        },
        source_handlers={"memory": InMemoryTensorSourceHandler({"u": perambulator})},
    )
    actual = execute_compilation(compilation, execution).output("C").item()
    reference = -torch.einsum(
        "ij,jsis->", vectors.reshape(3, 2).conj().T @ vectors.reshape(3, 2), perambulator,
    ).item()
    return compilation, actual, reference


_NETWORKS = (
    delta("a", "a"),
    delta("a", "b") * delta("b", "c") * delta("c", "a"),
    delta("a", "a") * delta("b", "b"),
    epsilon("a", "b", "c") * epsilon("a", "b", "c"),
    epsilon("a", "b", "c") * epsilon("b", "a", "c"),
    epsilon("a", "b", "c") * delta("a", "d") * epsilon("d", "b", "c"),
    epsilon("a", "a", "b") * epsilon("b", "c", "c"),
    reduce(mul, (
        epsilon("a", "b", "c"), epsilon("a", "d", "e"),
        epsilon("b", "d", "f"), epsilon("c", "e", "f"),
    )),
)


@pytest.mark.parametrize("expression", _NETWORKS)
@pytest.mark.parametrize("shared_basis", (False, True))
def test_disconnected_color_network_matches_independent_spatial_contraction(expression, shared_basis):
    operator = _loop() * expression
    assert validate(operator) == (True, [])
    factor = _independent_color_value(expression)
    _, actual, base_reference = _execute(Correlator(operator), shared_basis=shared_basis)
    assert base_reference == -8
    np.testing.assert_allclose(actual, factor * base_reference, rtol=0, atol=0)


def test_color_trace_multiplies_real_kernel_output_by_three():
    _, base, _ = _execute(Correlator(_loop()))
    _, traced, _ = _execute(Correlator(_loop() * delta("a", "a")))
    assert base == -8
    assert traced == -24


@pytest.mark.parametrize("expression", _NETWORKS)
def test_pure_color_correlator_is_its_scalar_contraction(expression):
    _, actual, _ = _execute(Correlator(Operator(expression)))
    assert actual == _independent_color_value(expression)


@pytest.mark.parametrize("shared_basis", (False, True))
def test_pure_color_insertion_preserves_original_operator_time_arity(shared_basis):
    _, actual, _ = _execute(
        Correlator(_loop(), Operator(delta("a", "a"))),
        operator_times={2: (4, 9)}, shared_basis=shared_basis,
    )
    assert actual == -24


@pytest.mark.parametrize("expression", (delta("a", "b"), link("y", "y", "a", "a")))
def test_endpoint_free_open_color_and_gauge_components_are_rejected(expression):
    correlator = Correlator(_loop(), Operator(expression))
    physics_ir = lower_correlator_to_physics_ir(correlator)
    with pytest.raises(NotImplementedError, match="Disconnected"):
        identify_operator_vertices(physics_ir.correlators[0])
    with pytest.raises(NotImplementedError, match="Disconnected"):
        compile_correlators(correlator, RuntimeCompilationConfig(mode_size=2))


def test_an_earlier_zero_color_factor_cannot_hide_an_unsupported_gauge_component():
    zero = epsilon("a", "a", "b") * epsilon("b", "c", "c")
    operator = _loop() * zero * Operator(link("y", "y", "g", "g"))
    with pytest.raises(NotImplementedError, match="Disconnected GaugeLink"):
        compile_correlators(Correlator(operator), RuntimeCompilationConfig(mode_size=2))
