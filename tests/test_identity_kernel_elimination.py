"""Exact LapH mode-identity detection and graph-rewrite tests."""

from __future__ import annotations

import pytest
import torch

from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.compiler.contraction_graph import (
    PrimitiveTensorNode,
    TensorLegBinding,
    WickContractionGraph,
)
from concord_lqcd.compiler.distillation_lowering import (
    KernelRecipe,
    ProjectedField,
    ProjectionContribution,
)
from concord_lqcd.compiler.physics_ir import (
    ColorTensorRecord,
    GaugeTensorRecord,
    OperationKind,
    OperationRecord,
    QuarkRecord,
)
from concord_lqcd.core import Correlator, Operator, c, cbar, gamma5, reset_registry, u, ubar


def _identity_recipe(
    *,
    momentum: tuple[int, int, int] = (0, 0, 0),
    first_operations: tuple[OperationRecord, ...] = (),
    color_tensors: tuple[ColorTensorRecord, ...] = (),
    gauge_tensors: tuple[GaugeTensorRecord, ...] = (),
    second_position: str = "x",
    second_color: str = "a",
) -> KernelRecipe:
    first_record = QuarkRecord("u", True, "x", "a", "s0")
    second_record = QuarkRecord("d", False, second_position, second_color, "s1")
    positions = tuple(dict.fromkeys(("x", second_position)))
    return KernelRecipe(
        vertex_id="identity-candidate",
        position_labels=positions,
        momenta={position: momentum for position in positions},
        projected_fields=(
            ProjectedField(
                record=first_record,
                contribution=ProjectionContribution.EIGENVECTOR_DAGGER,
                operations=first_operations,
            ),
            ProjectedField(
                record=second_record,
                contribution=ProjectionContribution.EIGENVECTOR,
            ),
        ),
        color_tensors=color_tensors,
        gauge_tensors=gauge_tensors,
        open_ports=("mode_0", "mode_1"),
    )


def _identity_node(node_id: str, first_index: str, second_index: str):
    return PrimitiveTensorNode(
        node_id=node_id,
        kind="kernel",
        resource_id=f"resource-{node_id}",
        bindings=(
            TensorLegBinding("mode_0", first_index),
            TensorLegBinding("mode_1", second_index),
        ),
        payload=_identity_recipe(),
    )


def _endpoint_node(node_id: str, index: str):
    return PrimitiveTensorNode(
        node_id=node_id,
        kind="propagation",
        resource_id=f"resource-{node_id}",
        bindings=(TensorLegBinding("mode", index),),
        payload=object(),
    )


def test_exact_identity_detector_accepts_only_plain_local_zero_momentum_vdagger_v():
    assert _identity_recipe().is_exact_mode_identity


@pytest.mark.parametrize(
    "recipe",
    (
        _identity_recipe(momentum=(1, 0, 0)),
        _identity_recipe(
            first_operations=(
                OperationRecord(OperationKind.DERIVATIVE, {"direction": "+x"}),
            )
        ),
        _identity_recipe(
            first_operations=(
                OperationRecord(OperationKind.TRANSPORT, {"path": ("+x",)}),
            )
        ),
        _identity_recipe(
            color_tensors=(ColorTensorRecord("delta", ("a", "a")),)
        ),
        _identity_recipe(
            gauge_tensors=(GaugeTensorRecord("x", "x", "a", "a"),)
        ),
        _identity_recipe(second_position="y"),
        _identity_recipe(second_color="b"),
    ),
    ids=(
        "momentum",
        "derivative",
        "transport",
        "color-tensor",
        "gauge-link",
        "nonlocal",
        "uncontracted-color",
    ),
)
def test_exact_identity_detector_rejects_unproven_cases(recipe):
    assert not recipe.is_exact_mode_identity


def test_identity_rewrite_unifies_contracted_mode_indices():
    graph = WickContractionGraph(
        coefficient=2.0 - 3.0j,
        nodes=(
            _endpoint_node("left", "left-mode"),
            _identity_node("identity", "left-mode", "right-mode"),
            _endpoint_node("right", "right-mode"),
        ),
    )

    rewritten = graph.eliminate_exact_mode_identity_kernels(mode_size=75)

    assert rewritten.coefficient == graph.coefficient
    assert tuple(node.node_id for node in rewritten.nodes) == ("left", "right")
    assert rewritten.nodes[0].legs == rewritten.nodes[1].legs
    rewritten.validate()


@pytest.mark.parametrize("mode_size", (1, 4, 64, 75))
def test_closed_identity_cycle_contributes_exact_mode_trace(mode_size):
    graph = WickContractionGraph(
        coefficient=-0.5j,
        nodes=(
            _identity_node("identity-0", "mode-a", "mode-b"),
            _identity_node("identity-1", "mode-b", "mode-a"),
        ),
    )

    rewritten = graph.eliminate_exact_mode_identity_kernels(mode_size=mode_size)

    assert rewritten.nodes == ()
    assert rewritten.coefficient == graph.coefficient * mode_size


def test_identity_chain_rewrite_matches_explicit_tensor_contraction():
    mode_size = 7
    generator = torch.Generator().manual_seed(20260909)
    left = torch.randn(mode_size, dtype=torch.float64, generator=generator)
    right = torch.randn(mode_size, dtype=torch.float64, generator=generator)
    identity = torch.eye(mode_size, dtype=torch.float64)

    explicit = torch.einsum("i,ij,j->", left, identity, right)
    rewritten = torch.einsum("i,i->", left, right)

    assert torch.equal(explicit, rewritten)


def test_identity_with_free_mode_indices_remains_explicit():
    graph = WickContractionGraph(
        coefficient=1.0,
        nodes=(_identity_node("identity", "free-a", "free-b"),),
    )

    assert graph.eliminate_exact_mode_identity_kernels(mode_size=64) is graph


def _meson_two_point(momentum: tuple[int, int, int]) -> Correlator:
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s0") * gamma5("s0", "s1") * u("y", "a", "s1"),
        momentum=momentum,
    )
    source = Operator(
        ubar("x", "b", "s2") * gamma5("s2", "s3") * c("x", "b", "s3"),
        momentum=tuple(-component for component in momentum),
    )
    return Correlator(sink, source)


def test_production_compilation_eliminates_zero_momentum_bilinear_kernels():
    compilation = compile_correlators(
        _meson_two_point((0, 0, 0)),
        RuntimeCompilationConfig(mode_size=75, shared_orthonormal_laph_basis=True),
    )

    assert compilation.required_kernel_resource_ids == ()
    assert {
        resource.kind for resource in compilation.global_cse.resources.values()
    } == {"propagation", "spin"}


def test_production_compilation_retains_nonzero_momentum_bilinear_kernels():
    compilation = compile_correlators(
        _meson_two_point((1, 0, 0)),
        RuntimeCompilationConfig(mode_size=75, shared_orthonormal_laph_basis=True),
    )

    assert len(compilation.required_kernel_resource_ids) == 2
    assert any(
        resource.kind == "kernel"
        for resource in compilation.global_cse.resources.values()
    )


@pytest.mark.parametrize("mode_size", (0, -1))
def test_identity_rewrite_rejects_nonpositive_mode_size(mode_size):
    graph = WickContractionGraph(1.0, ())

    with pytest.raises(ValueError, match="positive integer"):
        graph.eliminate_exact_mode_identity_kernels(mode_size=mode_size)


@pytest.mark.parametrize("mode_size", (True, 2.5, "64"))
def test_identity_rewrite_rejects_noninteger_mode_size(mode_size):
    graph = WickContractionGraph(1.0, ())

    with pytest.raises(TypeError, match="positive integer"):
        graph.eliminate_exact_mode_identity_kernels(mode_size=mode_size)


def test_mode_size_alone_does_not_eliminate_unproven_overlap_kernels():
    compilation = compile_correlators(
        _meson_two_point((0, 0, 0)),
        RuntimeCompilationConfig(mode_size=3),
    )

    assert len(compilation.required_kernel_resource_ids) == 2


@pytest.mark.parametrize("declaration", (None, 1, "true"))
def test_basis_contract_requires_a_boolean(declaration):
    with pytest.raises(TypeError, match="shared_orthonormal_laph_basis"):
        RuntimeCompilationConfig(
            mode_size=3, shared_orthonormal_laph_basis=declaration
        )
