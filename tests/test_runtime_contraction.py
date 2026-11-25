"""Numerical contraction mechanics and conservative direct-output lowering."""

from __future__ import annotations

from math import prod

import pytest
import torch

import concord_lqcd.backend.runtime.contraction as contraction_runtime


def _matrix_contraction(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    labels: tuple[str, str, str] = ("left", "contracted", "right"),
) -> torch.Tensor:
    left_index, contracted_index, right_index = labels
    return contraction_runtime.contract_labeled_tensors(
        (
            (left, (left_index, contracted_index)),
            (right, (contracted_index, right_index)),
        ),
        (left_index, right_index),
    )


def test_equivalent_contraction_signatures_compile_once(monkeypatch):
    contraction_runtime.cached_contract_expression.cache_clear()
    original = contraction_runtime.oe.contract_expression
    compilation_calls = 0

    def counted_contract_expression(*args, **kwargs):
        nonlocal compilation_calls
        compilation_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        contraction_runtime.oe,
        "contract_expression",
        counted_contract_expression,
    )
    first_left = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    first_right = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    second_left = first_left.to(torch.complex128) + 2j
    second_right = first_right.to(torch.complex128) - 3j

    first = _matrix_contraction(first_left, first_right)
    second = _matrix_contraction(
        second_left,
        second_right,
        labels=("row", "inner", "column"),
    )

    assert torch.equal(first, first_left @ first_right)
    assert torch.equal(second, second_left @ second_right)
    assert compilation_calls == 1


def test_shape_changes_compile_distinct_expressions(monkeypatch):
    contraction_runtime.cached_contract_expression.cache_clear()
    original = contraction_runtime.oe.contract_expression
    compilation_calls = 0

    def counted_contract_expression(*args, **kwargs):
        nonlocal compilation_calls
        compilation_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        contraction_runtime.oe,
        "contract_expression",
        counted_contract_expression,
    )

    _matrix_contraction(torch.ones(2, 3), torch.ones(3, 4))
    _matrix_contraction(torch.ones(2, 5), torch.ones(5, 4))

    assert compilation_calls == 2


def test_contraction_expression_cache_is_bounded():
    cache_info = contraction_runtime.cached_contract_expression.cache_info()

    assert cache_info.maxsize == contraction_runtime.CONTRACT_EXPRESSION_CACHE_SIZE
    assert cache_info.maxsize is not None


@pytest.mark.parametrize("batched", (False, True))
def test_direct_binary_gemm_matches_generic_contraction(batched):
    generator = torch.Generator().manual_seed(19)
    mode_size = 5
    spin_size = 2
    if batched:
        batch_size = 3
        left = torch.randn(
            batch_size,
            mode_size,
            mode_size,
            mode_size,
            dtype=torch.complex128,
            generator=generator,
        )
        right = torch.randn(
            batch_size,
            spin_size,
            spin_size,
            mode_size,
            mode_size,
            dtype=torch.complex128,
            generator=generator,
        )
        inputs = (
            (left, ("batch", "left", "right", "contracted")),
            (right, ("batch", "spin_a", "spin_b", "mode", "left")),
        )
        output_indices = (
            "batch",
            "right",
            "contracted",
            "spin_a",
            "spin_b",
            "mode",
        )
    else:
        left = torch.randn(
            mode_size,
            mode_size,
            mode_size,
            dtype=torch.complex128,
            generator=generator,
        )
        right = torch.randn(
            spin_size,
            spin_size,
            mode_size,
            mode_size,
            dtype=torch.complex128,
            generator=generator,
        )
        inputs = (
            (left, ("left", "right", "contracted")),
            (right, ("spin_a", "spin_b", "mode", "left")),
        )
        output_indices = (
            "right",
            "contracted",
            "spin_a",
            "spin_b",
            "mode",
        )

    expected = contraction_runtime.contract_labeled_tensors(inputs, output_indices)
    contraction = contraction_runtime.prepare_direct_binary_gemm(
        inputs,
        output_indices,
    )
    assert contraction is not None
    actual = torch.empty_like(expected)
    returned = contraction_runtime.execute_direct_binary_gemm_into(
        contraction,
        actual,
    )

    assert returned.data_ptr() == actual.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_direct_binary_gemm_retains_general_fallback_for_non_view_layout():
    generator = torch.Generator().manual_seed(23)
    left = torch.randn(3, 4, 5, dtype=torch.complex128, generator=generator)
    right = torch.randn(4, 6, dtype=torch.complex128, generator=generator)
    inputs = (
        (left, ("left", "contracted", "right")),
        (right, ("contracted", "output")),
    )
    output_indices = ("left", "right", "output")

    assert contraction_runtime.prepare_direct_binary_gemm(
        inputs,
        output_indices,
    ) is None
    result = contraction_runtime.contract_labeled_tensors(inputs, output_indices)
    expected = torch.einsum("icj,ck->ijk", left, right)
    torch.testing.assert_close(result, expected, rtol=1e-13, atol=1e-13)


def test_direct_binary_gemm_can_trade_small_cpu_layout_copy_for_large_output():
    generator = torch.Generator().manual_seed(29)
    batch_size = 2
    mode_size = 32
    spin_size = 4
    left = torch.randn(
        batch_size,
        mode_size,
        mode_size,
        mode_size,
        dtype=torch.complex128,
        generator=generator,
    )
    right = torch.randn(
        batch_size,
        spin_size,
        spin_size,
        mode_size,
        mode_size,
        dtype=torch.complex128,
        generator=generator,
    )
    inputs = (
        (left, ("batch", "left", "contracted", "right")),
        (right, ("batch", "spin_a", "spin_b", "mode", "contracted")),
    )
    output_indices = (
        "batch",
        "left",
        "right",
        "spin_a",
        "spin_b",
        "mode",
    )

    assert contraction_runtime.prepare_direct_binary_gemm(
        inputs,
        output_indices,
    ) is None
    contraction = contraction_runtime.prepare_direct_binary_gemm(
        inputs,
        output_indices,
        allow_layout_copies=True,
    )
    assert contraction is not None
    assert contraction.copy_left_layout
    assert not contraction.copy_right_layout
    assert (
        contraction.layout_copy_bytes
        * contraction_runtime.DIRECT_GEMM_MIN_OUTPUT_TO_LAYOUT_COPY_RATIO
        <= prod(contraction.output_shape) * left.element_size()
    )

    expected = contraction_runtime.contract_labeled_tensors(inputs, output_indices)
    actual = torch.empty_like(expected)
    contraction_runtime.execute_direct_binary_gemm_into(contraction, actual)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_direct_binary_gemm_rejects_expensive_layout_copies():
    generator = torch.Generator().manual_seed(31)
    left = torch.randn(3, 4, 5, dtype=torch.complex128, generator=generator)
    right = torch.randn(4, 6, dtype=torch.complex128, generator=generator)
    inputs = (
        (left, ("left", "contracted", "right")),
        (right, ("contracted", "output")),
    )

    assert contraction_runtime.prepare_direct_binary_gemm(
        inputs,
        ("left", "right", "output"),
        allow_layout_copies=True,
    ) is None
