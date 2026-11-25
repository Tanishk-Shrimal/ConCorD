"""Tensor-label contraction primitives for the generic runtime executors.

This module owns numerical contraction mechanics only. It may recognize an
already-planned binary contraction as matrix multiplication, but it does not
change tensor connectivity, output index order, or compiler-selected paths.
Unsupported layouts retain the exact opt_einsum fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import prod

import opt_einsum as oe
import torch


CONTRACT_EXPRESSION_CACHE_SIZE = 512
DIRECT_GEMM_MIN_OUTPUT_TO_LAYOUT_COPY_RATIO = 8


@lru_cache(maxsize=CONTRACT_EXPRESSION_CACHE_SIZE)
def cached_contract_expression(
    subscripts: str,
    shapes: tuple[tuple[int, ...], ...],
):
    """Compile one fixed contraction signature without retaining tensor data."""

    explicit_path = [tuple(range(len(shapes)))]
    return oe.contract_expression(
        subscripts,
        *shapes,
        optimize=explicit_path,
    )


def contract_labeled_tensors(
    inputs: tuple[tuple[torch.Tensor, tuple[str, ...]], ...],
    output_indices: tuple[str, ...],
) -> torch.Tensor:
    """Execute one compiler-selected contraction without another path search."""

    expression = labeled_contraction_expression(inputs, output_indices)
    return expression(*(tensor for tensor, _ in inputs), backend="torch")


def labeled_contraction_expression(
    inputs: tuple[tuple[torch.Tensor, tuple[str, ...]], ...],
    output_indices: tuple[str, ...],
):
    """Return the expression ``contract_labeled_tensors`` evaluates.

    It depends only on the index labels and shapes, so it may be built from
    storage-free tensors and reused for every tensor with those shapes.
    """

    if not inputs:
        raise ValueError("A contraction operation requires at least one input value.")

    index_ids: dict[str, int] = {}

    def index_id(label: str) -> int:
        if label not in index_ids:
            index_ids[label] = len(index_ids)
        return index_ids[label]

    input_subscripts: list[str] = []
    present: set[str] = set()
    for tensor, indices in inputs:
        if tensor.ndim != len(indices):
            raise ValueError(
                f"Contraction input rank {tensor.ndim} does not match its "
                f"{len(indices)} index labels."
            )
        present.update(indices)
        input_subscripts.append(
            "".join(oe.get_symbol(index_id(index)) for index in indices)
        )

    if any(index not in present for index in output_indices):
        raise ValueError(
            "Physical contraction output refers to an index absent from its inputs."
        )
    output_subscript = "".join(
        oe.get_symbol(index_id(index)) for index in output_indices
    )
    subscripts = f"{','.join(input_subscripts)}->{output_subscript}"
    shapes = tuple(tuple(tensor.shape) for tensor, _ in inputs)
    return cached_contract_expression(subscripts, shapes)


@dataclass(frozen=True)
class DirectBinaryGemm:
    """A binary labeled contraction proven representable as mm or bmm."""

    left: torch.Tensor
    right: torch.Tensor
    left_matrix_shape: tuple[int, ...]
    right_matrix_shape: tuple[int, ...]
    copy_left_layout: bool
    copy_right_layout: bool
    output_shape: tuple[int, ...]
    matrix_output_shape: tuple[int, ...]
    batched: bool
    # ``left`` is input ``left_input`` permuted by ``left_permutation``;
    # ``right`` is the other input permuted by ``right_permutation``.
    left_input: int = 0
    left_permutation: tuple[int, ...] = ()
    right_permutation: tuple[int, ...] = ()

    @property
    def layout_copy_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor, copy_layout in (
                (self.left, self.copy_left_layout),
                (self.right, self.copy_right_layout),
            )
            if copy_layout
        )


def prepare_direct_binary_gemm(
    inputs: tuple[tuple[torch.Tensor, tuple[str, ...]], ...],
    output_indices: tuple[str, ...],
    *,
    allow_layout_copies: bool = False,
) -> DirectBinaryGemm | None:
    """Recognize a direct-output matrix contraction or return ``None``."""

    if len(inputs) != 2 or len(set(output_indices)) != len(output_indices):
        return None

    def prepare_orientation(
        left_position: int,
        left_input: tuple[torch.Tensor, tuple[str, ...]],
        right_input: tuple[torch.Tensor, tuple[str, ...]],
    ) -> DirectBinaryGemm | None:
        left, left_indices = left_input
        right, right_indices = right_input
        if left.ndim != len(left_indices) or right.ndim != len(right_indices):
            return None
        if len(set(left_indices)) != len(left_indices):
            return None
        if len(set(right_indices)) != len(right_indices):
            return None
        if left.device != right.device or left.dtype != right.dtype:
            return None

        sizes: dict[str, int] = {}
        for tensor, indices in ((left, left_indices), (right, right_indices)):
            for index, size in zip(indices, tensor.shape, strict=True):
                if index in sizes and sizes[index] != size:
                    return None
                sizes[index] = size
        if any(index not in sizes for index in output_indices):
            return None

        shared = set(left_indices).intersection(right_indices)
        retained_shared = tuple(index for index in output_indices if index in shared)
        if not retained_shared:
            batch_index = None
        elif (
            len(retained_shared) == 1
            and output_indices
            and retained_shared[0] == output_indices[0]
            and left_indices[0] == retained_shared[0]
            and right_indices[0] == retained_shared[0]
        ):
            batch_index = retained_shared[0]
        else:
            return None

        contracted = tuple(
            index
            for index in left_indices
            if index in shared and index != batch_index
        )
        if not contracted or any(index in output_indices for index in contracted):
            return None
        if set(contracted) != (
            shared - ({batch_index} if batch_index is not None else set())
        ):
            return None

        left_free = tuple(index for index in left_indices if index not in shared)
        right_free = tuple(index for index in right_indices if index not in shared)
        expected_output = (
            *((batch_index,) if batch_index is not None else ()),
            *left_free,
            *right_free,
        )
        if output_indices != expected_output:
            return None

        left_order = (
            *((batch_index,) if batch_index is not None else ()),
            *left_free,
            *contracted,
        )
        right_order = (
            *((batch_index,) if batch_index is not None else ()),
            *contracted,
            *right_free,
        )
        left_permutation = tuple(left_indices.index(index) for index in left_order)
        right_permutation = tuple(right_indices.index(index) for index in right_order)
        left_view = left.permute(left_permutation)
        right_view = right.permute(right_permutation)

        rows = prod(sizes[index] for index in left_free)
        inner = prod(sizes[index] for index in contracted)
        columns = prod(sizes[index] for index in right_free)
        output_shape = tuple(sizes[index] for index in output_indices)
        if batch_index is None:
            left_matrix_shape = (rows, inner)
            right_matrix_shape = (inner, columns)
            matrix_output_shape = (rows, columns)
            batched = False
        else:
            batch_size = sizes[batch_index]
            left_matrix_shape = (batch_size, rows, inner)
            right_matrix_shape = (batch_size, inner, columns)
            matrix_output_shape = (batch_size, rows, columns)
            batched = True

        def requires_layout_copy(
            tensor: torch.Tensor,
            matrix_shape: tuple[int, ...],
        ) -> bool:
            try:
                tensor.view(matrix_shape)
            except RuntimeError:
                return True
            return False

        copy_left_layout = requires_layout_copy(left_view, left_matrix_shape)
        copy_right_layout = requires_layout_copy(right_view, right_matrix_shape)
        layout_copy_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor, copy_layout in (
                (left_view, copy_left_layout),
                (right_view, copy_right_layout),
            )
            if copy_layout
        )
        if layout_copy_bytes:
            if not allow_layout_copies:
                return None
            output_bytes = (
                prod(output_shape) * left.element_size()
                if output_shape
                else left.element_size()
            )
            if (
                layout_copy_bytes * DIRECT_GEMM_MIN_OUTPUT_TO_LAYOUT_COPY_RATIO
                > output_bytes
            ):
                return None

        return DirectBinaryGemm(
            left=left_view,
            right=right_view,
            left_matrix_shape=left_matrix_shape,
            right_matrix_shape=right_matrix_shape,
            copy_left_layout=copy_left_layout,
            copy_right_layout=copy_right_layout,
            output_shape=output_shape,
            matrix_output_shape=matrix_output_shape,
            batched=batched,
            left_input=left_position,
            left_permutation=left_permutation,
            right_permutation=right_permutation,
        )

    candidates = tuple(
        candidate
        for candidate in (
            prepare_orientation(0, inputs[0], inputs[1]),
            prepare_orientation(1, inputs[1], inputs[0]),
        )
        if candidate is not None
    )
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate.layout_copy_bytes)


def execute_direct_binary_gemm_into(
    contraction: DirectBinaryGemm,
    target: torch.Tensor,
) -> torch.Tensor:
    """Execute a recognized contraction directly into planned storage."""

    if tuple(target.shape) != contraction.output_shape:
        raise ValueError(
            f"Direct contraction target has shape {tuple(target.shape)}, expected "
            f"{contraction.output_shape}."
        )
    target_matrix = target.view(contraction.matrix_output_shape)
    left = (
        contraction.left.contiguous()
        if contraction.copy_left_layout
        else contraction.left
    ).view(contraction.left_matrix_shape)
    right = (
        contraction.right.contiguous()
        if contraction.copy_right_layout
        else contraction.right
    ).view(contraction.right_matrix_shape)
    if contraction.batched:
        torch.bmm(left, right, out=target_matrix)
    else:
        torch.mm(left, right, out=target_matrix)
    return target


__all__ = [
    "CONTRACT_EXPRESSION_CACHE_SIZE",
    "DIRECT_GEMM_MIN_OUTPUT_TO_LAYOUT_COPY_RATIO",
    "DirectBinaryGemm",
    "cached_contract_expression",
    "contract_labeled_tensors",
    "execute_direct_binary_gemm_into",
    "labeled_contraction_expression",
    "prepare_direct_binary_gemm",
]
