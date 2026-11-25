"""Exact spatial LapH field-operation realization tests."""

from __future__ import annotations

import pytest
import torch

from concord_lqcd.compiler.physics_ir import OperationKind, OperationRecord
from concord_lqcd.job.data import (
    _apply_laph_operations,
    _shift_path,
    _transport_one_hop,
    _transport_path,
)


def _field() -> torch.Tensor:
    values = torch.arange(1 * 2 * 2 * 3 * 3 * 2, dtype=torch.float64)
    values = values.reshape(1, 2, 2, 3, 3, 2)
    return values.to(torch.complex128) + 0.25j * values.flip(-1)


def _links(offset: float) -> torch.Tensor:
    links = torch.zeros((1, 2, 2, 3, 3, 3), dtype=torch.complex128)
    for z in range(2):
        for y in range(2):
            for x in range(3):
                matrix = torch.eye(3, dtype=torch.complex128)
                matrix[0, 1] = offset + z + 2 * y + 3 * x + 0.5j
                matrix[2, 0] = 0.25 * (offset + x + 1) - 0.125j
                links[0, z, y, x] = matrix
    return links


def _site(
    tensor: torch.Tensor,
    *,
    z: int,
    y: int,
    x: int,
) -> torch.Tensor:
    return tensor[0, z % 2, y % 2, x % 3]


def test_forward_and_backward_transport_match_explicit_periodic_formula():
    field = _field()
    link_x = _links(1.0)

    forward = _transport_one_hop(field, "+x", link_x)
    backward = _transport_one_hop(field, "-x", link_x)

    for z in range(2):
        for y in range(2):
            for x in range(3):
                expected_forward = _site(link_x, z=z, y=y, x=x) @ _site(
                    field, z=z, y=y, x=x + 1
                )
                previous_link = _site(link_x, z=z, y=y, x=x - 1)
                expected_backward = previous_link.conj().T @ _site(
                    field, z=z, y=y, x=x - 1
                )
                torch.testing.assert_close(
                    _site(forward, z=z, y=y, x=x), expected_forward
                )
                torch.testing.assert_close(
                    _site(backward, z=z, y=y, x=x), expected_backward
                )


def test_two_hop_transport_follows_path_left_to_right_from_field_position():
    field = _field()
    link_x = _links(1.0)
    link_y = _links(7.0)
    links = {"x": link_x, "y": link_y}

    transported = _transport_path(
        field,
        ("+x", "+y"),
        links.__getitem__,
    )

    for z in range(2):
        for y in range(2):
            for x in range(3):
                expected = (
                    _site(link_x, z=z, y=y, x=x)
                    @ _site(link_y, z=z, y=y, x=x + 1)
                    @ _site(field, z=z, y=y + 1, x=x + 1)
                )
                torch.testing.assert_close(
                    _site(transported, z=z, y=y, x=x), expected
                )


def test_dimensionless_signed_covariant_derivative_and_operation_order():
    field = _field()
    link_x = _links(1.0)
    link_y = _links(7.0)
    links = {"x": link_x, "y": link_y}

    positive = _apply_laph_operations(
        field,
        (OperationRecord(OperationKind.DERIVATIVE, {"direction": "+x"}),),
        links.__getitem__,
    )
    negative = _apply_laph_operations(
        field,
        (OperationRecord(OperationKind.DERIVATIVE, {"direction": "-x"}),),
        links.__getitem__,
    )
    expected = 0.5 * (
        _transport_one_hop(field, "+x", link_x)
        - _transport_one_hop(field, "-x", link_x)
    )
    torch.testing.assert_close(positive, expected)
    torch.testing.assert_close(negative, -expected)

    operations = (
        OperationRecord(OperationKind.TRANSPORT, {"path": ("+y",)}),
        OperationRecord(OperationKind.DERIVATIVE, {"direction": "+x"}),
    )
    actual_ordered = _apply_laph_operations(field, operations, links.__getitem__)
    transported_first = _transport_one_hop(field, "+y", link_y)
    expected_ordered = 0.5 * (
        _transport_one_hop(transported_first, "+x", link_x)
        - _transport_one_hop(transported_first, "-x", link_x)
    )
    torch.testing.assert_close(actual_ordered, expected_ordered)


def test_pure_shift_needs_no_gauge_links():
    field = _field()
    shifted = _shift_path(field, ("+x", "-y"))
    expected = torch.roll(field, shifts=(-1, 1), dims=(3, 2))
    torch.testing.assert_close(shifted, expected)


@pytest.mark.parametrize(
    "operation",
    (
        OperationRecord(OperationKind.DERIVATIVE, {"direction": "+t"}),
        OperationRecord(OperationKind.TRANSPORT, {"path": ("-t",)}),
        OperationRecord(OperationKind.SHIFT, {"path": ("+t",)}),
    ),
)
def test_temporal_field_operations_are_rejected_until_time_semantics_exist(operation):
    with pytest.raises(NotImplementedError, match="multi-timeslice"):
        _apply_laph_operations(_field(), (operation,), lambda _: _links(1.0))
