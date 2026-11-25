"""Check field conjugation against independently constructed spatial matrices."""

from itertools import product

import numpy as np
import pytest
import torch

from concord_lqcd.compiler.physics_lowering import extract_operations
from concord_lqcd.core import (
    Apply, CovariantDerivative, Operator, ParallelTransport, Path, Product,
    Projection, QuarkField, Shift, derivative, laph, point, shift, transport, u, ubar,
)
from concord_lqcd.job.data import (
    _PreparedDistillationTensorProvider, _apply_laph_operations, _operation_key,
)


def _complex_random(generator, shape):
    return generator.normal(size=shape) + 1j * generator.normal(size=shape)


def _dense_hop(links, axis):
    lattice_shape = links.shape[:3]
    volume = int(np.prod(lattice_shape))
    matrix = np.zeros((3 * volume, 3 * volume), dtype=np.complex128)
    for position in np.ndindex(lattice_shape):
        neighbor = list(position)
        neighbor[axis] = (neighbor[axis] + 1) % lattice_shape[axis]
        row = 3 * np.ravel_multi_index(position, lattice_shape)
        column = 3 * np.ravel_multi_index(tuple(neighbor), lattice_shape)
        matrix[row:row + 3, column:column + 3] = links[position]
    return matrix


@pytest.fixture
def spatial_reference():
    generator = np.random.default_rng(20261009)
    lattice_shape = (1, 3, 4)
    gauge = {}
    for axis in ("x", "y"):
        values = np.empty((*lattice_shape, 3, 3), dtype=np.complex128)
        for position in np.ndindex(lattice_shape):
            unitary, _ = np.linalg.qr(_complex_random(generator, (3, 3)))
            unitary[:, 0] /= np.linalg.det(unitary)
            values[position] = unitary
        gauge[axis] = values
    basis, _ = np.linalg.qr(_complex_random(generator, (36, 3)))
    hop_x = _dense_hop(gauge["x"], 2)
    hop_y = _dense_hop(gauge["y"], 1)
    derivative_x = 0.5 * (hop_x - hop_x.conj().T)
    identity_links = np.broadcast_to(np.eye(3), (*lattice_shape, 3, 3))
    matrices = {
        "local": np.eye(36),
        "transport_x": hop_x,
        "derivative_x": derivative_x,
        "shift_x": _dense_hop(identity_links, 2),
        "ordered_path_xy": hop_x @ hop_y,
        "nested_transport_y_x": hop_y @ hop_x,
        "nested_transport_y_derivative_x": hop_y @ derivative_x,
    }
    references = {name: matrix @ basis for name, matrix in matrices.items()}
    return basis, gauge, references


def _expressions(field):
    return {
        "local": field,
        "transport_x": transport(field, "+x"),
        "derivative_x": derivative(field, "+x"),
        "shift_x": shift(field, "+x"),
        "ordered_path_xy": transport(field, "+x", "+y"),
        "nested_transport_y_x": transport(transport(field, "+x"), "+y"),
        "nested_transport_y_derivative_x": transport(derivative(field, "+x"), "+y"),
    }


def _applied_field(expression):
    factors = expression.factors if isinstance(expression, Product) else (expression,)
    return next(factor for factor in factors if isinstance(factor, (Apply, QuarkField)))


@pytest.mark.parametrize(("name", "is_anti"), tuple(product(_expressions(u("x", "a", "alpha")), (False, True))))
def test_field_dagger_matches_independent_component_adjoint(spatial_reference, name, is_anti):
    basis, gauge, references = spatial_reference
    expected = references[name] if is_anti else references[name].conj()
    field = (ubar if is_anti else u)("x", "a", "alpha")
    expression = _expressions(field)[name]
    conjugated = Operator(expression).dagger()
    leaf, operations, _ = extract_operations(_applied_field(conjugated.expr))
    assert leaf.is_anti is not is_anti

    values = torch.tensor(basis.reshape(1, 1, 3, 4, 3, 3))
    links = {axis: torch.tensor(matrix[None]) for axis, matrix in gauge.items()}
    transformed = _apply_laph_operations(values, operations, links.__getitem__)
    provider = _PreparedDistillationTensorProvider(
        {_operation_key(operations): transformed.reshape(1, 12, 3, 3)}, {},
    )
    attributes = {"operations": [
        {"kind": operation.kind.value, **operation.parameters} for operation in operations
    ]}
    actual = provider.eigenvector("u", is_anti=leaf.is_anti, field_attributes=attributes)
    actual_array = actual.resolve_conj().numpy().reshape(36, 3)
    np.testing.assert_allclose(actual_array, expected, rtol=5e-13, atol=0)

    if not is_anti:
        expected_source = references[name].conj().T @ basis
        np.testing.assert_allclose(actual_array.T @ basis, expected_source, rtol=5e-13, atol=0)
        if name == "derivative_x":
            assert np.max(np.abs(expected_source)) > 0.01
            wrong_source = -expected_source
            assert np.max(np.abs(actual_array.T @ basis - wrong_source)) > 0.02


def test_field_dagger_keeps_representation_selectors_and_operation_order():
    expression = laph(transport(point(derivative(u("x", "a", "alpha"), "+x")), "+y", "+x"))
    operator = Operator(expression)
    original_leaf, original_operations, original_representation = extract_operations(expression)
    first_leaf, first_operations, first_representation = extract_operations(
        _applied_field(operator.dagger().expr)
    )
    second_leaf, second_operations, second_representation = extract_operations(
        _applied_field(operator.dagger().dagger().expr)
    )
    assert first_leaf.is_anti is True
    assert second_leaf.is_anti == original_leaf.is_anti
    assert first_operations == second_operations == original_operations
    assert first_representation == second_representation == original_representation


@pytest.mark.parametrize(
    ("operator", "expected"),
    (
        (CovariantDerivative("+x"), CovariantDerivative("-x")),
        (ParallelTransport(Path(["+x", "+y"])), ParallelTransport(Path(["-y", "-x"]))),
        (Shift(Path(["+x", "+y"])), Shift(Path(["-y", "-x"]))),
        (Projection("laph"), Projection("laph")),
    ),
)
def test_standalone_linear_operator_adjoint_is_unchanged(operator, expected):
    assert operator.dagger() == expected
    assert operator.dagger().dagger() == operator
