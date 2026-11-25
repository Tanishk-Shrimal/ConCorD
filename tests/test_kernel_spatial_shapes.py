"""Spatial kernel operands must agree by labeled site axis, including tiles."""

from itertools import permutations

import numpy as np
import pytest
import torch

from concord_lqcd.backend.ir import CompiledKernelProgram, TensorOp
from concord_lqcd.backend.runtime import ExecutionContext
from concord_lqcd.backend.runtime.kernels import evaluate_kernel_program
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider


def _eigenvector(flavor, position, color, mode, *, is_anti=False):
    return TensorOp('eigenvector', {
        'source': 'distillation', 'flavor': flavor, 'is_anti': is_anti,
        'field_id': mode, 'legs': ['T', position, color, mode],
    })


def _phase(position='x', momentum=(1, 0, 0)):
    return TensorOp('phase', {'momentum': list(momentum), 'legs': [position]})


def _program(rank=2):
    if rank == 2:
        fields = [
            _eigenvector('u', 'x', 'color', 'mode_0', is_anti=True),
            _eigenvector('d', 'x', 'color', 'mode_1'),
        ]
        colors = []
    elif rank == 3:
        fields = [
            _eigenvector(flavor, 'x', color, f'mode_{number}')
            for number, (flavor, color) in enumerate(zip(('u', 'd', 's'), ('a', 'b', 'c')))
        ]
        colors = [TensorOp('color_epsilon', {'legs': ['a', 'b', 'c']})]
    else:
        fields = [
            _eigenvector('u', 'x', 'a', 'mode_0', is_anti=True),
            _eigenvector('d', 'x', 'a', 'mode_1'),
            _eigenvector('u', 'x', 'b', 'mode_2', is_anti=True),
            _eigenvector('d', 'x', 'b', 'mode_3'),
        ]
        colors = []
    modes = [f'mode_{number}' for number in range(rank)]
    return CompiledKernelProgram(
        num_quarks=rank, operations=[*fields, _phase(), *colors],
        mode_ports=modes, spin_ports=[], output_legs=['T', *modes],
    )


def _arrays(dtype=np.complex128):
    generator = np.random.default_rng(20261010)
    fields = {
        flavor: (generator.standard_normal((2, 5, 3, 2))
                 + 1j * generator.standard_normal((2, 5, 3, 2))).astype(dtype)
        for flavor in ('u', 'd', 's')
    }
    phase = np.exp(-2j * np.pi * np.arange(5) / 5).astype(dtype)
    return fields, phase


def _provider(fields, phases):
    return DistillationTensorProvider(
        {flavor: torch.from_numpy(values) for flavor, values in fields.items()}, {},
        {momentum: torch.from_numpy(values) for momentum, values in phases.items()},
    )


def _context(dtype=torch.complex128):
    return ExecutionContext.create(
        'cpu', 0, (0, 1), lat_shape=(5, 1, 1), complex_dtype=dtype,
    )


def _reference(fields, phase, rank):
    if rank == 2:
        return np.einsum('txci,txcj,x->tij', fields['u'].conj(), fields['d'], phase, optimize=False)
    if rank == 3:
        epsilon = np.zeros((3, 3, 3), dtype=phase.dtype)
        for ordering in permutations(range(3)):
            inversions = sum(ordering[left] > ordering[right]
                             for left in range(3) for right in range(left + 1, 3))
            epsilon[ordering] = (-1) ** inversions
        return np.einsum(
            'txai,txbj,txck,x,abc->tijk',
            fields['u'], fields['d'], fields['s'], phase, epsilon, optimize=False,
        )
    return np.einsum(
        'txai,txaj,txbk,txbl,x->tijkl',
        fields['u'].conj(), fields['d'], fields['u'].conj(), fields['d'], phase, optimize=False,
    )


@pytest.mark.parametrize('phase_extent', (1, 4, 7))
@pytest.mark.parametrize('phase_first', (False, True))
def test_mismatched_phase_extent_fails_before_contraction(phase_extent, phase_first, monkeypatch):
    fields, _ = _arrays()
    provider = _provider(fields, {(1, 0, 0): np.ones(phase_extent, dtype=np.complex128)})
    program = _program()
    if phase_first:
        program.operations.insert(0, program.operations.pop())

    def forbidden_contraction(*arguments, **keywords):
        pytest.fail('shape mismatch reached numerical contraction')

    monkeypatch.setattr('concord_lqcd.backend.runtime.kernels.oe.contract', forbidden_contraction)
    with pytest.raises(ValueError, match="spatial leg 'x'.*extent"):
        evaluate_kernel_program(program, provider, _context())


@pytest.mark.parametrize('phase_state', ('provided', 'none', 'omitted'))
@pytest.mark.parametrize('reverse_fields', (False, True))
def test_single_site_field_cannot_broadcast_over_shared_volume(phase_state, reverse_fields, monkeypatch):
    fields, phase = _arrays()
    fields['d'] = fields['d'][:, :1]
    provider = _provider(fields, {(1, 0, 0): phase} if phase_state == 'provided' else {})
    program = _program()
    if phase_state == 'omitted':
        program.operations.pop()
    if reverse_fields:
        program.operations[:2] = reversed(program.operations[:2])

    def forbidden_contraction(*arguments, **keywords):
        pytest.fail('inconsistent field volumes reached numerical contraction')

    monkeypatch.setattr('concord_lqcd.backend.runtime.kernels.oe.contract', forbidden_contraction)
    with pytest.raises(ValueError, match="spatial leg 'x'.*extent"):
        evaluate_kernel_program(program, provider, _context())


@pytest.mark.parametrize('shape', ((), (5, 1), (1, 5)))
def test_phase_rank_must_match_its_single_declared_leg(shape):
    fields, _ = _arrays()
    provider = _provider(fields, {(1, 0, 0): np.ones(shape, dtype=np.complex128)})
    with pytest.raises(ValueError, match="phase.*rank.*1"):
        evaluate_kernel_program(_program(), provider, _context())


def test_eigenvector_rank_error_identifies_the_operand():
    fields, phase = _arrays()
    fields['u'] = fields['u'][:, :, 0]
    with pytest.raises(ValueError, match="eigenvector.*rank.*4"):
        evaluate_kernel_program(_program(), _provider(fields, {(1, 0, 0): phase}), _context())


@pytest.mark.parametrize('rank', (2, 3, 4))
@pytest.mark.parametrize('tile_sites', (2, 5))
@pytest.mark.parametrize('dtype', (np.complex64, np.complex128))
@pytest.mark.parametrize('pinned_path', (False, True))
def test_aligned_tiles_match_independent_nonzero_full_volume_sum(rank, tile_sites, dtype, pinned_path):
    fields, phase = _arrays(dtype)
    program = _program(rank)
    context = _context(torch.complex64 if dtype is np.complex64 else torch.complex128)
    path = (tuple(range(len(program.operations))),) if pinned_path else None
    full = evaluate_kernel_program(program, _provider(fields, {(1, 0, 0): phase}), context,
                                   contraction_path=path)
    accumulated = torch.zeros_like(full)
    for start in range(0, 5, tile_sites):
        stop = min(start + tile_sites, 5)
        tiled_fields = {flavor: values[:, start:stop] for flavor, values in fields.items()}
        tiled_phase = phase[start:stop]
        accumulated += evaluate_kernel_program(
            program, _provider(tiled_fields, {(1, 0, 0): tiled_phase}), context,
            contraction_path=path,
        )
    expected = _reference(fields, phase, rank)
    assert np.max(np.abs(expected)) > 1
    tolerance = 2e-5 if dtype is np.complex64 else 1e-12
    np.testing.assert_allclose(full.numpy(), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(accumulated.numpy(), expected, rtol=tolerance, atol=tolerance)


def test_noncontiguous_site_selection_uses_the_corresponding_phase_entries():
    fields, phase = _arrays()
    selected = [4, 1]
    sliced_fields = {flavor: values[:, selected] for flavor, values in fields.items()}
    actual = evaluate_kernel_program(
        _program(), _provider(sliced_fields, {(1, 0, 0): phase[selected]}), _context(),
    )
    expected = _reference(sliced_fields, phase[selected], 2)
    assert np.max(np.abs(expected)) > 1
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
    wrong = _reference(sliced_fields, phase[:2], 2)
    assert np.max(np.abs(expected - wrong)) > 1


@pytest.mark.parametrize('order', ((0, 1, 2, 3), (2, 3, 0, 1), (0, 2, 1, 3), (1, 3, 0, 2)))
def test_independent_spatial_legs_do_not_inherit_the_last_field_volume(order):
    fields, phase_x = _arrays()
    fields['d'] = fields['d'][:, :3]
    phase_y = np.exp(-2j * np.pi * np.arange(3) / 3)
    operations = [
        _eigenvector('u', 'x', 'color', 'mode_0', is_anti=True),
        _eigenvector('d', 'y', 'color', 'mode_1'),
        _phase('x'), _phase('y', (0, 1, 0)),
    ]
    program = CompiledKernelProgram(
        num_quarks=2, operations=[operations[number] for number in order],
        mode_ports=['mode_0', 'mode_1'], spin_ports=[], output_legs=['T', 'mode_0', 'mode_1'],
    )
    actual = evaluate_kernel_program(
        program, _provider(fields, {(1, 0, 0): phase_x, (0, 1, 0): phase_y}), _context(),
    )
    expected = np.einsum('txci,tycj,x,y->tij', fields['u'].conj(), fields['d'], phase_x, phase_y,
                         optimize=False)
    assert np.max(np.abs(expected)) > 1
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_existing_none_phase_identity_behavior_is_preserved():
    fields, _ = _arrays()
    program = _program()
    program.operations[-1] = _phase(momentum=(0, 0, 0))
    actual = evaluate_kernel_program(program, _provider(fields, {}), _context())
    expected = _reference(fields, np.ones(5, dtype=np.complex128), 2)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
