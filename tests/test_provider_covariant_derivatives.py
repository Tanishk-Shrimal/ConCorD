"""Independent periodic-site oracles for the default provider's derivatives."""

from itertools import product
import warnings

import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime import (
    ExecutionContext, RuntimeCompilationConfig, RuntimeExecutionConfig,
    compile_correlators, execute_compilation,
)
from concord_lqcd.backend.runtime.materialization import (
    CompiledKernelMaterializationProvider, InMemoryTensorSourceHandler,
)
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider
from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource
from concord_lqcd.core import Correlator, Operator, derivative, u, ubar


def _inputs(modes=2, batch=2):
    rng = np.random.default_rng(13102026)
    shape = (batch, 2, 3, 4, 3, modes)
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    links = {}
    for direction in ('x', 'y', 'z'):
        raw = rng.normal(size=shape[:4] + (3, 3))
        raw = raw + 1j * rng.normal(size=raw.shape)
        unitary = np.linalg.qr(raw)[0]
        unitary[..., :, 0] /= np.linalg.det(unitary)[..., None]
        links[direction] = unitary
    return field, links


def _reference(field, links, directions):
    """Evaluate each lattice site explicitly, without provider/roll helpers."""
    current = field.copy()
    batch, lz, ly, lx = field.shape[:4]
    for direction in directions:
        axis = {'z': 0, 'y': 1, 'x': 2}[direction[-1]]
        sign = -1 if direction.startswith('-') else 1
        extent = (lz, ly, lx)
        following = np.empty_like(current)
        for b, z, y, x in product(range(batch), range(lz), range(ly), range(lx)):
            here = (z, y, x)
            forward, backward = list(here), list(here)
            forward[axis] = (forward[axis] + 1) % extent[axis]
            backward[axis] = (backward[axis] - 1) % extent[axis]
            upper, lower = (b, *forward), (b, *backward)
            if links is None:
                fwd, bwd = current[upper], current[lower]
            else:
                link = links[direction[-1]]
                fwd = link[(b, *here)] @ current[upper]
                bwd = link[lower].conj().T @ current[lower]
            following[(b, *here)] = .5 * sign * (fwd - bwd)
        current = following
    return current


def _evaluate(provider, *, directions, context=None, is_anti=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return provider.eigenvector('u', derivatives=directions, context=context, is_anti=is_anti)


@pytest.mark.parametrize('modes', (2, 3, 5))
@pytest.mark.parametrize('directions', (('+x',), ('-y',), ('+z',), ('+x', '-z')))
@pytest.mark.parametrize('flattened', (False, True))
def test_periodic_free_derivatives_match_explicit_sites(modes, directions, flattened):
    field, _ = _inputs(modes)
    values = field.reshape(2, -1, 3, modes) if flattened else field
    provider = DistillationTensorProvider({'u': torch.from_numpy(values)}, {}, {})
    context = ExecutionContext.create('cpu', 0, (0, 1), lat_shape=(4, 3, 2))
    actual = _evaluate(provider, directions=directions, context=context).numpy()
    reference = _reference(field, None, directions).reshape(values.shape)
    np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('modes', (1, 2, 3, 5))
@pytest.mark.parametrize('link_layout', ('grid', 'flat', 'constant', 'shared_grid', 'shared_flat'))
def test_covariant_derivatives_apply_color_links_to_every_mode(modes, link_layout):
    field, links = _inputs(modes)
    supplied = {}
    for axis, link in links.items():
        if link_layout == 'flat':
            values = link.reshape(2, -1, 3, 3)
        elif link_layout == 'constant':
            values = link[0, 0, 0, 0]
            links[axis] = np.broadcast_to(values, link.shape)
        elif link_layout.startswith('shared'):
            values = link[0]
            links[axis] = np.broadcast_to(values, link.shape)
            if link_layout == 'shared_flat':
                values = values.reshape(-1, 3, 3)
        else:
            values = link
        supplied[axis] = torch.from_numpy(values.copy())
    values = field.reshape(2, -1, 3, modes)
    provider = DistillationTensorProvider(
        {'u': torch.from_numpy(values)}, {}, {}, supplied, spatial_extent=(4, 3, 2),
    )
    directions = ('+x', '-y', '+z')
    actual = _evaluate(provider, directions=directions, is_anti=True).resolve_conj().numpy()
    reference = _reference(field, links, directions).conj().reshape(values.shape)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)


def test_flattened_derivative_rejects_missing_or_inconsistent_geometry():
    field = torch.ones(1, 24, 3, 2, dtype=torch.complex128)
    provider = DistillationTensorProvider({'u': field}, {}, {})
    with pytest.raises(ValueError, match='require spatial extent'):
        _evaluate(provider, directions=('+x',))
    context = ExecutionContext.create('cpu', 0, (0,), lat_shape=(3, 3, 3))
    with pytest.raises(ValueError, match='site count'):
        _evaluate(provider, directions=('+x',), context=context)
    provider = DistillationTensorProvider({'u': field}, {}, {}, spatial_extent=(4, 3, 2))
    with pytest.raises(ValueError, match='disagrees with context'):
        _evaluate(provider, directions=('+x',), context=context)


@pytest.mark.parametrize('link_shape', ((2, 2), (1, 23, 3, 3), (3, 24, 3, 3)))
def test_supplied_invalid_links_fail_instead_of_becoming_unit_links(link_shape):
    field = torch.ones(1, 24, 3, 2, dtype=torch.complex128)
    provider = DistillationTensorProvider(
        {'u': field}, {}, {}, {'x': torch.ones(link_shape, dtype=torch.complex128)},
        spatial_extent=(4, 3, 2),
    )
    with pytest.raises(ValueError, match='gauge'):
        _evaluate(provider, directions=('+x',))


def test_precomputed_derivatives_need_no_geometry_or_fallback():
    precomputed = torch.full((1, 2, 3, 5), 2j, dtype=torch.complex128)
    provider = DistillationTensorProvider({'u_+x': precomputed}, {}, {})
    assert provider.eigenvector('u', derivatives=('+x',)) is precomputed


def test_complex_links_preserve_imaginary_derivative_of_real_fields():
    field = torch.ones((1, 4, 3, 2), dtype=torch.float64)
    link = torch.diag(torch.tensor([1j, -1j, 1], dtype=torch.complex128))
    provider = DistillationTensorProvider(
        {'u': field}, {}, {}, {'x': link}, spatial_extent=(4, 1, 1),
    )
    actual = _evaluate(provider, directions=('+x',))
    expected = torch.tensor([1j, -1j, 0], dtype=torch.complex128)
    torch.testing.assert_close(actual, expected.reshape(1, 1, 3, 1).expand_as(actual))


def test_compiled_derivative_loop_matches_independent_complex_contraction():
    field, links = _inputs(2, batch=1)
    values = field.reshape(1, -1, 3, 2)
    transformed = _reference(field, links, ('+x',)).reshape(values.shape)
    rng = np.random.default_rng(14102026)
    tau = rng.normal(size=(2, 4, 2, 4)) + 1j * rng.normal(size=(2, 4, 2, 4))
    reference = -np.einsum('txci,txcj,jsis->', values.conj(), transformed, tau)
    correlator = Correlator(Operator(
        ubar('x', 'color', 'spin') * derivative(u('x', 'color', 'spin'), '+x'),
    ))
    compilation = compile_correlators({'loop': correlator}, RuntimeCompilationConfig(mode_size=2))

    def provider_factory(request):
        return DistillationTensorProvider(
            {'u': torch.from_numpy(values)}, {}, {},
            {axis: torch.from_numpy(link) for axis, link in links.items()},
            spatial_extent=(4, 3, 2),
        )

    execution = RuntimeExecutionConfig(
        operator_times={1: (0,)},
        propagation_source_resolver=lambda resource_id, recipe: ExternalTensorSource(
            source_id=resource_id, kind='memory', location='u',
        ),
        kernel_providers={'distillation': CompiledKernelMaterializationProvider(provider_factory)},
        source_handlers={'memory': InMemoryTensorSourceHandler({'u': tau})},
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        actual = execute_compilation(compilation, execution).output('loop').item()
    assert abs(reference) > 1e-4
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
