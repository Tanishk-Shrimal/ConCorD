"""Compilation identities must not depend on unrelated Python activity."""

import json
import os
from itertools import count
import subprocess
import sys

import h5py
import numpy as np
import pytest

from concord_lqcd.core import (
    Correlator, CorrMatrix, Operator, build_baryon_operator, derivative, gamma5, gammaX, laph, point,
    shift, transport, u, ubar,
)
from concord_lqcd.compiler.prewick import lower_correlator_to_physics_ir
from concord_lqcd.compiler.wick_cache import WickCache
from concord_lqcd.backend.runtime.pipeline import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.job import (
    JobRunError, JobSpec, compile_job_plan, load_concord_plan, run_job, save_concord_plan,
    validate_plan_for_job,
)


def _bilinear(gamma=gamma5, momentum=(0, 0, 0)):
    return Operator(
        ubar('x', 'color', 'left_spin') * gamma('left_spin', 'right_spin')
        * u('x', 'color', 'right_spin'), momentum=momentum,
    )


def _labels(correlator):
    return tuple(tuple(index.label for index in operator.indices)
                 for operator in correlator.symbolic_operators)


def _job(tmp_path):
    source = tmp_path / 'operators.py'
    source.write_text(
        "from concord_lqcd.core import Correlator, Operator, gamma5, gammaX, u, ubar\n"
        "def build():\n"
        "    pion = Operator(ubar('x','color','left_spin') * gamma5('left_spin','right_spin') * u('x','color','right_spin'), momentum=(0,0,0))\n"
        "    vector = Operator(ubar('x','color','left_spin') * gammaX('left_spin','right_spin') * u('x','color','right_spin'), momentum=(1,0,0))\n"
        "    return {'mixed': Correlator(pion + (2+3j)*vector, pion.dagger())}\n",
        encoding='utf-8',
    )
    return JobSpec.model_validate({
        'schema_version': 1,
        'job': {'name': 'namespace_test'},
        'operators': {'file': str(source), 'factory': 'build'},
        'plan': {'file': str(tmp_path / 'plan.concord')},
        'lattice': {'spatial_extent': [2, 2, 2], 'temporal_extent': 4},
        'compile': {'mode_size': 2, 'contraction': {'strategy': 'greedy'}},
        'run': {'device': 'cpu', 'configurations': {'values': [1]},
                'times': {'generator': {'type': 'two_point', 'source': 0,
                                      'sinks': {'values': [1, 2]}}}},
        'data': {
            'eigenvectors': {'provider': 'hdf5_distillation', 'path': str(tmp_path / 'evec.h5'),
                             'dataset': '/eigenvectors',
                             'axes': ['time', 'z', 'y', 'x', 'color', 'mode']},
            'propagation': {'provider': 'hdf5_perambulator', 'ordinary': {
                'path': str(tmp_path / 'peram.h5'), 'dataset': '/tau_{source_time}',
                'axes': ['sink_time', 'source_spin', 'sink_spin', 'source_mode', 'sink_mode']}},
        },
        'output': {'file': str(tmp_path / 'result.h5')},
    })


def test_replanning_is_identical_after_unrelated_frontend_activity(tmp_path):
    spec = _job(tmp_path)
    first = compile_job_plan(spec)
    for _ in range(7):
        _ = Correlator(_bilinear()).symbolic_operators
    second = compile_job_plan(spec)
    assert first.manifest.job_fingerprint == second.manifest.job_fingerprint
    assert first.manifest.plan_sha256 == second.manifest.plan_sha256
    assert first.physical_plan == second.physical_plan


def test_cache_namespaces_are_independent_of_lookup_history():
    operators = (_bilinear(), _bilinear().dagger())
    first, _ = WickCache().get_or_contract_operators(operators)
    warm = WickCache()
    warm.get_or_contract_operators((_bilinear(gammaX),))
    second, _ = warm.get_or_contract_operators(operators)
    assert _labels(first) == _labels(second)


def test_compiler_does_not_consume_frontend_namespace_counter(monkeypatch):
    class ForbiddenCounter:
        def __next__(self):
            pytest.fail('compiler consumed the frontend namespace counter')

    monkeypatch.setattr(Correlator, '_counter', ForbiddenCounter())
    compilation = compile_correlators(
        {'pion': Correlator(_bilinear(), _bilinear().dagger())},
        RuntimeCompilationConfig(mode_size=2),
    )
    assert compilation.correlator_ids == ('pion',)


def test_prewick_is_repeatable_for_complex_momentum_combinations():
    zero = _bilinear(momentum=(0, 0, 0))
    moving = _bilinear(momentum=(1, 0, 0))
    correlator = Correlator(zero + (2 + 3j) * moving)
    first = lower_correlator_to_physics_ir(correlator)
    second = lower_correlator_to_physics_ir(correlator)
    assert first == second
    assert set(first.momenta.values()) == {(0, 0, 0), (1, 0, 0)}
    assert len(first.momenta) == 2
    assert tuple(term.coefficient for term in first.correlators) == (-1, -2-3j)


def test_warm_cache_and_cache_clear_preserve_compilation_identity():
    pion = _bilinear()
    vector = _bilinear(gammaX, (1, 0, 0))
    matrix = CorrMatrix([pion + (2 + 3j) * vector, 3 * pion - vector])
    config = RuntimeCompilationConfig(mode_size=2)
    first = compile_correlators(matrix, config)
    cache = WickCache()
    cache.get_or_contract_operators((vector,))
    second = compile_correlators(matrix, config, wick_cache=cache)
    assert first.physical_plan == second.physical_plan
    cache.clear()
    third = compile_correlators(matrix, config, wick_cache=cache)
    assert first.physical_plan == third.physical_plan


def test_open_spin_sum_preserves_its_interface_and_plan_identity():
    proton = build_baryon_operator(('u', 'd', 'u'), spin_labels=('alpha', 'beta', 'rho'))
    neutron = build_baryon_operator(('d', 'u', 'd'), spin_labels=('alpha', 'beta', 'rho'))
    combined = proton + (2 + 3j) * neutron
    correlator = Correlator(combined, combined.dagger())
    config = RuntimeCompilationConfig(mode_size=2, contraction_strategy='greedy')
    first = compile_correlators({'nucleon_sum': correlator}, config)
    _ = Correlator(_bilinear()).symbolic_operators
    second = compile_correlators({'nucleon_sum': correlator}, config)

    assert first.logical_dag.correlators[0].output_indices == ('spin:op0:rho', 'spin:op1:rho')
    assert first.physical_plan == second.physical_plan


@pytest.mark.parametrize(('internal', 'external'), [
    ('spin:c13:op0:rho', 'spin:op0:rho'),
    ('color:c7:op1:color', 'color:op1:color'),
    ('spin:basis_0123456789abcdef0123:op0:rho', 'spin:op0:rho'),
    ('color:basis_0123456789abcdef0123:op1:color', 'color:op1:color'),
])
def test_compiler_namespaces_do_not_change_public_output_labels(internal, external):
    from concord_lqcd.compiler.global_cse import _public_output_index

    assert _public_output_index(internal) == external


@pytest.mark.parametrize('label', [
    'spin:user:op0:rho',
    'spin:basis_deadbeef:op0:rho',
    'spin:basis_0123456789abcdef0123:other:rho',
    'spin:op0:rho',
])
def test_unrecognized_output_scopes_are_not_stripped(label):
    from concord_lqcd.compiler.global_cse import _public_output_index

    assert _public_output_index(label) == label


def test_saved_plan_roundtrip_and_fresh_replanning_agree(tmp_path):
    spec = _job(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)
    loaded = load_concord_plan(spec.plan.file)
    validate_plan_for_job(loaded, spec)
    repeated = compile_job_plan(spec)
    assert loaded.manifest.plan_sha256 == repeated.manifest.plan_sha256
    assert loaded.physical_plan == repeated.physical_plan


def test_older_counter_named_format_two_plans_remain_valid(tmp_path, monkeypatch):
    import concord_lqcd.compiler.wick_cache as cache_module

    spec = _job(tmp_path)
    old_counter = count(13)
    with monkeypatch.context() as patch:
        patch.setattr(cache_module, '_basis_namespace', lambda operators: f'c{next(old_counter)}')
        older = compile_job_plan(spec)
    current = compile_job_plan(spec)
    assert older.manifest.job_fingerprint == current.manifest.job_fingerprint
    assert older.manifest.plan_sha256 != current.manifest.plan_sha256
    save_concord_plan(older, spec.plan.file)
    loaded = load_concord_plan(spec.plan.file)
    validate_plan_for_job(loaded, spec)
    assert loaded.physical_plan == older.physical_plan

    generator = np.random.default_rng(20261010)
    eigenvectors = np.empty((4, 2, 2, 2, 3, 2), dtype=np.complex128)
    for time in range(4):
        raw = generator.standard_normal((24, 2)) + 1j * generator.standard_normal((24, 2))
        eigenvectors[time] = np.linalg.qr(raw)[0].reshape(2, 2, 2, 3, 2)
    with h5py.File(tmp_path / 'evec.h5', 'w') as handle:
        handle.create_dataset('eigenvectors', data=eigenvectors)
    with h5py.File(tmp_path / 'peram.h5', 'w') as handle:
        for time in range(4):
            raw = generator.standard_normal((4, 4, 4, 2, 2))
            handle.create_dataset(f'tau_{time}', data=raw + 1j * generator.standard_normal(raw.shape))

    spec = spec.model_copy(update={'run': spec.run.model_copy(update={'time_batch_size': 1})})
    assert run_job(spec, loaded).executed_work_units == 2
    assert run_job(spec, loaded).skipped_work_units == 2
    with pytest.raises(JobRunError, match='different compiled plan'):
        run_job(spec, current)

    current_spec = spec.model_copy(update={
        'output': spec.output.model_copy(update={'file': tmp_path / 'current_result.h5'}),
    })
    assert run_job(current_spec, current).executed_work_units == 2
    with h5py.File(spec.output.file, 'r') as older_output:
        with h5py.File(current_spec.output.file, 'r') as current_output:
            older_values = older_output['correlators/0/values'][:]
            assert np.max(np.abs(older_values)) > 1e-6
            np.testing.assert_allclose(
                older_values, current_output['correlators/0/values'][:], rtol=1e-12, atol=1e-12,
            )


def test_compilation_does_not_replace_user_symbolic_context():
    pion = _bilinear()
    correlator = Correlator(pion, pion.dagger())
    user_context = correlator.symbolic_operators
    before = _labels(correlator)
    compile_correlators(correlator, RuntimeCompilationConfig(mode_size=2))
    assert correlator.symbolic_operators is user_context
    assert _labels(correlator) == before


def test_standalone_frontend_correlators_stay_disjoint():
    first = Correlator(_bilinear())
    second = Correlator(_bilinear())
    assert set(_labels(first)[0]).isdisjoint(_labels(second)[0])


def test_namespace_digest_collision_fails_instead_of_aliasing(monkeypatch):
    import concord_lqcd.compiler.wick_cache as cache_module

    monkeypatch.setattr(cache_module, '_basis_namespace', lambda operators: 'basis_collision')
    cache = WickCache()
    pion_context, _ = cache.get_or_contract_operators((_bilinear(),))
    with pytest.raises(ValueError, match='namespace collision'):
        cache.get_or_contract_operators((_bilinear(gammaX),))
    assert cache.get_or_contract_operators((_bilinear(),))[0] is pion_context
    cache.clear()
    cache.get_or_contract_operators((_bilinear(gammaX),))


@pytest.mark.parametrize('transform', [
    lambda field: derivative(field, '+x'),
    lambda field: transport(field, '+x', '+y'),
    lambda field: shift(field, '-z'),
    lambda field: laph(field),
    lambda field: point(field),
    lambda field: transport(derivative(field, '+x'), '+y'),
])
def test_builtin_field_operation_names_are_repeatable(transform):
    field = transform(u('x', 'color', 'right_spin'))
    operator = Operator(
        ubar('x', 'color', 'left_spin') * gamma5('left_spin', 'right_spin') * field,
        momentum=(1, 0, 0),
    )
    operators = (operator, operator.dagger())
    first, _ = WickCache().get_or_contract_operators(operators)
    second, _ = WickCache().get_or_contract_operators(operators)
    assert _labels(first) == _labels(second)


@pytest.mark.parametrize('transformed', [False, True])
def test_full_plan_identity_is_stable_across_processes_and_hash_seeds(tmp_path, transformed):
    spec = _job(tmp_path)
    if transformed:
        spec.operators.file.write_text(
            "from concord_lqcd.core import Correlator, Operator, derivative, gamma5, laph, transport, u, ubar\n"
            "def build():\n"
            "    field = transport(derivative(laph(u('x','color','right_spin')), '+x'), '+y')\n"
            "    sink = Operator(ubar('x','color','left_spin') * gamma5('left_spin','right_spin') * field, momentum=(1,0,0))\n"
            "    return {'displaced': Correlator(sink, sink.dagger())}\n",
            encoding='utf-8',
        )
    code = (
        "import json, sys\n"
        "from concord_lqcd.core import Correlator, Operator, u, ubar\n"
        "from concord_lqcd.job import JobSpec, compile_job_plan\n"
        "for number in range(int(sys.argv[2])):\n"
        "    operator = Operator(ubar('x','color','spin') * u('x','color','spin'))\n"
        "    Correlator(operator).symbolic_operators\n"
        "plan = compile_job_plan(JobSpec.model_validate(json.loads(sys.argv[1])))\n"
        "print(json.dumps({'plan': plan.manifest.plan_sha256, 'job': plan.manifest.job_fingerprint}))\n"
    )
    identities = []
    for seed, unrelated in (('1', 0), ('937', 9)):
        environment = {**os.environ, 'PYTHONHASHSEED': seed}
        completed = subprocess.run(
            [sys.executable, '-c', code, json.dumps(spec.model_dump(mode='json')), str(unrelated)],
            env=environment, capture_output=True, text=True, check=True, timeout=60,
        )
        identities.append(json.loads(completed.stdout))
    assert identities[0] == identities[1]
