"""Public operator equality must preserve exposed tensor interfaces."""

from collections import Counter
from itertools import permutations

import h5py
import numpy as np
import pytest

from concord_lqcd.core import (
    ColorDelta, ColorEpsilon, Correlator, GammaMatrix, Operator, Product,
    delta, epsilon, gamma5, gammaX, spin_matrix, u, ubar,
)
from concord_lqcd.core.dirac_basis import get_spin_matrices


def _epsilon_array():
    values = np.zeros((3, 3, 3), dtype=np.complex128)
    for ordering in permutations(range(3)):
        inversions = sum(
            ordering[left] > ordering[right]
            for left in range(3) for right in range(left + 1, 3)
        )
        values[ordering] = (-1) ** inversions
    return values


def _tensor_value(operator, matrices, outputs=()):
    factors = operator.expr.factors if isinstance(operator.expr, Product) else (operator.expr,)
    labels = tuple(dict.fromkeys(index for factor in factors for index in factor.indices))
    numbers = {index: number for number, index in enumerate(labels)}
    arguments = []
    for factor in factors:
        if isinstance(factor, GammaMatrix):
            values = matrices[factor.kind]
        elif isinstance(factor, ColorDelta):
            values = np.eye(3)
        elif isinstance(factor, ColorEpsilon):
            values = _epsilon_array()
        else:
            raise AssertionError(type(factor))
        arguments.extend((values, [numbers[index] for index in factor.indices]))
    arguments.append([
        numbers[index] for label in outputs for index in labels if index.label == label
    ])
    return np.einsum(*arguments, optimize=False)


def _combination_value(combination, matrices, outputs=()):
    return sum(
        (coefficient * _tensor_value(operator, matrices, outputs)
         for coefficient, operator in combination.iter_terms()),
        np.asarray(0j),
    )


@pytest.mark.parametrize("tensor", (gammaX, delta, epsilon))
def test_transposed_open_interfaces_do_not_compare_equal(tensor):
    labels = ("alpha", "beta", "rho") if tensor is epsilon else ("alpha", "beta")
    transposed_labels = (labels[1], labels[0], *labels[2:])
    original = Operator(tensor(*labels))
    transposed = Operator(tensor(*transposed_labels))

    assert original != transposed
    assert original.structural_key != transposed.structural_key
    assert len({original, transposed}) == 2
    assert len((original - transposed).terms) == 2


def test_open_quark_bilinear_does_not_cancel_its_transpose():
    original = Operator(ubar("x", "color", "alpha") * u("x", "color", "beta"))
    transposed = Operator(ubar("x", "color", "beta") * u("x", "color", "alpha"))

    assert original != transposed
    assert len((original - transposed).terms) == 2
    assert not Correlator(original - transposed).is_zero


@pytest.mark.parametrize("space", ("spin", "color"))
def test_free_labels_are_not_implicitly_aligned(space):
    tensor = gammaX if space == "spin" else delta
    original = Operator(tensor("left", "right"))
    renamed = Operator(tensor("other_left", "other_right"))

    assert original != renamed
    assert len((original + renamed).terms) == 2
    assert original != original.reindexed("another_interface")


@pytest.mark.parametrize("space", ("spin", "color"))
def test_only_bound_labels_are_alpha_renamed(space):
    tensor = gammaX if space == "spin" else delta
    original = Operator(tensor("left", "internal") * tensor("internal", "right"))
    renamed = Operator(tensor("left", "other_internal") * tensor("other_internal", "right"))

    assert original == renamed
    assert hash(original) == hash(renamed)
    assert (original - renamed).terms == ()
    assert (original + renamed).terms == ((2 + 0j, original),)


def test_free_and_bound_tokens_cannot_collide():
    original = Operator(gammaX("alpha", "dummy") * gammaX("dummy", "beta"))
    changed_interface = Operator(gammaX("dummy", "alpha") * gammaX("alpha", "beta"))

    assert original != changed_interface
    assert len((original - changed_interface).terms) == 2


def test_closed_bilinears_keep_dummy_and_locality_label_equivalence():
    original = Operator(
        ubar("x", "color", "alpha") * gamma5("alpha", "beta") * u("x", "color", "beta"),
        momentum=(1, -2, 0),
    )
    renamed = Operator(
        ubar("y", "other_color", "other_alpha")
        * gamma5("other_alpha", "other_beta") * u("y", "other_color", "other_beta"),
        momentum=(1, -2, 0),
    )

    assert original == renamed
    assert hash(original) == hash(renamed)
    assert original == original.reindexed("another_scope")
    assert (original - renamed).terms == ()


@pytest.mark.parametrize("basis", ("DR", "DP"))
@pytest.mark.parametrize("closure_first", (False, True))
def test_spin_distributivity_preserves_a_nonzero_transposed_difference(basis, closure_first):
    matrices = get_spin_matrices(basis)
    charge_conjugation = matrices["C"]
    reference = np.sum((charge_conjugation - charge_conjugation.T) * charge_conjugation)
    assert abs(reference) > 1

    original = Operator(spin_matrix("C", "alpha", "beta"))
    transposed = Operator(spin_matrix("C", "beta", "alpha"))
    closure = Operator(spin_matrix("C", "alpha", "beta"))
    if closure_first:
        undistributed = closure * (original - transposed)
        distributed = closure * original - closure * transposed
    else:
        undistributed = (original - transposed) * closure
        distributed = original * closure - transposed * closure

    assert undistributed == distributed
    assert len(undistributed.terms) == 2
    np.testing.assert_allclose(_combination_value(undistributed, matrices), reference, rtol=0, atol=1e-14)
    np.testing.assert_allclose(_combination_value(distributed, matrices), reference, rtol=0, atol=1e-14)


def test_color_distributivity_preserves_a_nonzero_transposed_difference():
    values = _epsilon_array()
    reference = np.sum((values - values.transpose(1, 0, 2)) * values)
    assert reference == 12

    original = Operator(epsilon("alpha", "beta", "rho"))
    transposed = Operator(epsilon("beta", "alpha", "rho"))
    closure = Operator(epsilon("alpha", "beta", "rho"))
    undistributed = (original - transposed) * closure
    distributed = original * closure - transposed * closure

    assert undistributed == distributed
    np.testing.assert_array_equal(_combination_value(undistributed, {}), reference)


@pytest.mark.parametrize("basis", ("DR", "DP"))
def test_bound_renaming_in_an_open_chain_preserves_its_dense_value(basis):
    matrices = get_spin_matrices(basis)
    reference = matrices["C"] @ matrices["gammaX"]
    original = Operator(spin_matrix("C", "alpha", "dummy") * gammaX("dummy", "beta"))
    renamed = Operator(spin_matrix("C", "alpha", "other_dummy") * gammaX("other_dummy", "beta"))

    assert original == renamed
    np.testing.assert_allclose(
        _tensor_value(original, matrices, ("alpha", "beta")), reference, rtol=0, atol=1e-14,
    )
    np.testing.assert_allclose(
        _combination_value(original + renamed, matrices, ("alpha", "beta")),
        2 * reference, rtol=0, atol=1e-14,
    )


def test_more_than_two_occurrences_are_not_treated_as_a_valid_dummy():
    original = Operator(gamma5("alpha", "alpha") * gammaX("alpha", "beta"))
    renamed = Operator(gamma5("other_alpha", "other_alpha") * gammaX("other_alpha", "beta"))

    assert Counter(original.indices)[gammaX("alpha", "beta").left] == 3
    assert original != renamed


def test_compiler_expansion_retains_transposed_open_terms():
    from concord_lqcd.compiler.expansion import ExpansionPass

    original = Operator(ubar("x", "color", "alpha") * u("x", "color", "beta"))
    transposed = Operator(ubar("x", "color", "beta") * u("x", "color", "alpha"))
    terms = ExpansionPass.run_correlator(Correlator(original - transposed))

    assert len(terms) == 2
    assert dict((operators[0], coefficient) for coefficient, operators in terms) == {
        original: 1 + 0j, transposed: -1 + 0j,
    }


def test_compiler_rejects_a_sum_with_incompatible_exposed_interfaces():
    from concord_lqcd.backend.runtime import RuntimeCompilationConfig, compile_correlators

    original = Operator(ubar("x", "color", "alpha") * u("x", "color", "beta"))
    incompatible = Operator(
        ubar("x", "color", "other_alpha") * u("x", "color", "other_beta")
    )

    with pytest.raises(ValueError, match="external output interface"):
        compile_correlators(
            Correlator(original + incompatible), RuntimeCompilationConfig(mode_size=2),
        )


def test_wick_cache_reuses_bound_renaming_but_not_changed_open_attachments():
    from concord_lqcd.compiler.wick_cache import WickCache

    original = Operator(ubar("x", "color", "alpha") * u("x", "color", "beta"))
    renamed = Operator(ubar("y", "other_color", "alpha") * u("y", "other_color", "beta"))
    transposed = Operator(ubar("x", "color", "beta") * u("x", "color", "alpha"))
    cache = WickCache()
    original_result = cache.get_or_contract_operators((original,))
    renamed_result = cache.get_or_contract_operators((renamed,))
    transposed_result = cache.get_or_contract_operators((transposed,))

    assert renamed_result is original_result
    assert transposed_result is not original_result
    assert cache.hits == 1
    assert cache.misses == 2
    assert original_result[1] and transposed_result[1]


@pytest.mark.parametrize("basis", ("DR", "DP"))
def test_complete_loop_keeps_distributivity_and_matches_direct_perambulator_trace(tmp_path, basis):
    generator = np.random.default_rng(20261009)
    perambulators = {
        source_time: generator.normal(size=(2, 4, 4, 2, 2))
        + 1j * generator.normal(size=(2, 4, 4, 2, 2))
        for source_time in range(2)
    }
    matrix = get_spin_matrices(basis)["gammaX"]
    expected = np.asarray([
        -np.einsum("ba,abii->", matrix, perambulators[time][time], optimize=False)
        + np.einsum("ba,baii->", matrix, perambulators[time][time], optimize=False)
        for time in range(2)
    ])
    assert np.min(np.abs(expected)) > 0.1
    open_expected = np.asarray([
        -np.einsum("abii->ab", perambulators[time][time], optimize=False)
        + np.einsum("baii->ab", perambulators[time][time], optimize=False)
        for time in range(2)
    ])
    assert np.max(np.abs(open_expected)) > 0.1
    expected_by_id = {
        "before_distribution": expected, "after_distribution": expected,
        "open_difference": open_expected,
    }

    operator_file = tmp_path / "operators.py"
    operator_file.write_text(
        "from concord_lqcd.core import Correlator, Operator, gammaX, u, ubar\n"
        "def build():\n"
        "    original = Operator(ubar('x','color','alpha') * u('x','color','beta'))\n"
        "    transposed = Operator(ubar('x','color','beta') * u('x','color','alpha'))\n"
        "    closure = Operator(gammaX('beta','alpha'))\n"
        "    return {\n"
        "        'before_distribution': Correlator((original - transposed) * closure),\n"
        "        'after_distribution': Correlator(original * closure - transposed * closure),\n"
        "        'open_difference': Correlator(original - transposed),\n"
        "    }\n",
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "eigenvectors.h5", "w") as handle:
        handle.create_dataset("eigenvectors", data=np.broadcast_to(np.eye(3, 2), (2, 1, 1, 1, 3, 2)))
    with h5py.File(tmp_path / "perambulators.h5", "w") as handle:
        for time, values in perambulators.items():
            handle.create_dataset(f"tau_{time}", data=values)

    from concord_lqcd.job import JobSpec, run_job, write_job_plan

    spec = JobSpec.model_validate({
        "schema_version": 1,
        "job": {"name": "open_index_distributivity"},
        "operators": {"file": str(operator_file), "factory": "build"},
        "plan": {"file": str(tmp_path / "plan.concord")},
        "lattice": {"spatial_extent": [1, 1, 1], "temporal_extent": 2},
        "compile": {"mode_size": 2, "spin_basis": basis, "scalar_type": "complex128"},
        "run": {
            "device": "cpu", "time_batch_size": 1,
            "configurations": {"values": [1]},
            "times": {"contexts": [
                {"id": f"time_{time}", "operator_times": {1: [time]}}
                for time in range(2)
            ]},
        },
        "data": {
            "eigenvectors": {
                "provider": "hdf5_distillation", "path": str(tmp_path / "eigenvectors.h5"),
                "dataset": "/eigenvectors", "axes": ["time", "z", "y", "x", "color", "mode"],
            },
            "propagation": {"provider": "hdf5_perambulator", "ordinary": {
                "files": {"u": str(tmp_path / "perambulators.h5")},
                "dataset": "/tau_{source_time}",
                "axes": ["sink_time", "source_spin", "sink_spin", "source_mode", "sink_mode"],
            }},
        },
        "output": {"file": str(tmp_path / "result.h5"), "on_existing": "error"},
    })
    plan = write_job_plan(spec)
    summary = run_job(spec, plan)
    assert summary.executed_work_units == 2
    with h5py.File(spec.output.file, "r") as handle:
        assert np.all(handle["completed"][:])
        assert len(handle["correlators"]) == len(expected_by_id)
        for group in handle["correlators"].values():
            reference = expected_by_id[group.attrs["correlator_id"]]
            np.testing.assert_allclose(group["values"][0], reference, rtol=5e-13, atol=0)
