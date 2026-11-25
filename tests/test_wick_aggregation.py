"""Exact whole-term certificates, conservative fallback and executable parity."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch

from concord_lqcd.backend.runtime import (
    RuntimeCompilationConfig, RuntimeExecutionConfig, compile_correlators, execute_compilation,
)
from concord_lqcd.backend.runtime.materialization import KernelMaterializationProvider, PropagationSourceHandler
from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource
from concord_lqcd.compiler.contraction_graph import PrimitiveTensorNode, TensorLegBinding, WickContractionGraph
from concord_lqcd.compiler.global_cse import build_global_cse
from concord_lqcd.compiler.physics_ir import SpinTensorRecord
from concord_lqcd.compiler import _tensor_isomorphism, wick_aggregation
from concord_lqcd.compiler.wick_aggregation import (
    WickAggregationConfig, WickAggregationTimeLimitExceeded, _ExactComplex, _spin_contract,
    aggregate_certified_wick_terms, validate_wick_aggregation,
)
from concord_lqcd.core import Correlator, build_baryon_operator, reset_registry, spin_matrix
from concord_lqcd.core.dirac_basis import get_spin_matrices


def _spin_dag(coefficients=(1, 2), kinds=("Cg5", "Cg5")):
    labels = ("spin:op0:alpha", "spin:op0:beta")
    graphs = tuple(WickContractionGraph(coefficient, (PrimitiveTensorNode(
        node_id=f"spin{ordinal}", kind="spin", resource_id=f"spin:{kind}",
        bindings=tuple(TensorLegBinding(port, index) for port, index in zip(
            ("left", "right"), labels if ordinal == 0 else labels[::-1], strict=True)),
        payload=SpinTensorRecord(kind, *labels, operator_id=0),
    ),)) for ordinal, (coefficient, kind) in enumerate(zip(coefficients, kinds, strict=True)))
    return build_global_cse({"open": graphs})


def _aggregate(dag, **changes):
    return aggregate_certified_wick_terms(
        dag, WickAggregationConfig(enabled=True, **changes), spin_basis="DP",
        shared_orthonormal_laph_basis=True,
    )


def _validate(dag):
    validate_wick_aggregation(dag, config=WickAggregationConfig(enabled=True), spin_basis="DP",
                              shared_orthonormal_laph_basis=True, numerical_flavor_groups=())


def test_default_off_returns_the_original_dag():
    source = _spin_dag()
    result, report = aggregate_certified_wick_terms(
        source, WickAggregationConfig(), spin_basis="DP", shared_orthonormal_laph_basis=True,
    )
    assert result is source
    assert report == {"enabled": False}


def test_transposed_fixed_spin_term_is_combined_with_its_exact_phase():
    source = _spin_dag()
    result, report = _aggregate(source)
    assert len(source.correlators[0].terms) == 2
    assert len(result.correlators[0].terms) == 1
    assert result.correlators[0].terms[0].coefficient == -1
    assert report["correlators"][0]["merged_groups"][0]["tensor_phases_exact"] == [["1", "0"], ["-1", "0"]]
    _validate(result)
    matrix = get_spin_matrices("DP")["Cg5"]
    np.testing.assert_array_equal(matrix + 2 * matrix.T, -matrix)
    assert np.max(np.abs(matrix)) > 0.1


@pytest.mark.parametrize("change", [{"maximum_checks": 0}, {"maximum_match_states": 0}])
def test_exhausted_search_keeps_unproved_terms(change):
    source = _spin_dag()
    result, report = _aggregate(source, **change)
    assert result.correlators == source.correlators
    assert report["check_budget_exhausted"] or report["unproved_candidate_pairs"] == 1
    _validate(result)


def test_hard_time_limit_fails_loudly_instead_of_keeping_terms(monkeypatch):
    ticks = iter(range(0, 10 ** 9, 1000))
    for module in (wick_aggregation, _tensor_isomorphism):
        monkeypatch.setattr(module, "monotonic", lambda: next(ticks))
    with pytest.raises(WickAggregationTimeLimitExceeded, match="search_seconds"):
        _aggregate(_spin_dag(), search_seconds=1)


def test_result_depends_on_counts_not_on_elapsed_time(monkeypatch):
    expected = _aggregate(_spin_dag())
    ticks = iter(range(10 ** 9))
    for module in (wick_aggregation, _tensor_isomorphism):
        monkeypatch.setattr(module, "monotonic", lambda: next(ticks))
    result, report = _aggregate(_spin_dag(), search_seconds=10 ** 8)
    assert result == expected[0]
    assert {key: value for key, value in report.items() if key != "seconds"} == {
        key: value for key, value in expected[1].items() if key != "seconds"}


def test_all_zero_open_output_keeps_its_typed_interface():
    source = _spin_dag((1, 1))
    result, report = _aggregate(source)
    assert result.correlators == source.correlators
    assert report["correlators"][0]["retained_all_zero_output"]
    _validate(result)


def test_coefficient_and_spin_entries_are_not_tolerance_pruned():
    value = _ExactComplex.from_complex(1 / 3)
    assert value.real == Fraction.from_float(1 / 3) != Fraction(1, 3)
    assert _ExactComplex.from_complex(1e-30)
    matrix = np.asarray(get_spin_matrices("DP")["Cg5"]).copy()
    assert _spin_contract(matrix)[3] == "alternating"
    matrix[0, 0] = 1e-30
    assert _spin_contract(matrix)[3] == "ordered"


@pytest.mark.parametrize("coefficients", [(1, -(2 ** -53)), (1e308, -1e308)])
def test_nonrepresentable_sum_reuses_the_tensor_without_rounding_its_weights(coefficients):
    source = _spin_dag(coefficients)
    result, _ = _aggregate(source)
    terms = result.correlators[0].terms
    assert len(terms) == 2
    assert terms[0].components == terms[1].components
    expected = Fraction.from_float(coefficients[0]) - Fraction.from_float(coefficients[1])
    assert sum(Fraction.from_float(term.coefficient.real) for term in terms) == expected
    _validate(result)


def test_split_weights_survive_path_planning_and_logical_lowering():
    from concord_lqcd.backend.runtime.pipeline import _lower_execution_stages
    from concord_lqcd.compiler.contraction_order import ContractionSizeModel, optimize_global_contraction_order

    result, _ = _aggregate(_spin_dag((1, -(2 ** -53))))
    config = RuntimeCompilationConfig(mode_size=3, shared_orthonormal_laph_basis=True,
                                      wick_aggregation=WickAggregationConfig(enabled=True))
    order = optimize_global_contraction_order(result, ContractionSizeModel(3, 4), strategy="greedy")
    stages = _lower_execution_stages(order, config, physical_memory_limit_bytes=None)
    assert len(stages.logical_dag.primitives) == 1
    assert len(stages.logical_dag.terms) == 2
    expected = Fraction(1) + Fraction(1, 2 ** 53)
    assert sum(Fraction.from_float(term.coefficient.real) for term in stages.logical_dag.terms) == expected


def test_split_weights_roundtrip_through_the_plan_object_codec():
    import json
    from concord_lqcd.job.plan import _decode_value, _encode_value

    result, _ = _aggregate(_spin_dag((1, -(2 ** -53))))
    restored = _decode_value(json.loads(json.dumps(_encode_value(result))))
    _validate(restored)
    assert restored == result
    first, second = restored.correlators[0].terms
    assert first.components == second.components
    assert Fraction.from_float(first.coefficient.real) + Fraction.from_float(second.coefficient.real) == (
        Fraction(1) + Fraction(1, 2 ** 53))


@pytest.mark.parametrize("phase", [1, -1, 1j, -1j])
def test_all_unit_phases_align_complex_and_subnormal_weights_exactly(monkeypatch, phase):
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4).astype(np.complex128)
    monkeypatch.setattr("concord_lqcd.core.dirac_basis.get_spin_matrices",
                        lambda basis: {"first": matrix, "second": phase * matrix.T})
    coefficients = (0.25 + 0.5j, 1e-320 - 0.75j)
    result, _ = _aggregate(_spin_dag(coefficients, kinds=("first", "second")))
    expected = _ExactComplex.from_complex(coefficients[0]) + (
        _ExactComplex.from_complex(coefficients[1]) * _ExactComplex.from_complex(phase))
    observed = sum((_ExactComplex.from_complex(term.coefficient) for term in result.correlators[0].terms),
                   _ExactComplex())
    assert observed == expected
    _validate(result)


@pytest.mark.parametrize("change", ["missing_component", "different_output", "duplicate_port"])
def test_invalid_source_terms_cannot_be_hidden_by_cancellation(change):
    result, _ = _aggregate(_spin_dag())
    proof = result.aggregation_proof
    source = proof.source
    correlator = source.correlators[0]
    original = correlator.terms[0]
    if change == "duplicate_port":
        component_id = original.components[0].computation_id
        component = source.components[component_id]
        tensor = component.tensors[0]
        tensor = replace(tensor, bindings=(*tensor.bindings, tensor.bindings[0]))
        component = replace(component, tensors=(tensor,))
        source = replace(source, components={**source.components, component_id: component})
    else:
        use = original.components[0]
        if change == "missing_component":
            use = replace(use, computation_id="missing")
        else:
            first, *rest = use.output_bindings
            use = replace(use, output_bindings=(replace(first, index="different_output"), *rest))
        zero = replace(original, term_index=2, coefficient=0, components=(use,))
        source = replace(source, correlators=(replace(correlator, terms=(*correlator.terms, zero)),))
        record = proof.correlators[0]
        singleton = replace(record.classes[0], members=(2,), witnesses=())
        proof = replace(proof, correlators=(replace(record, classes=(*record.classes, singleton)),))
    result = replace(result, aggregation_proof=replace(proof, source=source))
    with pytest.raises(ValueError):
        _validate(result)


def test_no_basis_declaration_means_no_new_symmetry_reuse():
    source = _spin_dag()
    result, report = aggregate_certified_wick_terms(
        source, WickAggregationConfig(enabled=True), spin_basis="DP",
        shared_orthonormal_laph_basis=False,
    )
    assert result.correlators == source.correlators
    assert report["correlators"][0]["unsupported_contracts"]


def test_correlator_sums_are_never_combined_with_each_other():
    source = _spin_dag()
    source = replace(source, correlators=(source.correlators[0], replace(source.correlators[0], correlator_id="other")))
    result, _ = _aggregate(source)
    assert [item.correlator_id for item in result.correlators] == ["open", "other"]
    assert [len(item.terms) for item in result.correlators] == [1, 1]
    _validate(result)


@pytest.mark.parametrize("change", ["coefficient", "coverage", "phase", "resource"])
def test_certificate_replay_rejects_altered_algebra(change):
    result, _ = _aggregate(_spin_dag())
    proof = result.aggregation_proof
    if change == "coefficient":
        correlator = result.correlators[0]
        result = replace(result, correlators=(replace(correlator, terms=(replace(correlator.terms[0], coefficient=42),)),))
    elif change == "resource":
        result = replace(result, resources={})
    else:
        record = proof.correlators[0]
        group = record.classes[0]
        if change == "coverage":
            group = replace(group, members=(0, 0))
        else:
            witness = {**group.witnesses[0], "phase": -group.witnesses[0]["phase"]}
            group = replace(group, witnesses=(witness,))
        record = replace(record, classes=(group,))
        result = replace(result, aggregation_proof=replace(proof, correlators=(record,)))
    with pytest.raises(ValueError):
        _validate(result)


@pytest.mark.parametrize("changes", [{"search_seconds": float("nan")}, {"search_seconds": -1},
                                      {"search_seconds": 0}, {"maximum_checks": True},
                                      {"maximum_match_states": 1.5}])
def test_invalid_discovery_limits_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        WickAggregationConfig(**changes)


def _two_nucleons(triplet=False):
    reset_registry()
    if triplet:
        def projected(flavor, position, spectator):
            internal = position + "_internal"
            return (build_baryon_operator(
                ("u", "d", "u") if flavor == "p" else ("d", "u", "d"), "Cg5", pos_label=position,
                color_labels=tuple(f"{position}color{ordinal}" for ordinal in range(3)),
                spin_labels=(position + "alpha", position + "beta", internal),
            ) * spin_matrix("PplusUpper", internal, spectator))

        first = projected("p", "x", "rho") * projected("n", "y", "sigma") * spin_matrix("Cg3", "rho", "sigma")
        second = projected("n", "x", "rho") * projected("p", "y", "sigma") * spin_matrix("Cg3", "rho", "sigma")
        operator = (-2j) * (first - second)
        return {"two_nucleons": Correlator(operator, operator.dagger())}
    baryons = [build_baryon_operator(
        ("u", "d", "u"), "Cg5", pos_label=position,
        color_labels=tuple(f"{position}color{ordinal}" for ordinal in range(3)),
        spin_labels=(f"{position}alpha", f"{position}beta", spectator),
    ) for position, spectator in (("x", "rho"), ("y", "sigma"))]
    operator = baryons[0] * baryons[1] * spin_matrix("Cg5Upper", "rho", "sigma")
    return {"two_nucleons": Correlator(operator, operator.dagger())}


class _AlternatingKernel(KernelMaterializationProvider):
    def materialize(self, request):
        modes = request.expected_shape[0]
        generator = np.random.default_rng(40 + request.operator_times[0])
        raw = generator.normal(size=(modes,) * 3) + 1j * generator.normal(size=(modes,) * 3)
        alternating = (raw + raw.transpose(1, 2, 0) + raw.transpose(2, 0, 1)
                       - raw.transpose(1, 0, 2) - raw.transpose(2, 1, 0) - raw.transpose(0, 2, 1))
        recipe = request.recipe
        colors = tuple(field.record.color_label for field in recipe.projected_fields)
        order = tuple(colors.index(color) for color in recipe.color_tensors[0].indices)
        inversions = sum(order[left] > order[right] for left in range(3) for right in range(left + 1, 3))
        result = alternating.conj() if recipe.projected_fields[0].record.is_anti else alternating
        return torch.from_numpy(((-1) ** inversions) * result)


class _RandomPropagation(PropagationSourceHandler):
    def __init__(self, degenerate):
        self.degenerate = degenerate

    def materialize(self, request):
        flavor = request.recipe.segment_flavors[0]
        offset = 0 if self.degenerate or flavor == "u" else 1000
        generator = np.random.default_rng(902 + offset + 10 * request.source_time + request.sink_time)
        return torch.from_numpy(generator.normal(size=request.expected_shape)
                                + 1j * generator.normal(size=request.expected_shape))


def _execute(compilation, degenerate):
    sources = {resource_id: ExternalTensorSource(source_id=resource_id, kind="random", location=resource_id)
               for resource_id in compilation.required_propagation_resources}
    result = execute_compilation(compilation, RuntimeExecutionConfig(
        operator_times={2: (1, 0)}, propagation_sources=sources,
        kernel_providers={"distillation": _AlternatingKernel()},
        source_handlers={"random": _RandomPropagation(degenerate)},
    ))
    return result.output("two_nucleons").numpy().copy()


@pytest.mark.parametrize("basis", ["DP", "DR"])
@pytest.mark.parametrize("degenerate", [False, True])
@pytest.mark.parametrize("triplet", [False, True])
def test_two_nucleon_pipeline_matches_unaggregated_execution(basis, degenerate, triplet):
    correlators = _two_nucleons(triplet)
    # Six perambulators: below mode size 12 a sampled identity can be an artifact of the small basis
    # (at 3 every alternating kernel is a multiple of epsilon), so the cross-check uses 12.
    config = RuntimeCompilationConfig(mode_size=12, spin_basis=basis, shared_orthonormal_laph_basis=True,
                                      numerical_flavor_groups=(("u", "d"),) if degenerate else ())
    baseline = compile_correlators(correlators, config)
    optimized = compile_correlators(correlators, replace(config, wick_aggregation=WickAggregationConfig(enabled=True)))
    assert len(optimized.global_cse.correlators[0].terms) < len(baseline.global_cse.correlators[0].terms)
    diagnostics = optimized.wick_aggregation_diagnostics
    assert not diagnostics["check_budget_exhausted"] and diagnostics["unproved_candidate_pairs"] == 0
    expected_count = (18 if degenerate else 36) if triplet else (15 if degenerate else 16)
    assert len(baseline.global_cse.correlators[0].terms) == (144 if triplet else 48)
    assert len(optimized.global_cse.correlators[0].terms) == expected_count
    source = optimized.global_cse.aggregation_proof.source
    assert all(term.components in tuple(original.components for original in source.correlators[0].terms)
               for term in optimized.global_cse.correlators[0].terms)
    expected = _execute(baseline, degenerate)
    actual = _execute(optimized, degenerate)
    assert np.max(np.abs(expected)) > 1e-5
    assert np.max(np.abs(actual - expected)) / np.max(np.abs(expected)) < 2e-11


@pytest.mark.parametrize("change", ["missing", "repeated", "reverse", "factor"])
def test_saved_chain_path_cannot_hide_a_changed_spin_contraction(change):
    result, _ = _aggregate(_spin_dag())
    proof = result.aggregation_proof
    record = proof.correlators[0]
    group = record.classes[0]
    if change == "missing":
        paths = ()
    else:
        first, *rest = group.spin_chains
        step = first[0][0]
        if change == "repeated":
            first = (first[0], first[0])
        elif change == "reverse":
            first = (((step[0], not step[1]),),)
        else:
            first = (((999, step[1]),),)
        paths = (first, *rest)
    group = replace(group, spin_chains=paths)
    altered = replace(result, aggregation_proof=replace(proof, correlators=(replace(record, classes=(group,)),)))
    with pytest.raises(ValueError):
        _validate(altered)


def _momentum_nucleon(third, position, momentum, spin, tag, displace=None):
    from concord_lqcd.core import Operator, d, epsilon, transport, u

    quark = third(position, f"{tag}c", spin)
    if displace is not None:
        quark = transport(quark, displace)
    return Operator(epsilon(f"{tag}a", f"{tag}b", f"{tag}c") * spin_matrix("Cg5", f"{tag}al", f"{tag}be")
                    * u(position, f"{tag}a", f"{tag}al") * d(position, f"{tag}b", f"{tag}be") * quark,
                    momentum=momentum)


def _deuteron_matrix():
    """N(k) N(-k) with C g3, at rest and back to back, plus a transported proton leg."""
    from concord_lqcd.core import CorrMatrix, d, u

    reset_registry()
    operators = []
    for index, (momentum, displace) in enumerate((((0, 0, 0), None), ((1, 0, 0), None), ((0, 0, 0), "+x"))):
        back = tuple(-component for component in momentum)
        pairs = []
        for sign, first, second in ((1, u, d), (-1, d, u)):
            pairs.append(sign * (_momentum_nucleon(first, "x", momentum, "s1", f"o{index}{sign}x", displace)
                                 * _momentum_nucleon(second, "y", back, "s2", f"o{index}{sign}y")
                                 * spin_matrix("Cg3", "s1", "s2")))
        operators.append(pairs[0] + pairs[1])
    return {f"C_{i}_{j}": correlator for i, j, correlator in CorrMatrix(operators)}


class _RecipeKernel(KernelMaterializationProvider):
    """Random kernels with exactly the relations the merging contracts assume, no more."""

    def materialize(self, request):
        from concord_lqcd.compiler.wick_aggregation import _flavorless_signature, _local_baryon_sign

        recipe = request.recipe
        operator = int(recipe.projected_fields[0].record.is_anti)
        time = request.operator_times[0]
        sign = _local_baryon_sign(recipe, operator)
        shape = request.expected_shape
        if sign is not None:
            momentum = tuple(recipe.momenta.get(recipe.position_labels[0], (0, 0, 0)))
            generator = np.random.default_rng(abs(hash(("baryon", time, operator, momentum))) % 2 ** 32)
            raw = generator.normal(size=shape) + 1j * generator.normal(size=shape)
            value = sign * (raw + raw.transpose(1, 2, 0) + raw.transpose(2, 0, 1)
                            - raw.transpose(1, 0, 2) - raw.transpose(2, 1, 0) - raw.transpose(0, 2, 1))
        else:
            key = ("kernel", time, operator, _flavorless_signature(recipe))
            generator = np.random.default_rng(abs(hash(key)) % 2 ** 32)
            value = generator.normal(size=shape) + 1j * generator.normal(size=shape)
        return torch.from_numpy(value)


@pytest.mark.parametrize("basis", ["DP", "DR"])
@pytest.mark.parametrize("degenerate", [False, True])
def test_momentum_and_displaced_matrix_matches_unaggregated_execution(degenerate, basis):
    correlators = _deuteron_matrix()
    # Six perambulators per term: the cross-check runs at mode size 12 = m.
    config = RuntimeCompilationConfig(mode_size=12, spin_basis=basis, shared_orthonormal_laph_basis=True,
                                      numerical_flavor_groups=(("u", "d"),) if degenerate else ())
    baseline = compile_correlators(correlators, config)
    optimized = compile_correlators(correlators, replace(config, wick_aggregation=WickAggregationConfig(enabled=True)))
    diagnostics = optimized.wick_aggregation_diagnostics
    assert diagnostics["unproved_candidate_pairs"] == 0 and not diagnostics["check_budget_exhausted"]
    assert not any(record["unsupported_contracts"] for record in diagnostics["correlators"])
    before = sum(len(item.terms) for item in baseline.global_cse.correlators)
    after = sum(len(item.terms) for item in optimized.global_cse.correlators)
    assert after < before

    def execute(compilation):
        sources = {resource_id: ExternalTensorSource(source_id=resource_id, kind="random", location=resource_id)
                   for resource_id in compilation.required_propagation_resources}
        result = execute_compilation(compilation, RuntimeExecutionConfig(
            operator_times={2: (1, 0)}, propagation_sources=sources,
            kernel_providers={"distillation": _RecipeKernel()},
            source_handlers={"random": _RandomPropagation(degenerate)},
        ))
        return {name: result.output(name).numpy().copy() for name in correlators}

    expected, actual = execute(baseline), execute(optimized)
    for name in correlators:
        scale = np.max(np.abs(expected[name]))
        assert scale > 1e-5
        assert np.max(np.abs(actual[name] - expected[name])) / scale < 2e-11, name
