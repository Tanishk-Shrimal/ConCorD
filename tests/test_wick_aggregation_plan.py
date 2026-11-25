"""Saved certificates remain executable source records, not trusted diagnostics."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from concord_lqcd.compiler import wick_aggregation
from concord_lqcd.job import (
    JobSpec, PlanSerializationError, StalePlanError, compile_job_plan,
    load_concord_plan, save_concord_plan, validate_plan_for_job,
)


def _spec(tmp_path):
    operator = tmp_path / "operators.py"
    operator.write_text(
        "from concord_lqcd.core import build_baryon_operator, Correlator, reset_registry, spin_matrix\n"
        "def build():\n"
        "    reset_registry()\n"
        "    first = build_baryon_operator(('u','d','u'), 'Cg5', pos_label='x', "
        "color_labels=('a','b','c'), spin_labels=('alpha','beta','rho'))\n"
        "    second = build_baryon_operator(('u','d','u'), 'Cg5', pos_label='y', "
        "color_labels=('e','f','g'), spin_labels=('eta','theta','sigma'))\n"
        "    operator = first * second * spin_matrix('Cg5Upper', 'rho', 'sigma')\n"
        "    return {'nn': Correlator(operator, operator.dagger())}\n",
        encoding="utf-8",
    )
    return JobSpec.model_validate({
        "schema_version": 1, "job": {"name": "certificates"},
        "operators": {"file": operator}, "plan": {"file": tmp_path / "plan.concord"},
        "lattice": {"spatial_extent": [2, 2, 2], "temporal_extent": 4},
        "compile": {"mode_size": 3, "spin_basis": "DP", "numerical_flavor_groups": [["u", "d"]],
                    "wick_aggregation": {"enabled": True}},
        "run": {"device": "cpu", "configurations": {"values": [1]},
                "times": {"generator": {"type": "two_point", "source": 0, "sinks": {"values": [1]}}}},
        "data": {"eigenvectors": {"provider": "hdf5_distillation", "path": "shared_basis.h5",
                                  "dataset": "eigenvectors", "axes": ["time", "z", "y", "x", "color", "mode"]},
                 "propagation": {"provider": "hdf5_perambulator", "ordinary": {
                     "files": {"u": "shared_tau.h5", "d": "shared_tau.h5"},
                     "dataset": "tau_{source_time}",
                     "axes": ["sink_time", "sink_mode", "sink_spin", "source_mode", "source_spin"]}}},
        "output": {"file": tmp_path / "output.h5"},
    })


def test_saved_plan_replays_certificates_without_discovery_or_compiler_passes(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    assert len(plan.global_cse.correlators[0].terms) < len(plan.global_cse.aggregation_proof.source.correlators[0].terms)
    save_concord_plan(plan, spec.plan.file)

    def forbidden(*args, **kwargs):
        pytest.fail("Loading a compiled plan must not repeat discovery or optimization")

    monkeypatch.setattr(wick_aggregation, "find_witness", forbidden)
    monkeypatch.setattr(wick_aggregation, "aggregate_certified_wick_terms", forbidden)
    monkeypatch.setattr(wick_aggregation, "discover_spin_chains", forbidden)
    monkeypatch.setattr("concord_lqcd.compiler._spin_chains._discover_paths", forbidden)
    with patch("concord_lqcd.job.plan.compile_correlators", forbidden):
        loaded = load_concord_plan(spec.plan.file)
        validate_plan_for_job(loaded, spec)
    assert loaded.global_cse == plan.global_cse
    assert loaded.manifest.plan_sha256 == plan.manifest.plan_sha256


def test_rehashed_archive_cannot_hide_an_incorrect_aggregation_coefficient(tmp_path):
    from concord_lqcd.job.plan import _canonical_json_bytes, _sha256_bytes

    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    proof = plan.global_cse.aggregation_proof
    source = proof.source
    correlator = source.correlators[0]
    altered = replace(correlator.terms[0], coefficient=correlator.terms[0].coefficient + 1)
    source = replace(source, correlators=(replace(correlator, terms=(altered, *correlator.terms[1:])),))
    proof = replace(proof, source=source)
    plan.global_cse.__dict__["aggregation_proof"] = proof
    plan = replace(plan, manifest=replace(plan.manifest,
        plan_sha256=_sha256_bytes(_canonical_json_bytes(plan.execution_payload()))))
    with pytest.raises(PlanSerializationError, match="certificate"):
        plan.validate()


@pytest.mark.parametrize("change", ["switch", "budget", "flavors"])
def test_aggregation_choices_invalidate_the_saved_job_plan(tmp_path, change):
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    if change == "flavors":
        compile_spec = spec.compile.model_copy(update={"numerical_flavor_groups": ()})
    else:
        settings = spec.compile.wick_aggregation.model_copy(update={
            "enabled": False} if change == "switch" else {"maximum_checks": 1})
        compile_spec = spec.compile.model_copy(update={"wick_aggregation": settings})
    altered = spec.model_copy(update={"compile": compile_spec})
    with pytest.raises(StalePlanError):
        validate_plan_for_job(plan, altered)
