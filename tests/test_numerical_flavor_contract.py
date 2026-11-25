"""Declaration and source preflight tests; no tensor loading or execution."""

from __future__ import annotations

import importlib

import pytest

from concord_lqcd.backend.runtime.pipeline import RuntimeCompilationConfig
from concord_lqcd.compiler.numerical_flavors import (
    canonical_flavor,
    normalize_numerical_flavor_groups,
)
from concord_lqcd.job.flavor_equivalence import validate_numerical_flavor_sources
from concord_lqcd.job.schema import CompileSpec, JobSpec


def _job_spec(
    tmp_path, *, groups=(("u", "d"),), ordinary=None,
    eigenvectors=None, generalized=None, compile_options=None,
):
    ordinary_source = {
        "path": str(tmp_path / "arbitrary_shared_tensor_{configuration}.h5"),
        "dataset": "/tau_{source_time}/sink_{sink_time}",
        "axes": ["sink_mode", "sink_spin", "source_mode", "source_spin"],
    }
    if ordinary is not None:
        if "files" in ordinary:
            ordinary_source.pop("path")
        ordinary_source.update(ordinary)
    basis = {
        "provider": "hdf5_distillation",
        "path": str(tmp_path / "basis_{configuration}.h5"),
        "dataset": "/evec_{time}",
        "axes": ["z", "y", "x", "color", "mode"],
    }
    if eigenvectors is not None:
        basis = eigenvectors
    return JobSpec.model_validate({
        "schema_version": 1,
        "job": {"name": "numerical_flavor_contract"},
        "operators": {"file": tmp_path / "unused_operators.py"},
        "plan": {"file": tmp_path / "unused.concord"},
        "lattice": {"spatial_extent": [2, 1, 1], "temporal_extent": 8},
        "compile": {
            "mode_size": 2,
            "numerical_flavor_groups": groups,
            **(compile_options or {}),
        },
        "run": {
            "configurations": {"values": [1, 4]},
            "times": {
                "generator": {
                    "type": "two_point", "source": 1,
                    "sinks": {"values": [3, 5]},
                },
            },
        },
        "data": {
            "eigenvectors": basis,
            "propagation": {
                "provider": "hdf5_perambulator",
                "ordinary": ordinary_source,
                "generalized": generalized,
            },
        },
        "output": {"file": tmp_path / "unused_output.h5"},
    })


def test_empty_contract_is_the_default():
    assert normalize_numerical_flavor_groups() == ()
    assert CompileSpec(mode_size=2).numerical_flavor_groups == ()
    assert RuntimeCompilationConfig(mode_size=2).numerical_flavor_groups == ()
    assert canonical_flavor("u") == "u"
    assert canonical_flavor("d") == "d"


def test_normalization_is_sorted_idempotent_and_does_not_mutate_inputs():
    groups = [["u", "s", "d"], ["c", "b"]]
    expected = (("b", "c"), ("d", "s", "u"))
    normalized = normalize_numerical_flavor_groups(groups)
    assert normalized == expected
    assert normalize_numerical_flavor_groups(normalized) is normalized
    assert groups == [["u", "s", "d"], ["c", "b"]]
    assert CompileSpec(
        mode_size=2, numerical_flavor_groups=groups,
    ).numerical_flavor_groups == expected
    assert RuntimeCompilationConfig(
        mode_size=2, numerical_flavor_groups=groups,
    ).numerical_flavor_groups == expected


@pytest.mark.parametrize("groups", [None, "ud", b"ud", 2, {"u", "d"}, {"u": "d"}])
def test_normalization_rejects_nonsequence_containers(groups):
    with pytest.raises(ValueError, match="sequence of flavor groups"):
        normalize_numerical_flavor_groups(groups)


@pytest.mark.parametrize("group", [None, "ud", b"ud", 2, {"u", "d"}, {"u": "d"}])
def test_normalization_rejects_nonsequence_groups(group):
    with pytest.raises(ValueError, match="group must be a sequence"):
        normalize_numerical_flavor_groups([group])


@pytest.mark.parametrize("group", [[], ["u"]])
def test_normalization_rejects_small_groups(group):
    with pytest.raises(ValueError, match="at least two"):
        normalize_numerical_flavor_groups([group])


@pytest.mark.parametrize(
    "flavor", ["", " ", " u", "u ", "u\n", None, 1, True, b"u", ["u"]],
)
def test_normalization_rejects_inexact_labels(flavor):
    with pytest.raises(ValueError, match="nonempty exact strings"):
        normalize_numerical_flavor_groups([["d", flavor]])


@pytest.mark.parametrize(
    "groups, message",
    [([["u", "u"]], "duplicate"), ([["u", "d"], ["s", "u"]], "disjoint")],
)
def test_normalization_rejects_duplicate_or_overlapping_labels(groups, message):
    with pytest.raises(ValueError, match=message):
        normalize_numerical_flavor_groups(groups)


def test_canonical_flavors_require_explicit_case_sensitive_membership():
    groups = [["u", "d"], ["s", "c"]]
    assert [canonical_flavor(flavor, groups) for flavor in ("u", "d", "s", "c")] == [
        "d", "d", "c", "c",
    ]
    assert canonical_flavor("U", groups) == "U"
    assert canonical_flavor("strange", groups) == "strange"
    assert groups == [["u", "d"], ["s", "c"]]


@pytest.mark.parametrize("flavor", [None, 1, "", " u"])
def test_canonical_flavor_rejects_invalid_labels(flavor):
    with pytest.raises(ValueError, match="nonempty exact strings"):
        canonical_flavor(flavor)


def test_canonical_flavor_does_not_bypass_contract_validation():
    with pytest.raises(ValueError, match="disjoint"):
        canonical_flavor("u", [["u", "d"], ["u", "s"]])


def test_shared_sources_pass_without_reading_data_or_mutating_spec(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("source preflight must not open HDF5 data")

    monkeypatch.setattr("h5py.File", forbidden)
    spec = _job_spec(tmp_path)
    before = spec.model_dump()
    assert validate_numerical_flavor_sources(spec) is None
    assert spec.model_dump() == before
    assert not spec.operators.file.exists()
    assert not spec.output.file.exists()


def test_explicit_file_map_can_bind_arbitrarily_named_identical_tensors(tmp_path):
    common = str(tmp_path / "no_flavor_in_this_name_{configuration}.h5")
    spec = _job_spec(tmp_path, ordinary={"files": {"u": common, "d": common}})
    assert validate_numerical_flavor_sources(spec) is None


@pytest.mark.parametrize("source_field", ["qbar", "qdagger"])
@pytest.mark.parametrize("spin_basis", ["DR", "DP"])
def test_shared_time_spin_and_source_conventions_are_preserved(
    tmp_path, source_field, spin_basis,
):
    spec = _job_spec(
        tmp_path,
        ordinary={
            "dataset": "/fixed_{fixed_source_time}/src_{source_time}/sink_{sink_time}",
            "fixed_source_time": 1,
            "source_field": source_field,
            "axes": [
                "sink_time", "source_spin", "source_mode", "sink_spin", "sink_mode",
                "complex_pair",
            ],
        },
        compile_options={"spin_basis": spin_basis, "scalar_type": "complex64"},
    )
    before = spec.model_dump()
    validate_numerical_flavor_sources(spec)
    assert spec.model_dump() == before


@pytest.mark.parametrize("binding", ["path", "dataset", "files"])
def test_different_flavor_bindings_are_rejected(tmp_path, binding):
    if binding == "files":
        ordinary = {"files": {
            "u": str(tmp_path / "mass_equal_u.h5"),
            "d": str(tmp_path / "mass_equal_d.h5"),
        }}
    elif binding == "path":
        ordinary = {"path": str(tmp_path / "tau_{flavor}_{configuration}.h5")}
    else:
        ordinary = {"dataset": "/{flavor}/tau_{source_time}/sink_{sink_time}"}
    spec = _job_spec(tmp_path, ordinary=ordinary)
    with pytest.raises(ValueError, match="different ordinary HDF5 tensors"):
        validate_numerical_flavor_sources(spec)


def test_literal_flavor_text_in_dataset_is_not_a_flavor_binding(tmp_path):
    spec = _job_spec(tmp_path, ordinary={"dataset": "/{{flavor}}/tau_{source_time}"})
    validate_numerical_flavor_sources(spec)


def test_missing_nonrepresentative_flavor_is_rejected_even_if_optimized_away(tmp_path):
    spec = _job_spec(tmp_path, ordinary={"files": {"d": "shared.h5"}})
    assert canonical_flavor("u", spec.compile.numerical_flavor_groups) == "d"
    with pytest.raises(ValueError, match="exact flavor 'u'"):
        validate_numerical_flavor_sources(spec)


def test_all_declared_groups_are_checked_not_only_used_flavors(tmp_path):
    spec = _job_spec(
        tmp_path, groups=(("u", "d"), ("s", "c")),
        ordinary={"files": {"u": "shared.h5", "d": "shared.h5", "c": "shared.h5"}},
    )
    with pytest.raises(ValueError, match="exact flavor 's'"):
        validate_numerical_flavor_sources(spec)


def test_default_preflight_checks_every_configuration(tmp_path):
    spec = _job_spec(tmp_path, ordinary={"files": {
        "u": str(tmp_path / "tau_{configuration}.h5"),
        "d": str(tmp_path / "tau_1.h5"),
    }})
    validate_numerical_flavor_sources(spec, configuration=1)
    with pytest.raises(ValueError, match="configuration 4"):
        validate_numerical_flavor_sources(spec)
    with pytest.raises(ValueError, match="configuration 4"):
        validate_numerical_flavor_sources(spec, configuration=4)


@pytest.mark.parametrize("groups", [(), (("u", "d"),)])
def test_opaque_generalized_resolvers_require_a_separate_contract(tmp_path, groups):
    spec = _job_spec(
        tmp_path, groups=groups,
        generalized={"resolver": "unimported_current_module:resolve"},
    )
    if groups:
        with pytest.raises(ValueError, match="generalized current sources"):
            validate_numerical_flavor_sources(spec)
    else:
        validate_numerical_flavor_sources(spec)


@pytest.mark.parametrize("groups", [(), (("u", "d"),)])
def test_custom_basis_is_not_certified_implicitly(tmp_path, groups):
    spec = _job_spec(
        tmp_path, groups=groups,
        eigenvectors={"provider": "custom", "factory": "unimported_basis_module:build"},
    )
    if groups:
        with pytest.raises(ValueError, match="shared LapH basis"):
            validate_numerical_flavor_sources(spec)
    else:
        validate_numerical_flavor_sources(spec)


def test_configuration_axis_is_not_silently_accepted_for_dropped_flavors(tmp_path):
    spec = _job_spec(tmp_path, ordinary={"axes": [
        "configuration", "sink_mode", "sink_spin", "source_mode", "source_spin",
    ]})
    with pytest.raises(ValueError, match="configuration-axis"):
        validate_numerical_flavor_sources(spec)


def test_no_contract_does_not_require_bindings_for_unmentioned_flavors(tmp_path):
    spec = _job_spec(tmp_path, groups=(), ordinary={"files": {"d": "only_d.h5"}})
    validate_numerical_flavor_sources(spec)


@pytest.mark.parametrize("entry", ["compile", "run"])
@pytest.mark.parametrize("groups", [(), (("u", "d"),)])
def test_entry_points_preflight_before_operator_or_plan_loading(
    tmp_path, monkeypatch, entry, groups,
):
    module_name, function_name, load_name = (
        ("concord_lqcd.job.plan", "compile_job_plan", "_load_operator_snapshot")
        if entry == "compile"
        else ("concord_lqcd.job.runner", "run_job", "_load_prepared_plan")
    )
    module = importlib.import_module(module_name)

    def stop_before_loading(*args, **kwargs):
        raise RuntimeError("stopped before any compilation or numerical execution")

    monkeypatch.setattr(module, load_name, stop_before_loading)
    spec = _job_spec(tmp_path, groups=groups, ordinary={"files": {"d": "shared.h5"}})
    error_type, message = (
        (ValueError, "exact flavor 'u'")
        if groups else (RuntimeError, "stopped before any compilation")
    )
    with pytest.raises(error_type, match=message):
        getattr(module, function_name)(spec)


def test_preflight_requires_a_job_spec():
    with pytest.raises(ValueError, match="spec must be JobSpec"):
        validate_numerical_flavor_sources(object())
