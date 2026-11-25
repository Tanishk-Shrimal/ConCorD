"""Local helper freshness, isolated imports, and portable-plan source identity."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import sys
from types import ModuleType
from zipfile import ZipFile

import pytest

from concord_lqcd.job import (
    JobSpec, OperatorLoadError, PlanSerializationError, StalePlanError,
    compile_job_plan, compute_job_fingerprint, load_operator_collection,
    load_concord_plan, save_concord_plan, validate_plan_for_job,
)
from concord_lqcd.job.operator_sources import (
    OperatorSourceError, OperatorSourceSession, discover_operator_dependencies,
    validate_dependency_records,
)
from concord_lqcd.job.plan import PlanManifest


def _write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _helper(weight=2, weight_source=None):
    return (
        "from concord_lqcd.core import Correlator, Operator, u, ubar\n"
        + (weight_source if weight_source is not None else f"WEIGHT = {weight}\n")
        + "def build():\n"
        + "    operator = Operator(ubar('x','color','spin') * u('x','color','spin'))\n"
        + "    return {'loop': Correlator(WEIGHT * operator)}\n"
    )


def _spec(root, source="from helper import build\n", filename="operators.py"):
    operator_file = _write(root, filename, source)
    return JobSpec.model_validate({
        "schema_version": 1,
        "job": {"name": "local_helper_test"},
        "operators": {"file": str(operator_file), "factory": "build"},
        "plan": {"file": str(root / "plan.concord")},
        "lattice": {"spatial_extent": [1, 1, 1], "temporal_extent": 2},
        "compile": {"mode_size": 2},
        "run": {
            "device": "cpu", "configurations": {"values": [1]},
            "times": {"contexts": [{"id": "loop", "operator_times": {1: [0]}}]},
        },
        "data": {
            "eigenvectors": {
                "provider": "hdf5_distillation", "path": str(root / "eigenvectors.h5"),
                "dataset": "/eigenvectors", "axes": ["time", "z", "y", "x", "color", "mode"],
            },
            "propagation": {"provider": "hdf5_perambulator", "ordinary": {
                "path": str(root / "perambulators.h5"), "dataset": "/tau_{source_time}",
                "axes": ["sink_time", "source_spin", "sink_spin", "source_mode", "sink_mode"],
            }},
        },
        "output": {"file": str(root / "result.h5")},
    })


def _weights(plan):
    return tuple(abs(term.coefficient) for correlator in plan.global_cse.correlators
                 for term in correlator.terms)


@pytest.fixture(autouse=True)
def check_import_state():
    original_path = tuple(sys.path)
    private_names = {name for name in sys.modules if name.startswith("_concord_operator_sources_")}
    yield
    assert tuple(sys.path) == original_path
    assert {name for name in sys.modules if name.startswith("_concord_operator_sources_")} == private_names


def test_saved_plan_records_helpers_and_rejects_an_edit(tmp_path):
    helper_path = _write(tmp_path, "helper.py", _helper(2))
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)
    loaded = load_concord_plan(spec.plan.file)
    assert loaded.manifest.operator_dependencies == {
        "helper.py": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
    }
    validate_plan_for_job(loaded, spec)
    before = compute_job_fingerprint(spec)
    _write(tmp_path, "helper.py", _helper(5))
    assert compute_job_fingerprint(spec) != before
    with pytest.raises(StalePlanError, match="stale"):
        validate_plan_for_job(loaded, spec)


def test_deleted_helper_rejects_saved_plan(tmp_path):
    helper_path = _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    helper_path.unlink()
    with pytest.raises(StalePlanError, match="cannot validate"):
        validate_plan_for_job(plan, spec)


def test_transitive_edit_invalidates_and_recompiles_fresh(tmp_path):
    _write(tmp_path, "helper.py", _helper(weight_source="from constants import WEIGHT\n"))
    _write(tmp_path, "constants.py", "WEIGHT = 2\n")
    spec = _spec(tmp_path)
    original = compile_job_plan(spec)
    assert set(original.manifest.operator_dependencies) == {"helper.py", "constants.py"}
    _write(tmp_path, "constants.py", "WEIGHT = 5\n")
    with pytest.raises(StalePlanError, match="stale"):
        validate_plan_for_job(original, spec)
    assert _weights(original) == (2,)
    assert _weights(compile_job_plan(spec)) == (5,)


def test_same_named_helpers_and_preexisting_modules_remain_isolated(tmp_path, monkeypatch):
    cached = ModuleType("helper")
    cached.build = lambda: (_ for _ in ()).throw(AssertionError("global helper must not run"))
    monkeypatch.setitem(sys.modules, "helper", cached)
    for directory, weight in (("first", 2), ("second", 5), ("first", 7)):
        root = tmp_path / directory
        _write(root, "helper.py", _helper(weight))
        assert _weights(compile_job_plan(_spec(root))) == (weight,)
        assert sys.modules["helper"] is cached


@pytest.mark.parametrize("source", [
    "def build():\n    from helper import build as local_build\n    return local_build()\n",
    "import helper as local\nbuild = local.build\n",
    "from helper import *\n",
    "from .helper import build\n",
    "from . import helper\nbuild = helper.build\n",
    "import importlib\ndef build():\n    return importlib.import_module('helper').build()\n",
    "from importlib import import_module\ndef build():\n    return import_module('helper').build()\n",
    "import importlib.util\ndef build():\n    return importlib.import_module('helper').build()\n",
    "def build():\n    return __import__('helper').build()\n",
])
def test_common_local_import_styles(tmp_path, source):
    _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path, source)
    plan = compile_job_plan(spec)
    assert _weights(plan) == (2,)
    assert set(plan.manifest.operator_dependencies) == {"helper.py"}
    validate_plan_for_job(plan, spec)


@pytest.mark.parametrize("source", [
    "from family import build\n",
    "import family.mesons as mesons\nbuild = mesons.build\n",
    "from family.mesons import build\n",
])
def test_packages_and_relative_transitive_imports(tmp_path, source):
    _write(tmp_path, "family/__init__.py", "from .mesons import build\n")
    _write(tmp_path, "family/mesons.py", _helper(weight_source="from .constants import WEIGHT\n"))
    _write(tmp_path, "family/constants.py", "WEIGHT = 3\n")
    spec = _spec(tmp_path, source)
    plan = compile_job_plan(spec)
    assert _weights(plan) == (3,)
    assert set(plan.manifest.operator_dependencies) == {
        "family/__init__.py", "family/mesons.py", "family/constants.py",
    }
    _write(tmp_path, "family/__init__.py", "from .mesons import build\nSETTING = 1\n")
    with pytest.raises(StalePlanError):
        validate_plan_for_job(plan, spec)


def test_relative_dynamic_import_and_namespace_package(tmp_path):
    _write(tmp_path, "family/mesons.py", _helper(weight_source=(
        "import importlib\nWEIGHT = importlib.import_module('.constants', __package__).WEIGHT\n"
    )))
    _write(tmp_path, "family/constants.py", "WEIGHT = 4\n")
    spec = _spec(tmp_path, "from family.mesons import build\n")
    plan = compile_job_plan(spec)
    assert _weights(plan) == (4,)
    assert set(plan.manifest.operator_dependencies) == {"family/mesons.py", "family/constants.py"}
    _write(tmp_path, "family/__init__.py", "SETTING = 1\n")
    with pytest.raises(StalePlanError):
        validate_plan_for_job(plan, spec)


def test_circular_helpers_and_dataclass_decorators(tmp_path):
    _write(tmp_path, "helper.py", _helper(weight_source=(
        "from dataclasses import dataclass\n"
        "WEIGHT = 2\nimport second\n"
        "@dataclass\nclass Settings:\n    weight: int = second.WEIGHT\n"
        "WEIGHT = Settings().weight\n"
    )))
    _write(tmp_path, "second.py", "import helper\nWEIGHT = helper.WEIGHT + 3\n")
    assert _weights(compile_job_plan(_spec(tmp_path))) == (5,)


def test_dynamic_helper_edit_is_checked_without_factory_execution(tmp_path):
    marker = tmp_path / "factory_calls.txt"
    _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path, (
        "import importlib\nfrom pathlib import Path\n"
        "def build():\n"
        f"    with Path({str(marker)!r}).open('a') as handle:\n        handle.write('called\\n')\n"
        "    return importlib.import_module('helper').build()\n"
    ))
    plan = compile_job_plan(spec)
    assert marker.read_text() == "called\n"
    validate_plan_for_job(plan, spec)
    assert marker.read_text() == "called\n"
    _write(tmp_path, "helper.py", "raise AssertionError('validation must not execute helpers')\n")
    with pytest.raises(StalePlanError):
        validate_plan_for_job(plan, spec)
    assert marker.read_text() == "called\n"


def test_stale_bytecode_is_not_used(tmp_path):
    helper_path = _write(tmp_path, "helper.py", _helper(2))
    spec = _spec(tmp_path)
    py_compile.compile(str(helper_path), doraise=True)
    previous_stat = helper_path.stat()
    _write(tmp_path, "helper.py", _helper(5))
    assert helper_path.stat().st_size == previous_stat.st_size
    os.utime(helper_path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    assert _weights(compile_job_plan(spec)) == (5,)


def test_relocated_tree_has_same_fingerprint_and_accepts_plan(tmp_path):
    first = tmp_path / "first"
    _write(first, "helper.py", _helper())
    spec = _spec(first)
    plan = compile_job_plan(spec)
    second = tmp_path / "second"
    shutil.copytree(first, second)
    relocated = _spec(second)
    assert compute_job_fingerprint(relocated) == compute_job_fingerprint(spec)
    validate_plan_for_job(plan, relocated)


def test_unimported_helpers_do_not_affect_identity(tmp_path):
    _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path)
    before = compute_job_fingerprint(spec)
    _write(tmp_path, "unused.py", "this is deliberately invalid Python!\n")
    assert compute_job_fingerprint(spec) == before
    assert _weights(compile_job_plan(spec)) == (2,)


def test_conditional_imports_are_conservatively_tracked(tmp_path):
    _write(tmp_path, "helper.py", _helper())
    _write(tmp_path, "optional.py", "SETTING = 1\n")
    spec = _spec(tmp_path, "from helper import build\nif False:\n    import optional\n")
    plan = compile_job_plan(spec)
    assert set(plan.manifest.operator_dependencies) == {"helper.py", "optional.py"}
    _write(tmp_path, "optional.py", "SETTING = 2\n")
    with pytest.raises(StalePlanError):
        validate_plan_for_job(plan, spec)


@pytest.mark.parametrize("filename", ["operators-custom.py", "concord_lqcd.py"])
def test_arbitrary_main_filename_keeps_framework_identity(tmp_path, filename):
    original = sys.modules["concord_lqcd"]
    spec = _spec(tmp_path, _helper(), filename)
    assert _weights(compile_job_plan(spec)) == (2,)
    assert sys.modules["concord_lqcd"] is original


def test_factory_source_mutation_fails_planning(tmp_path):
    helper_path = _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path, (
        "from helper import build as local_build\nfrom pathlib import Path\n"
        "def build():\n"
        f"    Path({str(helper_path)!r}).write_text('WEIGHT = 9\\n')\n"
        "    return local_build()\n"
    ))
    with pytest.raises(OperatorLoadError, match="changed during planning"):
        compile_job_plan(spec)


def test_post_factory_source_mutation_fails_compilation(tmp_path, monkeypatch):
    import concord_lqcd.job.plan as plan_module

    helper_path = _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path)
    original_compile = plan_module.compile_correlators

    def changing_compile(*arguments):
        result = original_compile(*arguments)
        helper_path.write_text(_helper(9), encoding="utf-8")
        return result

    monkeypatch.setattr(plan_module, "compile_correlators", changing_compile)
    with pytest.raises(PlanSerializationError, match="changed while"):
        compile_job_plan(spec)


def test_loader_failure_cleans_private_imports(tmp_path):
    _write(tmp_path, "helper.py", "raise RuntimeError('helper failed')\n")
    spec = _spec(tmp_path)
    with pytest.raises(OperatorLoadError, match="helper failed"):
        load_operator_collection(spec)


def test_interleaved_sessions_are_independent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write(first, "helper.py", "WEIGHT = 2\n")
    _write(second, "helper.py", "WEIGHT = 5\n")
    first_main = _write(first, "operators.py", "import helper\n")
    second_main = _write(second, "operators.py", "import helper\n")
    with OperatorSourceSession(first_main) as first_session:
        with OperatorSourceSession(second_main) as second_session:
            assert first_session.load_main().helper.WEIGHT == 2
            assert second_session.load_main().helper.WEIGHT == 5
        assert first_session.load_main().helper.WEIGHT == 2


def test_external_symlink_helpers_are_not_silently_certified(tmp_path):
    external = _write(tmp_path, "outside.py", _helper())
    root = tmp_path / "operators"
    root.mkdir()
    (root / "helper.py").symlink_to(external)
    with pytest.raises(OperatorLoadError, match="outside"):
        compile_job_plan(_spec(root))


@pytest.mark.parametrize("records", [
    [], {"../helper.py": "0" * 64}, {"/helper.py": "0" * 64},
    {"./helper.py": "0" * 64}, {"family\\helper.py": "0" * 64},
    {"helper.txt": "0" * 64}, {"helper.py": "A" * 64},
    {"helper.py": "0" * 63}, {"helper.py": None}, {1: "0" * 64},
])
def test_malformed_dependency_records_are_rejected(records):
    with pytest.raises(OperatorSourceError):
        validate_dependency_records(records)


def test_manifest_rejects_untracked_legacy_and_invalid_dependency_records(tmp_path):
    _write(tmp_path, "helper.py", _helper())
    manifest = compile_job_plan(_spec(tmp_path)).manifest
    legacy = manifest.to_dict()
    del legacy["operator_dependencies"]
    legacy["plan_format_version"] = 1
    with pytest.raises(StalePlanError, match="helper dependency tracking"):
        PlanManifest.from_dict(legacy)
    with pytest.raises(PlanSerializationError, match="relative local"):
        replace(manifest, operator_dependencies={"../helper.py": "0" * 64}).validate()


def test_no_helper_source_is_executed_by_fingerprint_discovery(tmp_path):
    _write(tmp_path, "helper.py", "raise AssertionError('must not execute')\n")
    spec = _spec(tmp_path)
    assert set(discover_operator_dependencies(spec.operators.file)) == {"helper.py"}
    assert set(compute_job_fingerprint(spec).operator_dependencies) == {"helper.py"}


def test_dynamic_only_package_initializers_are_tracked(tmp_path):
    _write(tmp_path, "family/__init__.py", "SETTING = 1\n")
    _write(tmp_path, "family/mesons.py", _helper())
    spec = _spec(tmp_path, (
        "import importlib\ndef build():\n"
        "    return importlib.import_module('family.mesons').build()\n"
    ))
    plan = compile_job_plan(spec)
    assert set(plan.manifest.operator_dependencies) == {"family/__init__.py", "family/mesons.py"}
    validate_plan_for_job(plan, spec)
    _write(tmp_path, "family/__init__.py", "SETTING = 2\n")
    with pytest.raises(StalePlanError):
        validate_plan_for_job(plan, spec)


def test_runtime_choices_with_helpers_remain_outside_fingerprint(tmp_path):
    _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    payload = spec.model_dump(mode="python")
    payload["run"]["device"] = "gpu"
    payload["run"]["configurations"] = {"values": [5]}
    payload["output"]["file"] = tmp_path / "other_result.h5"
    payload["data"]["eigenvectors"]["path"] = str(tmp_path / "other_eigenvectors.h5")
    payload["data"]["propagation"]["ordinary"]["path"] = str(tmp_path / "other_perambulators.h5")
    modified = JobSpec.model_validate(payload)
    assert compute_job_fingerprint(modified) == compute_job_fingerprint(spec)
    validate_plan_for_job(plan, modified)


def test_invalid_relative_import_in_changed_helper_is_a_stale_plan(tmp_path):
    _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    _write(tmp_path, "helper.py", "from .. import outside\n")
    with pytest.raises(StalePlanError, match="escapes"):
        validate_plan_for_job(plan, spec)


def test_deleted_dynamic_helper_invalidates_plan(tmp_path):
    helper_path = _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path, (
        "import importlib\ndef build():\n    return importlib.import_module('helper').build()\n"
    ))
    plan = compile_job_plan(spec)
    helper_path.unlink()
    with pytest.raises(StalePlanError, match="cannot validate"):
        validate_plan_for_job(plan, spec)


def test_legacy_helper_archive_is_rejected_before_graph_decoding(tmp_path):
    _write(tmp_path, "helper.py", _helper())
    spec = _spec(tmp_path)
    plan = compile_job_plan(spec)
    save_concord_plan(plan, spec.plan.file)
    with ZipFile(spec.plan.file) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    del manifest["operator_dependencies"]
    manifest["plan_format_version"] = 1
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    members["plan.json"] = b"not a valid execution graph"
    with ZipFile(spec.plan.file, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(StalePlanError, match="helper dependency tracking"):
        load_concord_plan(spec.plan.file)


def test_python_file_takes_precedence_over_an_empty_namespace_directory(tmp_path):
    _write(tmp_path, "helper.py", _helper())
    (tmp_path / "helper").mkdir()
    assert _weights(compile_job_plan(_spec(tmp_path))) == (2,)


def test_package_star_import_loads_exported_submodules(tmp_path):
    _write(tmp_path, "family/__init__.py", "__all__ = ['mesons']\n")
    _write(tmp_path, "family/mesons.py", _helper())
    spec = _spec(tmp_path, "from family import *\nbuild = mesons.build\n")
    plan = compile_job_plan(spec)
    assert _weights(plan) == (2,)
    assert set(plan.manifest.operator_dependencies) == {"family/__init__.py", "family/mesons.py"}
    validate_plan_for_job(plan, spec)


@pytest.mark.parametrize("source", [
    "import family.missing\nbuild = family.missing.build\n",
    "import importlib\nbuild = importlib.import_module('family.missing').build\n",
])
def test_missing_local_child_never_uses_a_global_package(tmp_path, monkeypatch, source):
    _write(tmp_path, "family/__init__.py", "SETTING = 1\n")
    external_child = ModuleType("family.missing")
    external_child.build = lambda: (_ for _ in ()).throw(AssertionError("external child executed"))
    external_parent = ModuleType("family")
    external_parent.__path__ = []
    external_parent.missing = external_child
    monkeypatch.setitem(sys.modules, "family", external_parent)
    monkeypatch.setitem(sys.modules, "family.missing", external_child)
    with pytest.raises(OperatorLoadError, match="no local operator helper"):
        compile_job_plan(_spec(tmp_path, source))
    assert sys.modules["family"] is external_parent
    assert sys.modules["family.missing"] is external_child


def test_local_importlib_helper_retains_normal_local_precedence(tmp_path):
    _write(tmp_path, "importlib.py", _helper())
    spec = _spec(tmp_path, "from importlib import build\n")
    assert _weights(compile_job_plan(spec)) == (2,)
