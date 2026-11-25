"""File data resolvers retain fresh, isolated helper imports across jobs."""

import os
import py_compile
import sys
from types import ModuleType

import pytest

from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource
from concord_lqcd.job.data import JobDataError, _load_reference_callable


def _write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _resolver(root, imports, body):
    path = _write(root, "resolver.py", (
        "from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource\n"
        + imports
        + "def resolve(**kwargs):\n"
        + body
    ))
    return _load_reference_callable(f"{path}:resolve")


def _source_body(expression):
    return (
        f"    return ExternalTensorSource('source', 'hdf5', f'/tmp/tau_{{{expression}}}.h5', '/tau')\n"
    )


@pytest.fixture(autouse=True)
def check_import_cleanup():
    original_path = tuple(sys.path)
    original_private = {
        name for name in sys.modules
        if name.startswith(("_concord_operator_sources_", "_concord_job_data_"))
    }
    yield
    assert tuple(sys.path) == original_path
    assert {
        name for name in sys.modules
        if name.startswith(("_concord_operator_sources_", "_concord_job_data_"))
    } == original_private


def test_same_named_helpers_resolve_independently_without_polluting_existing_modules(tmp_path, monkeypatch):
    existing = ModuleType("resolver_helper")
    existing.WEIGHT = 99
    monkeypatch.setitem(sys.modules, "resolver_helper", existing)
    resolvers = []
    for name, weight in (("first", 2), ("second", 5)):
        root = tmp_path / name
        _write(root, "resolver_helper.py", "from resolver_constants import WEIGHT\n")
        _write(root, "resolver_constants.py", f"WEIGHT = {weight}\n")
        resolvers.append(_resolver(root, "from resolver_helper import WEIGHT\n", _source_body("WEIGHT")))

    for index in (0, 1, 0, 1):
        source = resolvers[index]()
        assert isinstance(source, ExternalTensorSource)
        assert source.location == f"/tmp/tau_{(2, 5)[index]}.h5"
        assert sys.modules["resolver_helper"] is existing
        assert "resolver_constants" not in sys.modules


def test_new_load_reads_transitive_helper_edits_instead_of_stale_bytecode(tmp_path):
    _write(tmp_path, "resolver_helper.py", "from resolver_constants import WEIGHT\n")
    constants = _write(tmp_path, "resolver_constants.py", "WEIGHT = 2\n")
    first = _resolver(tmp_path, "from resolver_helper import WEIGHT\n", _source_body("WEIGHT"))
    py_compile.compile(str(constants), doraise=True)
    previous = constants.stat()
    constants.write_text("WEIGHT = 5\n", encoding="utf-8")
    os.utime(constants, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    second = _load_reference_callable(f"{tmp_path / 'resolver.py'}:resolve")

    assert first().location == "/tmp/tau_2.h5"
    assert second().location == "/tmp/tau_5.h5"
    assert first().location == "/tmp/tau_2.h5"


@pytest.mark.parametrize("late_import", (
    "    from resolver_helper import WEIGHT\n",
    "    import importlib\n    WEIGHT = importlib.import_module('resolver_helper').WEIGHT\n",
    "    from importlib import import_module\n    WEIGHT = import_module('resolver_helper').WEIGHT\n",
))
def test_late_helper_imports_work_after_loader_returns_and_across_repeated_calls(tmp_path, late_import):
    resolvers = []
    for name, weight in (("first", 2), ("second", 5)):
        root = tmp_path / name
        _write(root, "resolver_helper.py", f"WEIGHT = {weight}\n")
        resolvers.append(_resolver(root, "", late_import + _source_body("WEIGHT")))

    for index in (0, 1, 1, 0):
        assert resolvers[index]().location == f"/tmp/tau_{(2, 5)[index]}.h5"
    assert "resolver_helper" not in sys.modules


def test_local_packages_relative_imports_and_dataclasses_have_private_module_identity(tmp_path):
    _write(tmp_path, "family/__init__.py", "from .factory import resolve\n")
    _write(tmp_path, "family/constants.py", "WEIGHT = 5\n")
    _write(tmp_path, "family/records.py", (
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, KW_ONLY\n"
        "from .constants import WEIGHT\n"
        "@dataclass\n"
        "class Record:\n"
        "    _: KW_ONLY\n"
        "    weight: int = WEIGHT\n"
    ))
    _write(tmp_path, "family/factory.py", (
        "from __future__ import annotations\n"
        "import importlib\n"
        "import sys\n"
        "from dataclasses import dataclass, KW_ONLY\n"
        "from concord_lqcd.backend.runtime.resource_binding import ExternalTensorSource\n"
        "from .records import Record\n"
        "def resolve(**kwargs):\n"
        "    assert sys.modules[__name__].resolve is resolve\n"
        "    records = importlib.import_module('.records', __package__)\n"
        "    assert records.Record is Record\n"
        "    @dataclass\n"
        "    class LateRecord:\n"
        "        _: KW_ONLY\n"
        "        weight: int = Record().weight\n"
        "    return ExternalTensorSource('source', 'hdf5', f'/tmp/tau_{LateRecord().weight}.h5', '/tau')\n"
    ))
    resolver_file = _write(tmp_path, "resolver.py", "from .family import resolve\n")
    resolve = _load_reference_callable(f"{resolver_file}:resolve")

    assert resolve().location == "/tmp/tau_5.h5"
    assert resolve().location == "/tmp/tau_5.h5"
    assert "family" not in sys.modules


def test_nested_wrapper_calls_retain_both_active_namespaces(tmp_path):
    second = _resolver(tmp_path / "second", "WEIGHT = 5\n", _source_body("WEIGHT"))
    first = _resolver(tmp_path / "first", "import sys\n", (
        "    own_module = sys.modules[__name__]\n"
        "    result = kwargs['peer']()\n"
        "    assert sys.modules[__name__] is own_module\n"
        "    return result\n"
    ))

    assert first(peer=second).location == "/tmp/tau_5.h5"
    assert first(peer=second).location == "/tmp/tau_5.h5"


def test_recursive_wrapper_calls_retain_same_active_namespace(tmp_path):
    resolve = _resolver(tmp_path, "import sys\n", (
        "    own_module = sys.modules[__name__]\n"
        "    if kwargs.get('depth', 0):\n"
        "        result = kwargs['peer'](depth=0)\n"
        "        assert sys.modules[__name__] is own_module\n"
        "        return result\n"
        + _source_body("2")
    ))

    assert resolve(peer=resolve, depth=1).location == "/tmp/tau_2.h5"


def test_eager_and_repeated_late_import_failures_clean_the_private_namespace(tmp_path):
    _write(tmp_path, "resolver_helper.py", "raise RuntimeError('helper failed')\n")
    with pytest.raises(JobDataError, match="helper failed"):
        _resolver(tmp_path, "import resolver_helper\n", _source_body("2"))

    late = _resolver(tmp_path, "", "    import resolver_helper\n" + _source_body("2"))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="helper failed"):
            late()
    assert "resolver_helper" not in sys.modules


def test_installed_module_references_keep_normal_module_loading():
    import math

    assert _load_reference_callable("math:sqrt") is math.sqrt
    assert _load_reference_callable("math:sqrt")(25) == 5
