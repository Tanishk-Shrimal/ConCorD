"""Executable contracts for ConCorD package ownership and documentation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "concord_lqcd"


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_name(path: Path) -> str:
    module = _module_name(path)
    if path.name == "__init__.py":
        return module
    return module.rpartition(".")[0]


def _import_targets(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_name(path)
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * node.level + (node.module or "")
                targets.append(importlib.util.resolve_name(relative, package))
            elif node.module:
                targets.append(node.module)
    return tuple(targets)


def _source_layer(path: Path) -> str:
    parts = path.relative_to(PACKAGE_ROOT).parts
    if len(parts) == 1:
        return "entry"
    if parts[0] in ("core", "compiler", "job"):
        return parts[0]
    if parts[0] != "backend":
        raise AssertionError(f"unclassified source module: {path.relative_to(ROOT)}")
    if len(parts) == 2:
        return "backend_root"
    if parts[1] in ("ir", "planner", "runtime"):
        return f"backend_{parts[1]}"
    raise AssertionError(f"unclassified backend module: {path.relative_to(ROOT)}")


def _matches(target: str, prefix: str) -> bool:
    return target == prefix or target.startswith(prefix + ".")


def _allowed_internal_import(path: Path, target: str) -> bool:
    layer = _source_layer(path)
    if layer == "entry":
        return True

    allowed = {
        "core": ("concord_lqcd.core",),
        "compiler": (
            "concord_lqcd.core",
            "concord_lqcd.compiler",
            "concord_lqcd.backend.ir",
        ),
        "backend_root": ("concord_lqcd.backend.ir",),
        "backend_ir": ("concord_lqcd.backend.ir",),
        "backend_planner": (
            "concord_lqcd.compiler",
            "concord_lqcd.backend.ir",
            "concord_lqcd.backend.planner",
        ),
        "backend_runtime": (
            "concord_lqcd.compiler",
            "concord_lqcd.backend.ir",
            "concord_lqcd.backend.planner",
            "concord_lqcd.backend.runtime",
        ),
        "job": (
            "concord_lqcd.core",
            "concord_lqcd.compiler",
            "concord_lqcd.backend.ir",
            "concord_lqcd.backend.planner",
            "concord_lqcd.backend.runtime",
            "concord_lqcd.job",
        ),
    }[layer]
    if any(_matches(target, prefix) for prefix in allowed):
        return True

    if layer == "backend_runtime" and _matches(
        target, "concord_lqcd.core.dirac_basis"
    ):
        return True
    return (
        layer == "backend_runtime"
        and path.name == "pipeline.py"
        and _matches(target, "concord_lqcd.core.correlator")
    )


def test_package_imports_follow_the_staged_dependency_direction():
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for target in _import_targets(path):
            if not _matches(target, "concord_lqcd"):
                continue
            if not _allowed_internal_import(path, target):
                violations.append(
                    f"{path.relative_to(ROOT)} ({_source_layer(path)}) -> {target}"
                )

    assert not violations, "forbidden cross-layer imports:\n" + "\n".join(violations)


def test_every_package_module_has_an_ownership_docstring():
    missing = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ast.get_docstring(tree) is None:
            missing.append(str(path.relative_to(ROOT)))
    assert not missing, "modules without ownership docstrings: " + ", ".join(missing)
