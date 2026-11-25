"""Local Python operator sources and isolated helper-module loading."""

from __future__ import annotations

import ast
import builtins
from collections.abc import Mapping
import hashlib
import importlib
import importlib.util
from pathlib import Path, PurePosixPath
import sys
from types import ModuleType
from uuid import uuid4


class OperatorSourceError(ValueError):
    """An operator source snapshot cannot be read or validated."""


def source_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_dependency_records(records) -> None:
    """Validate relative local-source identities without executing Python."""
    if not isinstance(records, Mapping):
        raise OperatorSourceError("operator dependencies must be a mapping")
    for name, digest in records.items():
        if not isinstance(name, str) or not name:
            raise OperatorSourceError("operator dependency names must be non-empty strings")
        path = PurePosixPath(name)
        if (path.is_absolute() or path.as_posix() != name or ".." in path.parts
                or "\\" in name or path.suffix != ".py"):
            raise OperatorSourceError("operator dependencies must name relative local .py files")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise OperatorSourceError("operator dependency digests must be lowercase SHA-256")


def _read_source(root: Path, name: str) -> bytes:
    path = root / name
    try:
        if not path.resolve().is_relative_to(root):
            raise OperatorSourceError(f"operator helper resolves outside its local directory: {path}")
        return path.read_bytes()
    except OSError as exc:
        raise OperatorSourceError(f"cannot read operator source {path}: {exc}") from exc


def _local_module(root: Path, name: str):
    parts = name.split(".")
    if not name or any(not part.isidentifier() for part in parts) or parts[0] == "concord_lqcd":
        return None
    base = root.joinpath(*parts)
    if base.is_dir():
        initializer = base / "__init__.py"
        if initializer.is_file():
            return (initializer, base)
    source = base.with_suffix(".py")
    if source.is_file():
        return (source, None)
    if base.is_dir():
        return (None, base)
    return None


def _relative_name(name: str, package: str, level: int) -> str:
    if not level:
        return name
    anchor = "_operator_root" + (f".{package}" if package else "")
    try:
        resolved = importlib.util.resolve_name("." * level + name, anchor)
    except ImportError as exc:
        raise OperatorSourceError("operator helper relative import escapes its local root") from exc
    if resolved == "_operator_root":
        return ""
    if not resolved.startswith("_operator_root."):
        raise OperatorSourceError("operator helper relative import escapes its local root")
    return resolved.removeprefix("_operator_root.")


def _collect_sources(path: Path, additional=()) -> dict[str, bytes]:
    root = path.parent
    pending = [path.name, *additional]
    sources = {}
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        content = _read_source(root, name)
        sources[name] = content
        try:
            tree = ast.parse(content, filename=str(root / name))
        except SyntaxError as exc:
            raise OperatorSourceError(f"invalid local operator source {root / name}: {exc}") from exc
        location = PurePosixPath(name)
        package = ".".join(location.parent.parts)
        targets = []
        for parent in location.parents:
            if parent == PurePosixPath("."):
                continue
            initializer = root / parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer.relative_to(root).as_posix())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = _relative_name(node.module or "", package, node.level)
                targets.append(target)
                targets.extend(
                    ".".join(part for part in (target, alias.name) if part)
                    for alias in node.names if alias.name != "*"
                )
        for target in targets:
            parts = target.split(".")
            for count in range(1, len(parts) + 1):
                module = _local_module(root, ".".join(parts[:count]))
                if module is not None and module[0] is not None:
                    pending.append(module[0].relative_to(root).as_posix())
    return sources


def discover_operator_dependencies(path: Path, additional=()) -> dict[str, str]:
    """Hash local direct/transitive imports conservatively, without importing them.

    Imports in inactive branches are included. Previously recorded dynamically
    imported helpers can be supplied as additional relative source paths.
    Framework/library modules and sources outside the operator directory are
    not certified by this local-source contract.
    """
    path = path.expanduser().resolve()
    return {
        name: source_digest(content)
        for name, content in sorted(_collect_sources(path, additional).items())
        if name != path.name
    }


class OperatorSourceSession:
    """Execute local sources from bytes in a private import namespace.

    Ordinary imports, relative package imports, and importlib.import_module
    resolve local helpers inside this session. External modules retain normal
    Python loading. Only private namespace entries are removed on exit. A
    session can be entered again to activate callables that retain its modules.
    """

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.root = self.path.parent
        self.sources = _collect_sources(self.path)
        self.namespace = f"_concord_operator_sources_{uuid4().hex}"
        self.modules: dict[str, ModuleType] = {}
        self._root_module = ModuleType(self.namespace)
        self._root_module.__package__ = self.namespace
        self._root_module.__path__ = [str(self.root)]
        self._active_entries = 0
        self.importlib = ModuleType("importlib")
        self.importlib.__dict__.update(importlib.__dict__)
        self.importlib.import_module = self._import_module

    def __enter__(self):
        if self._active_entries == 0:
            sys.modules[self.namespace] = self._root_module
            for name, module in self.modules.items():
                sys.modules[f"{self.namespace}.{name}"] = module
        self._active_entries += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._active_entries -= 1
        if self._active_entries:
            return
        for name in tuple(sys.modules):
            if name == self.namespace or name.startswith(f"{self.namespace}."):
                sys.modules.pop(name, None)

    @property
    def dependencies(self) -> dict[str, str]:
        return {name: source_digest(content) for name, content in sorted(self.sources.items())
                if name != self.path.name}

    @property
    def source_sha256(self) -> str:
        return source_digest(self.sources[self.path.name])

    def validate_snapshot(self) -> None:
        for name, content in self.sources.items():
            if _read_source(self.root, name) != content:
                raise OperatorSourceError(f"operator source changed during planning: {self.root / name}")
        current = discover_operator_dependencies(self.path, self.dependencies)
        if current != self.dependencies:
            raise OperatorSourceError("operator helper dependencies changed during planning")

    def _package(self, package: str) -> str:
        if package == self.namespace:
            return ""
        return package.removeprefix(f"{self.namespace}.")

    def load_main(self) -> ModuleType:
        return self._load("<main>", (self.path, None))

    def _load(self, name: str, location=None) -> ModuleType:
        if name in self.modules:
            return self.modules[name]
        location = location if location is not None else _local_module(self.root, name)
        if location is None:
            raise ModuleNotFoundError(f"no local operator helper {name!r}", name=name)
        source, package_path = location
        parent_name, _, child_name = name.rpartition(".")
        parent = self._load(parent_name) if parent_name else sys.modules[self.namespace]
        if name in self.modules:
            return self.modules[name]
        private_name = f"{self.namespace}.{name}"
        module = ModuleType(private_name)
        module.__package__ = private_name if package_path else parent.__name__
        module.__loader__ = self
        module.__spec__ = importlib.util.spec_from_loader(private_name, loader=None,
                                                        is_package=package_path is not None)
        if package_path:
            module.__path__ = [str(package_path)]
        self.modules[name] = module
        sys.modules[private_name] = module
        setattr(parent, child_name, module)
        if source is not None:
            try:
                relative = source.relative_to(self.root).as_posix()
                if relative not in self.sources:
                    for source_name, content in _collect_sources(self.path, (relative,)).items():
                        self.sources.setdefault(source_name, content)
                module.__file__ = str(source)
                module.__builtins__ = {**vars(builtins), "__import__": self._import}
                exec(compile(self.sources[relative], str(source), "exec"), module.__dict__)
            except BaseException:
                self.modules.pop(name, None)
                sys.modules.pop(private_name, None)
                if getattr(parent, child_name, None) is module:
                    delattr(parent, child_name)
                raise
        return module

    def _import_module(self, name: str, package: str | None = None):
        if name.startswith("."):
            if not package:
                raise TypeError("relative import requires a package")
            name = importlib.util.resolve_name(name, package)
        local_name = name.removeprefix(f"{self.namespace}.")
        if _local_module(self.root, local_name.split(".")[0]) is not None:
            return self._load(local_name)
        return importlib.import_module(name, package)

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        package = self._package((globals or {}).get("__package__", ""))
        target = _relative_name(name, package, level)
        is_local = _local_module(self.root, target.split(".")[0]) is not None
        if target == "importlib" and not is_local:
            return self.importlib
        if not target and level:
            module = self._load(package) if package else sys.modules[self.namespace]
        elif is_local:
            module = self._load(target)
        else:
            imported = builtins.__import__(name, globals, locals, fromlist, level)
            if target.startswith("importlib.") and not fromlist:
                self.importlib.__dict__.update(importlib.__dict__)
                self.importlib.import_module = self._import_module
                return self.importlib
            return imported
        if fromlist:
            attributes = tuple(fromlist)
            if "*" in fromlist:
                attributes += tuple(getattr(module, "__all__", ()))
            for attribute in attributes:
                child = ".".join(part for part in (target, attribute) if part)
                if attribute != "*" and not hasattr(module, attribute) and _local_module(self.root, child):
                    self._load(child)
            return module
        return self._load(target.split(".")[0])
