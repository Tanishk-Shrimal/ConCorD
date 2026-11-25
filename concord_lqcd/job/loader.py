"""Loading helpers for the V1 ConCorD user-facing job specification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import yaml
from pydantic import ValidationError

from concord_lqcd.core.correlator import CorrMatrix, Correlator

from .operator_sources import OperatorSourceError, OperatorSourceSession
from .schema import (
    CustomEigenvectorSpec,
    DataSpec,
    GeneralizedPropagationSpec,
    HDF5EigenvectorSpec,
    JobSpec,
    KernelCacheSpec,
    LaphIOMapGaugeSpec,
    OpenQCDGaugeSpec,
    OperatorSpec,
    OrdinaryPropagationSpec,
    OutputSpec,
    PlanSpec,
    PropagationSpec,
)


OperatorCollection: TypeAlias = Correlator | CorrMatrix | Mapping[str, Correlator]


class JobLoadError(ValueError):
    """Raised when a job YAML file cannot be parsed as a valid V1 specification."""


class OperatorLoadError(RuntimeError):
    """Raised when the configured symbolic operator module/factory cannot be loaded."""


def _resolve_path(path: Path, base_dir: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve(strict=False)


def _resolve_template_path(template: str, base_dir: Path) -> str:
    expanded = Path(template).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return str(expanded.resolve(strict=False))


def _resolve_import_reference(reference: str, base_dir: Path) -> str:
    target, attribute = reference.rsplit(":", 1)
    candidate = Path(target).expanduser()
    looks_like_file = (
        candidate.suffix.lower() == ".py"
        or candidate.is_absolute()
        or target.startswith(".")
    )
    if not looks_like_file:
        return reference
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return f"{candidate.resolve(strict=False)}:{attribute}"


def _resolve_job_paths(spec: JobSpec, base_dir: Path) -> JobSpec:
    operators = spec.operators.model_copy(
        update={"file": _resolve_path(spec.operators.file, base_dir)}
    )
    plan = PlanSpec(file=_resolve_path(spec.plan.file, base_dir))
    output = OutputSpec(
        file=_resolve_path(spec.output.file, base_dir),
        on_existing=spec.output.on_existing,
    )

    eigenvectors = spec.data.eigenvectors
    if isinstance(eigenvectors, HDF5EigenvectorSpec):
        eigenvectors = eigenvectors.model_copy(
            update={"path": _resolve_template_path(eigenvectors.path, base_dir)}
        )
    elif isinstance(eigenvectors, CustomEigenvectorSpec):
        eigenvectors = eigenvectors.model_copy(
            update={
                "factory": _resolve_import_reference(eigenvectors.factory, base_dir)
            }
        )

    ordinary = spec.data.propagation.ordinary
    ordinary_updates: dict[str, object] = {}
    if ordinary.path is not None:
        ordinary_updates["path"] = _resolve_template_path(ordinary.path, base_dir)
    if ordinary.files is not None:
        ordinary_updates["files"] = {
            flavor: _resolve_template_path(template, base_dir)
            for flavor, template in ordinary.files.items()
        }
    ordinary = ordinary.model_copy(update=ordinary_updates)

    generalized = spec.data.propagation.generalized
    if isinstance(generalized, GeneralizedPropagationSpec):
        generalized = generalized.model_copy(
            update={
                "resolver": _resolve_import_reference(generalized.resolver, base_dir)
            }
        )

    propagation = PropagationSpec(
        provider=spec.data.propagation.provider,
        ordinary=OrdinaryPropagationSpec.model_validate(ordinary.model_dump()),
        generalized=(
            GeneralizedPropagationSpec.model_validate(generalized.model_dump())
            if generalized is not None
            else None
        ),
    )
    gauge_links = spec.data.gauge_links
    if isinstance(gauge_links, (LaphIOMapGaugeSpec, OpenQCDGaugeSpec)):
        gauge_links = gauge_links.model_copy(
            update={"path": _resolve_template_path(gauge_links.path, base_dir)}
        )
    kernel_cache = spec.data.kernel_cache
    if isinstance(kernel_cache, KernelCacheSpec):
        kernel_cache = kernel_cache.model_copy(
            update={"directory": _resolve_path(kernel_cache.directory, base_dir)}
        )

    data = DataSpec(
        eigenvectors=eigenvectors,
        propagation=propagation,
        gauge_links=gauge_links,
        kernel_cache=kernel_cache,
    )

    resolved = spec.model_copy(
        update={
            "operators": operators,
            "plan": plan,
            "data": data,
            "output": output,
        }
    )
    # Revalidate the complete object after path normalization so future schema
    # changes cannot accidentally bypass cross-field validation here.
    return JobSpec.model_validate(resolved.model_dump(mode="python"))


def load_job_spec(path: str | Path) -> JobSpec:
    """Parse and validate one V1 ``job.yaml``.

    Relative filesystem paths are resolved relative to the YAML file, never the
    caller's current working directory.  HDF5 dataset names are not filesystem
    paths and are therefore left untouched.
    """

    job_path = Path(path).expanduser().resolve(strict=False)
    if not job_path.exists():
        raise JobLoadError(f"job YAML file does not exist: {job_path}")
    if not job_path.is_file():
        raise JobLoadError(f"job YAML path is not a file: {job_path}")

    try:
        payload = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise JobLoadError(f"failed to read job YAML {job_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise JobLoadError(f"invalid YAML in {job_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise JobLoadError("job YAML root must be a mapping/object")

    try:
        spec = JobSpec.model_validate(payload)
    except ValidationError as exc:
        raise JobLoadError(f"invalid ConCorD job specification:\n{exc}") from exc

    return _resolve_job_paths(spec, job_path.parent)


def _validate_operator_collection(value: object) -> OperatorCollection:
    if isinstance(value, (Correlator, CorrMatrix)):
        return value

    if isinstance(value, Mapping):
        if not value:
            raise OperatorLoadError("operator factory returned an empty correlator mapping")
        normalized: dict[str, Correlator] = {}
        for correlator_id, correlator in value.items():
            if not isinstance(correlator_id, str) or not correlator_id.strip():
                raise OperatorLoadError(
                    "operator factory mapping keys must be non-empty strings"
                )
            if not isinstance(correlator, Correlator):
                raise OperatorLoadError(
                    f"operator factory mapping entry {correlator_id!r} is not a Correlator"
                )
            normalized[correlator_id] = correlator
        return normalized

    raise OperatorLoadError(
        "operator factory must return Correlator, CorrMatrix, or Mapping[str, Correlator]"
    )


@dataclass(frozen=True)
class _LoadedOperators:
    collection: OperatorCollection
    source_sha256: str
    dependencies: Mapping[str, str]


def _load_operator_snapshot(spec: JobSpec | OperatorSpec) -> _LoadedOperators:
    operator_spec = spec.operators if isinstance(spec, JobSpec) else spec
    path = operator_spec.file.expanduser().resolve(strict=False)
    try:
        with OperatorSourceSession(path) as session:
            module = session.load_main()
            factory = getattr(module, operator_spec.factory, None)
            if factory is None:
                raise OperatorLoadError(
                    f"operator file {path} has no factory {operator_spec.factory!r}"
                )
            if not callable(factory):
                raise OperatorLoadError(f"operator factory {operator_spec.factory!r} is not callable")
            value = factory(**dict(operator_spec.parameters))
            collection = _validate_operator_collection(value)
            session.validate_snapshot()
            return _LoadedOperators(collection, session.source_sha256, session.dependencies)
    except OperatorLoadError:
        raise
    except OperatorSourceError as exc:
        raise OperatorLoadError(str(exc)) from exc
    except Exception as exc:
        raise OperatorLoadError(
            f"failed to load operator file {path} or factory {operator_spec.factory!r}: {exc}"
        ) from exc


def load_operator_collection(spec: JobSpec | OperatorSpec) -> OperatorCollection:
    """Execute the factory once with fresh, isolated local Python helpers."""
    return _load_operator_snapshot(spec).collection


__all__ = [
    "OperatorCollection",
    "JobLoadError",
    "OperatorLoadError",
    "load_job_spec",
    "load_operator_collection",
]
