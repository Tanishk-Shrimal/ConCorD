"""Versioned user-facing job specification for ConCorD.

This module is an application-layer schema.  It translates the frozen V1
``job.yaml`` design into strongly validated Python objects without importing or
changing compiler/runtime semantics.

Paths in a parsed :class:`JobSpec` are still exactly the paths supplied by the
user.  :func:`concord_lqcd.job.load_job_spec` resolves filesystem paths relative to
the YAML file before returning the specification.
"""

from __future__ import annotations

from pathlib import Path
from string import Formatter
import re
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from concord_lqcd.compiler.numerical_flavors import normalize_numerical_flavor_groups


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]

ScalarType = Literal["complex64", "complex128"]
SpinBasis = Literal["DR", "DP"]
PropagationSourceField = Literal["qbar", "qdagger"]
ContractionStrategy = Literal[
    "adaptive",
    "auto",
    "greedy",
    "dynamic-programming",
    "auto-hq",
    "reuse-aware",
]
ScheduleStrategy = Literal["memory_aware", "critical_path", "topological"]
BufferReuseStrategy = Literal["best_fit", "size_matched", "none"]
ExistingOutputPolicy = Literal["resume", "error", "overwrite"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntegerRange(_StrictModel):
    """Python-style, stop-exclusive integer range."""

    start: int
    stop: int
    step: int = 1

    @model_validator(mode="after")
    def _validate_range(self) -> "IntegerRange":
        if self.step == 0:
            raise ValueError("range step cannot be zero")
        if not tuple(range(self.start, self.stop, self.step)):
            raise ValueError(
                "range is empty; start/stop/step must describe at least one value"
            )
        return self

    def expand(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop, self.step))


class IntegerSequence(_StrictModel):
    """Either an explicit integer list or a stop-exclusive range, never both."""

    values: tuple[int, ...] | None = None
    range: IntegerRange | None = None

    @model_validator(mode="after")
    def _validate_choice(self) -> "IntegerSequence":
        supplied = int(self.values is not None) + int(self.range is not None)
        if supplied != 1:
            raise ValueError("exactly one of 'values' or 'range' must be supplied")
        if self.values is not None and not self.values:
            raise ValueError("values must contain at least one integer")
        return self

    def expand(self) -> tuple[int, ...]:
        if self.values is not None:
            return tuple(self.values)
        assert self.range is not None
        return self.range.expand()


class JobMetadata(_StrictModel):
    name: NonEmptyStr
    description: NonEmptyStr | None = None


class OperatorSpec(_StrictModel):
    file: Path
    factory: NonEmptyStr = "build"
    parameters: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_file(self) -> "OperatorSpec":
        if self.file.suffix.lower() != ".py":
            raise ValueError("operators.file must point to a .py file")
        object.__setattr__(self, "parameters", dict(self.parameters))
        return self


class PlanSpec(_StrictModel):
    file: Path

    @model_validator(mode="after")
    def _validate_file(self) -> "PlanSpec":
        if self.file.suffix.lower() != ".concord":
            raise ValueError("plan.file must use the .concord extension")
        return self


class LatticeSpec(_StrictModel):
    spatial_extent: tuple[PositiveInt, PositiveInt, PositiveInt]
    temporal_extent: PositiveInt


class ContractionConfig(_StrictModel):
    strategy: ContractionStrategy = "greedy"
    memory_limit: PositiveInt | None = None


class SchedulingConfig(_StrictModel):
    strategy: ScheduleStrategy = "memory_aware"
    include_dead: bool = False


class PhysicalConfig(_StrictModel):
    memory_space: NonEmptyStr = "default"
    reuse_strategy: BufferReuseStrategy = "best_fit"
    memory_limit_bytes: PositiveInt | None = None


class WickAggregationSpec(_StrictModel):
    """Optional exact whole-Wick-term discovery, performed only at plan creation."""

    enabled: bool = Field(default=False, strict=True)
    # Hard abort only: exceeding it fails plan creation and never changes a result.
    search_seconds: float = Field(default=3600.0, gt=0, allow_inf_nan=False)
    maximum_checks: int = Field(default=1_000_000, ge=0, strict=True)
    maximum_match_states: int = Field(default=10000, ge=0, strict=True)


class CompileSpec(_StrictModel):
    mode_size: PositiveInt
    spin_size: PositiveInt = 4
    spin_basis: SpinBasis = "DR"
    scalar_type: ScalarType = "complex128"
    contraction: ContractionConfig = Field(default_factory=ContractionConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    physical: PhysicalConfig = Field(default_factory=PhysicalConfig)
    numerical_flavor_groups: tuple[tuple[str, ...], ...] = ()
    wick_aggregation: WickAggregationSpec = Field(default_factory=WickAggregationSpec)

    @model_validator(mode="after")
    def _validate_numerical_flavor_groups(self) -> "CompileSpec":
        object.__setattr__(self, "numerical_flavor_groups",
                           normalize_numerical_flavor_groups(self.numerical_flavor_groups))
        return self

    @property
    def element_bytes(self) -> int:
        return 8 if self.scalar_type == "complex64" else 16


class ExpandedTimeContext(_StrictModel):
    """Concrete operator-time assignment generated from the user-facing spec."""

    id: NonEmptyStr
    operator_times: Mapping[int, tuple[int, ...]]

    @model_validator(mode="after")
    def _validate_operator_times(self) -> "ExpandedTimeContext":
        normalized: dict[int, tuple[int, ...]] = {}
        if not self.operator_times:
            raise ValueError("operator_times cannot be empty")
        for arity, times in self.operator_times.items():
            if not isinstance(arity, int) or isinstance(arity, bool) or arity <= 0:
                raise ValueError("operator_times keys must be positive integer arities")
            values = tuple(times)
            if len(values) != arity:
                raise ValueError(
                    f"arity {arity} requires {arity} operator times, got {len(values)}"
                )
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                raise ValueError("operator times must be integer lattice timeslices")
            normalized[arity] = values
        object.__setattr__(self, "operator_times", normalized)
        return self


class TwoPointTimeGenerator(_StrictModel):
    type: Literal["two_point"]
    source: NonNegativeInt | None = None
    sources: IntegerSequence | None = None
    sinks: IntegerSequence

    @model_validator(mode="after")
    def _validate_sources(self) -> "TwoPointTimeGenerator":
        supplied = int(self.source is not None) + int(self.sources is not None)
        if supplied != 1:
            raise ValueError(
                "two-point time generation requires exactly one of "
                "'source' or 'sources'"
            )
        return self

    def expand(self) -> tuple[ExpandedTimeContext, ...]:
        if self.source is not None:
            source_times = (self.source,)
        else:
            assert self.sources is not None
            source_times = self.sources.expand()
        return tuple(
            ExpandedTimeContext(
                id=f"2pt_src_{source}_snk_{sink}",
                operator_times={2: (sink, source)},
            )
            for source in source_times
            for sink in self.sinks.expand()
        )


class ThreePointTimeGenerator(_StrictModel):
    type: Literal["three_point"]
    source: NonNegativeInt
    sink: NonNegativeInt
    insertions: IntegerSequence

    def expand(self) -> tuple[ExpandedTimeContext, ...]:
        return tuple(
            ExpandedTimeContext(
                id=f"3pt_src_{self.source}_ins_{insertion}_snk_{self.sink}",
                operator_times={3: (self.sink, insertion, self.source)},
            )
            for insertion in self.insertions.expand()
        )


TimeGenerator = Annotated[
    TwoPointTimeGenerator | ThreePointTimeGenerator,
    Field(discriminator="type"),
]


class ExplicitTimeContext(ExpandedTimeContext):
    pass


class TimeSpec(_StrictModel):
    generator: TimeGenerator | None = None
    contexts: tuple[ExplicitTimeContext, ...] | None = None

    @model_validator(mode="after")
    def _validate_choice(self) -> "TimeSpec":
        supplied = int(self.generator is not None) + int(self.contexts is not None)
        if supplied != 1:
            raise ValueError("exactly one of times.generator or times.contexts is required")
        if self.contexts is not None and not self.contexts:
            raise ValueError("times.contexts must contain at least one context")
        expanded = self.expand()
        ids = tuple(context.id for context in expanded)
        if len(ids) != len(set(ids)):
            raise ValueError("expanded time-context ids must be unique")
        return self

    def expand(self) -> tuple[ExpandedTimeContext, ...]:
        if self.generator is not None:
            return self.generator.expand()
        assert self.contexts is not None
        return tuple(
            ExpandedTimeContext(id=context.id, operator_times=context.operator_times)
            for context in self.contexts
        )


class RunSpec(_StrictModel):
    device: NonEmptyStr = "cpu"
    time_batch_size: Literal["auto"] | PositiveInt = "auto"
    # Worker processes on one CPU node. They share the loaded plan and split the
    # pending work units. "auto" chooses both numbers from the plan, the cores
    # and the free memory; with a fixed worker count, threads_per_worker "auto"
    # means usable CPUs / workers. With one worker, threads_per_worker only sets
    # the PyTorch thread count.
    workers: Literal["auto"] | PositiveInt = "auto"
    threads_per_worker: Literal["auto"] | PositiveInt = "auto"
    configurations: IntegerSequence
    times: TimeSpec

    @model_validator(mode="after")
    def _validate_configurations(self) -> "RunSpec":
        values = self.configurations.expand()
        if any(value < 0 for value in values):
            raise ValueError("configuration ids must be non-negative integers")
        if self.workers != "auto" and self.workers > 1 and self.device.split(":", 1)[0] == "cuda":
            raise ValueError("run.workers > 1 runs on the CPU; use device 'cpu' or 'auto'")
        return self


_FORMATTER = Formatter()
_INSERTION_TIME = re.compile(r"insertion_time_[0-9]+$")


def _template_fields(template: str) -> tuple[str, ...]:
    fields: list[str] = []
    try:
        parsed = tuple(_FORMATTER.parse(template))
    except ValueError as exc:
        raise ValueError(f"invalid template syntax in {template!r}") from exc
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name:
            raise ValueError("empty '{}' placeholders are not supported")
        if format_spec or conversion:
            raise ValueError("template format specifiers/conversions are not supported")
        fields.append(field_name)
    return tuple(fields)


def _validate_template(
    template: str,
    *,
    allowed: set[str],
    allow_insertion_times: bool = False,
    role: str,
) -> str:
    if not template.strip():
        raise ValueError(f"{role} must be non-empty")
    for field_name in _template_fields(template):
        if field_name in allowed:
            continue
        if allow_insertion_times and _INSERTION_TIME.fullmatch(field_name):
            continue
        raise ValueError(f"unsupported placeholder {{{field_name}}} in {role}")
    return template


def _validate_axes(axes: tuple[str, ...], *, role: str) -> tuple[str, ...]:
    if not axes:
        raise ValueError(f"{role} must contain at least one semantic axis")
    if any(not isinstance(axis, str) or not axis.strip() for axis in axes):
        raise ValueError(f"{role} entries must be non-empty strings")
    if len(axes) != len(set(axes)):
        raise ValueError(f"{role} cannot contain duplicate semantic axes")
    return axes


def _validate_import_reference(value: str, *, role: str) -> str:
    text = value.strip()
    if ":" not in text:
        raise ValueError(f"{role} must have the form 'module-or-file:callable'")
    target, attribute = text.rsplit(":", 1)
    if not target.strip() or not attribute.strip():
        raise ValueError(f"{role} must have the form 'module-or-file:callable'")
    return text


class HDF5EigenvectorSpec(_StrictModel):
    """One shared orthonormal LapH basis for all flavors at each time.

    Selecting this adapter declares orthonormal modes and perambulators built
    in those same coordinates. Neither compilation nor reading scans the
    eigenvectors to verify orthonormality. Flavor-specific bases are not a
    supported built-in HDF5 job input.
    """

    provider: Literal["hdf5_distillation"]
    path: NonEmptyStr
    dataset: NonEmptyStr
    axes: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_storage(self) -> "HDF5EigenvectorSpec":
        _validate_template(
            self.path,
            allowed={"configuration", "time"},
            role="data.eigenvectors.path",
        )
        _validate_template(
            self.dataset,
            allowed={"configuration", "time"},
            role="data.eigenvectors.dataset",
        )
        _validate_axes(self.axes, role="data.eigenvectors.axes")
        return self


class CustomEigenvectorSpec(_StrictModel):
    provider: Literal["custom"]
    factory: NonEmptyStr

    @model_validator(mode="after")
    def _validate_factory(self) -> "CustomEigenvectorSpec":
        _validate_import_reference(self.factory, role="data.eigenvectors.factory")
        return self


EigenvectorSpec = Annotated[
    HDF5EigenvectorSpec | CustomEigenvectorSpec,
    Field(discriminator="provider"),
]


class LaphIOMapGaugeSpec(_StrictModel):
    """Native Chroma-LapH or QUDA-LapH smeared-gauge storage."""

    provider: Literal["laph_iomap"]
    path: NonEmptyStr

    @model_validator(mode="after")
    def _validate_storage(self) -> "LaphIOMapGaugeSpec":
        _validate_template(
            self.path,
            allowed={"configuration"},
            role="data.gauge_links.path",
        )
        return self


class OpenQCDGaugeSpec(_StrictModel):
    """Gauge links in the openQCD binary format (as used for CLS configurations).

    The links are used as stored; smearing them as for the eigenvectors is up to the user.
    """

    provider: Literal["openqcd"]
    path: NonEmptyStr

    @model_validator(mode="after")
    def _validate_storage(self) -> "OpenQCDGaugeSpec":
        _validate_template(
            self.path,
            allowed={"configuration"},
            role="data.gauge_links.path",
        )
        return self


GaugeLinkSpec = Annotated[
    LaphIOMapGaugeSpec | OpenQCDGaugeSpec,
    Field(discriminator="provider"),
]


class KernelCacheSpec(_StrictModel):
    """Optional persistent cache for materialized spatial LapH kernels."""

    directory: Path
    packing: Literal["none", "antisym3"] = "none"


_PROPAGATION_DATA_AXES = {
    "sink_mode",
    "sink_spin",
    "source_mode",
    "source_spin",
}
_PROPAGATION_EXTRA_AXES = {
    "configuration",
    "source_time",
    "sink_time",
    "complex_pair",
}


class OrdinaryPropagationSpec(_StrictModel):
    path: NonEmptyStr | None = None
    files: Mapping[str, NonEmptyStr] | None = None
    dataset: NonEmptyStr
    axes: tuple[str, ...]
    fixed_source_time: int | None = None
    source_field: PropagationSourceField = "qbar"

    @model_validator(mode="after")
    def _validate_storage(self) -> "OrdinaryPropagationSpec":
        supplied = int(self.path is not None) + int(self.files is not None)
        if supplied != 1:
            raise ValueError("ordinary propagation requires exactly one of 'path' or 'files'")

        if self.path is not None:
            _validate_template(
                self.path,
                allowed={"configuration", "flavor"},
                role="data.propagation.ordinary.path",
            )

        if self.files is not None:
            if not self.files:
                raise ValueError("ordinary propagation files cannot be empty")
            normalized_files: dict[str, str] = {}
            for flavor, template in self.files.items():
                if not isinstance(flavor, str) or not flavor.strip():
                    raise ValueError("ordinary propagation flavor keys must be non-empty strings")
                _validate_template(
                    template,
                    allowed={"configuration", "flavor"},
                    role=f"data.propagation.ordinary.files[{flavor!r}]",
                )
                normalized_files[flavor] = template
            object.__setattr__(self, "files", normalized_files)

        _validate_template(
            self.dataset,
            allowed={
                "configuration",
                "flavor",
                "source_time",
                "sink_time",
                "fixed_source_time",
            },
            allow_insertion_times=True,
            role="data.propagation.ordinary.dataset",
        )
        if isinstance(self.fixed_source_time, bool):
            raise ValueError("ordinary fixed_source_time must be an integer")
        axes = _validate_axes(self.axes, role="data.propagation.ordinary.axes")
        axis_set = set(axes)
        missing = _PROPAGATION_DATA_AXES - axis_set
        if missing:
            raise ValueError(
                "ordinary propagation axes must expose the four ConCorD data axes; "
                f"missing {sorted(missing)}"
            )
        for axis in axis_set - _PROPAGATION_DATA_AXES - _PROPAGATION_EXTRA_AXES:
            if not _INSERTION_TIME.fullmatch(axis):
                raise ValueError(f"unsupported ordinary propagation axis {axis!r}")
        return self


class GeneralizedPropagationSpec(_StrictModel):
    resolver: NonEmptyStr

    @model_validator(mode="after")
    def _validate_resolver(self) -> "GeneralizedPropagationSpec":
        _validate_import_reference(self.resolver, role="data.propagation.generalized.resolver")
        return self


class PropagationSpec(_StrictModel):
    provider: Literal["hdf5_perambulator"]
    ordinary: OrdinaryPropagationSpec
    generalized: GeneralizedPropagationSpec | None = None


class DataSpec(_StrictModel):
    eigenvectors: EigenvectorSpec
    propagation: PropagationSpec
    gauge_links: GaugeLinkSpec | None = None
    kernel_cache: KernelCacheSpec | None = None

    @model_validator(mode="after")
    def _validate_kernel_cache_provider(self) -> "DataSpec":
        if self.kernel_cache is not None and not isinstance(
            self.eigenvectors,
            HDF5EigenvectorSpec,
        ):
            raise ValueError(
                "data.kernel_cache currently requires HDF5 LapH eigenvectors"
            )
        return self


class OutputSpec(_StrictModel):
    file: Path
    on_existing: ExistingOutputPolicy = "resume"

    @model_validator(mode="after")
    def _validate_file(self) -> "OutputSpec":
        if self.file.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("output.file must use .h5 or .hdf5")
        return self


class VisualizationSpec(_StrictModel):
    """Presentation-only labels for symbolic operator insertions."""

    operator_labels: Mapping[str, tuple[NonEmptyStr, ...]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_operator_labels(self) -> "VisualizationSpec":
        normalized: dict[str, tuple[str, ...]] = {}
        for correlator_id, labels in self.operator_labels.items():
            key = correlator_id.strip()
            if not key:
                raise ValueError("visualization.operator_labels keys cannot be empty")
            values = tuple(labels)
            if not values:
                raise ValueError(
                    f"visualization.operator_labels[{key!r}] cannot be empty"
                )
            normalized[key] = values
        object.__setattr__(self, "operator_labels", normalized)
        return self


class JobSpec(_StrictModel):
    """Complete V1 single-file ConCorD job specification."""

    schema_version: Literal[1]
    job: JobMetadata
    operators: OperatorSpec
    plan: PlanSpec
    lattice: LatticeSpec
    compile: CompileSpec
    run: RunSpec
    data: DataSpec
    output: OutputSpec
    visualization: VisualizationSpec = Field(default_factory=VisualizationSpec)

    @model_validator(mode="after")
    def _validate_time_bounds(self) -> "JobSpec":
        temporal_extent = self.lattice.temporal_extent
        for context in self.run.times.expand():
            for arity, times in context.operator_times.items():
                for operator_id, time in enumerate(times):
                    if time < 0 or time >= temporal_extent:
                        raise ValueError(
                            f"time context {context.id!r}, arity {arity}, operator "
                            f"{operator_id} has timeslice {time}, outside [0, {temporal_extent})"
                        )
        return self

    @property
    def time_contexts(self) -> tuple[ExpandedTimeContext, ...]:
        return self.run.times.expand()

    @property
    def configurations(self) -> tuple[int, ...]:
        return self.run.configurations.expand()


__all__ = [
    "ScalarType",
    "SpinBasis",
    "ContractionStrategy",
    "ScheduleStrategy",
    "BufferReuseStrategy",
    "ExistingOutputPolicy",
    "IntegerRange",
    "IntegerSequence",
    "JobMetadata",
    "OperatorSpec",
    "PlanSpec",
    "LatticeSpec",
    "ContractionConfig",
    "SchedulingConfig",
    "PhysicalConfig",
    "CompileSpec",
    "WickAggregationSpec",
    "ExpandedTimeContext",
    "TwoPointTimeGenerator",
    "ThreePointTimeGenerator",
    "ExplicitTimeContext",
    "TimeSpec",
    "RunSpec",
    "HDF5EigenvectorSpec",
    "CustomEigenvectorSpec",
    "LaphIOMapGaugeSpec",
    "OpenQCDGaugeSpec",
    "GaugeLinkSpec",
    "KernelCacheSpec",
    "OrdinaryPropagationSpec",
    "GeneralizedPropagationSpec",
    "PropagationSpec",
    "DataSpec",
    "OutputSpec",
    "VisualizationSpec",
    "JobSpec",
]
