"""End-to-end orchestration for the new ConCorD compiler/runtime pipeline.

This module wires together the already-frozen compiler, planning, resource-
binding, primitive-materialization, and numerical-execution stages.  It adds no
new algebra and no alternate execution semantics.

The public boundary is deliberately split into two phases::

    symbolic correlators
        -> compile_correlators(...)
        -> RuntimeCompilation

    RuntimeCompilation + concrete runtime resources
        -> execute_compilation(...)
        -> RuntimePipelineResult

``run_correlators`` is the convenience composition of those two phases.  The
split is important because exact Stage-5 propagation resource ids only exist
after compilation; applications may inspect ``required_propagation_resources``
before binding datasets.

This is the sole compiler-to-runtime orchestration path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence, TypeAlias

import torch

from concord_lqcd.backend.planner.physical import (
    BufferReuseStrategy,
    MaterializePrimitiveOperation,
    PhysicalExecutionPlan,
    _online_buffer_capacity_bytes,
    build_physical_execution_plan,
)
from concord_lqcd.compiler import (
    ContractionSizeModel,
    GlobalComputationDAG,
    GlobalContractionOrderPlan,
    GlobalIntermediateDAG,
    LogicalExecutionDAG,
    LogicalSchedule,
    PhysicsIR,
    ScheduleStrategy,
    StaticExecutionAnalysis,
    StaticNumericalModel,
    analyze_static_execution,
    build_logical_execution_dag,
    build_logical_schedule,
    build_partial_intermediate_cse,
    lower_correlator_to_physics_ir,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.propagation_lowering import PropagationRecipe
from concord_lqcd.compiler.numerical_flavors import normalize_numerical_flavor_groups
from concord_lqcd.compiler.wick_aggregation import (
    WickAggregationConfig, aggregate_certified_wick_terms, validate_wick_aggregation,
)
from concord_lqcd.compiler.wick_cache import WickCache
from concord_lqcd.core.correlator import CorrMatrix, Correlator

from .materialization import (
    KernelMaterializationProvider,
    PrimitiveMaterializer,
    PropagationSourceHandler,
)
from .metrics import CompilationCostReport, build_compilation_cost_report
from .numerical_executor import GenericNumericalExecutor, NumericalExecutionResult
from .resource_binding import (
    ExternalTensorSource,
    KernelResourceBinding,
    OperatorTimeContext,
    PropagationResourceBinding,
    ResourceBindingPlan,
    build_resource_binding_plan,
)


CorrelatorCollection: TypeAlias = Correlator | CorrMatrix | Mapping[str, Correlator]
PropagationSourceResolver: TypeAlias = Callable[[str, PropagationRecipe], ExternalTensorSource]


_REUSE_AWARE_CANDIDATE_STRATEGIES = (
    "greedy",
    "dynamic-programming",
    "branch-1",
    "branch-2",
)
_REUSE_AWARE_EXACT_SEARCH_MAX_INPUTS = 12


@dataclass(frozen=True)
class RuntimeCompilationConfig:
    """Backend-independent and abstract-physical choices for one compilation.

    ``contraction_memory_limit`` is forwarded to opt_einsum's contraction-order
    optimizer and is intentionally distinct from
    ``physical_memory_limit_bytes``, which constrains the reusable physical
    value-buffer pool.  ``contraction_strategy="reuse-aware"`` evaluates a
    bounded deterministic candidate set after partial CSE, scheduling, and
    buffer reuse. Without a physical limit, the greedy plan's schedule-order
    buffer pool is the ceiling, before lifetime packing; with one, that byte
    limit is a hard ceiling. Improved packing does not tighten path admission.
    ``contraction_strategy="adaptive"`` freezes a portable baseline under an
    adaptive meta-strategy.  A later runtime layer may derive and time
    equivalent candidates from the serialized global DAG without changing
    symbolic or physics identity.

    ``shared_orthonormal_laph_basis=True`` declares that every flavor uses the
    same orthonormal LapH modes at each insertion, with matching perambulator
    coordinates. Only this contract permits replacing a plain local
    zero-momentum ``V^dagger V`` kernel by the mode identity. Custom data retain
    explicit overlap kernels by default; this declaration is not a numerical
    orthonormality check.
    """

    mode_size: int
    spin_size: int = 4
    contraction_strategy: str = "greedy"
    contraction_memory_limit: int | None = None
    scalar_type: str = "complex128"
    element_bytes: int = 16
    schedule_strategy: ScheduleStrategy = "memory_aware"
    include_dead: bool = False
    memory_space: str = "default"
    reuse_strategy: BufferReuseStrategy = "best_fit"
    physical_memory_limit_bytes: int | None = None
    spin_basis: str = "DR"
    shared_orthonormal_laph_basis: bool = False
    numerical_flavor_groups: tuple[tuple[str, ...], ...] = ()
    wick_aggregation: WickAggregationConfig = field(default_factory=WickAggregationConfig)

    def __post_init__(self) -> None:
        if type(self.shared_orthonormal_laph_basis) is not bool:
            raise TypeError("shared_orthonormal_laph_basis must be a boolean.")
        object.__setattr__(self, "numerical_flavor_groups",
                           normalize_numerical_flavor_groups(self.numerical_flavor_groups))
        if not isinstance(self.wick_aggregation, WickAggregationConfig):
            raise TypeError("wick_aggregation must be WickAggregationConfig.")
        numerical_model = StaticNumericalModel(
            self.mode_size, self.spin_size, self.scalar_type, self.element_bytes,
        )
        object.__setattr__(self, "mode_size", numerical_model.mode_size)
        object.__setattr__(self, "spin_size", numerical_model.spin_size)
        object.__setattr__(self, "element_bytes", numerical_model.element_bytes)
        if self.spin_basis not in ("DR", "DP"):
            raise ValueError("spin_basis must be 'DR' or 'DP'.")
        if not self.contraction_strategy:
            raise ValueError("contraction_strategy must be non-empty.")
        if self.contraction_memory_limit is not None and self.contraction_memory_limit <= 0:
            raise ValueError("contraction_memory_limit must be positive when provided.")
        if not self.memory_space:
            raise ValueError("memory_space must be non-empty.")
        if self.reuse_strategy not in ("best_fit", "size_matched", "none"):
            raise ValueError(f"Unsupported buffer reuse strategy {self.reuse_strategy!r}.")
        if (
            self.physical_memory_limit_bytes is not None
            and self.physical_memory_limit_bytes <= 0
        ):
            raise ValueError("physical_memory_limit_bytes must be positive when provided.")


@dataclass(frozen=True)
class RuntimeCompilation:
    """Inspectable compilation result spanning every frozen pre-runtime stage."""

    correlators: Mapping[str, Correlator]
    physics_ir: Mapping[str, PhysicsIR]
    global_cse: GlobalComputationDAG
    contraction_order: GlobalContractionOrderPlan
    intermediate_dag: GlobalIntermediateDAG
    logical_dag: LogicalExecutionDAG
    static_analysis: StaticExecutionAnalysis
    schedule: LogicalSchedule
    physical_plan: PhysicalExecutionPlan
    config: RuntimeCompilationConfig
    wick_aggregation_diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def correlator_ids(self) -> tuple[str, ...]:
        return tuple(self.correlators)

    @property
    def cost_report(self) -> CompilationCostReport:
        """Deterministic analytical costs for this selected execution plan."""

        return build_compilation_cost_report(
            self,
            input_wick_terms=sum(
                len(physics_ir.correlators)
                for physics_ir in self.physics_ir.values()
            ),
        )

    @property
    def required_propagation_resources(self) -> Mapping[str, PropagationRecipe]:
        """Exact scheduled propagation resource ids and their Stage-5 recipes."""

        ids = tuple(dict.fromkeys(
            step.operation.resource_id
            for step in self.physical_plan.steps
            if isinstance(step.operation, MaterializePrimitiveOperation)
            and step.operation.kind == "propagation"
        ))
        resources: dict[str, PropagationRecipe] = {}
        for resource_id in ids:
            try:
                resource = self.global_cse.resources[resource_id]
            except KeyError as exc:
                raise ValueError(
                    f"Scheduled propagation resource {resource_id!r} is absent from global CSE."
                ) from exc
            if resource.kind != "propagation" or not isinstance(
                resource.payload, PropagationRecipe
            ):
                raise TypeError(
                    f"Scheduled propagation resource {resource_id!r} has an invalid payload."
                )
            if resource.payload.resource_id != resource_id:
                raise ValueError(
                    f"Propagation recipe identity for {resource_id!r} is inconsistent."
                )
            resources[resource_id] = resource.payload
        return dict(resources)

    @property
    def required_kernel_resource_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            step.operation.resource_id
            for step in self.physical_plan.steps
            if isinstance(step.operation, MaterializePrimitiveOperation)
            and step.operation.kind == "kernel"
        ))

    def validate(self) -> None:
        if not self.correlators:
            raise ValueError("Runtime compilation contains no correlators.")
        if tuple(self.correlators) != tuple(self.physics_ir):
            raise ValueError("Runtime compilation correlator and PhysicsIR keys disagree.")
        validate_wick_aggregation(
            self.global_cse, config=self.config.wick_aggregation,
            spin_basis=self.config.spin_basis,
            shared_orthonormal_laph_basis=self.config.shared_orthonormal_laph_basis,
            numerical_flavor_groups=self.config.numerical_flavor_groups,
        )
        if self.contraction_order.source_dag is not self.global_cse:
            raise ValueError("Contraction-order plan is not attached to this global CSE DAG.")
        if self.intermediate_dag.source_plan is not self.contraction_order:
            raise ValueError("Partial-CSE DAG is not attached to this contraction-order plan.")
        if self.logical_dag.source_dag is not self.intermediate_dag:
            raise ValueError("Logical execution DAG is not attached to this intermediate DAG.")
        if self.static_analysis.source_dag is not self.logical_dag:
            raise ValueError("Static analysis is not attached to this logical execution DAG.")
        if self.schedule.source_analysis is not self.static_analysis:
            raise ValueError("Logical schedule is not attached to this static analysis.")
        if self.physical_plan.source_schedule is not self.schedule:
            raise ValueError("Physical plan is not attached to this logical schedule.")

        expected_size_model = ContractionSizeModel(
            mode_size=self.config.mode_size,
            spin_size=self.config.spin_size,
        )
        if self.contraction_order.size_model != expected_size_model:
            raise ValueError("Runtime compilation contraction size model disagrees with config.")
        if self.contraction_order.strategy != self.config.contraction_strategy:
            raise ValueError("Runtime compilation contraction strategy disagrees with config.")
        if self.contraction_order.memory_limit != self.config.contraction_memory_limit:
            raise ValueError("Runtime compilation contraction memory limit disagrees with config.")
        if self.static_analysis.numerical_model.scalar_type != self.config.scalar_type:
            raise ValueError("Runtime compilation scalar type disagrees with config.")
        if self.static_analysis.numerical_model.element_bytes != self.config.element_bytes:
            raise ValueError("Runtime compilation element byte size disagrees with config.")
        if self.schedule.strategy != self.config.schedule_strategy:
            raise ValueError("Runtime compilation schedule strategy disagrees with config.")
        if self.schedule.include_dead != self.config.include_dead:
            raise ValueError("Runtime compilation dead-node policy disagrees with config.")
        if self.physical_plan.memory_space != self.config.memory_space:
            raise ValueError("Runtime compilation physical memory space disagrees with config.")
        if self.physical_plan.reuse_strategy != self.config.reuse_strategy:
            raise ValueError("Runtime compilation buffer reuse policy disagrees with config.")

        self.logical_dag.validate()
        self.schedule.validate()
        self.physical_plan.validate()
        # Traverse this property as part of validation so malformed propagation
        # resource payloads fail before runtime binding.
        _ = self.required_propagation_resources


@dataclass(frozen=True)
class RuntimeExecutionConfig:
    """Concrete resource/materialization choices for one compiled pipeline."""

    operator_times: OperatorTimeContext | Mapping[int, Sequence[int]]
    propagation_sources: Mapping[str, ExternalTensorSource] = field(default_factory=dict)
    propagation_source_resolver: PropagationSourceResolver | None = None
    kernel_provider_id: str = "distillation"
    kernel_provider_ids: Mapping[str, str] = field(default_factory=dict)
    kernel_providers: Mapping[str, KernelMaterializationProvider] = field(default_factory=dict)
    source_handlers: Mapping[str, PropagationSourceHandler] = field(default_factory=dict)
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if not self.kernel_provider_id:
            raise ValueError("kernel_provider_id must be non-empty.")
        object.__setattr__(self, "propagation_sources", dict(self.propagation_sources))
        object.__setattr__(self, "kernel_provider_ids", dict(self.kernel_provider_ids))
        object.__setattr__(self, "kernel_providers", dict(self.kernel_providers))
        object.__setattr__(self, "source_handlers", dict(self.source_handlers))


@dataclass(frozen=True)
class RuntimePipelineResult:
    """Compilation, concrete resource binding, and numerical outputs together."""

    compilation: RuntimeCompilation
    binding_plan: ResourceBindingPlan
    numerical_result: NumericalExecutionResult

    @property
    def outputs(self) -> Mapping[str, torch.Tensor]:
        return self.numerical_result.outputs

    def output(self, correlator_id: str) -> torch.Tensor:
        return self.numerical_result.output(correlator_id)

    def validate(self) -> None:
        self.compilation.validate()
        self.binding_plan.validate()
        self.numerical_result.validate()
        if self.binding_plan.source_plan is not self.compilation.physical_plan:
            raise ValueError("Runtime binding plan is attached to a different physical plan.")
        if self.numerical_result.source_plan is not self.compilation.physical_plan:
            raise ValueError("Numerical result is attached to a different physical plan.")


def _normalize_correlators(
    correlators: CorrelatorCollection,
) -> dict[str, Correlator]:
    if isinstance(correlators, Correlator):
        return {"C0": correlators}

    if isinstance(correlators, CorrMatrix):
        return {
            f"C[{i},{j}]": correlator
            for i, j, correlator in correlators
        }

    if not isinstance(correlators, Mapping):
        raise TypeError(
            "correlators must be a Correlator, CorrMatrix, or mapping of ids to Correlator."
        )
    if not correlators:
        raise ValueError("At least one correlator is required.")

    normalized: dict[str, Correlator] = {}
    for correlator_id, correlator in correlators.items():
        if not isinstance(correlator_id, str) or not correlator_id:
            raise ValueError("Correlator ids must be non-empty strings.")
        if not isinstance(correlator, Correlator):
            raise TypeError(
                f"Correlator collection entry {correlator_id!r} is not a Correlator."
            )
        normalized[correlator_id] = correlator
    return normalized


@dataclass(frozen=True)
class RuntimeExecutionStages:
    """One complete lowering of a fixed contraction-order candidate.

    The record is hardware independent.  Later profile-guided policy may
    compare several such candidates, but device discovery and timing never
    enter the symbolic or PhysicsIR stages that produced ``source_dag``.
    """

    contraction_order: GlobalContractionOrderPlan
    intermediate_dag: GlobalIntermediateDAG
    logical_dag: LogicalExecutionDAG
    static_analysis: StaticExecutionAnalysis
    schedule: LogicalSchedule
    physical_plan: PhysicalExecutionPlan


@dataclass(frozen=True)
class RuntimeExecutionCandidate:
    """A named, mathematically equivalent complete execution candidate."""

    strategy: str
    stages: RuntimeExecutionStages

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("execution-candidate strategy must be non-empty")
        if not isinstance(self.stages, RuntimeExecutionStages):
            raise TypeError("execution candidate requires RuntimeExecutionStages")


def _lower_execution_stages(
    contraction_order: GlobalContractionOrderPlan,
    config: RuntimeCompilationConfig,
    *,
    physical_memory_limit_bytes: int | None,
) -> RuntimeExecutionStages:
    """Lower one fixed contraction order through the abstract physical plan."""

    intermediate_dag = build_partial_intermediate_cse(contraction_order)
    logical_dag = build_logical_execution_dag(intermediate_dag)
    static_analysis = analyze_static_execution(
        logical_dag,
        scalar_type=config.scalar_type,
        element_bytes=config.element_bytes,
    )
    schedule = build_logical_schedule(
        static_analysis,
        strategy=config.schedule_strategy,
        include_dead=config.include_dead,
    )
    physical_plan = build_physical_execution_plan(
        schedule,
        memory_space=config.memory_space,
        reuse_strategy=config.reuse_strategy,
        memory_limit_bytes=physical_memory_limit_bytes,
    )
    return RuntimeExecutionStages(
        contraction_order=contraction_order,
        intermediate_dag=intermediate_dag,
        logical_dag=logical_dag,
        static_analysis=static_analysis,
        schedule=schedule,
        physical_plan=physical_plan,
    )


def _reuse_aware_strategies(dag: GlobalComputationDAG) -> tuple[str, ...]:
    """Return the bounded deterministic candidate set for this global DAG."""

    largest_input_count = max(
        (len(component.tensors) for component in dag.components.values()),
        default=0,
    )
    if largest_input_count > _REUSE_AWARE_EXACT_SEARCH_MAX_INPUTS:
        # Dynamic-programming and branch searches are exponential in the number
        # of inputs.  Preserve predictable compilation time for larger networks.
        return ("greedy",)
    return _REUSE_AWARE_CANDIDATE_STRATEGIES


def _select_reuse_aware_execution_stages(
    dag: GlobalComputationDAG,
    size_model: ContractionSizeModel,
    config: RuntimeCompilationConfig,
) -> RuntimeExecutionStages:
    """Select a complete plan after global CSE, scheduling, and buffer reuse.

    With no explicit physical-memory budget, the greedy schedule's online
    buffer-pool capacity is the compatibility ceiling. Lifetime packing may
    improve candidate storage without tightening this ceiling. With a budget,
    every deterministic candidate fitting that hard ceiling is eligible and
    estimated scheduled contraction FLOPs are minimized.
    """

    candidates: list[tuple[str, RuntimeExecutionStages]] = []
    for strategy in _reuse_aware_strategies(dag):
        try:
            order = optimize_global_contraction_order(
                dag,
                size_model,
                strategy=strategy,
                memory_limit=config.contraction_memory_limit,
            )
        except (MemoryError, RuntimeError, ValueError):
            if strategy == "greedy":
                raise
            # Optional optimizers may reject a topology supported by the
            # compatibility path.  Such a rejection must not remove fallback.
            continue
        stages = _lower_execution_stages(
            order,
            config,
            # Candidate requirements must be known exactly before applying
            # the hard user limit and producing a useful failure message.
            physical_memory_limit_bytes=None,
        )
        candidates.append((strategy, stages))

    if not candidates:
        raise RuntimeError("Reuse-aware path selection produced no candidates.")

    baseline = next(
        stages for strategy, stages in candidates if strategy == "greedy"
    )
    if config.physical_memory_limit_bytes is None:
        capacity_limit = (
            _online_buffer_capacity_bytes(
                baseline.schedule, reuse_strategy=config.reuse_strategy
            )
            if config.reuse_strategy == "best_fit"
            else baseline.physical_plan.statistics.total_buffer_capacity_bytes
        )
    else:
        capacity_limit = config.physical_memory_limit_bytes

    feasible = [
        (strategy, stages)
        for strategy, stages in candidates
        if (
            stages.physical_plan.statistics.total_buffer_capacity_bytes
            <= capacity_limit
        )
    ]
    if not feasible:
        minimum_strategy, minimum_stages = min(
            candidates,
            key=lambda item: (
                item[1].physical_plan.statistics.total_buffer_capacity_bytes,
                item[0],
            ),
        )
        minimum_required = (
            minimum_stages.physical_plan.statistics.total_buffer_capacity_bytes
        )
        raise MemoryError(
            "No reuse-aware contraction candidate fits "
            f"physical_memory_limit_bytes={capacity_limit}: minimum required "
            f"{minimum_required} bytes by strategy {minimum_strategy!r}."
        )

    strategy_rank = {
        strategy: index
        for index, strategy in enumerate(_REUSE_AWARE_CANDIDATE_STRATEGIES)
    }
    selected_strategy, selected = min(
        feasible,
        key=lambda item: (
            item[1].schedule.statistics.total_estimated_contraction_flops,
            item[1].physical_plan.statistics.total_buffer_capacity_bytes,
            item[1].physical_plan.statistics.peak_active_capacity_bytes,
            item[1].static_analysis.statistics.critical_path_nodes,
            strategy_rank[item[0]],
        ),
    )

    selected_order = GlobalContractionOrderPlan(
        source_dag=dag,
        size_model=size_model,
        strategy="reuse-aware",
        memory_limit=config.contraction_memory_limit,
        component_plans=selected.contraction_order.component_plans,
    )
    final_stages = _lower_execution_stages(
        selected_order,
        config,
        physical_memory_limit_bytes=config.physical_memory_limit_bytes,
    )
    selected_component_strategies = {
        plan.strategy
        for plan in final_stages.contraction_order.component_plans.values()
    }
    if selected_component_strategies and selected_component_strategies != {
        selected_strategy
    }:
        raise RuntimeError(
            f"Reuse-aware selected strategy {selected_strategy!r} was not preserved."
        )
    return final_stages


def _meta_strategy_stages(
    stages: RuntimeExecutionStages,
    config: RuntimeCompilationConfig,
    *,
    meta_strategy: str,
) -> RuntimeExecutionStages:
    """Relabel and re-lower selected component paths under a meta-strategy.

    Re-lowering preserves the plan's source-identity chain and applies the
    configured hard physical-memory limit to the final candidate.
    """

    order = GlobalContractionOrderPlan(
        source_dag=stages.contraction_order.source_dag,
        size_model=stages.contraction_order.size_model,
        strategy=meta_strategy,
        memory_limit=config.contraction_memory_limit,
        component_plans=stages.contraction_order.component_plans,
    )
    return _lower_execution_stages(
        order,
        config,
        physical_memory_limit_bytes=config.physical_memory_limit_bytes,
    )


def build_adaptive_execution_candidates(
    dag: GlobalComputationDAG,
    size_model: ContractionSizeModel,
    config: RuntimeCompilationConfig,
) -> tuple[RuntimeExecutionCandidate, ...]:
    """Build the bounded portable candidates used by adaptive execution.

    Candidate construction is deterministic and hardware independent.  The
    baseline is the compatibility ``greedy`` path.  The second candidate is
    the existing global ``reuse-aware`` meta-strategy, which has already
    compared its bounded opt_einsum alternatives after partial CSE, scheduling,
    and buffer planning.  Candidates that violate an explicit hard physical
    memory limit are omitted.  Equivalent component-path selections are
    deduplicated before any later numerical calibration.
    """

    if not isinstance(dag, GlobalComputationDAG):
        raise TypeError("dag must be a GlobalComputationDAG")
    if not isinstance(size_model, ContractionSizeModel):
        raise TypeError("size_model must be a ContractionSizeModel")
    if not isinstance(config, RuntimeCompilationConfig):
        raise TypeError("config must be a RuntimeCompilationConfig")

    raw: list[tuple[str, RuntimeExecutionStages]] = []
    greedy_order = optimize_global_contraction_order(
        dag,
        size_model,
        strategy="greedy",
        memory_limit=config.contraction_memory_limit,
    )
    greedy = _lower_execution_stages(
        greedy_order,
        config,
        physical_memory_limit_bytes=None,
    )
    raw.append(("greedy", greedy))

    reuse_aware = _select_reuse_aware_execution_stages(dag, size_model, config)
    raw.append(("reuse-aware", reuse_aware))

    result: list[RuntimeExecutionCandidate] = []
    seen_component_plans: list[Mapping[str, object]] = []
    for strategy, stages in raw:
        capacity = stages.physical_plan.statistics.total_buffer_capacity_bytes
        limit = config.physical_memory_limit_bytes
        if limit is not None and capacity > limit:
            continue
        component_plans = stages.contraction_order.component_plans
        if any(component_plans == previous for previous in seen_component_plans):
            continue
        try:
            adaptive_stages = _meta_strategy_stages(
                stages,
                config,
                meta_strategy="adaptive",
            )
        except MemoryError:
            continue
        result.append(
            RuntimeExecutionCandidate(
                strategy=strategy,
                stages=adaptive_stages,
            )
        )
        seen_component_plans.append(component_plans)

    if not result:
        raise MemoryError(
            "No adaptive contraction candidate satisfies the configured "
            "physical-memory limit."
        )
    return tuple(result)


def compile_correlators(
    correlators: CorrelatorCollection,
    config: RuntimeCompilationConfig,
    *,
    wick_cache: WickCache | None = None,
) -> RuntimeCompilation:
    """Compile symbolic correlators through the complete frozen physical plan.

    A shared ``WickCache`` is used across the collection so equivalent basis
    operator sequences are Wick-contracted only once.  Linear-combination
    correlators are handled by ``lower_correlator_to_physics_ir`` before global
    CSE, preserving the existing frontend semantics.
    """

    if not isinstance(config, RuntimeCompilationConfig):
        raise TypeError("config must be RuntimeCompilationConfig.")
    normalized = _normalize_correlators(correlators)
    cache = wick_cache if wick_cache is not None else WickCache()
    if not isinstance(cache, WickCache):
        raise TypeError("wick_cache must be WickCache when provided.")

    physics_ir = {
        correlator_id: lower_correlator_to_physics_ir(correlator, cache)
        for correlator_id, correlator in normalized.items()
    }
    global_cse = lower_physics_ir_collection_to_global_cse(
        physics_ir,
        mode_size=config.mode_size,
        shared_orthonormal_laph_basis=config.shared_orthonormal_laph_basis,
    )
    global_cse, aggregation_diagnostics = aggregate_certified_wick_terms(
        global_cse, config.wick_aggregation,
        spin_basis=config.spin_basis,
        shared_orthonormal_laph_basis=config.shared_orthonormal_laph_basis,
        numerical_flavor_groups=config.numerical_flavor_groups,
    )
    size_model = ContractionSizeModel(
        mode_size=config.mode_size,
        spin_size=config.spin_size,
    )
    if config.contraction_strategy == "reuse-aware":
        stages = _select_reuse_aware_execution_stages(
            global_cse,
            size_model,
            config,
        )
    elif config.contraction_strategy == "adaptive":
        # The first candidate is the deterministic portable baseline.  Device
        # measurements are deliberately deferred to the runtime/job boundary.
        stages = build_adaptive_execution_candidates(
            global_cse,
            size_model,
            config,
        )[0].stages
    else:
        contraction_order = optimize_global_contraction_order(
            global_cse,
            size_model,
            strategy=config.contraction_strategy,
            memory_limit=config.contraction_memory_limit,
        )
        stages = _lower_execution_stages(
            contraction_order,
            config,
            physical_memory_limit_bytes=config.physical_memory_limit_bytes,
        )

    compilation = RuntimeCompilation(
        correlators=dict(normalized),
        physics_ir=dict(physics_ir),
        global_cse=global_cse,
        contraction_order=stages.contraction_order,
        intermediate_dag=stages.intermediate_dag,
        logical_dag=stages.logical_dag,
        static_analysis=stages.static_analysis,
        schedule=stages.schedule,
        physical_plan=stages.physical_plan,
        config=config,
        wick_aggregation_diagnostics=aggregation_diagnostics,
    )
    compilation.validate()
    return compilation


def _resolve_propagation_sources(
    compilation: RuntimeCompilation,
    config: RuntimeExecutionConfig,
) -> dict[str, ExternalTensorSource]:
    required = compilation.required_propagation_resources
    resolved = dict(config.propagation_sources)

    for resource_id, recipe in required.items():
        if resource_id in resolved:
            source = resolved[resource_id]
        elif config.propagation_source_resolver is not None:
            source = config.propagation_source_resolver(resource_id, recipe)
            resolved[resource_id] = source
        else:
            raise ValueError(
                f"No exact external tensor source was supplied for propagation resource "
                f"{resource_id!r}."
            )
        if not isinstance(source, ExternalTensorSource):
            raise TypeError(
                f"Propagation source for {resource_id!r} must be ExternalTensorSource."
            )

    return resolved


def bind_compilation_resources(
    compilation: RuntimeCompilation,
    config: RuntimeExecutionConfig,
) -> ResourceBindingPlan:
    """Bind one compiled physical plan to concrete runtime resource sources."""

    if not isinstance(compilation, RuntimeCompilation):
        raise TypeError("compilation must be RuntimeCompilation.")
    if not isinstance(config, RuntimeExecutionConfig):
        raise TypeError("config must be RuntimeExecutionConfig.")
    compilation.validate()

    propagation_sources = _resolve_propagation_sources(compilation, config)
    binding_plan = build_resource_binding_plan(
        compilation.physical_plan,
        operator_times=config.operator_times,
        propagation_sources=propagation_sources,
        kernel_provider_id=config.kernel_provider_id,
        kernel_provider_ids=config.kernel_provider_ids,
        spin_basis=compilation.config.spin_basis,
    )

    required_provider_ids = {
        binding.provider_id
        for binding in binding_plan.bindings.values()
        if isinstance(binding, KernelResourceBinding)
    }
    missing_kernel_providers = required_provider_ids - set(config.kernel_providers)
    if missing_kernel_providers:
        raise ValueError(
            "No kernel materialization provider is registered for provider ids "
            f"{sorted(missing_kernel_providers)}."
        )

    builtin_source_kinds = {"hdf5"}
    required_source_kinds = {
        binding.source.kind
        for binding in binding_plan.bindings.values()
        if isinstance(binding, PropagationResourceBinding)
    }
    missing_source_handlers = (
        required_source_kinds - builtin_source_kinds - set(config.source_handlers)
    )
    if missing_source_handlers:
        raise ValueError(
            "No propagation source handler is registered for source kinds "
            f"{sorted(missing_source_handlers)}."
        )

    return binding_plan


def execute_compilation(
    compilation: RuntimeCompilation,
    config: RuntimeExecutionConfig,
) -> RuntimePipelineResult:
    """Bind resources, materialize primitives, and execute one compiled plan."""

    binding_plan = bind_compilation_resources(compilation, config)
    materializer = PrimitiveMaterializer(
        binding_plan,
        kernel_providers=config.kernel_providers,
        source_handlers=config.source_handlers,
    )
    numerical_result = GenericNumericalExecutor(
        materializer,
        device=config.device,
    ).execute()
    result = RuntimePipelineResult(
        compilation=compilation,
        binding_plan=binding_plan,
        numerical_result=numerical_result,
    )
    result.validate()
    return result


def run_correlators(
    correlators: CorrelatorCollection,
    compilation_config: RuntimeCompilationConfig,
    execution_config: RuntimeExecutionConfig,
    *,
    wick_cache: WickCache | None = None,
) -> RuntimePipelineResult:
    """Compile, bind, and numerically execute symbolic correlators end to end."""

    compilation = compile_correlators(
        correlators,
        compilation_config,
        wick_cache=wick_cache,
    )
    return execute_compilation(compilation, execution_config)


__all__ = [
    "CorrelatorCollection",
    "PropagationSourceResolver",
    "RuntimeCompilationConfig",
    "RuntimeCompilation",
    "RuntimeExecutionCandidate",
    "RuntimeExecutionStages",
    "RuntimeExecutionConfig",
    "RuntimePipelineResult",
    "build_adaptive_execution_candidates",
    "compile_correlators",
    "bind_compilation_resources",
    "execute_compilation",
    "run_correlators",
]
