"""Deterministic analytical cost reporting for compiled ConCorD plans.

The report in this module is derived entirely from the already-selected
compiler, scheduler, and physical-planner records. It performs no timing,
hardware probing, tensor materialization, or alternative cost estimation.
Consequently it is suitable for deterministic regression baselines, while
wall-clock and allocator measurements remain a separate profiling concern.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

from concord_lqcd.backend.planner.physical import PhysicalExecutionPlan
from concord_lqcd.compiler.contraction_order import GlobalContractionOrderPlan
from concord_lqcd.compiler.global_cse import GlobalComputationDAG
from concord_lqcd.compiler.logical_execution import LogicalExecutionDAG
from concord_lqcd.compiler.logical_scheduling import LogicalSchedule
from concord_lqcd.compiler.partial_cse import GlobalIntermediateDAG
from concord_lqcd.compiler.static_analysis import StaticExecutionAnalysis


COMPILATION_COST_REPORT_VERSION = 1


class CompilationPlanView(Protocol):
    """Structural interface shared by runtime compilations and saved plans."""

    global_cse: GlobalComputationDAG
    contraction_order: GlobalContractionOrderPlan
    intermediate_dag: GlobalIntermediateDAG
    logical_dag: LogicalExecutionDAG
    static_analysis: StaticExecutionAnalysis
    schedule: LogicalSchedule
    physical_plan: PhysicalExecutionPlan


@dataclass(frozen=True)
class CompilationCostReport:
    """Comparable analytical costs for one complete compiled execution plan.

    ``input_wick_terms`` is available while compiling from PhysicsIR. It may be
    ``None`` for an older saved plan whose diagnostics predate this report.
    Every other field is reconstructed from the serialized execution plan.
    """

    schema_version: int
    correlators: int
    mode_size: int
    spin_size: int
    scalar_type: str
    element_bytes: int
    contraction_strategy: str
    schedule_strategy: str
    buffer_reuse_strategy: str
    input_wick_terms: int | None
    surviving_wick_terms: int
    resource_occurrences: int
    unique_resources: int
    component_occurrences: int
    unique_components: int
    contraction_step_occurrences: int
    unique_contraction_steps: int
    baseline_contraction_flops: int
    unique_contraction_flops: int
    scheduled_contraction_flops: int
    largest_intermediate_elements: int
    logical_values: int
    scheduled_operations: int
    critical_path_nodes: int
    max_ready_width: int
    total_materialized_value_bytes: int
    largest_value_bytes: int
    logical_peak_live_bytes: int
    physical_peak_active_bytes: int
    total_buffer_capacity_bytes: int
    final_active_capacity_bytes: int
    requested_output_bytes: int
    unique_buffers: int
    buffer_reuses: int

    @property
    def wick_terms_eliminated(self) -> int | None:
        if self.input_wick_terms is None:
            return None
        return self.input_wick_terms - self.surviving_wick_terms

    @property
    def resource_occurrences_eliminated(self) -> int:
        return self.resource_occurrences - self.unique_resources

    @property
    def component_occurrences_eliminated(self) -> int:
        return self.component_occurrences - self.unique_components

    @property
    def contraction_steps_eliminated(self) -> int:
        return self.contraction_step_occurrences - self.unique_contraction_steps

    @property
    def estimated_flops_saved(self) -> int:
        return self.baseline_contraction_flops - self.unique_contraction_flops

    @property
    def dead_contraction_flops_skipped(self) -> int:
        return self.unique_contraction_flops - self.scheduled_contraction_flops

    def validate(self) -> None:
        if self.schema_version != COMPILATION_COST_REPORT_VERSION:
            raise ValueError(
                f"Unsupported compilation cost report version {self.schema_version}."
            )

        text_fields = {
            "scalar_type",
            "contraction_strategy",
            "schedule_strategy",
            "buffer_reuse_strategy",
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in text_fields:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{item.name} must be a non-empty string.")
                continue
            if item.name == "input_wick_terms" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{item.name} must be a non-negative integer.")

        if (
            self.input_wick_terms is not None
            and self.surviving_wick_terms > self.input_wick_terms
        ):
            raise ValueError("Surviving Wick terms exceed the input Wick-term count.")
        if self.unique_resources > self.resource_occurrences:
            raise ValueError("Unique resources exceed resource occurrences.")
        if self.unique_components > self.component_occurrences:
            raise ValueError("Unique components exceed component occurrences.")
        if self.unique_contraction_steps > self.contraction_step_occurrences:
            raise ValueError("Unique contraction steps exceed step occurrences.")
        if self.unique_contraction_flops > self.baseline_contraction_flops:
            raise ValueError("Unique contraction FLOPs exceed the pre-CSE baseline.")
        if self.scheduled_contraction_flops > self.unique_contraction_flops:
            raise ValueError("Scheduled contraction FLOPs exceed unique plan FLOPs.")
        if self.logical_peak_live_bytes > self.total_materialized_value_bytes:
            raise ValueError("Logical peak bytes exceed all materialized value bytes.")
        if self.final_active_capacity_bytes > self.physical_peak_active_bytes:
            raise ValueError("Final active capacity exceeds physical peak capacity.")
        if self.physical_peak_active_bytes > self.total_buffer_capacity_bytes:
            raise ValueError("Physical peak capacity exceeds allocated buffer capacity.")
        if self.requested_output_bytes > self.final_active_capacity_bytes:
            raise ValueError("Requested outputs exceed final active buffer capacity.")
        if self.unique_buffers > self.scheduled_operations:
            raise ValueError("Unique buffers exceed scheduled operations.")
        if self.buffer_reuses != self.scheduled_operations - self.unique_buffers:
            raise ValueError("Buffer reuse count disagrees with scheduled operations.")


def build_compilation_cost_report(
    plan: CompilationPlanView,
    *,
    input_wick_terms: int | None = None,
) -> CompilationCostReport:
    """Derive one validated cost report without changing the compiled plan."""

    global_stats = plan.global_cse.statistics
    partial_stats = plan.intermediate_dag.statistics
    static_stats = plan.static_analysis.statistics
    schedule_stats = plan.schedule.statistics
    physical_stats = plan.physical_plan.statistics

    report = CompilationCostReport(
        schema_version=COMPILATION_COST_REPORT_VERSION,
        correlators=len(plan.global_cse.correlators),
        mode_size=plan.contraction_order.size_model.mode_size,
        spin_size=plan.contraction_order.size_model.spin_size,
        scalar_type=plan.static_analysis.numerical_model.scalar_type,
        element_bytes=plan.static_analysis.numerical_model.element_bytes,
        contraction_strategy=plan.contraction_order.strategy,
        schedule_strategy=plan.schedule.strategy,
        buffer_reuse_strategy=plan.physical_plan.reuse_strategy,
        input_wick_terms=input_wick_terms,
        surviving_wick_terms=sum(
            len(correlator.terms) for correlator in plan.global_cse.correlators
        ),
        resource_occurrences=global_stats.resource_occurrences,
        unique_resources=global_stats.unique_resources,
        component_occurrences=global_stats.component_occurrences,
        unique_components=global_stats.unique_components,
        contraction_step_occurrences=partial_stats.intermediate_occurrences,
        unique_contraction_steps=partial_stats.unique_intermediates,
        baseline_contraction_flops=partial_stats.baseline_step_flops,
        unique_contraction_flops=partial_stats.unique_step_flops,
        scheduled_contraction_flops=schedule_stats.total_estimated_contraction_flops,
        largest_intermediate_elements=(
            plan.contraction_order.largest_intermediate_elements
        ),
        logical_values=static_stats.values,
        scheduled_operations=physical_stats.scheduled_operations,
        critical_path_nodes=static_stats.critical_path_nodes,
        max_ready_width=schedule_stats.max_ready_width,
        total_materialized_value_bytes=static_stats.total_materialized_bytes,
        largest_value_bytes=static_stats.largest_value_bytes,
        logical_peak_live_bytes=schedule_stats.peak_live_bytes,
        physical_peak_active_bytes=physical_stats.peak_active_capacity_bytes,
        total_buffer_capacity_bytes=physical_stats.total_buffer_capacity_bytes,
        final_active_capacity_bytes=physical_stats.final_active_capacity_bytes,
        requested_output_bytes=physical_stats.requested_output_bytes,
        unique_buffers=physical_stats.unique_buffers,
        buffer_reuses=physical_stats.buffer_reuses,
    )
    report.validate()
    return report


__all__ = [
    "COMPILATION_COST_REPORT_VERSION",
    "CompilationCostReport",
    "CompilationPlanView",
    "build_compilation_cost_report",
]
