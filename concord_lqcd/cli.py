"""Command-line interface for the V1 ConCorD job workflow.

The CLI is intentionally a thin application layer over the frozen job APIs:

    concord plan job.yaml
    concord inspect calculation.concord
    concord visualize job.yaml
    concord run job.yaml

It does not implement compiler, resource-binding, materialization, or numerical
execution semantics of its own.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
import typer

from concord_lqcd.backend.runtime.hardware import HardwareProfile, detect_hardware_profile
from concord_lqcd.backend.runtime.resource_planning import (
    DeviceExecutionPlan,
    HardwareBudgetPolicy,
    ResourcePlanningError,
    plan_execution_resources,
    resolve_runtime_device,
)
from concord_lqcd.compiler.distillation_lowering import KernelRecipe
from concord_lqcd.compiler.propagation_lowering import PropagationRecipe
from concord_lqcd.job.data import JobDataError, build_job_runtime_resources
from concord_lqcd.job.loader import (
    JobLoadError,
    OperatorLoadError,
    load_job_spec,
    load_operator_collection,
)
from concord_lqcd.job._prepared import _load_prepared_plan
from concord_lqcd.job.plan import (
    PlanSerializationError,
    StalePlanError,
    ConcordPlan,
    load_concord_plan,
    validate_plan_for_job,
    write_job_plan,
)
from concord_lqcd.job.runner import (
    ConfigurationRuntimeDecision,
    JobRunError,
    JobRunSummary,
    iter_job_work_units,
    run_job,
)
from concord_lqcd.job.run_record import (
    JobRunRecord,
    RunRecordError,
    load_job_run_record,
    write_job_run_record,
)
from concord_lqcd.visualization import (
    PhysicsVisualizationError,
    write_job_visualization,
)


app = typer.Typer(
    name="concord",
    no_args_is_help=True,
    add_completion=False,
    help="Compile, inspect, and execute ConCorD lattice-QCD correlation jobs.",
)


def _error(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"  # pragma: no cover


def _count_option(value: str | None, option: str) -> int | str | None:
    """Parse an 'auto' or positive-integer option."""

    if value is None or value == "auto":
        return value
    if value.isdigit() and int(value) > 0:
        return int(value)
    _error(f"{option} must be 'auto' or a positive integer, not {value!r}")


def _print_worker_reports(summary: JobRunSummary) -> None:
    """One line on how evenly the workers ran and how much memory they used."""

    reports = summary.worker_reports
    seconds = [report.timing.numerical_execution_seconds for report in reports]
    typer.echo(
        f"  Worker time: {min(seconds):.1f}-{max(seconds):.1f} s numerical execution"
    )
    peaks = [report.max_rss_bytes for report in reports if report.max_rss_bytes is not None]
    private = [report.private_bytes for report in reports if report.private_bytes is not None]
    own_peaks = [
        report.peak_private_bytes for report in reports if report.peak_private_bytes is not None
    ]
    if peaks:
        line = f"  Worker mem:  peak {_format_bytes(max(peaks))}"
        if own_peaks:
            line += f", own {_format_bytes(max(own_peaks))} at peak"
        if private:
            line += f", own {_format_bytes(max(private))} at end"
        typer.echo(line)


def _print_configuration_runtime_decisions(summary: JobRunSummary) -> None:
    """Print distinct configuration policies without flooding large jobs."""

    groups: dict[
        tuple[object, ...],
        tuple[ConfigurationRuntimeDecision, list[int]],
    ] = {}
    for decision in summary.configuration_decisions:
        cache = decision.primitive_cache_plan
        prefetch = decision.propagation_prefetch_plan
        cache_signature = None if cache is None else (
            cache.max_bytes,
            cache.planned_bytes,
            len(cache.entries),
            cache.request_count,
            cache.cacheable_requests,
            cache.repeated_keys,
            cache.reason,
        )
        prefetch_signature = None if prefetch is None else (
            prefetch.enabled,
            prefetch.reason,
            prefetch.batch_count,
            prefetch.eligible_requests,
            prefetch.maximum_batch_requests,
            prefetch.host_staging_bytes,
            prefetch.device_staging_bytes,
        )
        signature = (
            decision.execution_mode,
            decision.batch_size,
            cache_signature,
            prefetch_signature,
            decision.kernel_batch_staging_limit_bytes,
        )
        if signature not in groups:
            groups[signature] = (decision, [])
        groups[signature][1].append(decision.configuration)

    if not groups:
        return

    typer.echo("  Runtime policies:")
    for decision, configurations in groups.values():
        if len(configurations) <= 6:
            scope = "config " + ",".join(str(value) for value in configurations)
        else:
            scope = f"{len(configurations)} configurations"
        typer.echo(
            f"    {scope}: {decision.execution_mode}, batch {decision.batch_size}"
        )
        cache = decision.primitive_cache_plan
        prefetch = decision.propagation_prefetch_plan
        if cache is None or prefetch is None:
            typer.echo("      resident cache/prefetch: not applicable")
            continue
        typer.echo(
            "      cache: "
            f"{len(cache.entries)}/{cache.repeated_keys} repeated keys, "
            f"{_format_bytes(cache.planned_bytes)} / {_format_bytes(cache.max_bytes)}"
        )
        typer.echo(f"      cache reason: {cache.reason}")
        state = "enabled" if prefetch.enabled else "disabled"
        typer.echo(f"      prefetch: {state} ({prefetch.reason})")
        if prefetch.enabled:
            typer.echo(
                "      staging: "
                f"host {_format_bytes(prefetch.host_staging_bytes)}, "
                f"device {_format_bytes(prefetch.device_staging_bytes)}"
            )
        kernel_limit = decision.kernel_batch_staging_limit_bytes
        if kernel_limit is None:
            typer.echo("      kernel batch staging: disabled")
        else:
            typer.echo(
                "      kernel batch staging limit: "
                f"{_format_bytes(kernel_limit)}"
            )


def _print_hardware_profile(profile: HardwareProfile) -> None:
    policy = HardwareBudgetPolicy()
    typer.echo("ConCorD hardware profile")
    typer.echo(f"  System:            {profile.system} {profile.machine}")
    typer.echo(f"  PyTorch threads:   {profile.torch_num_threads}")
    typer.echo(f"  Interop threads:   {profile.torch_num_interop_threads}")
    for device in profile.devices:
        typer.echo(f"\n  {device.device} ({device.name})")
        if device.logical_processors is not None:
            typer.echo(f"    usable CPUs      {device.logical_processors}")
        if device.compute_capability is not None:
            major, minor = device.compute_capability
            typer.echo(f"    compute          {major}.{minor}")
        if device.multiprocessor_count is not None:
            typer.echo(f"    multiprocessors  {device.multiprocessor_count}")
        if device.total_memory_bytes is None:
            typer.echo("    total memory     unknown")
        else:
            typer.echo(
                f"    total memory     {_format_bytes(device.total_memory_bytes)}"
            )
        if device.available_memory_bytes is None:
            typer.echo("    available now    unknown")
        else:
            reserve = policy.margin_for(device).reserve(device.available_memory_bytes)
            typer.echo(
                f"    available now    {_format_bytes(device.available_memory_bytes)}"
            )
            typer.echo(f"    safety reserve   {_format_bytes(reserve)}")
            typer.echo(
                f"    safe working set "
                f"{_format_bytes(device.available_memory_bytes - reserve)}"
            )
    for warning in profile.warnings:
        typer.echo(f"\n  Warning: {warning}")


def _print_execution_resource_plan(resource_plan: DeviceExecutionPlan) -> None:
    budget = resource_plan.selected_budget
    typer.echo("\nHardware-aware execution feasibility")
    typer.echo(f"  Compute device:    {resource_plan.selected_device}")
    typer.echo(f"  Storage device:    {resource_plan.storage_device}")
    typer.echo(f"  Execution mode:    {resource_plan.execution_mode}")
    typer.echo(
        f"  Buffer pool:       "
        f"{_format_bytes(resource_plan.estimate.buffer_pool_bytes)}"
    )
    typer.echo(
        f"  Largest transient: "
        f"{_format_bytes(resource_plan.estimate.largest_transient_value_bytes)}"
    )
    typer.echo(
        f"  Known working set: "
        f"{_format_bytes(resource_plan.estimate.known_working_set_bytes)}"
    )
    if budget.safe_working_set_limit_bytes is None:
        typer.echo("  Memory check:      unavailable for explicit device")
    else:
        typer.echo(
            f"  Safe device limit: "
            f"{_format_bytes(budget.safe_working_set_limit_bytes)}"
        )
        if resource_plan.execution_mode == "resident":
            typer.echo(
                f"  Safe pool budget:  "
                f"{_format_bytes(budget.recommended_buffer_limit_bytes or 0)}"
            )
    if resource_plan.tiled_plan is not None:
        tiled = resource_plan.tiled_plan
        host_budget = next(
            item
            for item in resource_plan.candidate_budgets
            if item.device.device == "cpu"
        )
        assert host_budget.safe_working_set_limit_bytes is not None
        typer.echo(
            f"  Safe host limit:   "
            f"{_format_bytes(host_budget.safe_working_set_limit_bytes)}"
        )
        typer.echo(
            f"  Full op peak:      {_format_bytes(tiled.full_peak_device_bytes)}"
        )
        typer.echo(
            f"  Peak tile memory:  {_format_bytes(tiled.peak_device_bytes)}"
        )
        typer.echo(f"  Tiled operations:  {tiled.tiled_operation_count}")
        typer.echo(f"  Contraction calls: {tiled.total_contraction_calls}")
    typer.echo(
        "  Scope:             provider/library workspace is covered by safety "
        "reserve, not exactly modeled"
    )


def _print_plan_summary(plan: ConcordPlan) -> None:
    plan.validate()
    resource_counts = Counter(resource.kind for resource in plan.resources.values())
    generalized = sum(
        int(recipe.is_generalized)
        for recipe in plan.required_propagation_resources.values()
    )
    physical = plan.physical_plan.statistics
    costs = plan.cost_report

    typer.echo("ConCorD execution plan")
    typer.echo(f"  Job:              {plan.manifest.job_name}")
    typer.echo(f"  Created:          {plan.manifest.created_utc}")
    typer.echo(f"  ConCorD version:  {plan.manifest.concord_version}")
    typer.echo(f"  Compiler semantics: {plan.manifest.compiler_semantics_version}")
    typer.echo(f"  Local helper files: {len(plan.manifest.operator_dependencies)}")
    typer.echo(f"  Plan SHA-256:     {plan.manifest.plan_sha256}")
    typer.echo(f"  Outputs:          {', '.join(plan.correlator_ids)}")
    typer.echo(f"  Distillation Nv:  {plan.config.mode_size}")
    typer.echo(f"  Spin size:        {plan.config.spin_size}")
    typer.echo(f"  Spin basis:       {plan.config.spin_basis}")
    typer.echo(f"  Scalar type:      {plan.config.scalar_type}")
    typer.echo(f"  Contraction:      {plan.contraction_order.strategy}")
    if plan.contraction_order.selected_strategies != (
        plan.contraction_order.strategy,
    ):
        typer.echo(
            "  Selected paths:    "
            + ", ".join(plan.contraction_order.selected_strategies)
        )
    typer.echo("  Resources:")
    typer.echo(f"    total           {len(plan.resources)}")
    typer.echo(f"    kernel          {resource_counts.get('kernel', 0)}")
    typer.echo(f"    spin            {resource_counts.get('spin', 0)}")
    typer.echo(f"    propagation     {resource_counts.get('propagation', 0)}")
    typer.echo(f"    generalized     {generalized}")
    typer.echo("  Execution:")
    typer.echo(f"    physical steps  {len(plan.physical_plan.steps)}")
    typer.echo(f"    buffers         {len(plan.physical_plan.buffers)}")
    typer.echo(
        f"    peak active     {_format_bytes(physical.peak_active_capacity_bytes)}"
    )
    typer.echo(
        f"    total capacity  {_format_bytes(physical.total_buffer_capacity_bytes)}"
    )
    typer.echo("  Analytical costs:")
    if costs.input_wick_terms is None:
        typer.echo(f"    Wick terms      {costs.surviving_wick_terms} surviving")
    else:
        typer.echo(
            f"    Wick terms      {costs.surviving_wick_terms}/"
            f"{costs.input_wick_terms} surviving"
        )
    typer.echo(
        f"    components      {costs.unique_components}/"
        f"{costs.component_occurrences} unique"
    )
    typer.echo(
        f"    intermediates   {costs.unique_contraction_steps}/"
        f"{costs.contraction_step_occurrences} unique"
    )
    typer.echo(
        f"    est. FLOPs      {costs.scheduled_contraction_flops:,} scheduled; "
        f"{costs.estimated_flops_saved:,} saved by intermediate CSE"
    )
    typer.echo(f"    ready width     {costs.max_ready_width}")
    typer.echo(
        f"    logical peak    {_format_bytes(costs.logical_peak_live_bytes)}"
    )


def _print_resources(plan: ConcordPlan) -> None:
    typer.echo("\nResources")
    for resource_id, resource in plan.resources.items():
        detail = ""
        payload = resource.payload
        if isinstance(payload, PropagationRecipe):
            kind = "generalized" if payload.is_generalized else "ordinary"
            detail = (
                f" {kind}; segment_flavors={tuple(payload.segment_flavors)!r}; "
                f"insertions={len(payload.insertion_specs)}"
            )
        elif isinstance(payload, KernelRecipe):
            momenta = tuple(payload.momenta.values())
            detail = f" vertex={payload.vertex_id!r}; momenta={momenta!r}"
        typer.echo(f"  {resource_id}  [{resource.kind}]{detail}")


def _print_schedule(plan: ConcordPlan) -> None:
    typer.echo("\nPhysical schedule")
    for step in plan.physical_plan.steps:
        operation = step.operation
        released = ",".join(step.released_values) if step.released_values else "-"
        typer.echo(
            f"  {step.step_index:4d}  {type(operation).__name__:<32} "
            f"node={operation.node_id}  release={released}"
        )


def _print_buffers(plan: ConcordPlan) -> None:
    typer.echo("\nPhysical buffers")
    uses = Counter(
        binding.buffer_id for binding in plan.physical_plan.value_bindings.values()
    )
    for buffer_id, buffer in plan.physical_plan.buffers.items():
        typer.echo(
            f"  {buffer_id}  capacity={_format_bytes(buffer.capacity_bytes)}  "
            f"dtype={buffer.scalar_type}  space={buffer.memory_space}  "
            f"logical_values={uses[buffer_id]}"
        )


def _prepare_plan_for_run(spec, *, replan: bool) -> tuple[ConcordPlan, str]:
    if replan:
        return write_job_plan(spec, overwrite=True), "replanned"
    if not spec.plan.file.exists():
        return write_job_plan(spec, overwrite=False), "created"
    plan = _load_prepared_plan(spec.plan.file)
    validate_plan_for_job(plan, spec)
    return plan, "loaded"


def _dry_run(
    spec,
    plan: ConcordPlan,
    *,
    device: str | torch.device | None,
) -> tuple[int, DeviceExecutionPlan]:
    requested_device = spec.run.device if device is None else device
    execution_plan = resolve_runtime_device(plan.physical_plan, requested_device)
    runtime_device = execution_plan.torch_device
    count = 0
    for unit in iter_job_work_units(spec):
        resources = build_job_runtime_resources(
            spec,
            plan,
            configuration=unit.configuration,
            time_context=unit.time_context,
            device=runtime_device,
            execution_plan=execution_plan,
        )
        # Binding resolves every scheduled exact resource occurrence, including
        # concrete operator times and storage descriptions, without materializing
        # numerical tensors or executing contractions.
        resources.bind(plan)
        count += 1
    return count, execution_plan


@app.command("hardware")
def hardware_command() -> None:
    """Report CPU/RAM and visible CUDA resources without allocating tensors."""

    _print_hardware_profile(detect_hardware_profile())


@app.command("plan")
def plan_command(
    job: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace an existing .concord plan explicitly.",
    ),
) -> None:
    """Compile JOB once and write the configured portable .concord plan."""

    try:
        spec = load_job_spec(job)
        plan = write_job_plan(spec, overwrite=force)
    except (JobLoadError, OperatorLoadError, PlanSerializationError, FileExistsError) as exc:
        _error(str(exc))
        return

    typer.echo(f"Wrote plan: {spec.plan.file}")
    _print_plan_summary(plan)


@app.command("inspect")
def inspect_command(
    plan_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    resources: bool = typer.Option(
        False,
        "--resources",
        help="List compiled tensor resources.",
    ),
    schedule: bool = typer.Option(
        False,
        "--schedule",
        help="Show the physical execution schedule.",
    ),
    buffers: bool = typer.Option(
        False,
        "--buffers",
        help="Show physical buffer assignments.",
    ),
    hardware: bool = typer.Option(
        False,
        "--hardware",
        help="Check this frozen plan against currently available CPU/GPU memory.",
    ),
) -> None:
    """Inspect a saved .concord plan without loading lattice data."""

    try:
        plan = load_concord_plan(plan_file)
    except PlanSerializationError as exc:
        _error(str(exc))
        return

    _print_plan_summary(plan)
    if resources:
        _print_resources(plan)
    if schedule:
        _print_schedule(plan)
    if buffers:
        _print_buffers(plan)
    if hardware:
        try:
            profile = detect_hardware_profile()
            resource_plan = plan_execution_resources(
                plan.physical_plan,
                profile,
                requested_device="auto",
            )
        except ResourcePlanningError as exc:
            _error(str(exc))
            return
        _print_hardware_profile(profile)
        _print_execution_resource_plan(resource_plan)


@app.command("visualize")
def visualize_command(
    job: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Standalone HTML output (default: JOB with .html suffix).",
    ),
    run_record: Path | None = typer.Option(
        None,
        "--run-record",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional JSON run record to add measured execution views.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace an existing HTML report explicitly.",
    ),
) -> None:
    """Write a physics-first report with optional plan and execution views."""

    try:
        spec = load_job_spec(job)
        collection = load_operator_collection(spec)
        plan = None
        if spec.plan.file.exists():
            plan = _load_prepared_plan(spec.plan.file)
            validate_plan_for_job(plan, spec)
        record = (
            load_job_run_record(run_record) if run_record is not None else None
        )
        target = output if output is not None else job.with_suffix(".html")
        written = write_job_visualization(
            spec,
            collection,
            target,
            plan=plan,
            run_record=record,
            overwrite=force,
        )
    except (
        FileExistsError,
        JobLoadError,
        OperatorLoadError,
        PlanSerializationError,
        PhysicsVisualizationError,
        RunRecordError,
    ) as exc:
        _error(str(exc))
        return
    typer.echo(f"Wrote visualization: {written}")


@app.command("run")
def run_command(
    job: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    replan: bool = typer.Option(
        False,
        "--replan",
        help="Explicitly rebuild and replace the configured .concord plan before running.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate/resolve the full work grid without tensor I/O or contractions.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Override run.device for this invocation only (for example cuda:1).",
    ),
    workers: str | None = typer.Option(
        None,
        "--workers",
        help="Override run.workers: 'auto' or the number of CPU worker processes.",
    ),
    threads_per_worker: str | None = typer.Option(
        None,
        "--threads-per-worker",
        help="Override run.threads_per_worker: 'auto' or threads per worker.",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Write a combined physics, tensor-network, and execution HTML report.",
    ),
    run_record: Path | None = typer.Option(
        None,
        "--run-record",
        help="Write the versioned machine-readable JSON execution record.",
    ),
    force_report: bool = typer.Option(
        False,
        "--force-report",
        help="Replace existing report and run-record files explicitly.",
    ),
) -> None:
    """Execute JOB, automatically creating a missing plan and resuming output."""

    workers = _count_option(workers, "--workers")
    threads_per_worker = _count_option(threads_per_worker, "--threads-per-worker")
    try:
        spec = load_job_spec(job)
        overrides = {
            name: value
            for name, value in (
                ("workers", workers),
                ("threads_per_worker", threads_per_worker),
            )
            if value is not None
        }
        if overrides:
            spec = spec.model_copy(update={"run": spec.run.model_copy(update=overrides)})
        if spec.run.workers != 1:
            # Keep this process free of OpenMP threads until any workers are forked.
            torch.set_num_threads(1)
        plan_started = perf_counter()
        plan, plan_action = _prepare_plan_for_run(spec, replan=replan)
        plan_preparation_seconds = perf_counter() - plan_started
    except StalePlanError as exc:
        _error(
            f"{exc}. Run 'concord plan {job} --force' or "
            f"'concord run {job} --replan'."
        )
        return
    except (
        JobLoadError,
        OperatorLoadError,
        PlanSerializationError,
        FileExistsError,
    ) as exc:
        _error(str(exc))
        return

    if dry_run and (report is not None or run_record is not None):
        _error("--report and --run-record require a numerical run, not --dry-run")
        return

    record_target = run_record
    if report is not None and record_target is None:
        record_target = report.with_suffix(".run.json")
    if report is not None:
        if report.suffix.lower() not in {".html", ".htm"}:
            _error("report output must use .html or .htm")
            return
        if report.exists() and not force_report:
            _error(f"calculation report already exists: {report}")
            return
    if record_target is not None:
        if record_target.suffix.lower() != ".json":
            _error("run-record output must use the .json extension")
            return
        if record_target.exists() and not force_report:
            _error(f"run record already exists: {record_target}")
            return

    if plan_action in {"created", "replanned"}:
        typer.echo(f"Plan {plan_action}: {spec.plan.file}")

    if dry_run:
        try:
            count, execution_plan = _dry_run(spec, plan, device=device)
        except (
            JobDataError,
            PlanSerializationError,
            ResourcePlanningError,
            ValueError,
            RuntimeError,
        ) as exc:
            _error(str(exc))
            return
        typer.echo("Dry run successful")
        typer.echo(f"  Plan:        {spec.plan.file}")
        typer.echo(f"  Compute:     {execution_plan.selected_device}")
        typer.echo(f"  Storage:     {execution_plan.storage_device}")
        typer.echo(f"  Mode:        {execution_plan.execution_mode}")
        typer.echo(f"  Work units:  {count}")
        typer.echo(f"  Output:      {spec.output.file}")
        return

    try:
        summary = run_job(spec, plan, device=device)
    except (JobDataError, JobRunError, PlanSerializationError, FileExistsError) as exc:
        _error(str(exc))
        return

    typer.echo("Job complete" if summary.all_complete else "Job incomplete")
    typer.echo(f"  Output:      {summary.output_file}")
    typer.echo(f"  Work units:  {summary.total_work_units}")
    typer.echo(f"  Executed:    {summary.executed_work_units}")
    typer.echo(f"  Skipped:     {summary.skipped_work_units}")
    typer.echo(f"  Completed:   {summary.completed_work_units}")
    typer.echo(f"  Contraction: {summary.contraction_path}")
    typer.echo(f"  Selection:   {summary.contraction_selection_source}")
    typer.echo(f"  Time batch:  {summary.time_batch_size}")
    typer.echo(f"  Batch policy:{summary.time_batch_selection_source:>14}")
    typer.echo(f"  Batches:     {summary.executed_batches}")
    typer.echo(f"  Compute:     {summary.selected_device}")
    typer.echo(f"  Storage:     {summary.storage_device}")
    typer.echo(f"  Mode:        {summary.execution_mode}")
    if summary.worker_reports:
        typer.echo(
            f"  Workers:     {summary.workers} x {summary.threads_per_worker} threads"
        )
        _print_worker_reports(summary)
    elif summary.threads_per_worker is not None:
        typer.echo(f"  Threads:     {summary.threads_per_worker}")
    if summary.worker_selection.startswith("auto"):
        typer.echo(f"  Split:       {summary.worker_selection}")
    _print_configuration_runtime_decisions(summary)
    if summary.resident_pool_allocations:
        typer.echo(f"  Pool allocs: {summary.resident_pool_allocations}")
        typer.echo(
            f"  Pool peak:   {_format_bytes(summary.resident_pool_peak_bytes)}"
        )
        typer.echo(
            "  Cache hits:  "
            f"{summary.primitive_cache_hits} "
            f"(misses {summary.primitive_cache_misses})"
        )
        typer.echo(
            f"  Cache peak:  {_format_bytes(summary.primitive_cache_peak_bytes)}"
        )
        if summary.primitive_cache_allocation_failures:
            typer.echo(
                "  Cache OOMs:  "
                f"{summary.primitive_cache_allocation_failures} (cache disabled)"
            )
    if summary.kernel_cache_enabled:
        typer.echo(
            "  Kernel cache: "
            f"{summary.kernel_cache_hits} hits, "
            f"{summary.kernel_cache_misses} misses, "
            f"{summary.kernel_cache_writes} saved"
        )
        if summary.kernel_cache_invalid_entries or summary.kernel_cache_write_failures:
            typer.echo(
                "  Cache issues: "
                f"{summary.kernel_cache_invalid_entries} invalid entries, "
                f"{summary.kernel_cache_write_failures} write failures"
            )
    if summary.hdf5_file_opens or summary.hdf5_file_reuses:
        typer.echo(
            f"  HDF5 files:  {summary.hdf5_file_opens} opens, "
            f"{summary.hdf5_file_reuses} reuses"
        )
    if (
        summary.propagation_prefetch_configurations
        or summary.propagation_prefetch_allocation_fallbacks
    ):
        typer.echo(
            "  Prefetch:    "
            f"{summary.propagation_prefetch_batches} batches, "
            f"{summary.propagation_prefetch_tensors} tensors"
        )
        typer.echo(
            "  Prefetched:  "
            f"{_format_bytes(summary.propagation_prefetch_bytes)}, "
            f"wait {summary.propagation_prefetch_wait_seconds:.3f} s"
        )
        if summary.propagation_prefetch_allocation_fallbacks:
            typer.echo(
                "  Prefetch fallback: "
                f"{summary.propagation_prefetch_allocation_fallbacks} "
                "(synchronous I/O retained)"
            )
    if summary.peak_tile_bytes is not None:
        typer.echo(f"  Peak tile:   {_format_bytes(summary.peak_tile_bytes)}")
        typer.echo(f"  Tiled ops:   {summary.tiled_operations}")
    typer.echo(
        f"  Known memory:{_format_bytes(summary.known_working_set_bytes):>14}"
    )
    typer.echo(f"  Wall time:   {summary.timing.execution_wall_seconds:.3f} s")

    if report is not None or record_target is not None:
        record = JobRunRecord.from_summary(
            summary,
            job_name=spec.job.name,
            plan_action=plan_action,
            plan_preparation_seconds=plan_preparation_seconds,
        )
        try:
            if record_target is not None:
                written_record = write_job_run_record(
                    record,
                    record_target,
                    overwrite=force_report,
                )
                typer.echo(f"  Run record:  {written_record}")
            if report is not None:
                collection = load_operator_collection(spec)
                written_report = write_job_visualization(
                    spec,
                    collection,
                    report,
                    plan=plan,
                    run_record=record,
                    overwrite=force_report,
                )
                typer.echo(f"  Report:      {written_report}")
        except (
            FileExistsError,
            OperatorLoadError,
            PhysicsVisualizationError,
            RunRecordError,
        ) as exc:
            _error(f"numerical run completed, but report generation failed: {exc}")


def main() -> None:
    """Console-script entry point."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
