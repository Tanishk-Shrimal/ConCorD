"""Backend-independent list scheduling for the logical execution DAG.

This layer consumes frozen ``StaticExecutionAnalysis`` and chooses one legal
sequential execution order.  It is deliberately still a *logical* schedule:
there is no CPU/GPU assignment, stream selection, buffer residency, storage
location, communication, prefetching, or runtime execution policy here.

The scheduler uses only compiler-visible information:

* dependency readiness;
* downstream criticality;
* value fanout/reuse;
* logical tensor sizes;
* contraction FLOP estimates;
* input-memory release potential.

It also derives schedule-specific value lifetimes and a conservative peak-live
byte estimate under a materialize-output-before-releasing-inputs model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
from numbers import Integral
from typing import Literal, Mapping

from concord_lqcd.compiler._validation import _has_static_validation
from concord_lqcd.compiler.logical_execution import LogicalContraction
from concord_lqcd.compiler.static_analysis import StaticExecutionAnalysis


ScheduleStrategy = Literal["memory_aware", "critical_path", "topological"]


@dataclass(frozen=True)
class LogicalValueLifetime:
    """Schedule-derived lifetime of one logical value.

    ``consumer_steps`` preserves repeated dependency occurrences.  If a value is
    used twice by the same Wick-term product, that schedule step therefore
    appears twice.  Requested correlator outputs are retained after the logical
    schedule and consequently have ``release_after_step is None``.
    """

    value_id: str
    produced_at: int
    consumer_steps: tuple[int, ...]
    retained_output: bool

    @property
    def first_used_at(self) -> int | None:
        return min(self.consumer_steps) if self.consumer_steps else None

    @property
    def last_used_at(self) -> int:
        return max(self.consumer_steps) if self.consumer_steps else self.produced_at

    @property
    def release_after_step(self) -> int | None:
        return None if self.retained_output else self.last_used_at


@dataclass(frozen=True)
class LogicalScheduleStep:
    """One selected node in a sequential hardware-independent schedule."""

    step_index: int
    node_id: str
    dependencies: tuple[str, ...]
    ready_width: int
    criticality: int
    fanout: int
    estimated_contraction_flops: int
    output_bytes: int
    releasable_input_bytes: int
    released_values: tuple[str, ...]
    live_bytes_before: int
    peak_live_bytes_during_step: int
    live_bytes_after: int


@dataclass(frozen=True)
class LogicalScheduleStatistics:
    scheduled_nodes: int
    skipped_dead_nodes: int
    peak_live_bytes: int
    final_live_bytes: int
    total_estimated_contraction_flops: int
    max_ready_width: int
    released_values: int


@dataclass(frozen=True)
class LogicalSchedule:
    """Deterministic sequential schedule plus logical lifetime information."""

    source_analysis: StaticExecutionAnalysis
    strategy: ScheduleStrategy
    include_dead: bool
    steps: tuple[LogicalScheduleStep, ...]
    lifetimes: Mapping[str, LogicalValueLifetime]
    output_nodes: tuple[str, ...]
    statistics: LogicalScheduleStatistics

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(step.node_id for step in self.steps)

    def lifetime(self, value_id: str) -> LogicalValueLifetime:
        return self.lifetimes[value_id]

    def validate(self) -> None:
        if _has_static_validation(self):
            return
        analysis = self.source_analysis
        analysis.validate()
        if self.strategy not in ("memory_aware", "critical_path", "topological"):
            raise ValueError(f"Unsupported logical schedule strategy {self.strategy!r}.")
        if type(self.include_dead) is not bool:
            raise ValueError("Logical schedule include_dead must be a boolean.")
        if self.output_nodes != analysis.output_nodes:
            raise ValueError("Logical schedule outputs disagree with the requested source outputs.")
        all_nodes = tuple(analysis.topological_order)
        expected = tuple(
            node_id
            for node_id in all_nodes
            if self.include_dead or analysis.dependency(node_id).is_live
        )
        order = self.order
        if len(order) != len(set(order)):
            raise ValueError("Logical schedule contains duplicate node ids.")
        if set(order) != set(expected):
            raise ValueError("Logical schedule does not contain exactly the selected DAG nodes.")
        if len(self.steps) != len(expected):
            raise ValueError("Logical schedule step count is inconsistent.")

        position = {node_id: idx for idx, node_id in enumerate(order)}
        selected_consumers: dict[str, set[str]] = {node_id: set() for node_id in order}
        for step in self.steps:
            if step.step_index != position[step.node_id]:
                raise ValueError("Logical schedule step indices are inconsistent.")
            if step.ready_width <= 0:
                raise ValueError("Logical schedule records a non-positive ready width.")
            expected_dependencies = tuple(
                dep
                for dep in analysis.dependency(step.node_id).dependencies
                if dep in position
            )
            if step.dependencies != expected_dependencies:
                raise ValueError(
                    f"Logical schedule dependencies for {step.node_id!r} are inconsistent."
                )
            for dependency in step.dependencies:
                if position[dependency] >= step.step_index:
                    raise ValueError(
                        f"Logical schedule executes {step.node_id!r} before dependency "
                        f"{dependency!r}."
                    )
                selected_consumers[dependency].add(step.node_id)

        for step in self.steps:
            expected_fanout = len(selected_consumers[step.node_id])
            if step.fanout != expected_fanout:
                raise ValueError(
                    f"Logical schedule fanout for {step.node_id!r} is inconsistent with "
                    "the selected execution subgraph."
                )

        if set(self.lifetimes) != set(order):
            raise ValueError("Logical schedule lifetime map does not match scheduled nodes.")

        expected_consumer_steps: dict[str, list[int]] = {node_id: [] for node_id in order}
        for step in self.steps:
            for dependency in step.dependencies:
                expected_consumer_steps[dependency].append(step.step_index)

        output_set = set(self.output_nodes)
        for node_id in order:
            lifetime = self.lifetimes[node_id]
            if lifetime.value_id != node_id:
                raise ValueError("Logical lifetime is stored under the wrong value id.")
            if lifetime.produced_at != position[node_id]:
                raise ValueError(f"Logical lifetime for {node_id!r} has wrong production step.")
            if lifetime.consumer_steps != tuple(expected_consumer_steps[node_id]):
                raise ValueError(f"Logical lifetime for {node_id!r} has wrong consumer steps.")
            if lifetime.retained_output != (node_id in output_set):
                raise ValueError(f"Logical lifetime for {node_id!r} has wrong output retention.")

        if not output_set.issubset(position):
            raise ValueError("Requested logical outputs are missing from the schedule.")

        unsatisfied = {node_id: len(set(self.steps[position[node_id]].dependencies)) for node_id in order}
        ready = {node_id for node_id in order if unsatisfied[node_id] == 0}
        resident: set[str] = set()
        live_bytes = 0
        for step in self.steps:
            for name in ("step_index", "ready_width", "criticality", "fanout",
                         "estimated_contraction_flops", "output_bytes", "releasable_input_bytes",
                         "live_bytes_before", "peak_live_bytes_during_step", "live_bytes_after"):
                value = getattr(step, name)
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"Logical schedule {name} must be an integer.")
            if step.node_id not in ready or step.ready_width != len(ready):
                raise ValueError("Logical schedule ready width disagrees with its source dependencies.")
            ready.remove(step.node_id)
            if not set(step.dependencies).issubset(resident):
                raise ValueError("Logical schedule uses an input after its recorded release.")
            inputs_to_release = tuple(
                dependency for dependency in dict.fromkeys(step.dependencies)
                if self.lifetimes[dependency].release_after_step == step.step_index
            )
            releases = inputs_to_release
            if self.lifetimes[step.node_id].release_after_step == step.step_index:
                releases += (step.node_id,)
            if step.released_values != releases:
                raise ValueError("Logical schedule releases disagree with final consumer lifetimes.")
            output_bytes = analysis.value(step.node_id).size_bytes
            release_bytes = sum(analysis.value(dependency).size_bytes for dependency in inputs_to_release)
            expected_criticality = analysis.dependency(step.node_id).distance_to_output
            if step.criticality != (-1 if expected_criticality is None else expected_criticality):
                raise ValueError("Logical schedule criticality disagrees with its source.")
            if step.estimated_contraction_flops != _contraction_flops(analysis, step.node_id):
                raise ValueError("Logical schedule contraction cost disagrees with its source.")
            if (step.output_bytes, step.releasable_input_bytes, step.live_bytes_before,
                step.peak_live_bytes_during_step, step.live_bytes_after) != (
                    output_bytes, release_bytes, live_bytes, live_bytes + output_bytes,
                    live_bytes + output_bytes - release_bytes
                    - (output_bytes if step.node_id in releases else 0),
                ):
                raise ValueError("Logical schedule byte accounting disagrees with value lifetimes.")
            resident.add(step.node_id)
            resident.difference_update(releases)
            live_bytes = step.live_bytes_after
            for consumer in selected_consumers[step.node_id]:
                unsatisfied[consumer] -= 1
                if unsatisfied[consumer] == 0:
                    ready.add(consumer)
        if resident != output_set:
            raise ValueError("Logical schedule does not finish with exactly its requested outputs live.")

        expected_statistics = LogicalScheduleStatistics(
            scheduled_nodes=len(order),
            skipped_dead_nodes=0 if self.include_dead else len(analysis.dead_nodes),
            peak_live_bytes=max((step.peak_live_bytes_during_step for step in self.steps), default=0),
            final_live_bytes=live_bytes,
            total_estimated_contraction_flops=sum(step.estimated_contraction_flops for step in self.steps),
            max_ready_width=max((step.ready_width for step in self.steps), default=0),
            released_values=sum(len(step.released_values) for step in self.steps),
        )
        if self.statistics != expected_statistics:
            raise ValueError("Logical schedule statistics disagree with its validated steps.")

        expected_final = sum(analysis.value(node_id).size_bytes for node_id in output_set)
        if live_bytes != expected_final:
            raise ValueError(
                "Logical schedule must retain exactly the requested outputs after execution."
            )


def _contraction_flops(analysis: StaticExecutionAnalysis, node_id: str) -> int:
    node = analysis.source_dag.node(node_id)
    if isinstance(node, LogicalContraction):
        return int(node.estimated_flops)
    return 0


def _candidate_release_bytes(
    analysis: StaticExecutionAnalysis,
    node_id: str,
    remaining_consumers: Mapping[str, int],
    scheduled_node_set: set[str],
    output_set: set[str],
) -> int:
    """Bytes that become dead immediately after executing ``node_id``.

    Repeated dependency occurrences are consumed together by one scheduled node,
    so they are counted before deciding whether this is the final consumer.
    """

    dependencies = [
        dep
        for dep in analysis.dependency(node_id).dependencies
        if dep in scheduled_node_set
    ]
    counts = Counter(dependencies)
    released = 0
    for dependency, occurrence_count in counts.items():
        if dependency in output_set:
            continue
        if remaining_consumers[dependency] == occurrence_count:
            released += analysis.value(dependency).size_bytes
    return released


def _priority_key(
    analysis: StaticExecutionAnalysis,
    node_id: str,
    *,
    strategy: ScheduleStrategy,
    remaining_consumers: Mapping[str, int],
    selected_fanout: Mapping[str, int],
    scheduled_node_set: set[str],
    output_set: set[str],
    topological_position: Mapping[str, int],
) -> tuple[object, ...]:
    record = analysis.dependency(node_id)
    output_bytes = analysis.value(node_id).size_bytes
    release_bytes = _candidate_release_bytes(
        analysis,
        node_id,
        remaining_consumers,
        scheduled_node_set,
        output_set,
    )
    criticality = record.distance_to_output if record.distance_to_output is not None else -1
    fanout = selected_fanout[node_id]
    flops = _contraction_flops(analysis, node_id)
    topo_tie_break = -topological_position[node_id]

    if strategy == "topological":
        return (topo_tie_break,)

    if strategy == "critical_path":
        return (
            criticality,
            fanout,
            release_bytes - output_bytes,
            flops,
            -output_bytes,
            topo_tie_break,
        )

    if strategy == "memory_aware":
        return (
            release_bytes - output_bytes,
            criticality,
            fanout,
            -output_bytes,
            flops,
            topo_tie_break,
        )

    raise ValueError(f"Unsupported logical schedule strategy {strategy!r}.")


def build_logical_schedule(
    analysis: StaticExecutionAnalysis,
    *,
    strategy: ScheduleStrategy = "memory_aware",
    include_dead: bool = False,
) -> LogicalSchedule:
    """Choose a deterministic dependency-valid sequential logical schedule.

    ``include_dead=False`` is the normal execution interpretation: nodes that do
    not reach any requested correlator output are not scheduled.  Static analysis
    still reports those nodes, so no information is lost.

    Peak-live bytes use a conservative tensor materialization model.  During a
    step, all currently-live inputs coexist with the newly-produced output; only
    after the operation completes may final-use inputs be released.  In-place or
    streaming kernels can improve on this later during physical planning.
    """

    if strategy not in ("memory_aware", "critical_path", "topological"):
        raise ValueError(f"Unsupported logical schedule strategy {strategy!r}.")

    analysis.validate()

    topological_position = {
        node_id: idx for idx, node_id in enumerate(analysis.topological_order)
    }
    selected = tuple(
        node_id
        for node_id in analysis.topological_order
        if include_dead or analysis.dependency(node_id).is_live
    )
    selected_set = set(selected)
    output_nodes = tuple(
        node_id for node_id in analysis.output_nodes if node_id in selected_set
    )
    output_set = set(output_nodes)

    dependencies: dict[str, tuple[str, ...]] = {
        node_id: tuple(
            dep
            for dep in analysis.dependency(node_id).dependencies
            if dep in selected_set
        )
        for node_id in selected
    }
    unique_dependencies = {
        node_id: set(deps) for node_id, deps in dependencies.items()
    }
    consumers: dict[str, set[str]] = {node_id: set() for node_id in selected}
    remaining_consumers: dict[str, int] = {node_id: 0 for node_id in selected}
    for node_id, deps in dependencies.items():
        for dependency in deps:
            consumers[dependency].add(node_id)
            remaining_consumers[dependency] += 1
    selected_fanout = {
        node_id: len(node_consumers)
        for node_id, node_consumers in consumers.items()
    }

    unsatisfied = {
        node_id: len(unique_dependencies[node_id]) for node_id in selected
    }
    ready: set[str] = {
        node_id for node_id in selected if unsatisfied[node_id] == 0
    }

    # Ready nodes wait in a heap of negated priority keys. Keys are unique through
    # the topological tie-break, so the heap selects exactly the node that a scan
    # of the whole ready set would select. Only the release-bytes term of a key
    # changes while a node waits: an input counts as released once the node is its
    # last unscheduled consumer, and that never reverts. A node is therefore
    # re-keyed only when it becomes the last consumer of an input; superseded heap
    # entries are skipped through their version number.
    unscheduled_consumers = {
        node_id: set(node_consumers) for node_id, node_consumers in consumers.items()
    }
    key_versions: dict[str, int] = {}
    ready_heap: list[tuple[tuple[object, ...], int, str]] = []

    def push_ready(node_id: str) -> None:
        version = key_versions.get(node_id, 0) + 1
        key_versions[node_id] = version
        key = _priority_key(
            analysis,
            node_id,
            strategy=strategy,
            remaining_consumers=remaining_consumers,
            selected_fanout=selected_fanout,
            scheduled_node_set=selected_set,
            output_set=output_set,
            topological_position=topological_position,
        )
        heapq.heappush(ready_heap, (tuple(-part for part in key), version, node_id))

    for node_id in ready:
        push_ready(node_id)

    steps: list[LogicalScheduleStep] = []
    produced_at: dict[str, int] = {}
    consumer_steps: dict[str, list[int]] = {node_id: [] for node_id in selected}
    live_values: set[str] = set()
    live_bytes = 0
    peak_live_bytes = 0
    total_flops = 0
    max_ready_width = 0
    released_value_count = 0

    while len(steps) < len(selected):
        if not ready:
            remaining = sorted(set(selected) - set(produced_at))
            raise ValueError(
                "Logical scheduler has no ready node before completing the selected DAG: "
                f"{remaining}"
            )

        ready_width = len(ready)
        max_ready_width = max(max_ready_width, ready_width)
        while True:
            _, version, node_id = heapq.heappop(ready_heap)
            if node_id in ready and key_versions[node_id] == version:
                break
        ready.remove(node_id)

        step_index = len(steps)
        node_dependencies = dependencies[node_id]
        missing_live = [dep for dep in set(node_dependencies) if dep not in live_values]
        if missing_live:
            raise RuntimeError(
                f"Ready node {node_id!r} has dependencies that are not live: {missing_live}."
            )

        record = analysis.dependency(node_id)
        criticality = record.distance_to_output if record.distance_to_output is not None else -1
        output_bytes = analysis.value(node_id).size_bytes
        releasable_input_bytes = _candidate_release_bytes(
            analysis,
            node_id,
            remaining_consumers,
            selected_set,
            output_set,
        )
        flops = _contraction_flops(analysis, node_id)
        total_flops += flops

        live_before = live_bytes
        live_values.add(node_id)
        live_bytes += output_bytes
        peak_during = live_bytes
        peak_live_bytes = max(peak_live_bytes, peak_during)
        produced_at[node_id] = step_index

        for dependency in node_dependencies:
            consumer_steps[dependency].append(step_index)
            remaining_consumers[dependency] -= 1
            if remaining_consumers[dependency] < 0:
                raise RuntimeError(
                    f"Logical consumer count for {dependency!r} became negative."
                )

        for dependency in dict.fromkeys(node_dependencies):
            waiting = unscheduled_consumers[dependency]
            waiting.discard(node_id)
            if len(waiting) == 1:
                (last_consumer,) = waiting
                if last_consumer in ready:
                    push_ready(last_consumer)

        released: list[str] = []
        for dependency in dict.fromkeys(node_dependencies):
            if remaining_consumers[dependency] == 0 and dependency not in output_set:
                if dependency not in live_values:
                    raise RuntimeError(
                        f"Logical value {dependency!r} reached final use while not live."
                    )
                live_values.remove(dependency)
                live_bytes -= analysis.value(dependency).size_bytes
                released.append(dependency)

        # This occurs only when dead nodes are explicitly included or for an
        # intentionally unconsumed non-output value in a hand-built DAG.
        if remaining_consumers[node_id] == 0 and node_id not in output_set:
            live_values.remove(node_id)
            live_bytes -= output_bytes
            released.append(node_id)

        released_value_count += len(released)
        steps.append(LogicalScheduleStep(
            step_index=step_index,
            node_id=node_id,
            dependencies=node_dependencies,
            ready_width=ready_width,
            criticality=criticality,
            fanout=selected_fanout[node_id],
            estimated_contraction_flops=flops,
            output_bytes=output_bytes,
            releasable_input_bytes=releasable_input_bytes,
            released_values=tuple(released),
            live_bytes_before=live_before,
            peak_live_bytes_during_step=peak_during,
            live_bytes_after=live_bytes,
        ))

        for consumer in consumers[node_id]:
            unsatisfied[consumer] -= 1
            if unsatisfied[consumer] < 0:
                raise RuntimeError(
                    f"Logical readiness count for {consumer!r} became negative."
                )
            if unsatisfied[consumer] == 0:
                ready.add(consumer)
                push_ready(consumer)

    if live_values != output_set:
        raise RuntimeError(
            "Logical scheduling finished with live values other than requested outputs: "
            f"live={sorted(live_values)}, outputs={sorted(output_set)}"
        )

    lifetimes = {
        node_id: LogicalValueLifetime(
            value_id=node_id,
            produced_at=produced_at[node_id],
            consumer_steps=tuple(consumer_steps[node_id]),
            retained_output=node_id in output_set,
        )
        for node_id in selected
    }

    schedule = LogicalSchedule(
        source_analysis=analysis,
        strategy=strategy,
        include_dead=include_dead,
        steps=tuple(steps),
        lifetimes=lifetimes,
        output_nodes=output_nodes,
        statistics=LogicalScheduleStatistics(
            scheduled_nodes=len(steps),
            skipped_dead_nodes=0 if include_dead else len(analysis.dead_nodes),
            peak_live_bytes=peak_live_bytes,
            final_live_bytes=live_bytes,
            total_estimated_contraction_flops=total_flops,
            max_ready_width=max_ready_width,
            released_values=released_value_count,
        ),
    )
    schedule.validate()
    return schedule
