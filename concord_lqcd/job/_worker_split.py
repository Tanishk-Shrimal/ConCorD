"""Choose the number of CPU worker processes and the threads each one uses.

``run.workers: auto`` decides from what the plan already knows (its steps,
FLOPs and memory per work unit), the node's cores and memory domains, the
free memory and the number of pending work units:

- When most of a plan's one-core time is the fixed cost of each step (many
  tiny steps), threads do not help, so every worker gets one thread and the
  node runs one worker per core.
- When most of it is arithmetic, a few threads per worker share the memory
  bandwidth better: on the 64-core test node four threads per worker beat
  8 x 8, 4 x 16 and 64 x 1 by 5-25 %.
- No worker spans two memory domains. A single process, too, keeps to one
  domain, which was 1.3-5.8x faster than 64 threads over the whole node.
- No more workers start than there are pending work units, or than fit in
  memory when each worker keeps the same safety margin a single process keeps.
- Small jobs run in one process, where starting workers would not pay off.

A worker also uses memory its plan does not count: the Python and PyTorch
runtime, and building the distillation kernels at the start of each work unit.
The system backs the plan's buffers with memory only as they are first
written, so in a worker's first work unit the kernels are built while most of
the buffers are not yet in memory, and in every later one on top of all of
them. When a rough allowance for this (``ROUGH_EXTRA_BYTES``) leaves too little
room for all the workers, fewer workers first run one work unit each, with
their buffers written before it, and measure what they really use; the rest of
the job then runs with as many workers as that measurement allows, never more
than the rule above.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

from concord_lqcd.backend.runtime.resource_planning import MemorySafetyMargin


# One-core costs on a Xeon Platinum 8358: about 0.2 ms of Python
# and PyTorch work per plan step, and 8e9 cost-report FLOP/s (32 real
# GFLOP/s; the report counts a complex multiply-add as 2 FLOPs). Plans differ
# by about 100x in FLOPs per step, so the choice does not hinge on these values.
STEP_SECONDS = 2e-4
CORE_FLOPS_PER_SECOND = 8e9
THREADS_FOR_ARITHMETIC = 4
# Below this much estimated one-core work, a job runs in one process.
MINIMUM_PARALLEL_SECONDS = 60.0
# Memory a worker uses beyond its plan's buffers from its second work unit
# on, on a 64-core node: the runtime (about 0.3-0.8 GiB once forked) and
# building the distillation kernels, which the kernel staging limit does not
# bound: 4.3 GiB for the N pi plan at N_v = 48 (16.3 GiB peak, 12.0 GiB of
# buffers), 1.6-3.3 GiB for plans at N_v = 32.
# With GROWTH_FRACTION it covers the N pi case. Only decides whether, and with
# how many workers, memory is measured first.
ROUGH_EXTRA_BYTES = 3 * 1024**3
# Allowance for what one measured work unit does not show: the caches and the
# next unit's prefetch, which a worker plans only when it holds more than one
# unit (0.1 GiB for the N pi plan at N_v = 48), and differences between workers.
GROWTH_FRACTION = 0.10
# Free memory kept for the parent process, which writes the results, and the system.
NODE_RESERVE_FRACTION = 0.05
NODE_RESERVE_MINIMUM_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class WorkerSplit:
    """Workers and threads per worker; ``threads`` None leaves the default.

    ``measure_with`` > 0: that many workers first run one work unit each to
    measure their memory, and the split is chosen again from the measurement.
    """

    workers: int
    threads: int | None
    automatic: bool
    reason: str
    measure_with: int = 0


def one_core_seconds(steps: int, flops: int) -> float:
    """Estimated seconds for one work unit on one core."""

    return steps * STEP_SECONDS + flops / CORE_FLOPS_PER_SECOND


def workers_in_memory(
    peak_bytes: int, available_bytes: int, margin: MemorySafetyMargin, most: int
) -> int:
    """Most workers (up to ``most``) whose equal memory shares each keep
    ``margin`` and still hold ``peak_bytes``; 0 if not even one fits."""

    for count in range(most, 0, -1):
        share = available_bytes // count
        if share - margin.reserve(share) >= peak_bytes:
            return count
    return 0


def workers_for_peak(peak_bytes: int, available_bytes: int, most: int) -> int:
    """Most workers (up to ``most``) that each fit ``peak_bytes`` of their own
    memory plus ``GROWTH_FRACTION`` in the free memory, less the node reserve."""

    reserve = max(NODE_RESERVE_MINIMUM_BYTES, ceil(available_bytes * NODE_RESERVE_FRACTION))
    usable = available_bytes - reserve
    need = max(1, ceil(peak_bytes * (1 + GROWTH_FRACTION)))
    return max(0, min(most, usable // need))


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def auto_worker_split(
    *,
    threads: int | Literal["auto"],
    pending: int,
    steps: int,
    flops: int,
    peak_bytes: int,
    available_bytes: int | None,
    margin: MemorySafetyMargin,
    cores: int,
    domain_cores: int,
    can_fork: bool,
    measured_peak_bytes: int | None = None,
) -> WorkerSplit:
    """Workers and threads for ``pending`` work units of one plan (``run.workers: auto``).

    ``cores`` are the usable physical cores, ``domain_cores`` the fewest of
    them in one memory domain, and ``can_fork`` is false when the run is not
    on this host's CPU or forked workers are unavailable.
    ``measured_peak_bytes`` is the most own memory a worker reached in one
    work unit with its buffers written first; without it, the split may ask
    for that measurement first.
    """

    fixed_threads = threads if threads != "auto" else None
    if not can_fork:
        return WorkerSplit(1, fixed_threads, True, "auto: one process (not a CPU run on Linux)")
    one_process_threads = fixed_threads or max(1, min(cores, domain_cores))
    one_core = one_core_seconds(steps, flops)
    work_seconds = pending * one_core
    if pending < 2 or cores < 2 or work_seconds < MINIMUM_PARALLEL_SECONDS:
        return WorkerSplit(
            1,
            one_process_threads,
            True,
            f"auto: one process ({_plural(pending, 'pending work unit')}, "
            f"about {work_seconds:.0f} s of one-core work)",
        )

    step_share = steps * STEP_SECONDS / one_core if one_core > 0 else 1.0
    if fixed_threads is not None:
        per_worker = fixed_threads
        why = "threads set in the job"
    elif step_share >= 0.5:
        per_worker = 1
        why = f"per-step cost is {step_share:.0%} of the one-core time"
    else:
        per_worker = min(THREADS_FOR_ARITHMETIC, max(1, domain_cores))
        why = f"arithmetic is {1 - step_share:.0%} of the one-core time"
    by_cores = max(1, cores // per_worker)
    count = min(pending, by_cores)
    limits = [f"{by_cores} by cores", f"{pending} pending"]
    if available_bytes is not None:
        by_memory = workers_in_memory(peak_bytes, available_bytes, margin, by_cores)
        limits.append(f"{by_memory} by memory")
        count = min(count, by_memory)
        if measured_peak_bytes is not None:
            by_measured = workers_for_peak(measured_peak_bytes, available_bytes, by_cores)
            limits.append(f"{by_measured} by measured memory, {_gib(measured_peak_bytes)} each")
            count = min(count, by_measured)
    if count <= 1:
        return WorkerSplit(
            1,
            one_process_threads,
            True,
            f"auto: one process ({', '.join(limits)})",
        )
    if fixed_threads is None and count < by_cores:
        # Fewer workers than the cores allow: give them the idle cores.
        per_worker = max(per_worker, min(domain_cores, cores // count))
    reason = (
        f"auto: {_plural(per_worker, 'thread')} per worker ({why}); "
        f"{count} workers ({', '.join(limits)})"
    )
    measure_with = 0
    if available_bytes is not None and measured_peak_bytes is None:
        rough = workers_for_peak(peak_bytes + ROUGH_EXTRA_BYTES, available_bytes, count)
        if rough < count:
            # The rough allowance does not fit them all: measure before starting the rest.
            measure_with = max(1, rough)
            reason += (
                f"; memory measured first by {_plural(measure_with, 'worker')} "
                f"({rough} fit with {_gib(ROUGH_EXTRA_BYTES)} each beyond the plan's buffers)"
            )
    return WorkerSplit(count, per_worker, True, reason, measure_with)


__all__ = [
    "CORE_FLOPS_PER_SECOND",
    "GROWTH_FRACTION",
    "MINIMUM_PARALLEL_SECONDS",
    "NODE_RESERVE_FRACTION",
    "NODE_RESERVE_MINIMUM_BYTES",
    "ROUGH_EXTRA_BYTES",
    "STEP_SECONDS",
    "THREADS_FOR_ARITHMETIC",
    "WorkerSplit",
    "auto_worker_split",
    "one_core_seconds",
    "workers_for_peak",
    "workers_in_memory",
]
