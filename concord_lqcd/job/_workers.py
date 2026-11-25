"""Forked CPU worker processes that share one loaded plan.

The parent loads and validates the plan once, then forks. Each child inherits
the plan through copy-on-write memory instead of reading and decoding it again,
runs its own share of the work with its own PyTorch thread count, and sends
every finished result back over a pipe. Only the parent touches the output file.

GNU OpenMP cannot be used in a forked child once the parent has run a
multi-threaded region. Each child therefore runs one small multi-threaded
operation before it reports ready; a parent that never hears back stops all
children and raises ``WorkerStartError`` instead of hanging.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import multiprocessing
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import pickle
from time import monotonic
import traceback
from typing import Any, Callable, Sequence
import warnings

import torch


READY_TIMEOUT_SECONDS = 120.0


class WorkerStartError(RuntimeError):
    """Raised when forked workers cannot start their thread pools."""


class WorkerFailure(RuntimeError):
    """Raised in the parent when a worker fails or exits early."""

    def __init__(self, index: int, error: BaseException, details: str = "") -> None:
        super().__init__(f"worker {index} failed: {error}")
        self.index = index
        self.error = error
        self.details = details


@dataclass(frozen=True)
class WorkerAssignment:
    """One worker's share: its threads, its CPUs (None = not pinned) and its items."""

    index: int
    threads: int
    cpus: tuple[int, ...] | None
    items: tuple[Any, ...]


def split_contiguous(items: Sequence[Any], workers: int) -> tuple[tuple[Any, ...], ...]:
    """Split items into at most ``workers`` contiguous, nearly equal, non-empty chunks."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    count = min(workers, len(items))
    chunks = []
    start = 0
    for index in range(count):
        stop = start + len(items) // count + (1 if index < len(items) % count else 0)
        chunks.append(tuple(items[start:stop]))
        start = stop
    return tuple(chunks)


def usable_cpus() -> tuple[int, ...]:
    """CPUs this process may run on, in increasing order."""

    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return tuple(range(os.cpu_count() or 1))


def _read_cpu_list(path: Path) -> tuple[int, ...]:
    cpus: list[int] = []
    for part in path.read_text().strip().split(","):
        if part:
            first, _, last = part.partition("-")
            cpus.extend(range(int(first), int(last or first) + 1))
    return tuple(cpus)


def cpu_topology() -> dict[int, tuple[int, tuple[int, int]]]:
    """Map each CPU to (memory domain, physical core); empty if Linux sysfs is unavailable."""

    try:
        domain = {
            cpu: int(path.parent.name.removeprefix("node"))
            for path in Path("/sys/devices/system/node").glob("node[0-9]*/cpulist")
            for cpu in _read_cpu_list(path)
        }
        topology = {}
        for cpu in usable_cpus():
            base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            core = (int((base / "physical_package_id").read_text()), int((base / "core_id").read_text()))
            topology[cpu] = (domain.get(cpu, 0), core)
        return topology
    except (OSError, ValueError):
        return {}


def _placement_order() -> tuple[tuple[int, ...], int]:
    """Usable CPUs ordered by memory domain, one per physical core first, then
    the hyperthread siblings; also the number of physical cores."""

    cpus = usable_cpus()
    topology = cpu_topology()
    if set(topology) != set(cpus):
        return cpus, len(cpus)
    first_of_core: dict[tuple[int, int], int] = {}
    for cpu in cpus:
        first_of_core.setdefault(topology[cpu][1], cpu)
    primary = sorted(first_of_core.values(), key=lambda cpu: (topology[cpu][0], cpu))
    siblings = sorted(set(cpus) - set(primary), key=lambda cpu: (topology[cpu][0], cpu))
    return (*primary, *siblings), len(primary)


def node_shape() -> tuple[int, int]:
    """Usable physical cores, and the fewest of them in one memory domain."""

    cpus = usable_cpus()
    topology = cpu_topology()
    if set(topology) != set(cpus):
        return len(cpus), len(cpus)
    cores_by_domain: dict[int, set[tuple[int, int]]] = {}
    for domain, core in topology.values():
        cores_by_domain.setdefault(domain, set()).add(core)
    cores = len({core for _, core in topology.values()})
    return cores, min(len(domain_cores) for domain_cores in cores_by_domain.values())


def worker_cpu_sets(workers: int, threads: int) -> tuple[tuple[int, ...] | None, ...]:
    """Give each worker its own block of ``threads`` CPUs, or none if they do not fit.

    Blocks are cut from the CPUs ordered by memory domain, so a worker's threads
    sit next to its memory, and use whole physical cores before hyperthread
    siblings. When the workers fit on the physical cores, their blocks are
    spread evenly over them, so every domain's cache and memory bandwidth is
    shared instead of crowding the first domains.
    """

    order, cores = _placement_order()
    if workers * threads > len(order):
        return (None,) * workers
    if workers * threads <= cores:
        starts = [index * cores // workers for index in range(workers)]
    else:
        starts = [index * threads for index in range(workers)]
    return tuple(tuple(sorted(order[start:start + threads])) for start in starts)


def _portable(error: BaseException) -> BaseException:
    try:
        pickle.dumps(error)
    except Exception:
        return RuntimeError(f"{type(error).__name__}: {error}")
    return error


def _start_thread_pool(threads: int) -> None:
    # Large enough for PyTorch's parallel_for to open an OpenMP region.
    if threads > 1:
        torch.ones(1 << 20).sum()


def _child_main(
    assignment: WorkerAssignment,
    connection: Connection,
    work: Callable[[WorkerAssignment, Callable[[Any], None]], Any],
) -> None:
    try:
        if assignment.cpus is not None:
            os.sched_setaffinity(0, assignment.cpus)
        torch.set_num_threads(assignment.threads)
        _start_thread_pool(assignment.threads)
        connection.send(("ready",))
        report = work(assignment, lambda message: connection.send(("message", message)))
        connection.send(("done", report))
    except BaseException as error:
        try:
            connection.send(("error", _portable(error), traceback.format_exc()))
        except Exception:
            pass
    finally:
        connection.close()


def run_workers(
    assignments: Sequence[WorkerAssignment],
    work: Callable[[WorkerAssignment, Callable[[Any], None]], Any],
    handle: Callable[[int, Any], None],
    *,
    ready_timeout: float | None = None,
) -> tuple[Any, ...]:
    """Run ``work(assignment, send)`` in one forked process per assignment.

    Every message a worker sends is passed to ``handle(index, message)`` in this
    process as it arrives. Returns the value each ``work`` call returned, in
    assignment order. The first failure stops all workers.
    """

    if ready_timeout is None:
        ready_timeout = READY_TIMEOUT_SECONDS
    context = multiprocessing.get_context("fork")
    processes: dict[int, multiprocessing.process.BaseProcess] = {}
    connections: dict[Connection, int] = {}
    # Frozen objects are skipped by the children's garbage collector, which
    # would otherwise write to (and so copy) every inherited plan object.
    gc.freeze()
    try:
        with warnings.catch_warnings():
            # The parent has idle library threads (OpenBLAS); readiness is checked below.
            warnings.filterwarnings(
                "ignore", message=r".*fork\(\) may lead to deadlocks", category=DeprecationWarning
            )
            for assignment in assignments:
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(
                    target=_child_main,
                    args=(assignment, writer, work),
                    name=f"concord-worker-{assignment.index}",
                    daemon=True,
                )
                process.start()
                writer.close()
                processes[assignment.index] = process
                connections[reader] = assignment.index
    except BaseException:
        for process in processes.values():
            process.kill()
            process.join()
        for reader in connections:
            reader.close()
        raise
    finally:
        gc.unfreeze()

    reports: dict[int, Any] = {}
    ready: set[int] = set()
    started = monotonic()
    try:
        while connections:
            starting = len(ready) < len(assignments)
            timeout = max(0.0, ready_timeout - (monotonic() - started)) if starting else None
            for reader in wait(list(connections), timeout=timeout):
                index = connections[reader]
                try:
                    kind, *payload = reader.recv()
                except EOFError:
                    del connections[reader]
                    reader.close()
                    if index not in reports:
                        processes[index].join(timeout=10)
                        raise WorkerFailure(
                            index,
                            RuntimeError(
                                "exited before finishing "
                                f"(exit code {processes[index].exitcode})"
                            ),
                        )
                    continue
                if kind == "ready":
                    ready.add(index)
                elif kind == "message":
                    handle(index, payload[0])
                elif kind == "done":
                    reports[index] = payload[0]
                else:
                    raise WorkerFailure(index, payload[0], payload[1])
            if len(ready) < len(assignments) and monotonic() - started >= ready_timeout:
                raise WorkerStartError(
                    f"{len(assignments) - len(ready)} of {len(assignments)} worker processes "
                    f"did not start within {ready_timeout:.0f} s. This happens when the parent "
                    "process ran multi-threaded PyTorch work before forking; start the job "
                    "with 'concord run', call run_job before other PyTorch work, or use "
                    "threads_per_worker: 1"
                )
    except BaseException:
        for process in processes.values():
            if process.is_alive():
                process.kill()
        raise
    finally:
        for process in processes.values():
            process.join()
        for reader in connections:
            reader.close()
    return tuple(reports[assignment.index] for assignment in assignments)


__all__ = [
    "READY_TIMEOUT_SECONDS",
    "WorkerAssignment",
    "WorkerFailure",
    "WorkerStartError",
    "cpu_topology",
    "node_shape",
    "run_workers",
    "split_contiguous",
    "usable_cpus",
    "worker_cpu_sets",
]
