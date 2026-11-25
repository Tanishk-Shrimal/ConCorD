"""Deterministic operation tiling for host-staged accelerator execution.

The frozen physical plan describes complete logical values in one abstract
memory space.  When that pool fits host RAM but not accelerator memory, this
module derives a second, runtime-only plan: complete values remain host-resident
while contraction operands and results are transferred in bounded index tiles.

Only LapH-mode indices are eligible for automatic tiling in this first stage.
Spin dimensions remain intact.  Tiling a contracted index creates partial
results that the numerical executor must sum; tiling an output index creates
disjoint output slices.  No symbolic identity, CSE decision, coefficient, or
contraction topology is changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import ceil, prod
from types import MappingProxyType
from typing import Mapping

from concord_lqcd.backend.planner.physical import (
    ContractOperation,
    PhysicalExecutionPlan,
    WickTermProductOperation,
)


TileableOperation = ContractOperation | WickTermProductOperation
_MAX_TILE_SEARCH_STATES = 100_000


class TilingPlanningError(MemoryError, ValueError):
    """Raised when no legal LapH-mode tiling satisfies a device budget."""


@dataclass(frozen=True)
class OperationTilePlan:
    """One exact blocked execution scheme for an atomic physical operation."""

    node_id: str
    index_sizes: tuple[tuple[str, int], ...]
    index_spaces: tuple[tuple[str, str], ...]
    tile_sizes: tuple[tuple[str, int], ...]
    contracted_indices: tuple[str, ...]
    output_indices: tuple[str, ...]
    input_tile_bytes: tuple[int, ...]
    output_tile_bytes: int
    full_device_bytes: int
    tiled_device_bytes: int
    output_tile_count: int
    reduction_tile_count: int
    contraction_calls: int

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("operation tile plan requires a non-empty node id")
        size_map = dict(self.index_sizes)
        space_map = dict(self.index_spaces)
        tile_map = dict(self.tile_sizes)
        if len(size_map) != len(self.index_sizes):
            raise ValueError("operation tile plan contains duplicate index sizes")
        if len(space_map) != len(self.index_spaces):
            raise ValueError("operation tile plan contains duplicate index spaces")
        if len(tile_map) != len(self.tile_sizes):
            raise ValueError("operation tile plan contains duplicate tile sizes")
        if set(size_map) != set(space_map) or set(size_map) != set(tile_map):
            raise ValueError("operation tile-plan index metadata is inconsistent")
        if any(not label for label in size_map):
            raise ValueError("operation tile-plan indices must be non-empty")
        if any(size <= 0 for size in size_map.values()):
            raise ValueError("operation tile-plan index sizes must be positive")
        if any(space not in ("mode", "spin") for space in space_map.values()):
            raise ValueError("operation tile plan contains an unsupported index space")
        if any(
            tile <= 0 or tile > size_map[index]
            for index, tile in tile_map.items()
        ):
            raise ValueError("operation tile sizes must lie within their index extents")
        present = set(size_map)
        contracted = set(self.contracted_indices)
        outputs = set(self.output_indices)
        if contracted & outputs or contracted | outputs != present:
            raise ValueError("operation tile plan has inconsistent index roles")
        if len(contracted) != len(self.contracted_indices):
            raise ValueError("operation tile plan repeats a contracted index")
        if len(outputs) != len(self.output_indices):
            raise ValueError("operation tile plan repeats an output index")
        if any(value < 0 for value in self.input_tile_bytes):
            raise ValueError("operation input tile byte counts must be non-negative")
        for field_name in (
            "output_tile_bytes",
            "full_device_bytes",
            "tiled_device_bytes",
            "output_tile_count",
            "reduction_tile_count",
            "contraction_calls",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.contraction_calls != (
            self.output_tile_count * self.reduction_tile_count
        ):
            raise ValueError("operation contraction-call count is inconsistent")

    @property
    def size_by_index(self) -> Mapping[str, int]:
        return dict(self.index_sizes)

    @property
    def space_by_index(self) -> Mapping[str, str]:
        return dict(self.index_spaces)

    @property
    def tile_by_index(self) -> Mapping[str, int]:
        return dict(self.tile_sizes)

    @property
    def tiled_indices(self) -> tuple[str, ...]:
        sizes = self.size_by_index
        return tuple(
            index
            for index, tile_size in self.tile_sizes
            if tile_size < sizes[index]
        )

    @property
    def is_tiled(self) -> bool:
        return bool(self.tiled_indices)


@dataclass(frozen=True)
class TiledExecutionPlan:
    """Runtime-only tile plan for a host-resident physical buffer pool."""

    source_plan: PhysicalExecutionPlan
    device_memory_limit_bytes: int
    operations: Mapping[str, OperationTilePlan]
    peak_device_bytes: int
    full_peak_device_bytes: int
    tiled_operation_count: int
    total_contraction_calls: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operations",
            MappingProxyType(dict(self.operations)),
        )
        if not isinstance(self.source_plan, PhysicalExecutionPlan):
            raise TypeError("tiled execution requires a PhysicalExecutionPlan")
        self.source_plan.validate()
        for field_name in (
            "device_memory_limit_bytes",
            "peak_device_bytes",
            "full_peak_device_bytes",
            "total_contraction_calls",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            not isinstance(self.tiled_operation_count, int)
            or isinstance(self.tiled_operation_count, bool)
            or self.tiled_operation_count < 0
        ):
            raise ValueError("tiled_operation_count must be a non-negative integer")
        source_operations = {
            step.operation.node_id: step.operation
            for step in self.source_plan.steps
            if isinstance(step.operation, (ContractOperation, WickTermProductOperation))
            and step.operation.inputs
        }
        if set(self.operations) != set(source_operations):
            raise ValueError("tiled execution operation map is incomplete")
        for node_id, operation in self.operations.items():
            if operation.node_id != node_id:
                raise ValueError("operation tile plan is stored under the wrong node id")
            source_operation = source_operations[node_id]
            index_sizes, index_spaces, contracted = _operation_index_metadata(
                self.source_plan,
                source_operation,
            )
            expected_operation = _make_operation_plan(
                self.source_plan,
                source_operation,
                index_sizes,
                index_spaces,
                contracted,
                operation.tile_by_index,
            )
            if operation != expected_operation:
                raise ValueError(
                    f"operation tile plan for {node_id!r} disagrees with its "
                    "physical operation"
                )
            if operation.tiled_device_bytes > self.device_memory_limit_bytes:
                raise ValueError("operation tile plan exceeds its device-memory limit")
        expected_peak = max(
            (operation.tiled_device_bytes for operation in self.operations.values()),
            default=1,
        )
        expected_full_peak = max(
            (operation.full_device_bytes for operation in self.operations.values()),
            default=1,
        )
        if self.peak_device_bytes != expected_peak:
            raise ValueError("tiled execution peak byte count is inconsistent")
        if self.full_peak_device_bytes != expected_full_peak:
            raise ValueError("tiled execution full-peak byte count is inconsistent")
        if self.tiled_operation_count != sum(
            operation.is_tiled for operation in self.operations.values()
        ):
            raise ValueError("tiled execution operation count is inconsistent")
        if self.total_contraction_calls != sum(
            operation.contraction_calls for operation in self.operations.values()
        ):
            raise ValueError("tiled execution contraction-call count is inconsistent")

    @property
    def uses_index_tiling(self) -> bool:
        return self.tiled_operation_count > 0


def _element_bytes(plan: PhysicalExecutionPlan, value_id: str) -> int:
    binding = plan.binding(value_id)
    elements = prod(binding.shape) if binding.shape else 1
    if binding.size_bytes % elements:
        raise ValueError(
            f"physical value {value_id!r} has a non-integral element byte size"
        )
    return binding.size_bytes // elements


def _operation_index_metadata(
    plan: PhysicalExecutionPlan,
    operation: TileableOperation,
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
]:
    sizes: dict[str, int] = {}
    spaces: dict[str, str] = {}
    order: list[str] = []

    def record(index: str, size: int, space: str) -> None:
        if index not in sizes:
            order.append(index)
            sizes[index] = size
            spaces[index] = space
            return
        if sizes[index] != size or spaces[index] != space:
            raise ValueError(
                f"operation {operation.node_id!r} binds index {index!r} to "
                "incompatible dimensions or spaces"
            )

    for use in operation.inputs:
        binding = plan.binding(use.value_id)
        ports = tuple(item.port for item in use.bindings)
        if ports != binding.axis_labels:
            raise ValueError(
                f"operation {operation.node_id!r} has stale input-port metadata"
            )
        for item, size, space in zip(
            use.bindings,
            binding.shape,
            binding.axis_spaces,
        ):
            record(item.index, size, space)

    output = plan.binding(operation.node_id)
    if len(operation.output_indices) != len(output.shape):
        raise ValueError(
            f"operation {operation.node_id!r} output rank disagrees with its binding"
        )
    for index, size, space in zip(
        operation.output_indices,
        output.shape,
        output.axis_spaces,
    ):
        record(index, size, space)

    outputs = set(operation.output_indices)
    if isinstance(operation, ContractOperation):
        contracted = operation.contracted_indices
    else:
        contracted = tuple(index for index in order if index not in outputs)
    if set(contracted) & outputs or set(contracted) | outputs != set(order):
        raise ValueError(
            f"operation {operation.node_id!r} does not classify every local index"
        )
    return (
        tuple((index, sizes[index]) for index in order),
        tuple((index, spaces[index]) for index in order),
        tuple(contracted),
    )


def _tile_count(size: int, tile_size: int) -> int:
    return ceil(size / tile_size)


def _operation_bytes(
    plan: PhysicalExecutionPlan,
    operation: TileableOperation,
    index_sizes: Mapping[str, int],
    tile_sizes: Mapping[str, int],
    contracted_indices: tuple[str, ...],
) -> tuple[tuple[int, ...], int, int, int, int]:
    input_bytes: list[int] = []
    for use in operation.inputs:
        elements = prod(
            tile_sizes[item.index]
            for item in use.bindings
        ) if use.bindings else 1
        input_bytes.append(elements * _element_bytes(plan, use.value_id))

    output_elements = prod(
        tile_sizes[index]
        for index in operation.output_indices
    ) if operation.output_indices else 1
    output_bytes = output_elements * _element_bytes(plan, operation.node_id)
    output_tiles = prod(
        _tile_count(index_sizes[index], tile_sizes[index])
        for index in operation.output_indices
    ) if operation.output_indices else 1
    reduction_tiles = prod(
        _tile_count(index_sizes[index], tile_sizes[index])
        for index in contracted_indices
    ) if contracted_indices else 1

    # Every tile input is copied to the accelerator.  The contraction result is
    # one output tile; when a contracted index is tiled, one accumulator and one
    # newly produced partial tile coexist after the first reduction tile.
    output_copies = 2 if reduction_tiles > 1 else 1
    peak = sum(input_bytes) + output_copies * output_bytes
    return tuple(input_bytes), output_bytes, output_tiles, reduction_tiles, peak


def _make_operation_plan(
    plan: PhysicalExecutionPlan,
    operation: TileableOperation,
    index_sizes: tuple[tuple[str, int], ...],
    index_spaces: tuple[tuple[str, str], ...],
    contracted_indices: tuple[str, ...],
    tile_sizes: Mapping[str, int],
) -> OperationTilePlan:
    sizes = dict(index_sizes)
    full_tiles = dict(sizes)
    full_input, full_output, _, _, full_peak = _operation_bytes(
        plan,
        operation,
        sizes,
        full_tiles,
        contracted_indices,
    )
    del full_input, full_output
    input_bytes, output_bytes, output_tiles, reduction_tiles, tiled_peak = (
        _operation_bytes(
            plan,
            operation,
            sizes,
            tile_sizes,
            contracted_indices,
        )
    )
    return OperationTilePlan(
        node_id=operation.node_id,
        index_sizes=index_sizes,
        index_spaces=index_spaces,
        tile_sizes=tuple((index, tile_sizes[index]) for index, _ in index_sizes),
        contracted_indices=contracted_indices,
        output_indices=operation.output_indices,
        input_tile_bytes=input_bytes,
        output_tile_bytes=output_bytes,
        full_device_bytes=full_peak,
        tiled_device_bytes=tiled_peak,
        output_tile_count=output_tiles,
        reduction_tile_count=reduction_tiles,
        contraction_calls=output_tiles * reduction_tiles,
    )


def _candidate_tiles(size: int) -> tuple[tuple[int, int], ...]:
    """Return ``(number_of_tiles, smallest_tile_size)`` choices.

    The smallest tile size for a fixed invocation count gives the lowest peak.
    Once a feasible invocation count is selected, the tile is enlarged again
    without changing that count.
    """

    choices: dict[int, int] = {}
    for requested_count in range(1, size + 1):
        tile_size = ceil(size / requested_count)
        actual_count = _tile_count(size, tile_size)
        previous = choices.get(actual_count)
        if previous is None or tile_size < previous:
            choices[actual_count] = tile_size
    return tuple(sorted(choices.items()))


def _choose_operation_tiles(
    plan: PhysicalExecutionPlan,
    operation: TileableOperation,
    device_memory_limit_bytes: int,
) -> OperationTilePlan:
    index_sizes, index_spaces, contracted = _operation_index_metadata(plan, operation)
    sizes = dict(index_sizes)
    spaces = dict(index_spaces)
    full = dict(sizes)
    full_plan = _make_operation_plan(
        plan,
        operation,
        index_sizes,
        index_spaces,
        contracted,
        full,
    )
    if full_plan.tiled_device_bytes <= device_memory_limit_bytes:
        return full_plan

    mode_indices = tuple(index for index, _ in index_sizes if spaces[index] == "mode")
    if not mode_indices:
        raise TilingPlanningError(
            f"operation {operation.node_id!r} requires {full_plan.full_device_bytes} "
            "accelerator bytes, exceeds the tile budget, and exposes no LapH-mode "
            "index that can be tiled"
        )

    options = tuple(_candidate_tiles(sizes[index]) for index in mode_indices)
    initial = tuple(0 for _ in mode_indices)
    queue: list[tuple[int, int, tuple[int, ...]]] = [(1, 0, initial)]
    visited = {initial}
    selected: dict[str, int] | None = None
    searched = 0

    while queue and searched < _MAX_TILE_SEARCH_STATES:
        _, _, state = heappop(queue)
        searched += 1
        tiles = dict(full)
        for axis, position in enumerate(state):
            _, tile_size = options[axis][position]
            tiles[mode_indices[axis]] = tile_size
        candidate = _make_operation_plan(
            plan,
            operation,
            index_sizes,
            index_spaces,
            contracted,
            tiles,
        )
        if candidate.tiled_device_bytes <= device_memory_limit_bytes:
            selected = tiles
            break
        for axis, position in enumerate(state):
            if position + 1 >= len(options[axis]):
                continue
            neighbor = list(state)
            neighbor[axis] += 1
            neighbor_state = tuple(neighbor)
            if neighbor_state in visited:
                continue
            visited.add(neighbor_state)
            calls = prod(
                options[item_axis][neighbor_state[item_axis]][0]
                for item_axis in range(len(mode_indices))
            )
            heappush(queue, (calls, sum(neighbor_state), neighbor_state))

    if selected is None:
        # Deterministic bounded fallback: reduce the most effective remaining
        # mode index to one until the operation fits or no legal reduction is
        # left.  This preserves a hard search-time bound for high-rank networks.
        selected = dict(full)
        while True:
            current = _make_operation_plan(
                plan,
                operation,
                index_sizes,
                index_spaces,
                contracted,
                selected,
            )
            if current.tiled_device_bytes <= device_memory_limit_bytes:
                break
            reductions: list[tuple[int, int, str, dict[str, int]]] = []
            for index in mode_indices:
                if selected[index] == 1:
                    continue
                candidate_tiles = dict(selected)
                candidate_tiles[index] = 1
                candidate = _make_operation_plan(
                    plan,
                    operation,
                    index_sizes,
                    index_spaces,
                    contracted,
                    candidate_tiles,
                )
                reductions.append(
                    (
                        candidate.tiled_device_bytes,
                        candidate.contraction_calls,
                        index,
                        candidate_tiles,
                    )
                )
            if not reductions:
                raise TilingPlanningError(
                    f"operation {operation.node_id!r} still requires "
                    f"{current.tiled_device_bytes} accelerator bytes with every "
                    "LapH-mode tile reduced to one; safe limit is "
                    f"{device_memory_limit_bytes} bytes"
                )
            selected = min(reductions, key=lambda item: item[:3])[3]

    # Enlarge each selected tile as far as possible without changing its number
    # of runtime contractions or violating the memory bound.
    for index in mode_indices:
        current_size = selected[index]
        count = _tile_count(sizes[index], current_size)
        if count == 1:
            continue
        largest_same_count = min(sizes[index], (sizes[index] - 1) // (count - 1))
        for candidate_size in range(largest_same_count, current_size, -1):
            candidate_tiles = dict(selected)
            candidate_tiles[index] = candidate_size
            candidate = _make_operation_plan(
                plan,
                operation,
                index_sizes,
                index_spaces,
                contracted,
                candidate_tiles,
            )
            if candidate.tiled_device_bytes <= device_memory_limit_bytes:
                selected = candidate_tiles
                break

    result = _make_operation_plan(
        plan,
        operation,
        index_sizes,
        index_spaces,
        contracted,
        selected,
    )
    if result.tiled_device_bytes > device_memory_limit_bytes:  # pragma: no cover
        raise RuntimeError("internal tile selection exceeded its memory limit")
    return result


def plan_tiled_execution(
    plan: PhysicalExecutionPlan,
    *,
    device_memory_limit_bytes: int,
) -> TiledExecutionPlan:
    """Plan exact host-staged contraction tiles under a hard device limit.

    This function models accelerator-resident operand tiles plus the operation
    result/accumulator tiles.  The caller must separately prove that the full
    physical plan fits the host-memory domain.  Backend-library workspace is
    intentionally outside this exact model and must remain covered by the
    hardware policy's safety reserve.
    """

    if not isinstance(plan, PhysicalExecutionPlan):
        raise TypeError("plan must be a PhysicalExecutionPlan")
    if (
        not isinstance(device_memory_limit_bytes, int)
        or isinstance(device_memory_limit_bytes, bool)
        or device_memory_limit_bytes <= 0
    ):
        raise ValueError("device_memory_limit_bytes must be a positive integer")
    plan.validate()

    operations: dict[str, OperationTilePlan] = {}
    for step in plan.steps:
        operation = step.operation
        if not isinstance(operation, (ContractOperation, WickTermProductOperation)):
            continue
        if not operation.inputs:
            continue
        operations[operation.node_id] = _choose_operation_tiles(
            plan,
            operation,
            device_memory_limit_bytes,
        )

    if not operations:
        raise TilingPlanningError(
            "physical plan contains no tensor contractions to stage on an accelerator"
        )
    result = TiledExecutionPlan(
        source_plan=plan,
        device_memory_limit_bytes=device_memory_limit_bytes,
        operations=operations,
        peak_device_bytes=max(item.tiled_device_bytes for item in operations.values()),
        full_peak_device_bytes=max(item.full_device_bytes for item in operations.values()),
        tiled_operation_count=sum(item.is_tiled for item in operations.values()),
        total_contraction_calls=sum(item.contraction_calls for item in operations.values()),
    )
    return result


__all__ = [
    "TilingPlanningError",
    "OperationTilePlan",
    "TiledExecutionPlan",
    "plan_tiled_execution",
]
