"""Contraction-order optimization for globally deduplicated tensor components.

This layer consumes the exact connected ``ComponentComputation`` objects from
``global_cse`` and chooses an algebraic contraction order for each unique
component.  It deliberately stops at a contraction tree/sequence:

* no execution scheduling;
* no device placement;
* no buffer allocation or spilling;
* no runtime tensor loading;
* no cross-component partial-intermediate CSE yet.

Path search is delegated to ``opt_einsum``.  ConCorD retains its own
backend-independent plan representation so a different path optimizer (for
example cotengra) can be plugged in later without changing the compiler IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral
from typing import Literal, Mapping, Sequence

import opt_einsum as oe

from concord_lqcd.compiler.contraction_graph import TensorLegBinding
from concord_lqcd.compiler.global_cse import (
    ComponentComputation,
    ComponentTensorUse,
    GlobalComputationDAG,
)


IndexSpace = Literal["mode", "spin"]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean or rounded value.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


@dataclass(frozen=True)
class ContractionSizeModel:
    """Numerical index dimensions used only for contraction-path search.

    Distillation mode indices have one common size within the current exact
    distillation job, while Dirac spin indices have size four.  These sizes are
    optimization inputs rather than physics/resource identity, so changing them
    never changes the contraction graph itself.
    """

    mode_size: int
    spin_size: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode_size", _positive_integer(self.mode_size, "mode_size"))
        object.__setattr__(self, "spin_size", _positive_integer(self.spin_size, "spin_size"))

    def size(self, space: IndexSpace) -> int:
        if space == "mode":
            return self.mode_size
        if space == "spin":
            return self.spin_size
        raise ValueError(f"Unsupported contraction index space {space!r}.")


@dataclass(frozen=True)
class ContractionStep:
    """One algebraic contraction/reduction in a component contraction plan.

    ``input_refs`` refer either to component tensor slots (``t0``, ``t1``, ...)
    or to earlier step ids.  ``input_indices`` and ``output_indices`` use the
    canonical component indices (``i0``, ``o0``, ...), not opt_einsum symbols.
    A normal tree step has two inputs; a one-input step is allowed for a trace or
    other unary reduction of a single tensor. Search-memory constraints can also
    make a path optimizer return a multi-input step.
    """

    step_id: str
    input_refs: tuple[str, ...]
    input_indices: tuple[tuple[str, ...], ...]
    contracted_indices: tuple[str, ...]
    output_indices: tuple[str, ...]
    estimated_flops: int
    output_elements: int


@dataclass(frozen=True)
class ComponentContractionPlan:
    """Backend-independent contraction order for one unique connected component."""

    computation_id: str
    optimizer_backend: str
    strategy: str
    steps: tuple[ContractionStep, ...]
    result_ref: str
    output_indices: tuple[str, ...]
    estimated_flops: int
    largest_intermediate_elements: int

    def validate(self, component: ComponentComputation) -> None:
        """Check references and index topology, not certify numerical cost metadata."""
        if component.computation_id != self.computation_id:
            raise ValueError("Contraction plan/component computation ids do not match.")
        slots = tuple(tensor.slot for tensor in component.tensors)
        if len(set(slots)) != len(slots):
            raise ValueError("Contraction component contains duplicate tensor slots.")
        if len(set(component.output_indices)) != len(component.output_indices):
            raise ValueError("Contraction component contains repeated output indices.")

        active: dict[str, tuple[str, ...]] = {
            tensor.slot: tuple(binding.index for binding in tensor.bindings)
            for tensor in component.tensors
        }
        used_step_ids: set[str] = set()

        for step in self.steps:
            if step.step_id in active or step.step_id in used_step_ids:
                raise ValueError(f"Duplicate contraction step id {step.step_id!r}.")
            if not step.input_refs:
                raise ValueError("A contraction step requires at least one input.")
            if len(set(step.input_refs)) != len(step.input_refs):
                raise ValueError("A contraction step contains a repeated input reference.")
            if len(step.input_refs) != len(step.input_indices):
                raise ValueError("Contraction step input refs/indices have inconsistent lengths.")

            for ref, indices in zip(step.input_refs, step.input_indices):
                if ref not in active:
                    raise ValueError(
                        f"Contraction step {step.step_id!r} references unavailable input {ref!r}."
                    )
                if active[ref] != indices:
                    raise ValueError(
                        f"Contraction step {step.step_id!r} has stale index metadata for {ref!r}."
                    )

            input_labels = {index for indices in step.input_indices for index in indices}
            output_labels = set(step.output_indices)
            if len(output_labels) != len(step.output_indices):
                raise ValueError("A contraction step contains repeated output indices.")
            if not output_labels <= input_labels:
                raise ValueError("Contraction step output indices are absent from its inputs.")
            remaining_labels = {
                index for ref, indices in active.items()
                if ref not in step.input_refs
                for index in indices
            }
            needed_labels = input_labels & (remaining_labels | set(component.output_indices))
            if not needed_labels <= output_labels:
                raise ValueError("Contraction step drops indices still needed by the component.")
            removed_labels = input_labels - output_labels
            if (
                len(set(step.contracted_indices)) != len(step.contracted_indices)
                or set(step.contracted_indices) != removed_labels
            ):
                raise ValueError("Contraction step contracted indices do not match its reduction.")

            for ref in step.input_refs:
                del active[ref]
            active[step.step_id] = step.output_indices
            used_step_ids.add(step.step_id)

        if set(active) != {self.result_ref}:
            raise ValueError(
                "Contraction plan must leave exactly one final tensor/result reference."
            )
        if active[self.result_ref] != self.output_indices:
            raise ValueError("Contraction plan final indices do not match its declared output.")
        if self.output_indices != component.output_indices:
            raise ValueError("Contraction plan output indices do not match the component output.")


@dataclass(frozen=True)
class GlobalContractionOrderPlan:
    """Contraction orders for every unique component in a GlobalComputationDAG."""

    source_dag: GlobalComputationDAG
    size_model: ContractionSizeModel
    strategy: str
    memory_limit: int | None
    component_plans: Mapping[str, ComponentContractionPlan]

    @property
    def selected_strategies(self) -> tuple[str, ...]:
        """Concrete component-level optimizers retained by this global plan."""

        return tuple(
            dict.fromkeys(
                plan.strategy for plan in self.component_plans.values()
            )
        )

    @property
    def estimated_unique_flops(self) -> int:
        """FLOPs to evaluate each unique component once, before later partial CSE."""
        return sum(plan.estimated_flops for plan in self.component_plans.values())

    @property
    def largest_intermediate_elements(self) -> int:
        if not self.component_plans:
            return 0
        return max(
            plan.largest_intermediate_elements
            for plan in self.component_plans.values()
        )


def _binding_space(tensor: ComponentTensorUse, binding: TensorLegBinding) -> IndexSpace:
    if tensor.kind == "kernel":
        # Standalone POINT kernels are rejected by contraction-graph lowering;
        # every explicit kernel leg in the current exact-distillation graph is
        # therefore a LapH mode index.
        return "mode"

    if tensor.kind == "spin":
        return "spin"

    if tensor.kind == "propagation":
        if binding.port in ("sink_mode", "source_mode"):
            return "mode"
        if binding.port in ("sink_spin", "source_spin"):
            return "spin"
        raise ValueError(
            f"Unknown propagation tensor port {binding.port!r} in contraction component."
        )

    raise ValueError(f"Unsupported component tensor kind {tensor.kind!r}.")


def infer_component_index_spaces(
    component: ComponentComputation,
) -> Mapping[str, IndexSpace]:
    """Infer and validate the numerical space of every canonical component index."""

    spaces: dict[str, IndexSpace] = {}
    for tensor in component.tensors:
        for binding in tensor.bindings:
            space = _binding_space(tensor, binding)
            previous = spaces.get(binding.index)
            if previous is not None and previous != space:
                raise ValueError(
                    f"Contraction index {binding.index!r} connects incompatible "
                    f"spaces {previous!r} and {space!r}."
                )
            spaces[binding.index] = space

    missing_outputs = set(component.output_indices) - set(spaces)
    if missing_outputs:
        raise ValueError(
            "Component declares output indices that are absent from its tensor bindings: "
            f"{sorted(missing_outputs)}"
        )
    return dict(spaces)


def infer_component_index_sizes(
    component: ComponentComputation,
    size_model: ContractionSizeModel,
) -> Mapping[str, int]:
    spaces = infer_component_index_spaces(component)
    return {index: size_model.size(space) for index, space in spaces.items()}


def _einsum_problem(
    component: ComponentComputation,
    size_model: ContractionSizeModel,
) -> tuple[str, tuple[tuple[int, ...], ...], Mapping[str, str], Mapping[str, int]]:
    if not component.tensors:
        raise ValueError("Cannot optimize an empty tensor component.")

    index_sizes = infer_component_index_sizes(component, size_model)

    ordered_indices: list[str] = []
    seen: set[str] = set()
    for tensor in component.tensors:
        for binding in tensor.bindings:
            if binding.index not in seen:
                seen.add(binding.index)
                ordered_indices.append(binding.index)
    for index in component.output_indices:
        if index not in seen:
            seen.add(index)
            ordered_indices.append(index)

    symbol_for = {
        index: oe.get_symbol(idx)
        for idx, index in enumerate(ordered_indices)
    }
    canonical_for_symbol = {symbol: index for index, symbol in symbol_for.items()}

    inputs: list[str] = []
    shapes: list[tuple[int, ...]] = []
    for tensor in component.tensors:
        inputs.append("".join(symbol_for[binding.index] for binding in tensor.bindings))
        shapes.append(tuple(index_sizes[binding.index] for binding in tensor.bindings))

    output = "".join(symbol_for[index] for index in component.output_indices)
    equation = f"{','.join(inputs)}->{output}"
    symbol_sizes = {
        symbol_for[index]: size
        for index, size in index_sizes.items()
    }
    return equation, tuple(shapes), canonical_for_symbol, symbol_sizes


def _symbols_to_indices(
    symbols: str,
    canonical_for_symbol: Mapping[str, str],
) -> tuple[str, ...]:
    try:
        return tuple(canonical_for_symbol[symbol] for symbol in symbols)
    except KeyError as exc:
        raise RuntimeError(
            "opt_einsum returned an index symbol not present in the component problem."
        ) from exc


def optimize_component_contraction_order(
    component: ComponentComputation,
    size_model: ContractionSizeModel,
    *,
    strategy: str = "auto",
    memory_limit: int | None = None,
) -> ComponentContractionPlan:
    """Choose a contraction order for one unique connected component.

    ``strategy`` is passed directly to ``opt_einsum.contract_path``.  The
    returned plan contains no opt_einsum-specific index names.

    ``memory_limit`` bounds the number of elements of every intermediate,
    including the component output.  Not every opt_einsum optimizer honours
    it (the random-greedy search behind ``auto-hq`` for more than 16 tensors
    ignores it), so the returned path is checked; a violating path is replaced
    by the memory-aware ``greedy`` search, and the plan records ``greedy`` as
    its strategy.  A limit that no path can meet raises ``ValueError``.
    """

    if memory_limit is not None:
        memory_limit = _positive_integer(memory_limit, "memory_limit")

    equation, shapes, canonical_for_symbol, symbol_sizes = _einsum_problem(
        component, size_model
    )

    kwargs: dict[str, object] = {
        "shapes": True,
        "optimize": strategy,
    }
    if memory_limit is not None:
        kwargs["memory_limit"] = memory_limit

    _, path_info = oe.contract_path(equation, *shapes, **kwargs)
    selected_strategy = strategy
    if memory_limit is not None and int(path_info.largest_intermediate) > memory_limit:
        if strategy != "greedy":
            _, path_info = oe.contract_path(
                equation, *shapes, shapes=True, optimize="greedy", memory_limit=memory_limit,
            )
            selected_strategy = "greedy"
        if int(path_info.largest_intermediate) > memory_limit:
            raise ValueError(
                f"contraction memory_limit={memory_limit} elements cannot be met for component "
                f"{component.computation_id!r}: the memory-aware path still needs "
                f"{int(path_info.largest_intermediate)} elements."
            )

    active: list[tuple[str, tuple[str, ...]]] = [
        (
            tensor.slot,
            tuple(binding.index for binding in tensor.bindings),
        )
        for tensor in component.tensors
    ]
    steps: list[ContractionStep] = []

    for step_idx, contraction in enumerate(path_info.contraction_list):
        positions, removed_symbols, step_equation, _, _ = contraction
        positions = tuple(int(position) for position in positions)
        if not positions:
            raise RuntimeError("opt_einsum returned an empty contraction step.")
        if len(set(positions)) != len(positions):
            raise RuntimeError("opt_einsum returned duplicate operand positions in one step.")
        if any(position < 0 or position >= len(active) for position in positions):
            raise RuntimeError("opt_einsum returned an out-of-range contraction operand.")

        lhs, rhs = step_equation.split("->", 1)
        lhs_terms = lhs.split(",")
        if len(lhs_terms) != len(positions):
            raise RuntimeError(
                "opt_einsum contraction metadata does not match the selected operands."
            )

        selected = [active[position] for position in positions]
        input_refs = tuple(ref for ref, _ in selected)
        input_indices = tuple(indices for _, indices in selected)

        # Validate that the package path metadata and our canonical active terms
        # describe exactly the same ordered operands.
        expected_inputs = tuple(
            _symbols_to_indices(term, canonical_for_symbol)
            for term in lhs_terms
        )
        if expected_inputs != input_indices:
            raise RuntimeError(
                "opt_einsum contraction path is inconsistent with the active component tensors."
            )

        output_indices = _symbols_to_indices(rhs, canonical_for_symbol)
        contracted_indices = tuple(sorted(
            (canonical_for_symbol[symbol] for symbol in removed_symbols),
            key=str,
        ))
        output_elements = prod(
            symbol_sizes[symbol]
            for symbol in rhs
        ) if rhs else 1

        involved_symbols = set("".join(lhs_terms))
        estimated_flops = int(oe.helpers.flop_count(
            involved_symbols,
            bool(removed_symbols),
            len(positions),
            dict(symbol_sizes),
        ))

        step_id = f"{component.computation_id}:step_{step_idx}"
        steps.append(ContractionStep(
            step_id=step_id,
            input_refs=input_refs,
            input_indices=input_indices,
            contracted_indices=contracted_indices,
            output_indices=output_indices,
            estimated_flops=estimated_flops,
            output_elements=output_elements,
        ))

        for position in sorted(positions, reverse=True):
            del active[position]
        active.append((step_id, output_indices))

    if len(active) != 1:
        raise RuntimeError(
            "opt_einsum path did not reduce the connected component to one final tensor."
        )

    result_ref, final_indices = active[0]
    if final_indices != component.output_indices:
        raise RuntimeError(
            "Optimized contraction path final indices do not match component outputs."
        )

    plan = ComponentContractionPlan(
        computation_id=component.computation_id,
        optimizer_backend="opt_einsum",
        strategy=selected_strategy,
        steps=tuple(steps),
        result_ref=result_ref,
        output_indices=component.output_indices,
        estimated_flops=int(path_info.opt_cost),
        largest_intermediate_elements=int(path_info.largest_intermediate),
    )
    plan.validate(component)
    return plan


def optimize_global_contraction_order(
    dag: GlobalComputationDAG,
    size_model: ContractionSizeModel,
    *,
    strategy: str = "auto",
    memory_limit: int | None = None,
) -> GlobalContractionOrderPlan:
    """Optimize every globally unique connected component exactly once."""

    if memory_limit is not None:
        memory_limit = _positive_integer(memory_limit, "memory_limit")

    plans = {
        computation_id: optimize_component_contraction_order(
            component,
            size_model,
            strategy=strategy,
            memory_limit=memory_limit,
        )
        for computation_id, component in dag.components.items()
    }
    return GlobalContractionOrderPlan(
        source_dag=dag,
        size_model=size_model,
        strategy=strategy,
        memory_limit=memory_limit,
        component_plans=plans,
    )
