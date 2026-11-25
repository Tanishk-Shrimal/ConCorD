"""Evaluate declarative kernel programs with concrete runtime tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING
import opt_einsum as oe
import torch

from concord_lqcd.backend.ir import CompiledKernelProgram

if TYPE_CHECKING:
    from concord_lqcd.backend.runtime.context import ExecutionContext
    from concord_lqcd.backend.runtime.providers import TensorProvider


def evaluate_kernel_program(
    program: CompiledKernelProgram,
    provider: TensorProvider,
    context: ExecutionContext,
    is_sink: bool = True,
    *,
    contraction_path: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    """Evaluate a kernel, optionally pinning its spatial contraction path.

    The default retains the existing opt_einsum path selection. Spatial
    kernels carry no spin structure: the compiler keeps spin tensors as
    separate primitive nodes, so spin operations and spin ports are rejected.
    """
    program.validate()
    if program.spin_ports or any(op.kind == "spin_matrix" for op in program.operations):
        raise ValueError(
            "Spatial kernel programs must not contain spin operations or spin ports; "
            "spin tensors are separate primitive nodes in the contraction graph."
        )

    operands = []
    operand_legs: list[list[int]] = []
    index_map: dict[str, int] = {}
    next_index = 0
    has_time_leg = False
    T_batch = len(context.t_snk_list)
    Nc = 3
    dtype = context.complex_dtype
    spatial_legs = {
        op.params["legs"][0] for op in program.operations if op.kind == "phase"
    } | {
        op.params["legs"][1] for op in program.operations if op.kind == "eigenvector"
    }
    spatial_extents: dict[str, tuple[int, str]] = {}

    def idx(label: str) -> int:
        nonlocal next_index
        if label not in index_map:
            index_map[label] = next_index
            next_index += 1
        return index_map[label]

    def append_operand(tensor: torch.Tensor, legs: list[str], kind: str) -> None:
        nonlocal has_time_leg
        if tensor.ndim != len(legs):
            raise ValueError(
                f"Kernel {kind!r} operand has rank {tensor.ndim}, expected "
                f"{len(legs)} from declared legs {tuple(legs)!r}."
            )
        for leg, extent in zip(legs, tensor.shape, strict=True):
            if leg not in spatial_legs:
                continue
            previous = spatial_extents.get(leg)
            if previous is None:
                spatial_extents[leg] = (extent, kind)
            elif extent != previous[0]:
                raise ValueError(
                    f"Kernel spatial leg {leg!r} has extent {extent} in {kind!r} "
                    f"operand, expected {previous[0]} from {previous[1]!r} operand. "
                    "Providers must supply matching field and phase site selections."
                )
        has_time_leg = has_time_leg or "T" in legs
        operands.append(tensor)
        operand_legs.append([idx(leg) for leg in legs])

    for op in program.operations:
        params = op.params
        legs = params["legs"]

        if op.kind == "eigenvector":
            V = provider.eigenvector(
                flavor=params.get("flavor"),
                is_anti=params.get("is_anti", False),
                derivatives=params.get("derivatives", params.get("field_attributes", {}).get("derivatives", ())),
                is_sink=is_sink,
                context=context,
                field_attributes=params.get("field_attributes"),
            )
            append_operand(V, legs, op.kind)
            T_batch, _, Nc, _ = V.shape
            dtype = V.dtype

        elif op.kind == "phase":
            momentum = tuple(params["momentum"])
            phase = provider.phase(momentum, context)
            if phase is None:
                continue
            append_operand(phase, legs, op.kind)

        elif op.kind == "gauge_link":
            U_link = provider.gauge_link(params.get("direction", "+x"), context)
            append_operand(U_link, legs, op.kind)

        elif op.kind == "color_delta":
            append_operand(provider.color_delta(Nc, dtype, context), legs, op.kind)

        elif op.kind == "color_epsilon":
            append_operand(provider.color_epsilon(dtype, context), legs, op.kind)

        else:
            raise ValueError(f"Unsupported TensorOp kind {op.kind!r}.")

    if not operands:
        raise ValueError("Kernel program produced no tensor operands.")

    output_labels = program.output_legs or (["T"] + program.mode_ports)
    contract_output = [leg for leg in output_labels if leg != "T" or has_time_leg]

    oe_args = []
    for operand, legs in zip(operands, operand_legs):
        oe_args.extend([operand, legs])
    oe_args.append([idx(leg) for leg in contract_output])
    if contraction_path is None:
        res = oe.contract(*oe_args, backend="torch")
    else:
        res = oe.contract(*oe_args, backend="torch", optimize=list(contraction_path))

    if "T" in output_labels and not has_time_leg:
        res = res.unsqueeze(0).expand(T_batch, *res.shape)

    return res
