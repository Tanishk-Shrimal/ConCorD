"""Declarative kernel-program IR used by primitive materialization."""

from .model import CompiledKernelProgram, TensorOp, TensorOpKind

KernelProgram = CompiledKernelProgram
KernelOperation = TensorOp

__all__ = [
    "TensorOpKind",
    "TensorOp",
    "KernelOperation",
    "CompiledKernelProgram",
    "KernelProgram",
]
