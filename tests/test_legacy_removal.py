"""Regression tests for removal of the superseded execution stack."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]

REMOVED_MODULES = (
    "concord_lqcd.visualize",
    "concord_lqcd.core.distillation",
    "concord_lqcd.compiler.compiler",
    "concord_lqcd.backend.analysis.analyzer",
    "concord_lqcd.backend.io",
    "concord_lqcd.backend.kernels",
    "concord_lqcd.backend.driver.distillation",
    "concord_lqcd.backend.optimizer.pipeline",
    "concord_lqcd.backend.ir.verify",
    "concord_lqcd.backend.planner.planner",
    "concord_lqcd.backend.runtime.legacy_executor",
    "concord_lqcd.backend.runtime.resource_manager",
    "concord_lqcd.backend.runtime.kernel_cache",
    "concord_lqcd.backend.runtime.kernel_evaluator",
    "concord_lqcd.backend.runtime.virtual_kernel_evaluator",
    "concord_lqcd.backend.runtime.task_executor",
    "concord_lqcd.backend.runtime.output_assembler",
    "concord_lqcd.backend.runtime.storage_manager",
)


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_legacy_modules_are_not_importable(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_removed_public_compatibility_exports_are_absent():
    import concord_lqcd
    import concord_lqcd.backend as backend
    import concord_lqcd.backend.runtime as runtime
    import concord_lqcd.compiler as compiler
    import concord_lqcd.core as core

    for namespace, names in (
        (concord_lqcd, ("compile_corr_matrix", "to_mermaid", "to_graphviz", "Propagator")),
        (core, ("compile_corr_matrix", "Propagator")),
        (compiler, ("compile_corr_matrix",)),
        (backend, ("CompiledJob", "Planner", "DistillationBackend", "optimize_job")),
        (runtime, ("Executor", "StorageManager")),
    ):
        for name in names:
            assert not hasattr(namespace, name)


def test_current_pipeline_and_shared_runtime_import_in_fresh_process():
    result = _run_python(
        """
        from concord_lqcd import Correlator, Operator
        from concord_lqcd.backend.ir import CompiledKernelProgram, TensorOp
        from concord_lqcd.backend.planner import (
            PhysicalExecutionPlan,
            build_physical_execution_plan,
        )
        from concord_lqcd.backend.runtime import (
            ExecutionContext,
            RuntimeCompilationConfig,
            compile_correlators,
        )

        assert all(value is not None for value in (
            Correlator,
            Operator,
            CompiledKernelProgram,
            TensorOp,
            PhysicalExecutionPlan,
            build_physical_execution_plan,
            ExecutionContext,
            RuntimeCompilationConfig,
            compile_correlators,
        ))
        """
    )
    assert result.returncode == 0, result.stderr
