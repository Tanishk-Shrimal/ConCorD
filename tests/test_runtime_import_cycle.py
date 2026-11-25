"""Regression tests for package import ordering around the new runtime pipeline."""

from __future__ import annotations

import subprocess
import sys


def test_compiler_then_runtime_pipeline_exports_import_in_fresh_process():
    code = "\n".join(
        [
            "from concord_lqcd.compiler import ContractionSizeModel",
            "from concord_lqcd.backend.runtime import CompilationCostReport, RuntimeCompilationConfig",
            "from concord_lqcd.backend.runtime import HardwareProfile, plan_execution_resources",
            "from concord_lqcd.backend.runtime import BatchedNumericalExecutor",
            "from concord_lqcd.backend.runtime import plan_time_batch_candidates",
            "from concord_lqcd.backend.runtime import ResidentExecutionSession",
            "from concord_lqcd.backend.runtime import PrimitiveTensorCache",
            "from concord_lqcd.backend.runtime import PropagationPrefetchSession",
            "assert ContractionSizeModel is not None",
            "assert CompilationCostReport is not None",
            "assert RuntimeCompilationConfig is not None",
            "assert HardwareProfile is not None",
            "assert plan_execution_resources is not None",
            "assert BatchedNumericalExecutor is not None",
            "assert plan_time_batch_candidates is not None",
            "assert ResidentExecutionSession is not None",
            "assert PrimitiveTensorCache is not None",
            "assert PropagationPrefetchSession is not None",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_runtime_pipeline_exports_then_compiler_import_in_fresh_process():
    code = "\n".join(
        [
            "from concord_lqcd.backend.runtime import CompilationCostReport, RuntimeCompilationConfig",
            "from concord_lqcd.backend.runtime import HardwareProfile, plan_execution_resources",
            "from concord_lqcd.backend.runtime import BatchedNumericalExecutor",
            "from concord_lqcd.backend.runtime import plan_time_batch_candidates",
            "from concord_lqcd.backend.runtime import ResidentExecutionSession",
            "from concord_lqcd.backend.runtime import PrimitiveTensorCache",
            "from concord_lqcd.backend.runtime import PropagationPrefetchSession",
            "from concord_lqcd.compiler import ContractionSizeModel",
            "assert CompilationCostReport is not None",
            "assert RuntimeCompilationConfig is not None",
            "assert HardwareProfile is not None",
            "assert plan_execution_resources is not None",
            "assert BatchedNumericalExecutor is not None",
            "assert plan_time_batch_candidates is not None",
            "assert ResidentExecutionSession is not None",
            "assert PrimitiveTensorCache is not None",
            "assert PropagationPrefetchSession is not None",
            "assert ContractionSizeModel is not None",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
