"""Public runtime boundary for the current staged execution pipeline."""

from __future__ import annotations

from importlib import import_module


_DIRECT_EXPORTS = {
    "WickAggregationConfig": (
        "concord_lqcd.compiler.wick_aggregation", "WickAggregationConfig",
    ),
    "WickAggregationTimeLimitExceeded": (
        "concord_lqcd.compiler.wick_aggregation", "WickAggregationTimeLimitExceeded",
    ),
    "BatchPlanningError": (
        "concord_lqcd.backend.runtime.batching",
        "BatchPlanningError",
    ),
    "BatchedExecutionMemoryEstimate": (
        "concord_lqcd.backend.runtime.batching",
        "BatchedExecutionMemoryEstimate",
    ),
    "TimeBatchCandidate": (
        "concord_lqcd.backend.runtime.batching",
        "TimeBatchCandidate",
    ),
    "TimeBatchFeasibilityPlan": (
        "concord_lqcd.backend.runtime.batching",
        "TimeBatchFeasibilityPlan",
    ),
    "estimate_batched_execution_memory": (
        "concord_lqcd.backend.runtime.batching",
        "estimate_batched_execution_memory",
    ),
    "plan_time_batch_candidates": (
        "concord_lqcd.backend.runtime.batching",
        "plan_time_batch_candidates",
    ),
    "BatchedNumericalExecutionResult": (
        "concord_lqcd.backend.runtime.batched_executor",
        "BatchedNumericalExecutionResult",
    ),
    "BatchedNumericalExecutor": (
        "concord_lqcd.backend.runtime.batched_executor",
        "BatchedNumericalExecutor",
    ),
    "ResidentExecutionResult": (
        "concord_lqcd.backend.runtime.residency",
        "ResidentExecutionResult",
    ),
    "ResidentExecutionSessionStatistics": (
        "concord_lqcd.backend.runtime.residency",
        "ResidentExecutionSessionStatistics",
    ),
    "ResidentExecutionSession": (
        "concord_lqcd.backend.runtime.residency",
        "ResidentExecutionSession",
    ),
    "CALIBRATION_CACHE_SCHEMA_VERSION": (
        "concord_lqcd.backend.runtime.calibration",
        "CALIBRATION_CACHE_SCHEMA_VERSION",
    ),
    "CalibrationCache": (
        "concord_lqcd.backend.runtime.calibration",
        "CalibrationCache",
    ),
    "CalibrationCacheError": (
        "concord_lqcd.backend.runtime.calibration",
        "CalibrationCacheError",
    ),
    "CalibrationCandidate": (
        "concord_lqcd.backend.runtime.calibration",
        "CalibrationCandidate",
    ),
    "CalibrationKey": (
        "concord_lqcd.backend.runtime.calibration",
        "CalibrationKey",
    ),
    "CalibrationRecord": (
        "concord_lqcd.backend.runtime.calibration",
        "CalibrationRecord",
    ),
    "CalibrationSelection": (
        "concord_lqcd.backend.runtime.calibration",
        "CalibrationSelection",
    ),
    "default_calibration_cache_path": (
        "concord_lqcd.backend.runtime.calibration",
        "default_calibration_cache_path",
    ),
    "deployment_fingerprint": (
        "concord_lqcd.backend.runtime.calibration",
        "deployment_fingerprint",
    ),
    "performance_environment_fingerprint": (
        "concord_lqcd.backend.runtime.calibration",
        "performance_environment_fingerprint",
    ),
    "physical_plan_fingerprint": (
        "concord_lqcd.backend.runtime.calibration",
        "physical_plan_fingerprint",
    ),
    "select_calibration_candidate": (
        "concord_lqcd.backend.runtime.calibration",
        "select_calibration_candidate",
    ),
    "HardwareDeviceProfile": (
        "concord_lqcd.backend.runtime.hardware",
        "HardwareDeviceProfile",
    ),
    "HardwareProfile": (
        "concord_lqcd.backend.runtime.hardware",
        "HardwareProfile",
    ),
    "detect_hardware_profile": (
        "concord_lqcd.backend.runtime.hardware",
        "detect_hardware_profile",
    ),
    "ResourcePlanningError": (
        "concord_lqcd.backend.runtime.resource_planning",
        "ResourcePlanningError",
    ),
    "ExecutionMode": (
        "concord_lqcd.backend.runtime.resource_planning",
        "ExecutionMode",
    ),
    "InsufficientDeviceMemoryError": (
        "concord_lqcd.backend.runtime.resource_planning",
        "InsufficientDeviceMemoryError",
    ),
    "MemorySafetyMargin": (
        "concord_lqcd.backend.runtime.resource_planning",
        "MemorySafetyMargin",
    ),
    "HardwareBudgetPolicy": (
        "concord_lqcd.backend.runtime.resource_planning",
        "HardwareBudgetPolicy",
    ),
    "ExecutionMemoryEstimate": (
        "concord_lqcd.backend.runtime.resource_planning",
        "ExecutionMemoryEstimate",
    ),
    "DeviceMemoryBudget": (
        "concord_lqcd.backend.runtime.resource_planning",
        "DeviceMemoryBudget",
    ),
    "DeviceExecutionPlan": (
        "concord_lqcd.backend.runtime.resource_planning",
        "DeviceExecutionPlan",
    ),
    "estimate_execution_memory": (
        "concord_lqcd.backend.runtime.resource_planning",
        "estimate_execution_memory",
    ),
    "plan_execution_resources": (
        "concord_lqcd.backend.runtime.resource_planning",
        "plan_execution_resources",
    ),
    "resolve_runtime_device": (
        "concord_lqcd.backend.runtime.resource_planning",
        "resolve_runtime_device",
    ),
    "revalidate_execution_resources": (
        "concord_lqcd.backend.runtime.resource_planning",
        "revalidate_execution_resources",
    ),
    "TilingPlanningError": (
        "concord_lqcd.backend.runtime.tiling",
        "TilingPlanningError",
    ),
    "OperationTilePlan": (
        "concord_lqcd.backend.runtime.tiling",
        "OperationTilePlan",
    ),
    "TiledExecutionPlan": (
        "concord_lqcd.backend.runtime.tiling",
        "TiledExecutionPlan",
    ),
    "plan_tiled_execution": (
        "concord_lqcd.backend.runtime.tiling",
        "plan_tiled_execution",
    ),
    "ExecutionContext": ("concord_lqcd.backend.runtime.context", "ExecutionContext"),
    "KernelMaterializationProvider": (
        "concord_lqcd.backend.runtime.materialization",
        "KernelMaterializationProvider",
    ),
    "PropagationSourceHandler": (
        "concord_lqcd.backend.runtime.materialization",
        "PropagationSourceHandler",
    ),
    "CompiledKernelMaterializationProvider": (
        "concord_lqcd.backend.runtime.materialization",
        "CompiledKernelMaterializationProvider",
    ),
    "KernelBatchMaterialization": (
        "concord_lqcd.backend.runtime.materialization",
        "KernelBatchMaterialization",
    ),
    "InMemoryTensorSourceHandler": (
        "concord_lqcd.backend.runtime.materialization",
        "InMemoryTensorSourceHandler",
    ),
    "HDF5TensorSourceHandler": (
        "concord_lqcd.backend.runtime.materialization",
        "HDF5TensorSourceHandler",
    ),
    "PropagationBatchMaterialization": (
        "concord_lqcd.backend.runtime.materialization",
        "PropagationBatchMaterialization",
    ),
    "HDF5FilePoolStatistics": (
        "concord_lqcd.backend.runtime.materialization",
        "HDF5FilePoolStatistics",
    ),
    "HDF5FilePool": (
        "concord_lqcd.backend.runtime.materialization",
        "HDF5FilePool",
    ),
    "MaterializationCacheKey": (
        "concord_lqcd.backend.runtime.materialization",
        "MaterializationCacheKey",
    ),
    "PrimitiveCacheEntryPlan": (
        "concord_lqcd.backend.runtime.materialization",
        "PrimitiveCacheEntryPlan",
    ),
    "PrimitiveCachePlan": (
        "concord_lqcd.backend.runtime.materialization",
        "PrimitiveCachePlan",
    ),
    "PrimitiveTensorCacheStatistics": (
        "concord_lqcd.backend.runtime.materialization",
        "PrimitiveTensorCacheStatistics",
    ),
    "PrimitiveTensorCache": (
        "concord_lqcd.backend.runtime.materialization",
        "PrimitiveTensorCache",
    ),
    "plan_primitive_tensor_cache": (
        "concord_lqcd.backend.runtime.materialization",
        "plan_primitive_tensor_cache",
    ),
    "MaterializedPrimitiveValue": (
        "concord_lqcd.backend.runtime.materialization",
        "MaterializedPrimitiveValue",
    ),
    "MaterializedPrimitiveBatch": (
        "concord_lqcd.backend.runtime.materialization",
        "MaterializedPrimitiveBatch",
    ),
    "PrimitiveMaterializer": (
        "concord_lqcd.backend.runtime.materialization",
        "PrimitiveMaterializer",
    ),
    "materialize_primitive_batch": (
        "concord_lqcd.backend.runtime.materialization",
        "materialize_primitive_batch",
    ),
    "PropagationPrefetchPlan": (
        "concord_lqcd.backend.runtime.prefetch",
        "PropagationPrefetchPlan",
    ),
    "PropagationPrefetchStatistics": (
        "concord_lqcd.backend.runtime.prefetch",
        "PropagationPrefetchStatistics",
    ),
    "PropagationPrefetchSession": (
        "concord_lqcd.backend.runtime.prefetch",
        "PropagationPrefetchSession",
    ),
    "plan_propagation_prefetch": (
        "concord_lqcd.backend.runtime.prefetch",
        "plan_propagation_prefetch",
    ),
    "RuntimeBufferPool": (
        "concord_lqcd.backend.runtime.numerical_executor",
        "RuntimeBufferPool",
    ),
    "NumericalExecutionResult": (
        "concord_lqcd.backend.runtime.numerical_executor",
        "NumericalExecutionResult",
    ),
    "GenericNumericalExecutor": (
        "concord_lqcd.backend.runtime.numerical_executor",
        "GenericNumericalExecutor",
    ),
    "COMPILATION_COST_REPORT_VERSION": (
        "concord_lqcd.backend.runtime.metrics",
        "COMPILATION_COST_REPORT_VERSION",
    ),
    "CompilationCostReport": (
        "concord_lqcd.backend.runtime.metrics",
        "CompilationCostReport",
    ),
    "build_compilation_cost_report": (
        "concord_lqcd.backend.runtime.metrics",
        "build_compilation_cost_report",
    ),
}

_PIPELINE_EXPORTS = {
    "build_adaptive_execution_candidates",
    "CorrelatorCollection",
    "PropagationSourceResolver",
    "RuntimeCompilation",
    "RuntimeCompilationConfig",
    "RuntimeExecutionCandidate",
    "RuntimeExecutionStages",
    "RuntimeExecutionConfig",
    "RuntimePipelineResult",
    "bind_compilation_resources",
    "compile_correlators",
    "execute_compilation",
    "run_correlators",
}


def __getattr__(name: str):
    if name in _PIPELINE_EXPORTS:
        value = getattr(import_module("concord_lqcd.backend.runtime.pipeline"), name)
    elif name in _DIRECT_EXPORTS:
        module_name, attribute = _DIRECT_EXPORTS[name]
        value = getattr(import_module(module_name), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


__all__ = [*_DIRECT_EXPORTS, *_PIPELINE_EXPORTS]
