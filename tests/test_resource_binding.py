"""Focused tests for exact resource-id to runtime-source binding."""

import pytest

from concord_lqcd.backend.planner import build_physical_execution_plan
from concord_lqcd.backend.planner.physical import MaterializePrimitiveOperation
from concord_lqcd.backend.runtime.resource_binding import (
    ExternalTensorSource,
    KernelMaterializationRequest,
    KernelResourceBinding,
    OperatorTimeContext,
    PropagationMaterializationRequest,
    PropagationResourceBinding,
    SpinMaterializationRequest,
    SpinResourceBinding,
    build_resource_binding_plan,
    prepare_resource_binding_template,
)
from concord_lqcd.compiler import (
    ContractionSizeModel,
    analyze_static_execution,
    build_logical_execution_dag,
    build_logical_schedule,
    build_partial_intermediate_cse,
    lower_physics_ir_collection_to_global_cse,
    optimize_global_contraction_order,
)
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.core import (
    Correlator,
    Operator,
    c,
    cbar,
    gamma5,
    gammaX,
    point,
    reset_registry,
    s,
    sbar,
    u,
    ubar,
)
from concord_lqcd.core.wick import wick_contract


def _physical_plan(correlators, *, mode_size=5):
    physics = {
        key: lower_to_physics_ir(correlator, wick_contract(correlator))
        for key, correlator in correlators.items()
    }
    global_dag = lower_physics_ir_collection_to_global_cse(physics)
    order = optimize_global_contraction_order(
        global_dag,
        ContractionSizeModel(mode_size=mode_size),
        strategy="greedy",
    )
    logical = build_logical_execution_dag(build_partial_intermediate_cse(order))
    analysis = analyze_static_execution(logical)
    schedule = build_logical_schedule(analysis, strategy="memory_aware")
    return build_physical_execution_plan(schedule)


def _two_point():
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1") * gamma5("s1", "s2") * u("y", "a", "s2")
    )
    source = Operator(
        ubar("x", "b", "s3") * gamma5("s3", "s4") * c("x", "b", "s4")
    )
    return Correlator(sink, source)


def _three_point(*, current_momentum=(1, 0, 0), source_momentum=(-1, 0, 0)):
    reset_registry()
    sink = Operator(
        cbar("y", "a", "s1")
        * gamma5("s1", "s2")
        * s("y", "a", "s2"),
        momentum=(0, 0, 0),
    )
    current = Operator(
        point(sbar("z", "b", "s3"))
        * gammaX("s3", "s4")
        * point(u("z", "b", "s4")),
        momentum=current_momentum,
    )
    source = Operator(
        ubar("x", "c", "s5")
        * gamma5("s5", "s6")
        * c("x", "c", "s6"),
        momentum=source_momentum,
    )
    return Correlator(sink, current, source)


def _used_propagation_ids(plan):
    return tuple(dict.fromkeys(
        step.operation.resource_id
        for step in plan.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
        and step.operation.kind == "propagation"
    ))


def _sources_for(plan):
    return {
        resource_id: ExternalTensorSource(
            source_id=f"source:{resource_id}",
            kind="hdf5",
            location=f"/data/{resource_id}.h5",
            dataset="perambulators",
            metadata={"layout": "distillation"},
        )
        for resource_id in _used_propagation_ids(plan)
    }


def test_resource_binding_resolves_all_scheduled_primitive_values():
    physical = _physical_plan({"C00": _two_point()}, mode_size=6)
    times = OperatorTimeContext({2: (12, 3)})
    plan = build_resource_binding_plan(
        physical,
        operator_times=times,
        propagation_sources=_sources_for(physical),
    )

    primitive_operations = [
        step.operation
        for step in physical.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
    ]
    assert plan.statistics.primitive_materializations == len(primitive_operations)
    assert set(plan.materializations) == {op.value_id for op in primitive_operations}
    assert set(plan.bindings) == {op.resource_id for op in primitive_operations}

    for operation in primitive_operations:
        request = plan.materialization(operation.value_id)
        assert request.resource_id == operation.resource_id
        assert request.output_buffer_id == operation.output_buffer_id
        assert request.expected_shape == physical.binding(operation.value_id).shape
        assert request.scalar_type == physical.binding(operation.value_id).scalar_type


def test_resource_binding_template_reuses_configuration_invariants():
    physical = _physical_plan({"C00": _two_point()}, mode_size=6)
    template = prepare_resource_binding_template(
        physical,
        propagation_sources=_sources_for(physical),
    )

    first = template.bind({2: (12, 3)})
    second = template.bind({2: (13, 3)})

    assert first.bindings is second.bindings
    assert first.materializations is not second.materializations
    assert first.time_context != second.time_context
    first.validate()
    first.validate()
    with pytest.raises(TypeError):
        first.bindings["new"] = object()


def test_kernel_resource_exemplar_does_not_override_occurrence_placement():
    # A single two-point function does not guarantee a repeated kernel resource:
    # source and sink may have different ordered V/V^dag interfaces.  Combine a
    # two-point and a three-point correlator with matching zero-momentum source/
    # sink spatial structures so the same kernel resource is genuinely reused at
    # different arity-scoped operator placements.
    physical = _physical_plan({
        "C2": _two_point(),
        "C3": _three_point(current_momentum=(0, 0, 0), source_momentum=(0, 0, 0)),
    })
    plan = build_resource_binding_plan(
        physical,
        operator_times={2: (14, 2), 3: (21, 9, 3)},
        propagation_sources=_sources_for(physical),
    )

    kernel_requests = [
        request
        for request in plan.materializations.values()
        if isinstance(request, KernelMaterializationRequest)
    ]
    by_resource = {}
    for request in kernel_requests:
        by_resource.setdefault(request.resource_id, []).append(request)

    reused = [
        requests
        for requests in by_resource.values()
        if len({request.placement for request in requests}) > 1
    ]
    assert reused

    # Prefer a resource whose raw operator slot also changes (the common source
    # kernel appears at op1 in the 2-point graph and op2 in the 3-point graph).
    requests = next(
        (
            group
            for group in reused
            if len({request.operator_ids for request in group}) > 1
        ),
        reused[0],
    )
    assert {request.placement[1] for request in requests} == {2, 3}
    for request in requests:
        arity = request.placement[1]
        assert request.operator_times == tuple(
            plan.time_context.time_for(arity, operator_id)
            for operator_id in request.operator_ids
        )


def test_spin_binding_freezes_registered_matrix_but_uses_occurrence_slot():
    physical = _physical_plan({"C00": _two_point()})
    plan = build_resource_binding_plan(
        physical,
        operator_times={2: (9, 1)},
        propagation_sources=_sources_for(physical),
    )

    requests = [
        request
        for request in plan.materializations.values()
        if isinstance(request, SpinMaterializationRequest)
    ]
    assert requests
    for request in requests:
        binding = plan.binding(request.resource_id)
        assert isinstance(binding, SpinResourceBinding)
        assert len(request.matrix) == 4
        assert all(len(row) == 4 for row in request.matrix)
        if request.operator_id is not None:
            assert request.operator_time == plan.time_context.time_for(2, request.operator_id)


def test_propagation_binding_is_exact_resource_id_not_flavor_fallback():
    physical = _physical_plan({"C00": _two_point()})
    resource_ids = _used_propagation_ids(physical)
    assert resource_ids

    fake_flavor_catalog = {
        "u": ExternalTensorSource("u", "hdf5", "/data/u.h5")
    }
    with pytest.raises(ValueError, match="No exact external tensor source"):
        build_resource_binding_plan(
            physical,
            operator_times={2: (10, 0)},
            propagation_sources=fake_flavor_catalog,
        )


def test_ordinary_propagation_request_resolves_source_and_sink_times_from_placement():
    physical = _physical_plan({"C00": _two_point()})
    plan = build_resource_binding_plan(
        physical,
        operator_times={2: (11, 4)},
        propagation_sources=_sources_for(physical),
    )

    requests = [
        request
        for request in plan.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
    ]
    assert requests
    for request in requests:
        binding = plan.binding(request.resource_id)
        assert isinstance(binding, PropagationResourceBinding)
        assert not request.recipe.is_generalized
        assert request.insertion_operator_ids == ()
        assert request.insertion_times == ()
        assert request.source_time == plan.time_context.time_for(
            2, request.source_operator_id
        )
        assert request.sink_time == plan.time_context.time_for(
            2, request.sink_operator_id
        )
        assert request.source == binding.source


def test_generalized_propagation_request_preserves_ordered_insertion_time():
    physical = _physical_plan({"C3": _three_point()}, mode_size=6)
    sources = _sources_for(physical)
    plan = build_resource_binding_plan(
        physical,
        operator_times={3: (15, 8, 2)},
        propagation_sources=sources,
    )

    generalized = [
        request
        for request in plan.materializations.values()
        if isinstance(request, PropagationMaterializationRequest)
        and request.recipe.is_generalized
    ]
    assert len(generalized) == 1
    request = generalized[0]
    assert request.recipe.segment_flavors == ("u", "s")
    assert len(request.insertion_operator_ids) == 1
    assert request.insertion_times == tuple(
        plan.time_context.time_for(3, operator_id)
        for operator_id in request.insertion_operator_ids
    )
    assert request.source_time == plan.time_context.time_for(
        3, request.source_operator_id
    )
    assert request.sink_time == plan.time_context.time_for(3, request.sink_operator_id)
    assert request.source == sources[request.resource_id]


def test_kernel_provider_override_is_keyed_by_exact_resource_id():
    physical = _physical_plan({"C00": _two_point()})
    kernel_operation = next(
        step.operation
        for step in physical.steps
        if isinstance(step.operation, MaterializePrimitiveOperation)
        and step.operation.kind == "kernel"
    )
    plan = build_resource_binding_plan(
        physical,
        operator_times={2: (7, 0)},
        propagation_sources=_sources_for(physical),
        kernel_provider_ids={kernel_operation.resource_id: "special-kernel-provider"},
    )

    binding = plan.binding(kernel_operation.resource_id)
    assert isinstance(binding, KernelResourceBinding)
    assert binding.provider_id == "special-kernel-provider"
    for request in plan.materializations.values():
        if request.resource_id == kernel_operation.resource_id:
            assert isinstance(request, KernelMaterializationRequest)
            assert request.provider_id == "special-kernel-provider"


def test_missing_operator_time_context_fails_before_runtime_loading():
    physical = _physical_plan({"C00": _two_point()})
    with pytest.raises(ValueError, match="No operator-time context"):
        build_resource_binding_plan(
            physical,
            operator_times={},
            propagation_sources=_sources_for(physical),
        )
