"""Linear bookkeeping without weakening plan validation or source agreement."""

from dataclasses import fields, replace

import pytest

from concord_lqcd.backend.planner.physical import build_physical_execution_plan
from concord_lqcd.compiler import (
    ContractionSizeModel,
    GlobalCSEStatistics,
    GlobalComputationDAG,
    GlobalCorrelatorUse,
    GlobalWickTermUse,
    LogicalPrimitiveEvaluation,
    analyze_static_execution,
    build_logical_execution_dag,
    build_logical_schedule,
    build_partial_intermediate_cse,
    optimize_global_contraction_order,
)


class CountingBuffers(dict):
    reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return super().__getitem__(key)


class CountingTuple(tuple):
    def __new__(cls, values):
        instance = super().__new__(cls, values)
        instance.iterations = 0
        return instance

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def constant_terms(count):
    source = GlobalComputationDAG(
        resources={},
        components={},
        correlators=(GlobalCorrelatorUse("constant", tuple(
            GlobalWickTermUse(index, complex(index + 1), ()) for index in range(count)
        )),),
        statistics=GlobalCSEStatistics(0, 0, 0, 0),
    )
    order = optimize_global_contraction_order(source, ContractionSizeModel(mode_size=3), strategy="greedy")
    return build_logical_execution_dag(build_partial_intermediate_cse(order))


@pytest.mark.parametrize("count", (32, 128, 256))
@pytest.mark.parametrize("reuse_strategy", ("best_fit", "size_matched", "none"))
def test_buffer_validation_reads_scale_with_operations(count, reuse_strategy):
    logical = constant_terms(count)
    schedule = build_logical_schedule(analyze_static_execution(logical), strategy="topological")
    plan = build_physical_execution_plan(schedule, reuse_strategy=reuse_strategy)
    buffers = CountingBuffers(plan.buffers)

    replace(plan, buffers=buffers).validate()

    assert buffers.reads < 8 * len(plan.steps)
    assert max(step.active_capacity_before for step in plan.steps) >= count * 16
    assert plan.statistics.final_active_capacity_bytes == 16


def test_assembly_lookup_indexes_each_immutable_sequence_only_once():
    original = constant_terms(32)
    terms = CountingTuple(original.terms)
    correlators = CountingTuple(original.correlators)
    logical = replace(original, terms=terms, correlators=correlators)

    for _ in range(1000):
        assert logical.node(original.terms[-1].node_id) is original.terms[-1]
        assert logical.node(original.correlators[-1].node_id) is original.correlators[-1]

    assert terms.iterations == correlators.iterations == 1
    assert "_assembly_nodes" not in {field.name for field in fields(logical)}
    with pytest.raises(TypeError):
        logical._assembly_nodes["invented"] = original.terms[0]
    with pytest.raises(KeyError, match="missing"):
        logical.node("missing")


def test_replacing_assembly_records_does_not_reuse_an_outdated_index():
    logical = constant_terms(2)
    original = logical.terms[0]
    assert logical.node(original.node_id) is original
    changed = replace(original, coefficient=original.coefficient + 1)
    altered = replace(logical, terms=(changed,) + logical.terms[1:])

    assert altered.node(original.node_id) is changed
    with pytest.raises(ValueError, match="compiler source"):
        altered.validate()


def test_mutable_node_maps_are_not_cached_with_assembly_records():
    logical = constant_terms(2)
    assert logical.node(logical.terms[0].node_id) is logical.terms[0]
    node = LogicalPrimitiveEvaluation("new", "new", "resource", "spin", ("spin_op", 0), ())
    logical.primitives[node.node_id] = node
    assert logical.node(node.node_id) is node
    replacement = replace(node, resource_id="changed")
    logical.primitives[node.node_id] = replacement
    assert logical.node(node.node_id) is replacement
    del logical.primitives[node.node_id]

    with pytest.raises(KeyError, match="new"):
        logical.node(node.node_id)
    logical.validate()


def test_mutable_assembly_sequences_keep_live_lookup_semantics():
    original = constant_terms(2)
    terms = list(original.terms)
    logical = replace(original, terms=terms)
    assert logical.node(terms[0].node_id) is terms[0]
    changed = replace(terms[0], coefficient=23j)
    terms[0] = changed

    assert logical.node(changed.node_id) is changed
    with pytest.raises(ValueError, match="compiler source"):
        logical.validate()


@pytest.mark.parametrize("field_name", ("active_capacity_before", "peak_active_capacity_during_step", "active_capacity_after"))
def test_repeated_validation_still_rejects_changed_capacity_records(field_name):
    schedule = build_logical_schedule(analyze_static_execution(constant_terms(8)))
    plan = build_physical_execution_plan(schedule)
    plan.validate()
    first = plan.steps[0]
    damaged = replace(first, **{field_name: getattr(first, field_name) + 16})

    with pytest.raises(ValueError, match="capacity"):
        replace(plan, steps=(damaged,) + plan.steps[1:]).validate()
