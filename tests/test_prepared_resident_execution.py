"""Prepared resident execution gives bitwise the step-by-step results."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.plan_costs import _baryon_correlator, _tetraquark_correlator
from concord_lqcd.backend.planner import build_physical_execution_plan
from concord_lqcd.backend.runtime import prepared_executor
from concord_lqcd.backend.runtime.pipeline import RuntimeCompilationConfig, compile_correlators
from concord_lqcd.backend.runtime.prepared_executor import check_capacity_bookkeeping
from concord_lqcd.backend.runtime.residency import ResidentExecutionSession
from tests.test_generic_numerical_executor import (
    _SeededComplexKernelProvider,
    _complex_propagation_loader,
    _materializer_for,
    _schedule,
)


CONTEXTS = ({2: (3, 1)}, {2: (4, 1)}, {2: (5, 2)})


def _plan(name):
    if name == "meson":
        return build_physical_execution_plan(
            _schedule(("C0", "C1"), mode_size=3),
            reuse_strategy="best_fit",
        )
    factory = {"baryon": _baryon_correlator, "tetraquark": _tetraquark_correlator}[name]
    return compile_correlators(
        {"C": factory()},
        RuntimeCompilationConfig(mode_size=2),
    ).physical_plan


def _run(plan, *, prepared, batches, inspect=None):
    """Execute CONTEXTS in the given batch sizes and return each work unit's outputs."""

    session = ResidentExecutionSession(
        plan,
        device="cpu",
        maximum_batch_size=max(batches),
        prepared=prepared,
    )
    materializers = tuple(
        _materializer_for(
            plan,
            operator_times=context,
            kernel_provider=_SeededComplexKernelProvider(1),
            propagation_loader=_complex_propagation_loader,
        )
        for context in CONTEXTS
    )
    outputs = []
    start = 0
    for size in batches:
        result = session.execute(materializers[start:start + size])
        if inspect is not None:
            inspect(session)
        for index in range(size):
            values = result.outputs if size == 1 else result.outputs_at(index)
            outputs.append({key: tensor.clone() for key, tensor in values.items()})
        session.release()
        start += size
    session.close()
    return outputs


@pytest.mark.parametrize("name", ("meson", "baryon", "tetraquark"))
@pytest.mark.parametrize(
    "batches",
    ((1, 1, 1), (2, 1), (3,)),
    ids=("single", "full-and-short", "one-batch"),
)
def test_prepared_execution_is_bitwise_identical_to_checked_execution(name, batches):
    plan = _plan(name)
    checked = _run(plan, prepared=False, batches=batches)
    prepared = _run(plan, prepared=True, batches=batches)

    assert len(checked) == len(prepared) == len(CONTEXTS)
    for expected, observed in zip(checked, prepared):
        assert expected.keys() == observed.keys()
        for key in expected:
            assert torch.equal(expected[key], observed[key])
    assert any(torch.max(torch.abs(value)).item() > 1e-8 for value in checked[0].values())


def test_prepared_execution_leaves_the_pool_as_checked_execution_does():
    plan = _plan("baryon")
    states = {}
    for prepared in (False, True):
        observed = []

        def inspect(session):
            observed.append((
                session.pool.active_value_ids,
                session.pool.active_capacity_bytes,
                dict(session.pool.active_owner),
            ))

        _run(plan, prepared=prepared, batches=(2, 1), inspect=inspect)
        states[prepared] = observed
    assert states[True] == states[False]


def test_steps_are_prepared_once_per_process_and_bound_once_per_batch_size(monkeypatch):
    plan = _plan("meson")
    calls = []
    original = prepared_executor._prepare_steps

    def counting(plan, batch_size, device_type):
        calls.append((batch_size, device_type))
        return original(plan, batch_size, device_type)

    monkeypatch.setattr(prepared_executor, "_prepare_steps", counting)
    bound = []
    for _ in range(2):
        _run(
            plan,
            prepared=True,
            batches=(2, 1),
            inspect=lambda session: bound.append(len(session._programs)),
        )

    assert sorted(calls) == [(1, "cpu"), (2, "cpu")]
    assert bound == [1, 2, 1, 2]


def test_environment_selects_checked_execution(monkeypatch):
    plan = _plan("meson")
    monkeypatch.setenv("CONCORD_CHECKED_EXECUTION", "1")
    assert not ResidentExecutionSession(plan, device="cpu", maximum_batch_size=1).prepared
    monkeypatch.setenv("CONCORD_CHECKED_EXECUTION", "0")
    assert ResidentExecutionSession(plan, device="cpu", maximum_batch_size=1).prepared
    monkeypatch.delenv("CONCORD_CHECKED_EXECUTION")
    assert ResidentExecutionSession(plan, device="cpu", maximum_batch_size=1).prepared
    assert not ResidentExecutionSession(
        plan, device="cpu", maximum_batch_size=1, prepared=False
    ).prepared


def test_capacity_replay_rejects_a_plan_whose_bookkeeping_disagrees():
    plan = _plan("meson")
    check_capacity_bookkeeping(plan)

    first = plan.steps[0]
    broken = SimpleNamespace(
        steps=(
            replace(first, active_capacity_after=first.active_capacity_after + 16),
            *plan.steps[1:],
        ),
        binding=plan.binding,
        buffer=plan.buffer,
        source_schedule=plan.source_schedule,
    )
    with pytest.raises(RuntimeError, match="after step"):
        check_capacity_bookkeeping(broken)
