"""Unit tests for the declarative kernel-program ISA."""

import pytest
import torch

from concord_lqcd.backend.ir import CompiledKernelProgram, TensorOp
from concord_lqcd.backend.runtime.kernels import evaluate_kernel_program
from concord_lqcd.backend.runtime import ExecutionContext
from concord_lqcd.backend.runtime.providers import DistillationTensorProvider


def test_tensor_op_invariant_and_resource_id():
    op1 = TensorOp("eigenvector", {
        "source": "distillation",
        "flavor": "u",
        "is_anti": False,
        "field_id": 0,
        "field_attributes": {"derivative": ["+x"]},
        "legs": ["T", "x", "a", "q_0"],
    })
    op2 = TensorOp("eigenvector", {
        "source": "distillation",
        "flavor": "u",
        "is_anti": False,
        "field_id": 0,
        "field_attributes": {"derivative": ["+x"]},
        "legs": ["T", "y", "b", "q_1"],
    })

    assert "INVARIANT:" in TensorOp.__doc__
    assert isinstance(op1.resource_id, str)
    assert len(op1.resource_id) == 16
    assert op1.resource_id == op2.resource_id


def test_gauge_link_tensor_op_validation():
    op = TensorOp("gauge_link", {
        "direction": "+x",
        "legs": ["c_in", "c_out"],
    })
    assert op.kind == "gauge_link"
    assert op.params["direction"] == "+x"

    with pytest.raises(ValueError, match="missing required params"):
        TensorOp("gauge_link", {"legs": ["c_in", "c_out"]}).validate()


def test_evaluate_kernel_program_with_gauge_link():
    program = CompiledKernelProgram(
        num_quarks=2,
        operations=[
            TensorOp("eigenvector", {
                "source": "distillation",
                "flavor": "u",
                "is_anti": False,
                "field_id": 0,
                "legs": ["T", "x", "a", "q_0"],
            }),
            TensorOp("eigenvector", {
                "source": "distillation",
                "flavor": "d",
                "is_anti": True,
                "field_id": 1,
                "legs": ["T", "x", "b", "q_1"],
            }),
            TensorOp("gauge_link", {
                "direction": "+x",
                "legs": ["a", "b"],
            }),
        ],
        mode_ports=["q_0", "q_1"],
        spin_ports=[],
        output_legs=["T", "q_0", "q_1"],
    )
    eigenvectors = torch.ones((1, 10, 3, 4), dtype=torch.complex128)
    provider = DistillationTensorProvider(
        {"u": eigenvectors, "d": eigenvectors},
        {"u": eigenvectors, "d": eigenvectors},
        {},
    )
    context = ExecutionContext.create("cpu", 0, [0])

    result = evaluate_kernel_program(program, provider, context)

    assert result.shape == (1, 4, 4)


def test_optional_kernel_path_keeps_default_selection_unchanged(monkeypatch):
    import opt_einsum as oe

    program = CompiledKernelProgram(
        num_quarks=2,
        operations=[
            TensorOp("eigenvector", {
                "source": "distillation", "flavor": "u", "is_anti": False,
                "field_id": 0, "legs": ["T", "x", "a", "q_0"],
            }),
            TensorOp("eigenvector", {
                "source": "distillation", "flavor": "d", "is_anti": True,
                "field_id": 1, "legs": ["T", "x", "b", "q_1"],
            }),
            TensorOp("gauge_link", {"direction": "+x", "legs": ["a", "b"]}),
        ],
        mode_ports=["q_0", "q_1"], spin_ports=[], output_legs=["T", "q_0", "q_1"],
    )
    generator = torch.Generator().manual_seed(20261003)
    fields = {
        flavor: torch.randn((1, 4, 3, 3), dtype=torch.complex128, generator=generator)
        for flavor in ("u", "d")
    }
    provider = DistillationTensorProvider(fields, fields, {})
    context = ExecutionContext.create("cpu", 0, [0])
    original = oe.contract
    choices = []

    def capture_choice(*arguments, **keywords):
        choices.append(keywords.get("optimize"))
        return original(*arguments, **keywords)

    monkeypatch.setattr(oe, "contract", capture_choice)
    legacy = evaluate_kernel_program(program, provider, context)
    fixed_path = ((0, 1), (0, 1))
    planned = evaluate_kernel_program(program, provider, context, contraction_path=fixed_path)
    assert choices == [None, list(fixed_path)]
    torch.testing.assert_close(planned, legacy, rtol=1e-12, atol=0)
