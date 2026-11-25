import pytest

from concord_lqcd.backend.ir import CompiledKernelProgram, TensorOp


def test_tensor_op_requires_explicit_legs():
    with pytest.raises(ValueError, match="missing required params"):
        TensorOp("phase", {"momentum": (0, 0, 0)}).validate()


def test_tensor_op_validates_rank():
    with pytest.raises(ValueError, match="expects 2 legs"):
        TensorOp("color_delta", {"legs": ["a"]}).validate()


def test_compiled_kernel_program_validates_mode_ports():
    program = CompiledKernelProgram(
        num_quarks=1,
        operations=[TensorOp("phase", {"momentum": (0, 0, 0), "legs": ["x"]})],
        mode_ports=[],
        spin_ports=[],
        output_legs=["T"],
    )

    with pytest.raises(ValueError, match="mode_ports"):
        program.validate()
