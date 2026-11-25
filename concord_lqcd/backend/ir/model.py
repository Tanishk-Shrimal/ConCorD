"""Declarative primitive-tensor programs used by runtime materialization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Mapping
import hashlib
import json

TensorOpKind = Literal[
    "eigenvector",
    "gauge_link",
    "phase",
    "color_delta",
    "color_epsilon",
    "spin_matrix",
]

_REQUIRED_OP_PARAMS: dict[str, set[str]] = {
    "eigenvector": {"source", "flavor", "is_anti", "field_id", "legs"},
    "gauge_link": {"direction", "legs"},
    "phase": {"momentum", "legs"},
    "color_delta": {"legs"},
    "color_epsilon": {"legs"},
    "spin_matrix": {"matrix", "legs"},
}
_OP_RANKS: dict[str, int] = {
    "eigenvector": 4,
    "gauge_link": 2,
    "phase": 1,
    "color_delta": 2,
    "color_epsilon": 3,
    "spin_matrix": 2,
}


@dataclass(frozen=True)
class TensorOp:
    """A primitive numerical tensor operation in a compiled kernel program.

    INVARIANT: A TensorOp declaratively identifies a tensor resource required during
    kernel evaluation. It specifies tensor identity, intrinsic attributes, and index
    structure. A TensorOp never encodes an algorithm or execution order.
    """

    kind: TensorOpKind
    params: dict[str, Any]

    @property
    def resource_id(self) -> str:
        """Deterministic hash key uniquely identifying the underlying tensor resource for caching and CSE."""
        payload = {
            "kind": self.kind,
            "params": {k: str(v) for k, v in self.params.items() if k != "legs"},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def validate(self) -> None:
        if self.kind not in _REQUIRED_OP_PARAMS:
            raise ValueError(f"Unsupported TensorOp kind {self.kind!r}.")

        missing = _REQUIRED_OP_PARAMS[self.kind] - set(self.params)
        if missing:
            raise ValueError(f"TensorOp {self.kind!r} is missing required params: {sorted(missing)}")

        legs = self.params["legs"]
        if not isinstance(legs, list) or not all(isinstance(leg, str) for leg in legs):
            raise ValueError(f"TensorOp {self.kind!r} params['legs'] must be a list of strings.")
        expected_rank = _OP_RANKS[self.kind]
        if len(legs) != expected_rank:
            raise ValueError(
                f"TensorOp {self.kind!r} expects {expected_rank} legs, got {len(legs)}."
            )

        if "field_attributes" in self.params:
            attrs = self.params["field_attributes"]
            if not isinstance(attrs, (dict, Mapping)):
                raise ValueError("TensorOp 'field_attributes' must be a mapping/dict.")

        if self.kind == "phase":
            momentum = self.params["momentum"]
            if not isinstance(momentum, (list, tuple)) or len(momentum) != 3:
                raise ValueError("TensorOp 'phase' requires a 3-component momentum.")

        if self.kind == "spin_matrix":
            matrix = self.params["matrix"]
            if not isinstance(matrix, dict) or set(matrix) != {"real", "imag"}:
                raise ValueError("TensorOp 'spin_matrix' matrix must contain real/imag arrays.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"kind": self.kind, "params": self.params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TensorOp':
        op = cls(kind=data["kind"], params=data.get("params", {}))
        op.validate()
        return op


@dataclass
class CompiledKernelProgram:
    """Syntax-free executable IR for evaluating one kernel tensor."""

    num_quarks: int
    operations: list[TensorOp]
    mode_ports: list[str]
    spin_ports: list[str]
    output_legs: list[str] | None = None

    def validate(self) -> None:
        if self.num_quarks < 0:
            raise ValueError("CompiledKernelProgram.num_quarks must be non-negative.")
        if len(self.mode_ports) != self.num_quarks:
            raise ValueError("CompiledKernelProgram.mode_ports must match num_quarks.")
        for op in self.operations:
            op.validate()
        if self.output_legs is not None:
            if not isinstance(self.output_legs, list) or not all(isinstance(leg, str) for leg in self.output_legs):
                raise ValueError("CompiledKernelProgram.output_legs must be a list of strings.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "num_quarks": self.num_quarks,
            "operations": [op.to_dict() for op in self.operations],
            "mode_ports": self.mode_ports,
            "spin_ports": self.spin_ports,
            "output_legs": self.output_legs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CompiledKernelProgram':
        program = cls(
            num_quarks=data.get("num_quarks", 0),
            operations=[TensorOp.from_dict(op) for op in data.get("operations", [])],
            mode_ports=data.get("mode_ports", []),
            spin_ports=data.get("spin_ports", []),
            output_legs=data.get("output_legs"),
        )
        program.validate()
        return program

