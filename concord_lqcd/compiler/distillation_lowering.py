"""Lower PhysicsIR operator vertices into distillation kernel recipes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from itertools import permutations
from typing import Any, Mapping, Sequence

from concord_lqcd.compiler.physics_ir import (
    OperationKind,
    Representation,
    OperationRecord,
    QuarkRecord,
    SpinTensorRecord,
    ColorTensorRecord,
    GaugeTensorRecord,
    LogicalCorrelator,
    PhysicsIR,
)
from concord_lqcd.backend.ir import CompiledKernelProgram, TensorOp


# ── Stage 1 & Stage 2 Abstractions ─────────────────────────────────────

class ProjectionContribution(Enum):
    POINT = "point"
    EIGENVECTOR = "V"
    EIGENVECTOR_DAGGER = "V_dag"


@dataclass(frozen=True)
class Endpoint:
    """An endpoint (projected field) belonging to an operator vertex.
    
    INVARIANT 5 (Information Locality):
      - Endpoint owns field representation, QuarkRecord, and field-local operations (\nabla, U, S).
    """
    record: QuarkRecord
    operations: tuple[OperationRecord, ...] = ()

    @property
    def pos_label(self) -> str:
        return self.record.pos_label

    @property
    def representation(self) -> Representation:
        return self.record.representation


@dataclass(frozen=True)
class ProjectedField:
    """Represents a single field after distillation realization.
    
    INVARIANT 6 (Single Translation Principle):
      - Translated exactly once from Representation (LAPH/POINT) and is_anti into ProjectionContribution.
      - Retains field-local operations (\nabla, U, S) and field identity.
    """
    record: QuarkRecord
    contribution: ProjectionContribution
    operations: tuple[OperationRecord, ...] = ()


@dataclass(frozen=True)
class OperatorVertex:
    """One connected spatial-color component identified in Stage 1.

    A component is defined by incidence of field endpoints, position indices,
    color indices, color tensors, and explicit gauge tensors. Spin tensors do
    not participate in the partition. Most local operators contain one
    position label per component, while explicitly nonlocal gauge-connected
    structures may contain several.
    """
    vertex_id: str
    position_labels: tuple[str, ...]
    endpoints: tuple[Endpoint, ...]
    spin_tensors: tuple[SpinTensorRecord, ...] = ()
    color_tensors: tuple[ColorTensorRecord, ...] = ()
    gauge_tensors: tuple[GaugeTensorRecord, ...] = ()
    momenta: Mapping[str, tuple[int, int, int]] = field(default_factory=dict)

    @property
    def num_quarks(self) -> int:
        return len(self.endpoints)

    @property
    def pos_label(self) -> str:
        """Backward-compatible local position label for one-position components."""
        if len(self.position_labels) != 1:
            raise ValueError(
                "This spatial-color component contains multiple position labels; "
                "use `position_labels` instead of `pos_label`."
            )
        label = self.position_labels[0]
        return label.split(":")[-1] if ":" in label else label

    @property
    def momentum(self) -> tuple[int, int, int]:
        """Backward-compatible momentum for one-position components."""
        if len(self.position_labels) != 1:
            raise ValueError(
                "This spatial-color component contains multiple position labels; "
                "use `momenta` instead of `momentum`."
            )
        return self.momenta.get(self.position_labels[0], (0, 0, 0))


def _component_vertex_id(position_labels: tuple[str, ...]) -> str:
    if len(position_labels) == 1:
        return position_labels[0]
    return "component[" + "|".join(position_labels) + "]"


def _spatial_color_components(
    correlator: LogicalCorrelator,
) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Partition every endpoint and tensor, including endpoint-free factors."""
    entities: list[tuple[str, Any, tuple[tuple[str, str], ...]]] = []

    def add_entity(
        kind: str,
        payload: Any,
        tokens: Sequence[tuple[str, str]],
    ) -> int:
        idx = len(entities)
        entities.append((kind, payload, tuple(tokens)))
        return idx

    for line in correlator.quark_lines:
        for endpoint in (
            Endpoint(record=line.quark, operations=line.quark_operations),
            Endpoint(record=line.antiquark, operations=line.antiquark_operations),
        ):
            add_entity(
                "endpoint",
                endpoint,
                (
                    ("position", endpoint.record.pos_label),
                    ("color", endpoint.record.color_label),
                ),
            )

    for tensor in correlator.color_tensors:
        add_entity(
            "color",
            tensor,
            tuple(("color", label) for label in tensor.indices),
        )

    for tensor in correlator.gauge_tensors:
        add_entity(
            "gauge",
            tensor,
            (
                ("position", tensor.start_pos),
                ("position", tensor.end_pos),
                ("color", tensor.start_color),
                ("color", tensor.end_color),
            ),
        )

    parent = list(range(len(entities)))
    rank = [0] * len(entities)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    token_owner: dict[tuple[str, str], int] = {}
    for idx, (_, _, tokens) in enumerate(entities):
        for token in tokens:
            if token in token_owner:
                union(idx, token_owner[token])
            else:
                token_owner[token] = idx

    ordered_roots: list[int] = []
    seen_roots: set[int] = set()
    for idx in range(len(entities)):
        root = find(idx)
        if root not in seen_roots:
            seen_roots.add(root)
            ordered_roots.append(root)

    members_by_root: dict[int, list[tuple[str, Any]]] = defaultdict(list)
    for idx, (kind, payload, _) in enumerate(entities):
        members_by_root[find(idx)].append((kind, payload))

    return tuple(tuple(members_by_root[root]) for root in ordered_roots)


@lru_cache(maxsize=128)
def _closed_color_factor(
    color_tensors: tuple[ColorTensorRecord, ...],
    gauge_tensors: tuple[GaugeTensorRecord, ...],
) -> int:
    """Contract a disconnected SU(3) delta/epsilon network with exact integers."""
    if gauge_tensors:
        raise NotImplementedError(
            "Disconnected GaugeLink components are not executable by the current backend."
        )
    counts = Counter(label for tensor in color_tensors for label in tensor.indices)
    if any(count > 2 for count in counts.values()):
        raise ValueError("Disconnected color indices must occur exactly twice.")
    if any(count == 1 for count in counts.values()):
        raise NotImplementedError(
            "Disconnected open-color components are not supported; every color index "
            "must be contracted exactly twice."
        )

    # Delta constraints identify color variables. A class which touches no
    # epsilon is an independent color sum and contributes exactly Nc = 3.
    parent = {label: label for label in counts}

    def find(label: str) -> str:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for tensor in color_tensors:
        if tensor.kind == "delta":
            left, right = tensor.indices
            parent[find(left)] = find(right)
        elif tensor.kind != "epsilon":
            raise ValueError(f"Unsupported disconnected color tensor kind {tensor.kind!r}.")

    classes = tuple(dict.fromkeys(find(label) for label in counts))
    numbers = {label: index for index, label in enumerate(classes)}
    epsilon_indices = tuple(
        tuple(numbers[find(label)] for label in tensor.indices)
        for tensor in color_tensors if tensor.kind == "epsilon"
    )
    if any(len(set(indices)) != 3 for indices in epsilon_indices):
        return 0
    epsilon_classes = {index for indices in epsilon_indices for index in indices}
    delta_factor = 3 ** (len(classes) - len(epsilon_classes))
    assignments = tuple(
        (
            values,
            -1 if sum(values[i] > values[j] for i in range(3) for j in range(i + 1, 3)) % 2 else 1,
        )
        for values in permutations(range(3))
    )

    @lru_cache(maxsize=None)
    def contract(remaining: tuple[int, ...], boundary: tuple[tuple[int, int], ...]) -> int:
        if not remaining:
            return 1
        known = dict(boundary)
        selected = max(
            remaining,
            key=lambda index: sum(label in known for label in epsilon_indices[index]),
        )
        indices = epsilon_indices[selected]
        rest = tuple(index for index in remaining if index != selected)
        needed = {label for index in rest for label in epsilon_indices[index]}
        value = 0
        for colors, sign in assignments:
            if any(label in known and known[label] != color for label, color in zip(indices, colors)):
                continue
            updated = {**known, **dict(zip(indices, colors))}
            next_boundary = tuple(sorted(
                (label, color) for label, color in updated.items() if label in needed
            ))
            value += sign * contract(rest, next_boundary)
        return value

    return delta_factor * contract(tuple(range(len(epsilon_indices))), ())


def disconnected_color_scalar(correlator: LogicalCorrelator) -> int:
    """Return the exact scalar from all endpoint-free color components.

    These constants are folded into the Wick graph coefficient instead of
    requiring an external spatial tensor or a Euclidean-time placement.
    Unsupported endpoint-free tensors fail before any component can be dropped.
    """
    result = 1
    for members in _spatial_color_components(correlator):
        if any(kind == "endpoint" for kind, _ in members):
            continue
        result *= _closed_color_factor(
            tuple(payload for kind, payload in members if kind == "color"),
            tuple(payload for kind, payload in members if kind == "gauge"),
        )
    return result


def identify_operator_vertices(correlator: LogicalCorrelator) -> tuple[OperatorVertex, ...]:
    """Stage 1: identify every spatial-color component containing fields.

    Position/color incidence determines the partition independently of spin or
    hadron type. Endpoint-free closed delta/epsilon components are validated
    here and folded by ``disconnected_color_scalar`` during graph lowering.
    Endpoint-free open-color and gauge components are explicitly unsupported.
    """
    vertices: list[OperatorVertex] = []
    for members in _spatial_color_components(correlator):
        endpoints = tuple(payload for kind, payload in members if kind == "endpoint")
        color_tensors = tuple(payload for kind, payload in members if kind == "color")
        gauge_tensors = tuple(payload for kind, payload in members if kind == "gauge")
        if not endpoints:
            _closed_color_factor(color_tensors, gauge_tensors)
            continue

        position_labels_list: list[str] = []
        for endpoint in endpoints:
            if endpoint.record.pos_label not in position_labels_list:
                position_labels_list.append(endpoint.record.pos_label)
        for tensor in gauge_tensors:
            for label in (tensor.start_pos, tensor.end_pos):
                if label not in position_labels_list:
                    position_labels_list.append(label)
        position_labels = tuple(position_labels_list)

        operator_ids = {
            endpoint.record.operator_id
            for endpoint in endpoints
            if endpoint.record.operator_id is not None
        }
        if operator_ids:
            spin_tensors = tuple(
                tensor
                for tensor in correlator.spin_tensors
                if tensor.operator_id is None or tensor.operator_id in operator_ids
            )
        else:
            spin_tensors = tuple(correlator.spin_tensors)

        momenta = {
            label: correlator.momenta[label]
            for label in position_labels
            if label in correlator.momenta
        }

        vertices.append(OperatorVertex(
            vertex_id=_component_vertex_id(position_labels),
            position_labels=position_labels,
            endpoints=endpoints,
            spin_tensors=spin_tensors,
            color_tensors=color_tensors,
            gauge_tensors=gauge_tensors,
            momenta=momenta,
        ))

    return tuple(vertices)


@dataclass(frozen=True)
class RealizedOperatorVertex:
    """Stage 2: An operator vertex whose fields have been realized into distillation contributions."""
    vertex: OperatorVertex
    projected_fields: tuple[ProjectedField, ...]


def realize_projected_fields(vertex: OperatorVertex) -> RealizedOperatorVertex:
    """Stage 2: Realize projected fields for an OperatorVertex into explicit ProjectionContribution values."""
    p_fields: list[ProjectedField] = []
    for ep in vertex.endpoints:
        if ep.representation == Representation.POINT:
            contrib = ProjectionContribution.POINT
        elif ep.record.is_anti:
            contrib = ProjectionContribution.EIGENVECTOR_DAGGER
        else:
            contrib = ProjectionContribution.EIGENVECTOR

        p_fields.append(ProjectedField(
            record=ep.record,
            contribution=contrib,
            operations=ep.operations,
        ))

    return RealizedOperatorVertex(
        vertex=vertex,
        projected_fields=tuple(p_fields),
    )


@dataclass(frozen=True)
class KernelRecipe:
    """Stage-3 symbolic spatial kernel built from fundamental tensor objects.

    A recipe represents one Stage-1 spatial-color connected component after
    Stage-2 field realization.  It is independent of hadron type: no meson,
    baryon, tetraquark, or rank-specific branch is encoded here.

    The kernel contains only spatial/color information:
      * realized field factors (V, V^dagger, or point contribution),
      * field-local spatial transformations,
      * color tensors and explicit gauge tensors,
      * all position variables and their momentum projections.

    Spin tensors are deliberately excluded.  The open ports are inferred from
    the realized fields: LapH fields expose mode ports, while point fields keep
    a color-like external port until the propagation layer decides how that
    object is supplied.
    """
    vertex_id: str
    position_labels: tuple[str, ...]
    momenta: Mapping[str, tuple[int, int, int]]
    projected_fields: tuple[ProjectedField, ...]
    color_tensors: tuple[ColorTensorRecord, ...] = ()
    gauge_tensors: tuple[GaugeTensorRecord, ...] = ()
    open_ports: tuple[str, ...] = ()

    @property
    def num_quarks(self) -> int:
        return len(self.projected_fields)

    @property
    def pos_label(self) -> str:
        """Backward-compatible local label for one-position kernels."""
        if len(self.position_labels) != 1:
            raise ValueError(
                "This kernel contains multiple position labels; use "
                "`position_labels` instead of `pos_label`."
            )
        label = self.position_labels[0]
        return label.split(":")[-1] if ":" in label else label

    @property
    def momentum(self) -> tuple[int, int, int]:
        """Backward-compatible momentum for one-position kernels."""
        if len(self.position_labels) != 1:
            raise ValueError(
                "This kernel contains multiple position labels; use `momenta` "
                "instead of `momentum`."
            )
        return self.momenta.get(self.position_labels[0], (0, 0, 0))

    @property
    def is_exact_mode_identity(self) -> bool:
        """Whether this recipe is an identity under the shared LapH contract.

        This structural predicate does not prove equality or orthonormality of
        numerical flavor bases. The caller must declare those preconditions
        before using the elementary relation ``V^dagger V = I`` to eliminate
        the kernel. Any explicit color tensor, gauge link,
        field transformation, non-locality, or momentum phase keeps the
        kernel explicit until a separate exact rule proves it reducible.
        """
        if len(self.position_labels) != 1 or len(self.projected_fields) != 2:
            return False
        if self.color_tensors or self.gauge_tensors:
            return False
        if self.open_ports != ("mode_0", "mode_1"):
            return False

        position_label = self.position_labels[0]
        if self.momenta.get(position_label, (0, 0, 0)) != (0, 0, 0):
            return False
        if any(momentum != (0, 0, 0) for momentum in self.momenta.values()):
            return False

        first_field, second_field = self.projected_fields
        if first_field.operations or second_field.operations:
            return False
        if first_field.record.pos_label != position_label:
            return False
        if second_field.record.pos_label != position_label:
            return False
        if first_field.record.color_label != second_field.record.color_label:
            return False

        contributions = {
            first_field.contribution,
            second_field.contribution,
        }
        return contributions == {
            ProjectionContribution.EIGENVECTOR,
            ProjectionContribution.EIGENVECTOR_DAGGER,
        }


def construct_kernel_recipe(realized_vertex: RealizedOperatorVertex) -> KernelRecipe:
    """Stage 3: materialize one generic spatial/color kernel recipe.

    The algorithm is rank-agnostic.  Every realized field contributes one
    external port and the existing symbolic position/color incidence determines
    the contractions.  No object-type recognition is performed.
    """
    v = realized_vertex.vertex
    open_ports: list[str] = []

    for idx, pf in enumerate(realized_vertex.projected_fields):
        if pf.contribution in (
            ProjectionContribution.EIGENVECTOR,
            ProjectionContribution.EIGENVECTOR_DAGGER,
        ):
            open_ports.append(f"mode_{idx}")
        else:
            # POINT is retained as a distinct external spatial/color object.
            # It is not reinterpreted as an eigenvector mode.
            open_ports.append(f"color_{idx}")

    return KernelRecipe(
        vertex_id=v.vertex_id,
        position_labels=v.position_labels,
        momenta=dict(v.momenta),
        projected_fields=realized_vertex.projected_fields,
        color_tensors=v.color_tensors,
        gauge_tensors=v.gauge_tensors,
        open_ports=tuple(open_ports),
    )


# ── Kernel resource identity ──────────────────────────────────────────

def kernel_resource_signature(kernel_recipe: KernelRecipe) -> str:
    """Return the canonical, alpha-renaming-invariant kernel signature."""

    position_names: dict[str, str] = {}
    color_names: dict[str, str] = {}

    def canonical_position(label: str) -> str:
        if label not in position_names:
            position_names[label] = f"p{len(position_names)}"
        return position_names[label]

    def canonical_color(label: str) -> str:
        if label not in color_names:
            color_names[label] = f"c{len(color_names)}"
        return color_names[label]

    field_payloads = []
    for pf in kernel_recipe.projected_fields:
        field_payloads.append({
            "flavor": pf.record.flavor,
            "contribution": pf.contribution.value,
            "position": canonical_position(pf.record.pos_label),
            "color": canonical_color(pf.record.color_label),
            "operations": [
                {"kind": op.kind.value, **dict(op.parameters)}
                for op in pf.operations
            ],
        })

    color_payloads = [
        {
            "kind": tensor.kind,
            "indices": tuple(canonical_color(label) for label in tensor.indices),
        }
        for tensor in kernel_recipe.color_tensors
    ]
    gauge_payloads = [
        {
            "start_pos": canonical_position(tensor.start_pos),
            "end_pos": canonical_position(tensor.end_pos),
            "start_color": canonical_color(tensor.start_color),
            "end_color": canonical_color(tensor.end_color),
        }
        for tensor in kernel_recipe.gauge_tensors
    ]
    momentum_payload = [
        (canonical_position(label), tuple(kernel_recipe.momenta.get(label, (0, 0, 0))))
        for label in kernel_recipe.position_labels
    ]

    payload = {
        "fields": field_payloads,
        "color": color_payloads,
        "gauge": gauge_payloads,
        "momenta": momentum_payload,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def identify_kernel_resource(kernel_recipe: KernelRecipe) -> str:
    """Stage 6: assign identity to a reusable spatial/color kernel.

    The identity is alpha-renaming invariant and excludes spin, but retains
    flavor because tensor providers may supply different eigenvectors per flavor.
    Momentum is included because it changes the numerical spatial tensor.
    """
    encoded = kernel_resource_signature(kernel_recipe).encode("utf-8")
    return "spatial_kernel_" + hashlib.sha256(encoded).hexdigest()[:16]


def emit_backend_kernel_program(kernel_recipe: KernelRecipe) -> CompiledKernelProgram:
    """Stage 7 compatibility bridge for the Stage-3 spatial/color recipe.

    Spin tensors are intentionally absent: this program represents only the
    reusable spatial/color kernel.  Explicit gauge tensors remain unsupported
    by the current numerical ISA even though Stage 3 can represent them.
    """
    tensor_ops: list[TensorOp] = []
    num_quarks = kernel_recipe.num_quarks

    if kernel_recipe.gauge_tensors:
        raise NotImplementedError(
            "Explicit GaugeLink tensor nodes are represented by the symbolic "
            "kernel recipe but are not yet executable by the current backend ISA."
        )

    for idx, pf in enumerate(kernel_recipe.projected_fields):
        field_attrs: dict[str, Any] = {}
        if pf.operations:
            deriv_list = []
            operation_list = []
            for op in pf.operations:
                operation = {"kind": op.kind.value, **dict(op.parameters)}
                if "path" in operation:
                    operation["path"] = list(operation["path"])
                operation_list.append(operation)
                if op.kind == OperationKind.DERIVATIVE:
                    direction = op.parameters.get("direction", "+x")
                    deriv_list.append((direction.lstrip("+-"), direction.startswith("-")))
            if deriv_list:
                field_attrs["derivatives"] = deriv_list
            field_attrs["operations"] = operation_list

        tensor_ops.append(TensorOp(
            kind="eigenvector",
            params={
                "source": (
                    "distillation"
                    if pf.contribution in (
                        ProjectionContribution.EIGENVECTOR,
                        ProjectionContribution.EIGENVECTOR_DAGGER,
                    )
                    else "point"
                ),
                "flavor": pf.record.flavor,
                "is_anti": pf.contribution == ProjectionContribution.EIGENVECTOR_DAGGER,
                "field_id": idx,
                "field_attributes": field_attrs,
                "legs": [
                    "T",
                    pf.record.pos_label,
                    pf.record.color_label,
                    kernel_recipe.open_ports[idx],
                ],
            },
        ))

    for position_label in kernel_recipe.position_labels:
        tensor_ops.append(TensorOp(
            kind="phase",
            params={
                "momentum": list(kernel_recipe.momenta.get(position_label, (0, 0, 0))),
                "legs": [position_label],
            },
        ))

    for tensor in kernel_recipe.color_tensors:
        kind = "color_delta" if tensor.kind == "delta" else "color_epsilon"
        tensor_ops.append(TensorOp(
            kind=kind,
            params={"legs": list(tensor.indices)},
        ))

    return CompiledKernelProgram(
        num_quarks=num_quarks,
        operations=tensor_ops,
        mode_ports=list(kernel_recipe.open_ports),
        spin_ports=[],
        output_legs=list(kernel_recipe.open_ports),
    )
