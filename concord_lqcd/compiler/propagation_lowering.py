"""Describe external propagation resources for exact fermion segments.

This module consumes chains derived from the canonical fermion topology and
describes which precomputed propagation tensor is required. It never evaluates
or constructs a perambulator/generalized perambulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from concord_lqcd.compiler.physics_ir import Representation, SpinTensorRecord
from concord_lqcd.compiler.distillation_lowering import (
    Endpoint,
    KernelRecipe,
    identify_kernel_resource,
)
from concord_lqcd.compiler.fermion_topology import FermionVertexTransition, QuarkSegment


@dataclass(frozen=True)
class PropagationInsertionSpec:
    """One ordered current insertion absorbed into a generalized resource.

    ``kernel`` carries only the spatial/color specification.  ``spin_tensors``
    carries the independent spin structure.  Incoming/outgoing flavor and the
    exact kernel ports are retained explicitly so flavor-changing currents and
    repeated quark legs are unambiguous.
    """

    kernel: KernelRecipe
    incoming_flavor: str
    outgoing_flavor: str
    incoming_port: str
    outgoing_port: str
    incoming_spin_label: str
    outgoing_spin_label: str
    spin_tensors: tuple[SpinTensorRecord, ...] = ()

    @property
    def spatial_resource_id(self) -> str:
        return identify_kernel_resource(self.kernel)


@dataclass(frozen=True)
class PropagationRecipe:
    """Logical requirement for one precomputed propagation tensor.

    A recipe is deliberately independent of filenames, HDF5 groups, source
    times, device placement, or execution strategy.  ``resource_id`` is a
    deterministic logical identity that the provider/IO layer may map to an
    on-disk dataset.

    With no insertion the resource is an ordinary perambulator.  With one or
    more explicitly POINT bilinear insertions the resource is a generalized
    perambulator containing those insertions in path order.
    """

    start_kernel: KernelRecipe
    end_kernel: KernelRecipe
    start_port: str
    end_port: str
    segment_flavors: tuple[str, ...]
    source_representation: Representation
    sink_representation: Representation
    insertion_specs: tuple[PropagationInsertionSpec, ...] = ()

    @property
    def inserted_kernels(self) -> tuple[KernelRecipe, ...]:
        """Compatibility/readability view of the ordered insertion kernels."""
        return tuple(spec.kernel for spec in self.insertion_specs)

    @property
    def flavor(self) -> str:
        """Compatibility view for flavor-preserving/first-segment callers."""
        return self.segment_flavors[0] if self.segment_flavors else ""

    @property
    def flavors(self) -> tuple[str, ...]:
        return self.segment_flavors

    @property
    def incoming_flavor(self) -> str:
        return self.segment_flavors[0] if self.segment_flavors else ""

    @property
    def outgoing_flavor(self) -> str:
        return self.segment_flavors[-1] if self.segment_flavors else ""

    @property
    def is_generalized(self) -> bool:
        return bool(self.insertion_specs)

    @property
    def resource_kind(self) -> str:
        return "generalized_perambulator" if self.is_generalized else "perambulator"

    @property
    def resource_signature(self) -> str:
        """Canonical resource specification, excluding endpoint placement."""
        payload = {
            "kind": self.resource_kind,
            "source_representation": self.source_representation.value,
            "sink_representation": self.sink_representation.value,
            "segment_flavors": list(self.segment_flavors),
            "insertions": [_insertion_resource_payload(spec) for spec in self.insertion_specs],
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @property
    def resource_id(self) -> str:
        """Deterministic logical identity of the required external tensor."""
        encoded = self.resource_signature.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:20]
        return f"{self.resource_kind}_{digest}"


def _unique_kernels(kernel_map: Mapping[str, KernelRecipe]) -> tuple[KernelRecipe, ...]:
    kernels: list[KernelRecipe] = []
    seen: set[int] = set()
    for kernel in kernel_map.values():
        marker = id(kernel)
        if marker not in seen:
            seen.add(marker)
            kernels.append(kernel)
    return tuple(kernels)


def _field_matches_endpoint(kernel: KernelRecipe, endpoint: Endpoint) -> list[int]:
    """Return candidate field slots for one exact path endpoint.

    The in-memory PhysicsIR objects preserve record identity, which is the
    strongest match.  A unique structural match is retained as a fallback for
    hand-built/deserialized fixtures.  Ambiguity is never resolved by field
    order alone.
    """
    identity_matches = [
        idx
        for idx, field in enumerate(kernel.projected_fields)
        if field.record is endpoint.record and field.operations == endpoint.operations
    ]
    if identity_matches:
        return identity_matches

    return [
        idx
        for idx, field in enumerate(kernel.projected_fields)
        if field.record == endpoint.record and field.operations == endpoint.operations
    ]


def _kernel_and_port_for_endpoint(
    endpoint: Endpoint,
    kernels: Sequence[KernelRecipe],
) -> tuple[KernelRecipe, str]:
    candidates: list[tuple[KernelRecipe, int]] = []
    for kernel in kernels:
        matches = _field_matches_endpoint(kernel, endpoint)
        if len(matches) > 1:
            raise ValueError(
                "A kernel contains multiple structurally identical endpoint fields; "
                "Stage 5 cannot infer an exact propagation port from value equality."
            )
        if matches:
            candidates.append((kernel, matches[0]))

    if len(candidates) != 1:
        raise ValueError(
            "Stage 5 requires exactly one kernel/port for every quark-path endpoint; "
            f"found {len(candidates)} candidates for {endpoint.record!r}."
        )

    kernel, field_index = candidates[0]
    return kernel, kernel.open_ports[field_index]


def _insertion_kernel_and_ports(
    insertion: FermionVertexTransition,
    kernels: Sequence[KernelRecipe],
) -> tuple[KernelRecipe, str, str]:
    incoming_kernel, incoming_port = _kernel_and_port_for_endpoint(
        insertion.quark_endpoint, kernels
    )
    outgoing_kernel, outgoing_port = _kernel_and_port_for_endpoint(
        insertion.antiquark_endpoint, kernels
    )
    if incoming_kernel is not outgoing_kernel:
        raise NotImplementedError(
            "A precomputed local-current generalized perambulator requires the two "
            "current endpoints to belong to one spatial-color component."
        )
    return incoming_kernel, incoming_port, outgoing_port


def _canonical_spin_payload(spec: PropagationInsertionSpec) -> dict[str, object]:
    names: dict[str, str] = {}

    def spin(label: str) -> str:
        if label not in names:
            names[label] = f"s{len(names)}"
        return names[label]

    # Anchor alpha-renaming to the physical current roles before visiting any
    # intermediate spin labels introduced by gamma/dagger chains.
    outgoing = spin(spec.outgoing_spin_label)
    incoming = spin(spec.incoming_spin_label)
    tensors = [
        {
            "kind": tensor.kind,
            "left": spin(tensor.left_spin),
            "right": spin(tensor.right_spin),
        }
        for tensor in spec.spin_tensors
    ]
    return {
        "outgoing_endpoint": outgoing,
        "incoming_endpoint": incoming,
        "tensors": tensors,
    }


def _insertion_resource_payload(spec: PropagationInsertionSpec) -> dict[str, object]:
    try:
        incoming_port_index = spec.kernel.open_ports.index(spec.incoming_port)
        outgoing_port_index = spec.kernel.open_ports.index(spec.outgoing_port)
    except ValueError as exc:
        raise ValueError("Insertion port is not present in its KernelRecipe.") from exc

    return {
        "incoming_flavor": spec.incoming_flavor,
        "outgoing_flavor": spec.outgoing_flavor,
        "spatial_resource": spec.spatial_resource_id,
        "incoming_spatial_leg": incoming_port_index,
        "outgoing_spatial_leg": outgoing_port_index,
        "spin": _canonical_spin_payload(spec),
    }


def _build_insertion_spec(
    insertion: FermionVertexTransition,
    kernels: Sequence[KernelRecipe],
) -> PropagationInsertionSpec:
    if (
        insertion.quark_endpoint.representation != Representation.POINT
        or insertion.antiquark_endpoint.representation != Representation.POINT
    ):
        raise NotImplementedError(
            "Stage 5 absorbs only explicitly POINT local bilinear insertions into "
            "precomputed generalized perambulators.  Use point(qbar) * Gamma * "
            "point(q) for this workflow."
        )

    kernel, incoming_port, outgoing_port = _insertion_kernel_and_ports(
        insertion, kernels
    )
    return PropagationInsertionSpec(
        kernel=kernel,
        incoming_flavor=insertion.incoming_flavor,
        outgoing_flavor=insertion.outgoing_flavor,
        incoming_port=incoming_port,
        outgoing_port=outgoing_port,
        incoming_spin_label=insertion.quark_endpoint.record.spin_label,
        outgoing_spin_label=insertion.antiquark_endpoint.record.spin_label,
        spin_tensors=insertion.spin_tensors,
    )


def _infer_propagation_recipe(
    start_endpoint: Endpoint,
    end_endpoint: Endpoint,
    segments: Sequence[QuarkSegment],
    insertions: Sequence[FermionVertexTransition],
    kernel_map: Mapping[str, KernelRecipe],
) -> PropagationRecipe:
    """Infer an external resource for one linear chain from fermion topology."""
    if not segments:
        raise ValueError("A propagation resource requires at least one Wick segment.")

    kernels = _unique_kernels(kernel_map)
    start_kernel, start_port = _kernel_and_port_for_endpoint(start_endpoint, kernels)
    end_kernel, end_port = _kernel_and_port_for_endpoint(end_endpoint, kernels)

    insertion_specs = tuple(
        _build_insertion_spec(transition, kernels)
        for transition in insertions
    )

    if len(segments) != len(insertion_specs) + 1:
        raise ValueError(
            "A linear generalized propagation path must contain exactly one more "
            "propagator segment than absorbed current insertions."
        )

    for index, transition in enumerate(insertions):
        if segments[index].sink_endpoint != transition.quark_endpoint:
            raise ValueError("Insertion quark endpoint does not match the preceding segment.")
        if segments[index + 1].source_endpoint != transition.antiquark_endpoint:
            raise ValueError("Insertion antiquark endpoint does not match the following segment.")

    segment_flavors = tuple(segment.flavor for segment in segments)
    for idx, spec in enumerate(insertion_specs):
        if segment_flavors[idx] != spec.incoming_flavor:
            raise ValueError("Insertion incoming flavor does not match the preceding Wick segment.")
        if segment_flavors[idx + 1] != spec.outgoing_flavor:
            raise ValueError("Insertion outgoing flavor does not match the following Wick segment.")

    return PropagationRecipe(
        start_kernel=start_kernel,
        end_kernel=end_kernel,
        start_port=start_port,
        end_port=end_port,
        segment_flavors=segment_flavors,
        source_representation=start_endpoint.representation,
        sink_representation=end_endpoint.representation,
        insertion_specs=insertion_specs,
    )
