"""Optional, certified whole-term reuse after flavor-respecting Wick lowering.

Discovery proposes tensor-network matches; a separate incidence and permutation
verifier must certify every accepted match. Unsupported resources and exhausted
searches retain their original work. No numerical samples establish equality.
Only deterministic counts can end a search early, so the result depends on the
source and the configured counts alone; the wall-clock limit is a hard abort.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
from math import isfinite
from time import monotonic
from typing import Mapping

import numpy as np

from ._tensor_isomorphism import (
    Factor, Network, SearchAborted, SearchBudgetExceeded, find_witness, graph,
    index_kinds, invariant, parity, verify_witness,
)
from ._spin_chains import _ExactComplex, discover_spin_chains, replay_spin_chains
from .distillation_lowering import KernelRecipe, ProjectionContribution
from .distillation_lowering import identify_kernel_resource, kernel_resource_signature
from .global_cse import GlobalComputationDAG, GlobalCSEStatistics
from .numerical_flavors import canonical_flavor
from .physics_ir import Representation, SpinTensorRecord
from .propagation_lowering import PropagationRecipe


@dataclass(frozen=True)
class WickAggregationConfig:
    """One-time discovery limits; exhaustion cannot authorize a rewrite.

    ``maximum_checks`` bounds the witness searches and ``maximum_match_states``
    the refined nodes of each search. Exhausting either keeps the unproved terms
    separate and is reported. ``search_seconds`` never changes a result:
    exceeding it aborts plan creation with ``WickAggregationTimeLimitExceeded``.
    """

    enabled: bool = False
    search_seconds: float = 3600.0
    maximum_checks: int = 1_000_000
    maximum_match_states: int = 10000

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("Wick aggregation enabled must be a boolean.")
        value = self.search_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("search_seconds must be finite and positive.")
        if not isfinite(value) or value <= 0:
            raise ValueError("Invalid Wick aggregation search_seconds.")
        for name in ("maximum_checks", "maximum_match_states"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")


class _UnsupportedContract(ValueError):
    pass


class WickAggregationTimeLimitExceeded(RuntimeError):
    """The hard wall-clock limit expired; no plan is produced from a partial search."""


@dataclass(frozen=True)
class CertifiedTermClass:
    members: tuple[int, ...]
    witnesses: tuple[Mapping[str, object], ...] = ()
    spin_chains: tuple[tuple[tuple[tuple[int, bool], ...], ...], ...] = ()


@dataclass(frozen=True)
class CorrelatorAggregationProof:
    correlator_id: str
    classes: tuple[CertifiedTermClass, ...]


@dataclass(frozen=True)
class WickAggregationProof:
    source: GlobalComputationDAG
    correlators: tuple[CorrelatorAggregationProof, ...]
    spin_basis: str
    shared_orthonormal_laph_basis: bool
    numerical_flavor_groups: tuple[tuple[str, ...], ...]


def _certificate_value(value):
    if isinstance(value, Mapping):
        return {key: _certificate_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_certificate_value(item) for item in value)
    return value


def _spin_contract(value: np.ndarray) -> tuple[str, bool, _ExactComplex, str]:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise _UnsupportedContract("non-finite or non-four-spin constant")
    choices = []
    for transpose in (False, True):
        oriented = matrix.T if transpose else matrix
        for phase in (1, -1, 1j, -1j):
            candidate = (oriented / phase).copy()
            candidate.real[candidate.real == 0] = 0
            candidate.imag[candidate.imag == 0] = 0
            choices.append((candidate.astype("<c16").tobytes(), transpose, phase, candidate))
    encoded, transpose, phase, canonical = min(choices, key=lambda item: (item[0], item[1]))
    symmetry = "symmetric" if np.array_equal(canonical.T, canonical) else (
        "alternating" if np.array_equal(canonical.T, -canonical) else "ordered"
    )
    return "fixed-spin:" + encoded.hex(), transpose, _ExactComplex.from_complex(phase), symmetry


def _local_baryon_sign(resource: KernelRecipe, operator: int) -> int | None:
    """Sign mapping a local three-mode epsilon kernel to its alternating form, else None.

    Three undisplaced LapH legs at one position, one epsilon and no gauge links:
    the kernel is alternating in its mode ports at every momentum projection.
    """
    contribution = (ProjectionContribution.EIGENVECTOR_DAGGER if operator
                    else ProjectionContribution.EIGENVECTOR)
    if (resource.num_quarks != 3 or len(resource.position_labels) != 1 or len(resource.open_ports) != 3
            or len(resource.color_tensors) != 1 or resource.color_tensors[0].kind != "epsilon"
            or resource.gauge_tensors
            or any(field.operations or field.contribution != contribution
                   or field.record.is_anti != bool(operator)
                   or field.record.pos_label != resource.position_labels[0]
                   for field in resource.projected_fields)):
        return None
    colors = tuple(field.record.color_label for field in resource.projected_fields)
    epsilon = resource.color_tensors[0].indices
    if len(set(colors)) != 3 or len(epsilon) != 3 or set(epsilon) != set(colors):
        return None
    return parity(tuple(colors.index(label) for label in epsilon))


def _flavorless_signature(resource: KernelRecipe) -> str:
    """The kernel's ordered, alpha-renamed signature without flavor labels.

    Fields keep their port order and epsilon indices their order, so equal
    signatures are the same tensor with the same port order. Kernels contain
    only eigenvectors and gauge links, so under the declared shared LapH basis
    they do not depend on the quark flavors.
    """
    payload = json.loads(kernel_resource_signature(resource))
    for field in payload["fields"]:
        del field["flavor"]
    return json.dumps(payload, sort_keys=True)


def _validate_source(dag):
    from .contraction_graph import PrimitiveTensorNode, WickContractionGraph
    from .contraction_order import infer_component_index_spaces

    spaces = {}
    for identifier, resource in dag.resources.items():
        if identifier != resource.resource_id:
            raise ValueError("Wick aggregation source resource is stored under the wrong identity.")
        if resource.kind == "kernel":
            if not isinstance(resource.payload, KernelRecipe) or identify_kernel_resource(resource.payload) != identifier:
                raise ValueError("Wick aggregation source has an inconsistent kernel resource.")
        elif resource.kind == "propagation":
            if not isinstance(resource.payload, PropagationRecipe) or resource.payload.resource_id != identifier:
                raise ValueError("Wick aggregation source has an inconsistent propagation resource.")
        elif resource.kind != "spin" or not isinstance(resource.payload, SpinTensorRecord):
            raise ValueError("Wick aggregation source has an unknown resource kind.")
    for identifier, component in dag.components.items():
        if component.computation_id != identifier:
            raise ValueError("Wick aggregation source component is stored under the wrong identity.")
        nodes = []
        for tensor in component.tensors:
            if tensor.resource_id not in dag.resources:
                raise ValueError("Wick aggregation source component references a missing resource.")
            resource = dag.resources[tensor.resource_id]
            if tensor.kind != resource.kind:
                raise ValueError("Wick aggregation source tensor changes its resource kind.")
            expected_ports = (resource.payload.open_ports if tensor.kind == "kernel" else (
                ("left", "right") if tensor.kind == "spin" else
                ("sink_mode", "sink_spin", "source_mode", "source_spin")
            ))
            if tuple(binding.port for binding in tensor.bindings) != expected_ports:
                raise ValueError("Wick aggregation source tensor has missing, extra or reordered ports.")
            nodes.append(PrimitiveTensorNode(tensor.slot, tensor.kind, tensor.resource_id,
                                             tensor.bindings, resource.payload))
        graph_value = WickContractionGraph(1, tuple(nodes))
        graph_value.validate()
        outputs = {index for index, degree in graph_value.index_degrees.items() if degree == 1}
        if len(set(component.output_indices)) != len(component.output_indices) or set(component.output_indices) != outputs:
            raise ValueError("Wick aggregation source component has an inconsistent output interface.")
        spaces[identifier] = infer_component_index_spaces(component)
    identities = set()
    for correlator in dag.correlators:
        if correlator.correlator_id in identities:
            raise ValueError("Wick aggregation source has duplicate correlator identities.")
        identities.add(correlator.correlator_id)
        indices = set()
        interface = None
        for term in correlator.terms:
            if type(term.term_index) is not int or term.term_index < 0 or term.term_index in indices:
                raise ValueError("Wick aggregation source has invalid term identities.")
            indices.add(term.term_index)
            coefficient = complex(term.coefficient)
            if not isfinite(coefficient.real) or not isfinite(coefficient.imag):
                raise ValueError("Wick aggregation source has a non-finite coefficient.")
            exposed = {}
            for use in term.components:
                if use.computation_id not in dag.components:
                    raise ValueError("Wick aggregation source term references a missing component.")
                component = dag.components[use.computation_id]
                ports = tuple(binding.port for binding in use.output_bindings)
                if len(set(ports)) != len(ports) or set(ports) != set(component.output_indices):
                    raise ValueError("Wick aggregation source term has an incomplete component output map.")
                for binding in use.output_bindings:
                    if not binding.index or binding.index in exposed:
                        raise ValueError("Wick aggregation source term repeats or omits an external axis.")
                    exposed[binding.index] = spaces[use.computation_id][binding.port]
            current = tuple(sorted(exposed.items()))
            if interface is not None and current != interface:
                raise ValueError("Wick aggregation source terms have different typed output interfaces.")
            interface = current


class _ResourceContracts:
    def __init__(self, dag, *, spin_basis, shared_basis, numerical_flavor_groups):
        from concord_lqcd.core.dirac_basis import get_spin_matrices

        self.dag = dag
        self.shared_basis = shared_basis
        self.groups = numerical_flavor_groups
        self.matrices = get_spin_matrices(spin_basis)
        self.spin_cache = {}

    def term(self, term):
        if not self.shared_basis:
            raise _UnsupportedContract("shared orthonormal LapH basis not declared")
        factors = []
        phase = _ExactComplex.from_complex(1)
        exposed = []
        for occurrence, use in enumerate(term.components):
            component = self.dag.components[use.computation_id]
            external = {binding.port: "external:" + binding.index for binding in use.output_bindings}
            exposed.extend(external.values())
            for tensor in component.tensors:
                resource = self.dag.resources[tensor.resource_id].payload
                labels = {binding.port: external.get(binding.index, f"internal:{occurrence}:{binding.index}")
                          for binding in tensor.bindings}
                if tensor.kind == "kernel":
                    if not isinstance(resource, KernelRecipe):
                        raise _UnsupportedContract("unknown kernel recipe")
                    if tensor.placement[0] != "kernel_ops" or len(tensor.placement[1]) != 1:
                        raise _UnsupportedContract("kernel has multiple operator placements")
                    operator = tensor.placement[1][0]
                    if operator not in (0, 1):
                        raise _UnsupportedContract("kernel is not in a two-point operator slot")
                    if set(labels) != set(resource.open_ports) or len(labels) != len(resource.open_ports):
                        raise _UnsupportedContract("kernel interface does not match its ports")
                    if any(field.record.representation != Representation.LAPH
                           or field.contribution not in (ProjectionContribution.EIGENVECTOR,
                                                         ProjectionContribution.EIGENVECTOR_DAGGER)
                           for field in resource.projected_fields):
                        raise _UnsupportedContract("kernel has a non-LapH leg")
                    ports = tuple(labels[port] for port in resource.open_ports)
                    sign = _local_baryon_sign(resource, operator)
                    if sign is not None:
                        phase = phase * _ExactComplex.from_complex(sign)
                        momentum = tuple(resource.momenta.get(resource.position_labels[0], (0, 0, 0)))
                        factors.append(Factor(f"local-baryon:{operator}:{momentum}", ports, ("mode",) * 3, "alternating"))
                    else:
                        factors.append(Factor(f"kernel:{operator}:{_flavorless_signature(resource)}", ports,
                                              ("mode",) * len(ports), "ordered"))
                elif tensor.kind == "spin":
                    if not isinstance(resource, SpinTensorRecord) or resource.kind not in self.matrices:
                        raise _UnsupportedContract("unknown fixed spin constant")
                    if resource.kind not in self.spin_cache:
                        self.spin_cache[resource.kind] = _spin_contract(self.matrices[resource.kind])
                    identifier, transpose, extra, symmetry = self.spin_cache[resource.kind]
                    ordered = (labels["left"], labels["right"])
                    factors.append(Factor(identifier, ordered[::-1] if transpose else ordered,
                                          ("spin", "spin"), symmetry))
                    phase = phase * extra
                elif tensor.kind == "propagation":
                    if (not isinstance(resource, PropagationRecipe) or resource.is_generalized
                            or len(resource.segment_flavors) != 1
                            or resource.source_representation != Representation.LAPH
                            or resource.sink_representation != Representation.LAPH):
                        raise _UnsupportedContract("propagation is not ordinary LapH propagation")
                    if tensor.placement[0] != "path_ops" or tensor.placement[2] != ():
                        raise _UnsupportedContract("propagation has insertion placements")
                    start = resource.start_kernel.projected_fields[resource.start_kernel.open_ports.index(resource.start_port)]
                    end = resource.end_kernel.projected_fields[resource.end_kernel.open_ports.index(resource.end_port)]
                    # Field operations act on eigenvectors inside the kernels; the LapH perambulator is unchanged.
                    if not start.record.is_anti or end.record.is_anti:
                        raise _UnsupportedContract("propagation endpoints are not a quark-antiquark pair")
                    source, sink = tensor.placement[1], tensor.placement[3]
                    if source not in (0, 1) or sink not in (0, 1):
                        raise _UnsupportedContract("propagation is not in two-point slots")
                    flavor = canonical_flavor(resource.segment_flavors[0], self.groups)
                    identifier = repr(("ordinary-qbar", flavor, source, sink))
                    factors.append(Factor(identifier, tuple(labels[port] for port in (
                        "sink_mode", "sink_spin", "source_mode", "source_spin")),
                        ("mode", "spin", "mode", "spin"), "ordered"))
                else:
                    raise _UnsupportedContract("unknown primitive tensor kind")
        if len(set(exposed)) != len(exposed):
            raise _UnsupportedContract("repeated component output indices")
        network = Network(tuple(factors), tuple(sorted(exposed)))
        index_kinds(network)
        return network, phase


def _apply_correlator_proof(correlator, proof, contracts):
    members = [member for group in proof.classes for member in group.members]
    if any(type(member) is not int for member in members) or sorted(members) != list(range(len(correlator.terms))):
        raise ValueError("Wick aggregation certificate does not partition the source terms.")
    kept = []
    audit = []
    has_merges = any(len(group.members) > 1 for group in proof.classes)
    for group in proof.classes:
        if not group.members or len(group.witnesses) != len(group.members) - 1:
            raise ValueError("Wick aggregation certificate has missing or extra witnesses.")
        if len(group.spin_chains) != (len(group.members) if len(group.members) > 1 else 0):
            raise ValueError("Wick aggregation certificate has missing or extra spin-chain proofs.")
        representative = correlator.terms[group.members[0]]
        if len(group.members) == 1:
            if representative.coefficient != 0 or not has_merges:
                kept.append(replace(representative, term_index=len(kept)))
            continue
        network, phase = contracts.term(representative)
        network, spin_phase = replay_spin_chains(network, group.spin_chains[0])
        phase = phase * spin_phase
        aligned = [_ExactComplex.from_complex(representative.coefficient)]
        signs = [1]
        tensor_phases = [_ExactComplex.from_complex(1)]
        for member, witness, paths in zip(group.members[1:], group.witnesses, group.spin_chains[1:], strict=True):
            term = correlator.terms[member]
            other, extra = contracts.term(term)
            other, spin_phase = replay_spin_chains(other, paths)
            extra = extra * spin_phase
            sign = verify_witness(network, other, witness)
            tensor_phase = (extra * _ExactComplex.from_complex(sign)) / phase
            aligned.append(_ExactComplex.from_complex(term.coefficient) * tensor_phase)
            signs.append(sign)
            tensor_phases.append(tensor_phase)
        coefficient = sum(aligned, _ExactComplex())
        storage = "single_exact"
        if coefficient:
            try:
                converted = complex(coefficient)
                compact = (isfinite(converted.real) and isfinite(converted.imag)
                           and _ExactComplex.from_complex(converted) == coefficient)
            except OverflowError:
                compact = False
            if compact:
                kept.append(replace(representative, term_index=len(kept), coefficient=converted))
            else:
                storage = "split_exact"
                for weight in aligned:
                    converted = complex(weight)
                    if _ExactComplex.from_complex(converted) != weight:
                        raise ValueError("A unit-phase aligned coefficient is not exactly representable.")
                    kept.append(replace(representative, term_index=len(kept), coefficient=converted))
        audit.append({"members": list(group.members), "phases": signs,
                      "tensor_phases_exact": [[str(value.real), str(value.imag)] for value in tensor_phases],
                      "coefficient_storage": storage,
                      "coefficient_exact_binary": [str(coefficient.real), str(coefficient.imag)]})
    retained_zero_output = not kept and bool(correlator.terms)
    if retained_zero_output:
        kept = list(correlator.terms)
    return replace(correlator, terms=tuple(kept)), audit, retained_zero_output


def _dag_from_proof(proof):
    source = proof.source
    if source.aggregation_proof is not None:
        raise ValueError("Nested Wick aggregation certificates are not supported.")
    _validate_source(source)
    if tuple(item.correlator_id for item in proof.correlators) != tuple(item.correlator_id for item in source.correlators):
        raise ValueError("Wick aggregation certificate changes correlator identities.")
    contracts = _ResourceContracts(source, spin_basis=proof.spin_basis,
                                   shared_basis=proof.shared_orthonormal_laph_basis,
                                   numerical_flavor_groups=proof.numerical_flavor_groups)
    correlators = tuple(_apply_correlator_proof(correlator, record, contracts)[0]
                        for correlator, record in zip(source.correlators, proof.correlators, strict=True))
    live_components = {use.computation_id for correlator in correlators
                       for term in correlator.terms for use in term.components}
    components = {identifier: value for identifier, value in source.components.items() if identifier in live_components}
    live_resources = {tensor.resource_id for component in components.values() for tensor in component.tensors}
    resources = {identifier: value for identifier, value in source.resources.items() if identifier in live_resources}
    return replace(source, resources=resources, components=components, correlators=correlators,
                   statistics=GlobalCSEStatistics(source.statistics.resource_occurrences, len(resources),
                                                 source.statistics.component_occurrences, len(components)),
                   aggregation_proof=proof)


def validate_wick_aggregation(dag, *, config, spin_basis, shared_orthonormal_laph_basis, numerical_flavor_groups):
    """Replay certificates only, never Wick expansion, discovery or path search."""
    proof = dag.aggregation_proof
    if not config.enabled:
        if proof is not None:
            raise ValueError("Disabled Wick aggregation cannot carry a rewrite certificate.")
        return
    if not isinstance(proof, WickAggregationProof):
        raise ValueError("Enabled Wick aggregation requires a source certificate.")
    if (proof.spin_basis, proof.shared_orthonormal_laph_basis, proof.numerical_flavor_groups) != (
        spin_basis, shared_orthonormal_laph_basis, numerical_flavor_groups,
    ):
        raise ValueError("Wick aggregation certificate disagrees with its physics contracts.")
    expected = _dag_from_proof(proof)
    if expected != dag:
        raise ValueError("Wick aggregation result disagrees with its certified source algebra.")


def aggregate_certified_wick_terms(
    dag: GlobalComputationDAG,
    config: WickAggregationConfig,
    *,
    spin_basis: str,
    shared_orthonormal_laph_basis: bool,
    numerical_flavor_groups: tuple[tuple[str, ...], ...] = (),
) -> tuple[GlobalComputationDAG, dict]:
    """Aggregate only proved terms, independently within each correlator sum.

    Named outputs, factor multiplicities and source/sink placements are pinned.
    Coefficients are summed as exact binary rationals, without tolerance pruning.
    An entirely cancelled correlator retains its original typed output work.
    """
    if not config.enabled:
        return dag, {"enabled": False}
    started = monotonic()
    deadline = started + config.search_seconds
    contracts = _ResourceContracts(dag, spin_basis=spin_basis,
                                   shared_basis=shared_orthonormal_laph_basis,
                                   numerical_flavor_groups=numerical_flavor_groups)
    checks = 0
    unproved = 0
    exhausted = False
    records = []
    proofs = []
    correlators = []
    try:
        for correlator in dag.correlators:
            buckets = defaultdict(list)
            groups = []
            skipped = Counter()
            for ordinal, term in enumerate(correlator.terms):
                group = {"members": [ordinal], "witnesses": [], "representative": term,
                         "phase": _ExactComplex.from_complex(1)}
                if checks >= config.maximum_checks:
                    exhausted = True
                    groups.append(group)
                    continue
                if term.coefficient == 0:
                    groups.append(group)
                    continue
                try:
                    network, phase = contracts.term(term)
                    network, spin_phase, chains = discover_spin_chains(network, deadline=deadline)
                    phase = phase * spin_phase
                    network_graph = graph(network, deadline=deadline)
                    key = invariant(network_graph, deadline=deadline)
                except _UnsupportedContract as exc:
                    skipped[str(exc)] += 1
                    groups.append(group)
                    continue
                group.update(network=network, graph=network_graph, phase=phase, spin_chains=[chains])
                for previous in buckets[key]:
                    if checks >= config.maximum_checks:
                        exhausted = True
                        break
                    checks += 1
                    try:
                        witness = find_witness(previous["network"], network,
                                               previous["graph"], network_graph,
                                               max_states=config.maximum_match_states, deadline=deadline)
                    except SearchBudgetExceeded:
                        unproved += 1
                        continue
                    if witness is not None:
                        verify_witness(previous["network"], network, witness)
                        previous["members"].append(ordinal)
                        previous["witnesses"].append(_certificate_value(witness))
                        previous["spin_chains"].append(chains)
                        break
                else:
                    groups.append(group)
                    buckets[key].append(group)
                    continue
                if exhausted:
                    groups.append(group)
            proof = CorrelatorAggregationProof(correlator.correlator_id, tuple(
                CertifiedTermClass(tuple(group["members"]), tuple(group["witnesses"]),
                                   tuple(group["spin_chains"]) if len(group["members"]) > 1 else ()) for group in groups
            ))
            proofs.append(proof)
            reduced, audit_groups, retained_zero_output = _apply_correlator_proof(correlator, proof, contracts)
            correlators.append(reduced)
            records.append({"correlator_id": correlator.correlator_id,
                            "input_terms": len(correlator.terms), "output_terms": len(reduced.terms),
                            "merged_groups": audit_groups, "unsupported_contracts": dict(skipped),
                            "retained_all_zero_output": retained_zero_output})
    except SearchAborted as exc:
        raise WickAggregationTimeLimitExceeded(
            f"Wick aggregation exceeded search_seconds={config.search_seconds}; no plan was created. "
            "Raise compile.wick_aggregation.search_seconds or disable the pass."
        ) from exc
    proof = WickAggregationProof(dag, tuple(proofs), spin_basis,
                                 shared_orthonormal_laph_basis, numerical_flavor_groups)
    result = _dag_from_proof(proof)
    return result, {"enabled": True, "exact_checks": checks, "unproved_candidate_pairs": unproved,
                    "check_budget_exhausted": exhausted, "seconds": monotonic() - started,
                    "coefficient_policy": "exact-binary-rational", "correlators": records}
