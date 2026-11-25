"""Generic pre-Wick lowering for arbitrary symbolic N-point correlators."""

from __future__ import annotations

from concord_lqcd.core.correlator import Correlator
from concord_lqcd.core.wick import WickTerm
from concord_lqcd.compiler.expansion import ExpansionPass
from concord_lqcd.compiler.physics_ir import PhysicsIR, validate_physics_ir
from concord_lqcd.compiler.physics_lowering import lower_to_physics_ir
from concord_lqcd.compiler.wick_cache import WickCache


def lower_correlator_to_physics_ir(
    correlator: Correlator,
    wick_cache: WickCache | None = None,
) -> PhysicsIR:
    """Expand, Wick-contract, and lower an arbitrary symbolic N-point correlator.

    Linear-combination weights are applied to the corresponding basis Wick terms
    before PhysicsIR lowering.  PhysicsIR therefore remains unaware of the
    frontend linear-algebra representation.
    """
    if wick_cache is None:
        wick_cache = WickCache()

    logical_correlators = []
    combined_momenta: dict[str, tuple[int, int, int]] = {}

    for weight, operators in ExpansionPass.run_correlator(correlator):
        basis_correlator, wick_terms = wick_cache._get_or_contract_normalized(operators)
        weighted_terms = [
            WickTerm(
                sign=term.sign,
                coefficient=term.coefficient * weight,
                expr=term.expr,
                components=term.components,
                pairing=term.pairing,
            )
            for term in wick_terms
        ]

        basis_ir = lower_to_physics_ir(basis_correlator, weighted_terms)
        logical_correlators.extend(basis_ir.correlators)
        combined_momenta.update(basis_ir.momenta)

    physics_ir = PhysicsIR(
        correlators=tuple(logical_correlators),
        momenta=combined_momenta,
    )
    validate_physics_ir(physics_ir)
    return physics_ir
