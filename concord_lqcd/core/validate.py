"""
Index validation for operators.

Usage::

    from concord_lqcd import *

    pion = Operator(dbar('x','a','α') * gamma5('α','β') * u('x','a','β'))
    ok, errors = validate(pion)
    assert ok

Validation rules
----------------
1. No spin or color index may appear more than twice (Einstein summation).
2. Free indices (count = 1) are allowed — baryons, projected operators,
   etc. naturally carry open Dirac spinor indices.
3. Position indices are treated as labels for field locality and are not
   subject to Einstein multiplicity rules.
4. Operator construction checks index type consistency in that expression
   alone (same label → same space, or error).
5. QuarkField must carry exactly 3 indices: (position, color, spin).
6. GammaMatrix must carry exactly 2 indices: both spin.
7. ColorDelta must carry exactly 2 indices: both color.
8. ColorEpsilon must carry exactly 3 indices: all color.
9. GaugeLink must carry exactly 4 indices: (position, position, color, color).
10. Propagator must carry exactly 6 indices: (position, color, spin; position, color, spin).
"""

from __future__ import annotations

from collections import Counter

from .expr import Expr, Product
from .fields import QuarkField
from .gamma import GammaMatrix
from .color import ColorDelta, ColorEpsilon
from .operator import Operator
from .indices import Index, SPIN, COLOR, POSITION
from .gauge import GaugeLink
from .propagator import Propagator
from .operations import Apply

def _collect_leaves(expr: Expr) -> list[Expr]:
    """Flatten the expression tree into a list of leaf nodes."""
    if isinstance(expr, Apply):
        return _collect_leaves(expr.operand)
    if isinstance(expr, Product):
        leaves: list[Expr] = []
        for f in expr.factors:
            leaves.extend(_collect_leaves(f))
        return leaves
    return [expr]


def _validate_expr(expr: Expr) -> list[str]:
    """Validate a single expression and return a list of error strings."""
    errors: list[str] = []

    leaves = _collect_leaves(expr)
    all_indices: list[Index] = []

    for leaf in leaves:
        all_indices.extend(leaf.indices)

        if isinstance(leaf, QuarkField):
            if len(leaf.indices) != 3:
                errors.append(
                    f"{leaf}: must have exactly 3 indices (position, color, spin), "
                    f"got {len(leaf.indices)}."
                )
            else:
                p, c, s = leaf.indices
                if p.space != POSITION:
                    errors.append(f"{leaf}: first index '{p.label}' must be a position index, got '{p.space}'.")
                if c.space != COLOR:
                    errors.append(f"{leaf}: second index '{c.label}' must be a color index, got '{c.space}'.")
                if s.space != SPIN:
                    errors.append(f"{leaf}: third index '{s.label}' must be a spin index, got '{s.space}'.")

        elif isinstance(leaf, GammaMatrix):
            if len(leaf.indices) != 2:
                errors.append(
                    f"{leaf}: must have exactly 2 spin indices, "
                    f"got {len(leaf.indices)}."
                )
            else:
                for idx in leaf.indices:
                    if idx.space != SPIN:
                        errors.append(
                            f"{leaf}: index '{idx.label}' must be a spin index, got '{idx.space}'."
                        )

        elif isinstance(leaf, ColorDelta):
            if len(leaf.indices) != 2:
                errors.append(
                    f"{leaf}: must have exactly 2 color indices, "
                    f"got {len(leaf.indices)}."
                )
            else:
                for idx in leaf.indices:
                    if idx.space != COLOR:
                        errors.append(
                            f"{leaf}: index '{idx.label}' must be a color index, got '{idx.space}'."
                        )

        elif isinstance(leaf, ColorEpsilon):
            if len(leaf.indices) != 3:
                errors.append(
                    f"{leaf}: must have exactly 3 color indices, "
                    f"got {len(leaf.indices)}."
                )
            else:
                for idx in leaf.indices:
                    if idx.space != COLOR:
                        errors.append(
                            f"{leaf}: index '{idx.label}' must be a color index, got '{idx.space}'."
                        )

        elif isinstance(leaf, GaugeLink):
            if len(leaf.indices) != 4:
                errors.append(
                    f"{leaf}: must have exactly 4 indices (pos, pos, color, color), "
                    f"got {len(leaf.indices)}."
                )
            else:
                p1, p2, c1, c2 = leaf.indices
                if p1.space != POSITION or p2.space != POSITION:
                    errors.append(f"{leaf}: first two indices must be position indices.")
                if c1.space != COLOR or c2.space != COLOR:
                    errors.append(f"{leaf}: last two indices must be color indices.")

        elif isinstance(leaf, Propagator):
            if len(leaf.indices) != 6:
                errors.append(
                    f"{leaf}: must have exactly 6 indices (pos, color, spin; pos, color, spin), "
                    f"got {len(leaf.indices)}."
                )
            else:
                p1, c1, s1, p2, c2, s2 = leaf.indices
                if p1.space != POSITION or p2.space != POSITION:
                    errors.append(f"{leaf}: propagator position slots must be position indices.")
                if c1.space != COLOR or c2.space != COLOR:
                    errors.append(f"{leaf}: propagator color slots must be color indices.")
                if s1.space != SPIN or s2.space != SPIN:
                    errors.append(f"{leaf}: propagator spin slots must be spin indices.")

        else:
            errors.append(f"Unsupported expression leaf type: {type(leaf).__name__}.")

    counts = Counter(all_indices)
    warnings: list[str] = []
    for idx, count in counts.items():
        if idx.space == POSITION:
            continue
        if count > 2:
            errors.append(
                f"Index '{idx.label}' ({idx.space}) appears {count} "
                f"times — maximum allowed is 2 (Einstein summation)."
            )
        elif count == 1 and idx.space == COLOR:
            warnings.append(
                f"Color index '{idx.label}' is uncontracted (appears once). "
                f"Operator may not be gauge-invariant."
            )
        # Free spin indices (count == 1) are allowed — baryons,
        # projected operators, etc. carry open Dirac spinor indices.

    return errors, warnings


def validate(op: Operator) -> tuple[bool, list[str]]:
    """Validate the index structure of an Operator.

    Parameters
    ----------
    op : Operator
        The operator to validate.

    Returns
    -------
    ok : bool
        True if no errors were found (warnings don't count as errors).
    messages : list[str]
        Human-readable list of errors and warnings.  Errors start with
        ``'ERROR:'``, warnings with ``'WARNING:'``.
    """
    errors, warnings = _validate_expr(op.expr)
    messages: list[str] = []
    for e in errors:
        messages.append(f"ERROR: {e}")
    for w in warnings:
        messages.append(f"WARNING: {w}")
    return len(errors) == 0, messages
