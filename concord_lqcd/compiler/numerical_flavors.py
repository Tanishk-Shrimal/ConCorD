"""Explicit tensor equality declarations, independent of physical Wick flavors.

Canonical labels are comparison keys for ordinary local propagation factors
only. They must not replace field flavors, Wick pairings, kernel recipes, or
resource identities. Generalized propagation needs a separate equality
contract; this declaration does not authorize aliases on unsupported graphs.
"""

from __future__ import annotations

from collections.abc import Sequence


def normalize_numerical_flavor_groups(
    groups: Sequence[Sequence[str]] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic, disjoint groups without coercing flavor labels."""
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("numerical_flavor_groups must be a sequence of flavor groups")
    normalized = []
    seen = set()
    for group in groups:
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise ValueError("each numerical flavor group must be a sequence")
        if len(group) < 2:
            raise ValueError("each numerical flavor group requires at least two flavors")
        if any(
            type(flavor) is not str or not flavor or flavor != flavor.strip()
            for flavor in group
        ):
            raise ValueError("numerical flavor labels must be nonempty exact strings")
        if len(set(group)) != len(group):
            raise ValueError("a numerical flavor group contains duplicate labels")
        if seen.intersection(group):
            raise ValueError("numerical flavor groups must be disjoint")
        seen.update(group)
        normalized.append(tuple(sorted(group)))
    result = tuple(sorted(normalized))
    if (
        type(groups) is tuple
        and all(type(group) is tuple for group in groups)
        and groups == result
    ):
        return groups
    return result


def canonical_flavor(
    flavor: str, groups: Sequence[Sequence[str]] = (),
) -> str:
    """Return a comparison label, never an instruction to rewrite a resource."""
    if type(flavor) is not str or not flavor or flavor != flavor.strip():
        raise ValueError("numerical flavor labels must be nonempty exact strings")
    for group in normalize_numerical_flavor_groups(groups):
        if flavor in group:
            return group[0]
    return flavor


__all__ = ["normalize_numerical_flavor_groups", "canonical_flavor"]
