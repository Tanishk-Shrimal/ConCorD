"""
Index system for the symbolic frontend.

Indices are immutable values identified by label and space. Frontend factories
do not consult shared state. Each Operator checks its own index-label scope.
IndexRegistry also supports explicit indexed construction; the thread-local
registry helpers are available to callers but are not used by frontend factories.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass


# ── Index ──────────────────────────────────────────────────────────────

SPIN = "spin"
COLOR = "color"
POSITION = "position"

VALID_SPACES = {SPIN, COLOR, POSITION}


@dataclass(frozen=True, slots=True)
class Index:
    """A symbolic index identified by a label and a space type."""

    label: str
    space: str  # 'spin' or 'color'

    def __post_init__(self):
        if self.space not in VALID_SPACES:
            raise ValueError(
                f"Invalid index space '{self.space}'. "
                f"Must be one of {VALID_SPACES}."
            )

    def __repr__(self) -> str:
        return self.label


# ── Registry ───────────────────────────────────────────────────────────

class IndexConflictError(Exception):
    """Raised when a label is used with conflicting space types."""


def _fresh_index(space: str, used_labels: set[str], prefix: str) -> Index:
    """Allocate a dummy index without capturing any label in the expression."""
    number = 1
    while f"{prefix}_{number}" in used_labels:
        number += 1
    label = f"{prefix}_{number}"
    used_labels.add(label)
    return Index(label, space)


class IndexRegistry:
    """
    Maps string labels to Index objects within one explicit scope.

    * First time a label is seen, it is created with the given space.
    * If the same label is requested again with the *same* space, the
      existing Index is returned (identity preserved).
    * If the same label is requested with a *different* space, an
      IndexConflictError is raised.
    """
    
    def __init__(self, indices: Iterable[Index] = ()) -> None:
        self._indices: dict[str, Index] = {}
        self._counter: int = 0  # Track auto-generated indices
        for index in indices:
            self._resolve(index.label, index.space)

    # -- public convenience methods --

    def fresh(self, space: str, prefix: str = "__internal") -> Index:
        """Generate a strictly unique index that will not collide with user indices."""
        while True:
            self._counter += 1
            label = f"{prefix}_{self._counter}"
            if label not in self._indices:
                return self._resolve(label, space)

    def spin(self, label: str) -> Index:
        """Get or create a spin index with the given label."""
        return self._resolve(label, SPIN)

    def color(self, label: str) -> Index:
        """Get or create a color index with the given label."""
        return self._resolve(label, COLOR)

    def position(self, label: str) -> Index:
        """Get or create a position/spacetime index with the given label."""
        return self._resolve(label, POSITION)

    # -- low-level --

    def _resolve(self, label: str, space: str) -> Index:
        existing = self._indices.get(label)
        if existing is not None:
            if existing.space != space:
                raise IndexConflictError(
                    f"Index '{label}' was already registered as "
                    f"'{existing.space}', cannot reuse as '{space}'."
                )
            return existing
        idx = Index(label, space)
        self._indices[label] = idx
        return idx

    def lookup(self, label: str) -> Index | None:
        """Return the Index for *label*, or None if unknown."""
        return self._indices.get(label)

    def all_indices(self) -> list[Index]:
        """Return all registered indices."""
        return list(self._indices.values())

    def clear(self) -> None:
        """Remove all registered indices."""
        self._indices.clear()

    def __contains__(self, label: str) -> bool:
        return label in self._indices

    def __repr__(self) -> str:
        items = ", ".join(
            f"{k}: {v.space}" for k, v in self._indices.items()
        )
        return f"IndexRegistry({{{items}}})"


# ── Global (thread-local) registry ─────────────────────────────────────

_thread_local = threading.local()


def get_registry() -> IndexRegistry:
    """Return an explicit thread-local registry, independent of frontend factories."""
    if not hasattr(_thread_local, "registry"):
        _thread_local.registry = IndexRegistry()
    return _thread_local.registry


def reset_registry() -> None:
    """Clear the explicit thread-local registry; ordinary operators need no reset."""
    get_registry().clear()
