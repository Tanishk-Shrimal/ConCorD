"""Bounded exact fixed-spin identities for comparison, never materialization.

Discovery contracts short isolated paths; replay checks their original ports
and recomputes each entry independently. Primitive representatives are retained.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
from time import monotonic

import numpy as np

from ._tensor_isomorphism import Factor, Network, SearchAborted


@dataclass(frozen=True)
class _ExactComplex:
    real: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)

    @classmethod
    def from_complex(cls, value: complex) -> _ExactComplex:
        converted = complex(value)
        return cls(Fraction.from_float(converted.real), Fraction.from_float(converted.imag))

    def __add__(self, other: _ExactComplex) -> _ExactComplex:
        return _ExactComplex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other: _ExactComplex) -> _ExactComplex:
        return _ExactComplex(self.real * other.real - self.imag * other.imag,
                             self.real * other.imag + self.imag * other.real)

    def __truediv__(self, other: _ExactComplex) -> _ExactComplex:
        norm = other.real ** 2 + other.imag ** 2
        return _ExactComplex((self.real * other.real + self.imag * other.imag) / norm,
                             (self.imag * other.real - self.real * other.imag) / norm)

    def __bool__(self) -> bool:
        return bool(self.real or self.imag)

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))


_ONE = _ExactComplex.from_complex(1)
_ZERO = _ExactComplex()
_MAXIMUM_CHAIN_FACTORS = 6


def _check(deadline):
    if deadline is not None and monotonic() >= deadline:
        raise SearchAborted("The fixed-spin chain hard deadline expired")


def _is_spin(factor):
    return (factor.resource.startswith("fixed-spin:") and len(factor.labels) == 2
            and factor.kinds == ("spin", "spin"))


@lru_cache(maxsize=256)
def _matrix(resource):
    values = np.frombuffer(bytes.fromhex(resource.removeprefix("fixed-spin:")), dtype="<c16")
    if values.size != 16 or not np.isfinite(values).all():
        raise ValueError("A spin-chain constant must be a finite four-spin matrix.")
    return tuple(tuple(_ExactComplex.from_complex(value) for value in row)
                 for row in values.reshape(4, 4))


def _transpose(matrix):
    return tuple(zip(*matrix, strict=True))


@lru_cache(maxsize=512)
def _product(matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = tuple(tuple(sum((result[row][inner] * matrix[inner][column]
                                  for inner in range(4)), _ZERO)
                             for column in range(4)) for row in range(4))
    return result


@lru_cache(maxsize=512)
def _scalar_replay(matrices):
    @lru_cache(maxsize=None)
    def entry(position, left, right):
        if position == len(matrices):
            return _ONE if left == right else _ZERO
        return sum((matrices[position][left][inner] * entry(position + 1, inner, right)
                    for inner in range(4)), _ZERO)

    return tuple(tuple(entry(0, left, right) for right in range(4)) for left in range(4))


@lru_cache(maxsize=512)
def _canonical(matrix):
    candidates = []
    for transposed in (False, True):
        oriented = _transpose(matrix) if transposed else matrix
        for phase in (1, -1, 1j, -1j):
            constant = _ExactComplex.from_complex(phase)
            normalized = tuple(tuple(value / constant for value in row) for row in oriented)
            encoded = json.dumps(tuple(tuple((str(value.real), str(value.imag)) for value in row)
                                       for row in normalized), separators=(",", ":"))
            candidates.append((encoded, transposed, constant, normalized))
    encoded, transposed, phase, normalized = min(candidates, key=lambda item: (item[0], item[1]))
    opposite = tuple(tuple(value * _ExactComplex.from_complex(-1) for value in row) for row in normalized)
    symmetry = "symmetric" if _transpose(normalized) == normalized else (
        "alternating" if _transpose(normalized) == opposite else "ordered")
    return "exact-spin:" + encoded, transposed, phase, symmetry


def _discover_paths(network, deadline):
    remaining = {ordinal for ordinal, factor in enumerate(network.factors) if _is_spin(factor)}
    incidence = defaultdict(list)
    for ordinal in remaining:
        for label in network.factors[ordinal].labels:
            incidence[label].append(ordinal)
    paths = []
    while remaining:
        _check(deadline)
        component = set()
        stack = [min(remaining)]
        while stack:
            ordinal = stack.pop()
            if ordinal in component:
                continue
            component.add(ordinal)
            remaining.remove(ordinal)
            for label in network.factors[ordinal].labels:
                stack.extend(other for other in incidence[label] if other in remaining)
        if len(component) > _MAXIMUM_CHAIN_FACTORS:
            continue
        labels = {label for ordinal in component for label in network.factors[ordinal].labels}
        ends = sorted(label for label in labels if len(incidence[label]) == 1)
        if len(ends) != 2 or any(len(incidence[label]) > 2 for label in labels):
            continue
        current = ends[0]
        visited = set()
        steps = []
        while current != ends[1]:
            choices = [ordinal for ordinal in incidence[current] if ordinal not in visited]
            if len(choices) != 1:
                break
            ordinal = choices[0]
            labels = network.factors[ordinal].labels
            reverse = labels[0] != current
            steps.append((ordinal, reverse))
            visited.add(ordinal)
            current = labels[0] if reverse else labels[1]
        if visited == component:
            paths.append(tuple(steps))
    return tuple(paths)


def _render(network, paths, *, replay, deadline):
    if not isinstance(paths, tuple):
        raise ValueError("Spin-chain proofs require a tuple of paths.")
    counts = Counter(network.outputs)
    counts.update(label for factor in network.factors for label in factor.labels)
    covered = set()
    replacements = []
    phase = _ONE
    for steps in paths:
        _check(deadline)
        if not isinstance(steps, tuple) or not 1 <= len(steps) <= _MAXIMUM_CHAIN_FACTORS:
            raise ValueError("A spin-chain path exceeds its bounded scope.")
        matrices = []
        members = set()
        current = None
        start = None
        internal = []
        for step in steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValueError("Invalid spin-chain step.")
            ordinal, reverse = step
            if (type(ordinal) is not int or not 0 <= ordinal < len(network.factors)
                    or ordinal in covered or ordinal in members or type(reverse) is not bool):
                raise ValueError("Invalid or repeated spin-chain factor.")
            factor = network.factors[ordinal]
            if not _is_spin(factor):
                raise ValueError("A spin chain cannot cross a propagation or kernel factor.")
            left, right = factor.labels[::-1] if reverse else factor.labels
            if current is None:
                start = left
            else:
                if left != current:
                    raise ValueError("A spin-chain step changes its original index path.")
                internal.append(left)
            current = right
            matrix = _matrix(factor.resource)
            matrices.append(_transpose(matrix) if reverse else matrix)
            members.add(ordinal)
        boundaries = (start, current)
        if (start == current or len(set(internal)) != len(internal)
                or set(boundaries).intersection(internal)
                or any(counts[label] != 2 for label in (*boundaries, *internal))):
            raise ValueError("A spin chain hides a public axis, branch or external use.")
        matrix = _scalar_replay(tuple(matrices)) if replay else _product(tuple(matrices))
        _check(deadline)
        resource, transposed, extra, symmetry = _canonical(matrix)
        replacements.append(Factor(resource, boundaries[::-1] if transposed else boundaries,
                                   ("spin", "spin"), symmetry))
        phase = phase * extra
        covered.update(members)
    return (Network(tuple(factor for ordinal, factor in enumerate(network.factors) if ordinal not in covered)
                    + tuple(replacements), network.outputs), phase)


def discover_spin_chains(network, *, deadline):
    """Propose short isolated constant paths; unsupported components stay intact."""
    paths = _discover_paths(network, deadline)
    counts = Counter(network.outputs)
    counts.update(label for factor in network.factors for label in factor.labels)
    paths = tuple(steps for steps in paths if all(counts[label] == 2 for ordinal, _ in steps
                                                for label in network.factors[ordinal].labels))
    folded, phase = _render(network, paths, replay=False, deadline=deadline)
    return folded, phase, paths


def replay_spin_chains(network, paths):
    """Validate saved paths and exact scalar entries without discovery or search."""
    return _render(network, paths, replay=True, deadline=None)
