"""Bounded, configuration-local storage for normalized raw LapH fields."""

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RawEigenvectorCacheStatistics:
    hits: int
    misses: int
    evictions: int
    resident_bytes: int
    peak_resident_bytes: int
    allocation_failures: int


class RawEigenvectorCache:
    def __init__(self, maximum_bytes: int = 0) -> None:
        self._values: OrderedDict[tuple[object, ...], torch.Tensor] = OrderedDict()
        self._resident_bytes = 0
        self._peak_resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._allocation_failures = 0
        self.configure(maximum_bytes)

    def configure(self, maximum_bytes: int) -> None:
        if (
            not isinstance(maximum_bytes, int)
            or isinstance(maximum_bytes, bool)
            or maximum_bytes < 0
        ):
            raise ValueError("raw eigenvector cache budget must be a non-negative integer")
        self.clear()
        self.maximum_bytes = maximum_bytes

    def get(self, key: tuple[object, ...]) -> torch.Tensor | None:
        value = self._values.get(key)
        if value is None:
            self._misses += 1
        else:
            self._hits += 1
            self._values.move_to_end(key)
        return value

    def put(self, key: tuple[object, ...], value: torch.Tensor) -> None:
        size = value.numel() * value.element_size()
        if size > self.maximum_bytes or key in self._values:
            return
        while self._values and self._resident_bytes + size > self.maximum_bytes:
            _, removed = self._values.popitem(last=False)
            self._resident_bytes -= removed.numel() * removed.element_size()
            self._evictions += 1
            del removed
        try:
            owned = value.detach().clone(memory_format=torch.contiguous_format)
        except (MemoryError, torch.OutOfMemoryError):
            self._allocation_failures += 1
            self.clear()
            self.maximum_bytes = 0
            return
        self._values[key] = owned
        self._resident_bytes += size
        self._peak_resident_bytes = max(self._peak_resident_bytes, self._resident_bytes)

    def clear(self) -> None:
        self._values.clear()
        self._resident_bytes = 0

    @property
    def statistics(self) -> RawEigenvectorCacheStatistics:
        return RawEigenvectorCacheStatistics(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            resident_bytes=self._resident_bytes,
            peak_resident_bytes=self._peak_resident_bytes,
            allocation_failures=self._allocation_failures,
        )
