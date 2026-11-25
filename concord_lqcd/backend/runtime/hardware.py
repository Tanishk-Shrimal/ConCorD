"""Read-only hardware capability discovery for runtime deployment planning.

This module deliberately reports hardware; it does not choose contraction
paths, mutate PyTorch thread pools, allocate tensors, or enter compiler/physics
identity.  All byte counts are point-in-time observations and callers must
treat available memory as volatile.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
from typing import Literal

import torch


DeviceKind = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class HardwareDeviceProfile:
    """One CPU host-memory domain or visible CUDA device."""

    device: str
    kind: DeviceKind
    name: str
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    logical_processors: int | None = None
    compute_capability: tuple[int, int] | None = None
    multiprocessor_count: int | None = None

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("hardware device name must be non-empty")
        if self.kind not in ("cpu", "cuda"):
            raise ValueError(f"unsupported hardware device kind {self.kind!r}")
        if not self.name:
            raise ValueError("hardware device description must be non-empty")
        for field_name in ("total_memory_bytes", "available_memory_bytes"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        if self.total_memory_bytes == 0:
            raise ValueError("known total device memory must be positive")
        if (
            self.total_memory_bytes is not None
            and self.available_memory_bytes is not None
            and self.available_memory_bytes > self.total_memory_bytes
        ):
            raise ValueError("available device memory exceeds total device memory")
        for field_name in ("logical_processors", "multiprocessor_count"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer or None")
        if self.compute_capability is not None:
            if (
                not isinstance(self.compute_capability, tuple)
                or len(self.compute_capability) != 2
                or any(
                    not isinstance(item, int)
                    or isinstance(item, bool)
                    or item < 0
                    for item in self.compute_capability
                )
            ):
                raise ValueError("compute_capability must contain two non-negative integers")
        if self.kind == "cpu":
            if self.device != "cpu":
                raise ValueError("the host CPU memory domain must use device='cpu'")
            if self.compute_capability is not None or self.multiprocessor_count is not None:
                raise ValueError("CPU profiles cannot declare CUDA capabilities")
        else:
            prefix, separator, index = self.device.partition(":")
            if prefix != "cuda" or not separator or not index.isdigit():
                raise ValueError(
                    "CUDA device profiles must use canonical names such as 'cuda:0'"
                )

    @property
    def memory_is_known(self) -> bool:
        return (
            self.total_memory_bytes is not None
            and self.available_memory_bytes is not None
        )


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of resources visible to the current process."""

    host: HardwareDeviceProfile
    accelerators: tuple[HardwareDeviceProfile, ...] = ()
    system: str = "unknown"
    machine: str = "unknown"
    torch_num_threads: int | None = None
    torch_num_interop_threads: int | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "accelerators", tuple(self.accelerators))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.host.kind != "cpu" or self.host.device != "cpu":
            raise ValueError("HardwareProfile.host must be the canonical CPU profile")
        devices = [self.host.device]
        for accelerator in self.accelerators:
            if accelerator.kind == "cpu":
                raise ValueError("accelerator profiles cannot use kind='cpu'")
            devices.append(accelerator.device)
        if len(devices) != len(set(devices)):
            raise ValueError("hardware profile contains duplicate device names")
        for field_name in ("torch_num_threads", "torch_num_interop_threads"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer or None")
        if not self.system or not self.machine:
            raise ValueError("hardware system and machine descriptions must be non-empty")
        if any(not isinstance(item, str) or not item for item in self.warnings):
            raise ValueError("hardware discovery warnings must be non-empty strings")

    @property
    def devices(self) -> tuple[HardwareDeviceProfile, ...]:
        return (self.host, *self.accelerators)

    def device(self, name: str) -> HardwareDeviceProfile:
        for device in self.devices:
            if device.device == name:
                return device
        raise KeyError(name)


def _usable_cpu_count() -> int:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            value = len(sched_getaffinity(0))
        except OSError:
            value = 0
        if value > 0:
            return value
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        value = process_cpu_count()
        if value is not None and value > 0:
            return value
    return os.cpu_count() or 1


def _proc_memory_bytes(path: Path = Path("/proc/meminfo")) -> tuple[int, int] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    values: dict[str, int] = {}
    for line in lines:
        name, separator, payload = line.partition(":")
        if not separator:
            continue
        fields = payload.strip().split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if len(fields) > 1 and fields[1].lower() == "kb":
            value *= 1024
        values[name] = value
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    if total is None or available is None or total <= 0 or available < 0:
        return None
    return total, min(available, total)


def _proc_cpu_model(path: Path = Path("/proc/cpuinfo")) -> str | None:
    """Return the first stable processor description exposed by Linux."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    preferred = ("model name", "hardware", "processor")
    values: dict[str, str] = {}
    for line in lines:
        name, separator, payload = line.partition(":")
        if not separator:
            continue
        key = name.strip().lower()
        value = payload.strip()
        if key == "processor" and value.isdigit():
            continue
        if key in preferred and value and key not in values:
            values[key] = value
    return next((values[key] for key in preferred if key in values), None)


def _read_nonnegative_integer(path: Path) -> int | None:
    try:
        payload = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if payload == "max":
        return None
    try:
        value = int(payload)
    except ValueError:
        return None
    return value if value >= 0 else None


def _self_cgroup_memory_bytes(
    membership_path: Path = Path("/proc/self/cgroup"),
    mount_root: Path = Path("/sys/fs/cgroup"),
) -> tuple[int, int] | None:
    """Return a finite cgroup memory limit and remaining bytes when available."""

    try:
        memberships = membership_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None

    candidates: list[tuple[Path, str, str]] = []
    for line in memberships:
        hierarchy, separator, remainder = line.partition(":")
        if not separator:
            continue
        controllers, separator, relative = remainder.partition(":")
        if not separator:
            continue
        relative_path = Path(relative.lstrip("/"))
        if hierarchy == "0" and not controllers:
            candidates.append(
                (
                    mount_root / relative_path,
                    "memory.max",
                    "memory.current",
                )
            )
        elif "memory" in controllers.split(","):
            candidates.append(
                (
                    mount_root / "memory" / relative_path,
                    "memory.limit_in_bytes",
                    "memory.usage_in_bytes",
                )
            )

    for directory, limit_name, usage_name in candidates:
        limit = _read_nonnegative_integer(directory / limit_name)
        usage = _read_nonnegative_integer(directory / usage_name)
        if limit is None or usage is None or limit <= 0:
            continue
        # Some cgroup-v1 hosts encode "unlimited" as a near-int64 maximum.
        if limit >= 1 << 60:
            continue
        return limit, max(0, limit - usage)
    return None


def _sysconf_total_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def _host_profile() -> tuple[HardwareDeviceProfile, tuple[str, ...]]:
    warnings: list[str] = []
    memory = _proc_memory_bytes()
    if memory is None:
        total = _sysconf_total_memory_bytes()
        available = None
        warnings.append(
            "host available memory could not be measured; automatic CPU memory "
            "planning is unavailable on this platform"
        )
    else:
        total, available = memory
    cgroup_memory = _self_cgroup_memory_bytes()
    if cgroup_memory is not None:
        cgroup_total, cgroup_available = cgroup_memory
        if total is None or cgroup_total < total:
            total = cgroup_total
            available = (
                cgroup_available
                if available is None
                else min(available, cgroup_available)
            )
    description = (
        _proc_cpu_model()
        or platform.processor().strip()
        or platform.machine().strip()
        or "CPU"
    )
    return (
        HardwareDeviceProfile(
            device="cpu",
            kind="cpu",
            name=description,
            total_memory_bytes=total,
            available_memory_bytes=available,
            logical_processors=_usable_cpu_count(),
        ),
        tuple(warnings),
    )


def _cuda_profiles() -> tuple[tuple[HardwareDeviceProfile, ...], tuple[str, ...]]:
    profiles: list[HardwareDeviceProfile] = []
    warnings: list[str] = []
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - backend/driver-specific
        return (), (f"CUDA discovery failed: {exc}",)
    if not available:
        if count:
            warnings.append(
                f"PyTorch reports {count} visible CUDA device(s), but CUDA is unavailable"
            )
        return (), tuple(warnings)

    for index in range(count):
        canonical = f"cuda:{index}"
        try:
            properties = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                allocated_bytes = torch.cuda.memory_allocated(index)
                reserved_bytes = torch.cuda.memory_reserved(index)
            reusable_cache = max(0, int(reserved_bytes) - int(allocated_bytes))
            process_available = min(
                int(total_bytes),
                int(free_bytes) + reusable_cache,
            )
            profiles.append(
                HardwareDeviceProfile(
                    device=canonical,
                    kind="cuda",
                    name=str(properties.name),
                    total_memory_bytes=int(total_bytes),
                    available_memory_bytes=process_available,
                    compute_capability=(
                        int(properties.major),
                        int(properties.minor),
                    ),
                    multiprocessor_count=int(properties.multi_processor_count),
                )
            )
        except Exception as exc:  # pragma: no cover - backend/driver-specific
            warnings.append(f"CUDA device {canonical} could not be inspected: {exc}")
    return tuple(profiles), tuple(warnings)


def detect_hardware_profile() -> HardwareProfile:
    """Inspect CPU/RAM and every CUDA device visible to the current process."""

    host, host_warnings = _host_profile()
    accelerators, accelerator_warnings = _cuda_profiles()
    return HardwareProfile(
        host=host,
        accelerators=accelerators,
        system=platform.system() or "unknown",
        machine=platform.machine() or "unknown",
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        warnings=(*host_warnings, *accelerator_warnings),
    )


__all__ = [
    "DeviceKind",
    "HardwareDeviceProfile",
    "HardwareProfile",
    "detect_hardware_profile",
]
