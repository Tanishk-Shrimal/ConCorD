"""Keep the temporaries of kernel building out of the C library's heap.

This matters only with glibc, and only for a process that asks for it
(``fix_block_limit``); a worker process of ``concord run`` does.

glibc gives a large block its own mapping, which goes back to the system when
the block is freed, and takes smaller blocks from its heap, where freed space
stays with the process. Which blocks count as large changes while a process
runs: the limit starts at 128 KiB and rises, up to 32 MiB, to the size of each
separately mapped block that is freed.

Building a spatial kernel allocates temporaries the size of a few eigenvector
fields, tens of MiB, and frees them when the kernel is done. Once they come
from the heap, a small tensor that outlives them (a kernel kept for the
worker's later work units) can be placed in the space one of them left. The
next temporary then no longer fits there and the heap grows by its size, for
every kernel. In the pi pi plan at N_v = 32 a worker with several work units
held up to 2 GiB of such freed space in its first unit, which a worker
measured with one unit does not show.

A fixed low limit for the whole run avoids this but slows the contraction,
which allocates blocks of a few MiB at a high rate: 20 % in the N pi plan at
N_v = 48. So the limit is low only while a kernel is built.

Fixing the limit also stops glibc from raising, with it, the free space it
keeps at the top of the heap before returning it to the system (128 KiB at
first, twice the limit once raised). With 128 KiB the contraction's blocks at
the top of the heap are returned and faulted in again at every free, which
cost that N pi plan 10 %. So that is fixed as well, at
the value glibc reaches by itself.
"""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from functools import cache
from typing import Callable, Iterator

# glibc's M_MMAP_THRESHOLD and M_TRIM_THRESHOLD options of mallopt.
_M_MMAP_THRESHOLD = -3
_M_TRIM_THRESHOLD = -1

# Blocks from this size up get their own mapping: outside kernel building the
# largest limit glibc raises itself to (and accepts), inside it 1 MiB.
BLOCK_LIMIT_BYTES = 32 * 1024**2
KERNEL_BUILDING_BLOCK_LIMIT_BYTES = 1024**2
# Free space kept at the top of the heap: what glibc keeps once it has raised
# its block limit to BLOCK_LIMIT_BYTES.
HEAP_TOP_KEPT_BYTES = 2 * BLOCK_LIMIT_BYTES

_limit_fixed = False


@cache
def libc_function(name: str) -> Callable[..., int] | None:
    """A function of the C library this process uses, on Linux; None if absent."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        return getattr(ctypes.CDLL(None), name)
    except (OSError, AttributeError):
        return None


def fix_block_limit() -> bool:
    """Fix this process's limit for separately mapped blocks; True if done.

    glibc then no longer changes the limit itself, and ``kernel_building``
    lowers it for its duration. The free space kept at the top of the heap is
    fixed with it. Meant for a worker process: the settings stay for the life
    of the process.
    """

    global _limit_fixed
    mallopt = libc_function("mallopt")
    _limit_fixed = (
        mallopt is not None
        and mallopt(_M_MMAP_THRESHOLD, BLOCK_LIMIT_BYTES) == 1
        and mallopt(_M_TRIM_THRESHOLD, HEAP_TOP_KEPT_BYTES) == 1
    )
    return _limit_fixed


@contextmanager
def kernel_building() -> Iterator[None]:
    """Map large blocks separately while a kernel is built.

    Does nothing unless ``fix_block_limit`` succeeded in this process.
    """

    if not _limit_fixed:
        yield
        return
    mallopt = libc_function("mallopt")
    assert mallopt is not None
    mallopt(_M_MMAP_THRESHOLD, KERNEL_BUILDING_BLOCK_LIMIT_BYTES)
    try:
        yield
    finally:
        mallopt(_M_MMAP_THRESHOLD, BLOCK_LIMIT_BYTES)
