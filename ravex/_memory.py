"""Host memory that a run frees and the process keeps anyway (GPU-443).

Every checkpoint and every round of the outer loop allocates and frees copies
of the model's state in host memory: tens of megabytes per tensor, gigabytes
in all. glibc's malloc serves a request that large with ``mmap`` and gives it
back on ``free``, until the first such free: then it raises its own threshold
to that size, so every later request up to it comes from the heap, and a heap
gives memory back to the system only from its top. A run's resident memory
then grows with every save and never comes down. Measured on two ranks of the
outer loop with a 192 MB model: 16x its parameters at the peak, 10.6x with
the threshold held where it starts.

So :func:`hold_the_threshold` fixes the threshold at glibc's own default and
the trim threshold with it, once, when a run starts. What it costs is an
``mmap`` and a ``munmap`` per large allocation, which is what the first one
of a process gets anyway; the small ones, which are most of them, never
reach it. Only on a machine with CUDA: there a step's activations are CUDA's,
while training on the CPU they are malloc's, and every one over the threshold
was an ``mmap`` and a ``munmap`` per step - 8 DDP ranks went from 52 to 31
steps/s.

A person who set ``MALLOC_MMAP_THRESHOLD_`` or ``MALLOC_TRIM_THRESHOLD_`` has
said what they want, and is left alone. Anything but Linux with glibc is too:
its allocator is a different one, with different rules.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("ravex")

#: glibc's ``mallopt`` parameters, from ``malloc.h``.
M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3

#: glibc's default mmap threshold: the value it starts every process at.
THRESHOLD = 128 * 1024

_held = False


def hold_the_threshold() -> bool:
    """Keep large allocations on ``mmap`` for the whole run. Whether it was done."""
    global _held
    if _held:
        return True
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("MALLOC_MMAP_THRESHOLD_") or os.environ.get("MALLOC_TRIM_THRESHOLD_"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        mallopt = libc.mallopt
    except (OSError, AttributeError):
        # musl, or a libc without mallopt: not glibc's rules either.
        return False
    mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
    mallopt.restype = ctypes.c_int
    if mallopt(M_MMAP_THRESHOLD, THRESHOLD) != 1 or mallopt(M_TRIM_THRESHOLD, THRESHOLD) != 1:
        logger.debug("mallopt refused the thresholds; memory freed may stay with the process")
        return False
    _held = True
    return True
