"""Each rank's own random state, beside a checkpoint only rank 0 writes (GPU-548).

A replicated (DDP) checkpoint, and a sharded one gathered on rank 0, is written
by rank 0 alone, so the state it holds has rank 0's random generators in it.
That is enough when every rank seeds alike and reads its data through a
sampler. It is not when a rank draws from its own seed - random augmentation,
a sampler of its own, ``torch.manual_seed(seed + rank)`` - because the resume
then hands every rank rank 0's generator: they draw rank 0's batches, and the
run that continues is another one. Measured on 8 ranks on 2026-11-27: the
resume landed on the right step every time, and 28 steps of 400 matched.

Asking the other ranks for their state when rank 0 saves would be a
collective, and the saves that matter most are the ones on the way out, when
one rank has its SIGTERM and its peers do not: a gather there waits for ranks
that are gone. So each rank writes its own, a few kilobytes, to a file of its
own beside the store, at every checkpoint it reaches, with no peer involved;
and on resume each rank reads its own back for the step the checkpoint is at.

The file sits beside the store (``<storage.path>.rng``), not in it: the store
is the checkpoint engine's, and nothing else writes there. It lives where the
store does, so on a shared filesystem a replacement machine finds it, and on a
machine's own disk it goes with the machine. A rank that finds no state of its
own for the step says so and continues with rank 0's: its random draws, and
nothing else, then differ from the run that stopped.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("ravex")

#: Steps a rank's file keeps: a resume reads the newest checkpoint, or one a
#: little older when the newest is incomplete, never one from long ago.
KEEP = 8


def directory(storage_path: str) -> str:
    return os.path.normpath(storage_path) + ".rng"


def _file(storage_path: str, rank: int) -> str:
    return os.path.join(directory(storage_path), "rank_%d.pt" % rank)


def save(storage_path: str, rank: int, step: int, rng: Dict[str, Any]) -> None:
    """Record this rank's random state for ``step``, replacing the file atomically.

    Never raises: losing this costs one rank's random draws after a resume,
    which is reported then; failing a checkpoint over it would cost the run.
    """
    import torch

    path = _file(storage_path, rank)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kept = _read(path) or {}
        kept[int(step)] = _plain(rng)
        for old in sorted(kept)[:-KEEP]:
            del kept[old]
        temporary = "%s.%d.tmp" % (path, os.getpid())
        torch.save(kept, temporary)
        os.replace(temporary, path)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        logger.warning("Could not record rank %d's random state at step %d: %s", rank, step, exc)


def load(storage_path: str, rank: int, step: int) -> Optional[Dict[str, Any]]:
    """This rank's random state at ``step``, or None when it never recorded one."""
    kept = _read(_file(storage_path, rank))
    rng = None if kept is None else kept.get(int(step))
    return None if rng is None else _restored(rng)


def _plain(rng: Dict[str, Any]) -> Dict[str, Any]:
    """Generator states as plain lists, so the file loads with ``weights_only``.
    NumPy's is there only when the program's own NumPy was there to read."""
    out = dict(rng)
    numpy = out.get("numpy")
    if numpy is not None:
        name, keys, pos, has_gauss, cached = numpy
        out["numpy"] = (name, [int(k) for k in keys], int(pos), int(has_gauss), float(cached))
    python = out.get("python")
    if python is not None:
        out["python"] = (python[0], list(python[1]), python[2])
    return out


def _restored(rng: Dict[str, Any]) -> Dict[str, Any]:
    """Back to what the setters take. NumPy's key stays a list: its
    ``set_state`` accepts one, and Ravex does not import NumPy."""
    out = dict(rng)
    python = out.get("python")
    if python is not None:
        out["python"] = (python[0], tuple(python[1]), python[2])
    return out


def _read(path: str) -> Optional[Dict[int, Dict[str, Any]]]:
    import torch

    if not os.path.exists(path):
        return None
    try:
        # Tensors and plain lists only (see `_plain`), so nothing in the file
        # can run code when it is read.
        return torch.load(path, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - a torn file is a missing one
        logger.warning("Could not read %s: %s", path, exc)
        return None
