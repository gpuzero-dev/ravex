"""Reading a ``torch.save`` file without running code from it (GPU-538).

A checkpoint is written by whoever can write to its directory or bucket, and
``torch.load(weights_only=False)`` runs whatever a pickle in it names: a
planted object there would execute in the process that resumes. Every
``torch.save`` file Ravex reads - its own ``torch_save`` backend, and the
checkpoints of other tools it resumes from - goes through :func:`load`,
torch's ``weights_only`` unpickler, which builds tensors and plain data only.

Beyond what that unpickler allows by itself, a training checkpoint holds
NumPy's RNG state (``np.random.get_state()``: a tuple with a ``uint32``
array), so NumPy's array, its dtypes and the function that rebuilds an array
are allowed too. Nothing else is: something outside the list fails the
restore with :class:`UnsafeCheckpoint`, naming the file and what it held.
There is no fallback to an unrestricted load.
"""

from __future__ import annotations

import pickle
from typing import Any, List


class UnsafeCheckpoint(pickle.UnpicklingError):
    """A ``torch.save`` file names something that is neither a tensor nor
    plain data, which loading it would have imported and called."""


def _numpy_globals() -> List[Any]:
    """What NumPy's RNG state pickles through, in NumPy 1 and 2."""
    try:
        import numpy as np
    except ImportError:
        return []
    found: List[Any] = [np.ndarray, np.dtype]
    for module in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            multiarray = __import__(module, fromlist=["_reconstruct"])
        except ImportError:
            continue
        found += [multiarray._reconstruct, multiarray.scalar]
        break
    dtypes = getattr(np, "dtypes", None)
    if dtypes is not None:
        found += [
            kind
            for name in dir(dtypes)
            if name.endswith("DType")
            and isinstance(kind := getattr(dtypes, name), type)
            and issubclass(kind, np.dtype)
        ]
    return found


def load(path: Any) -> Any:
    """``torch.load`` of ``path`` onto the CPU, restricted to tensors and plain
    data (see the module docstring)."""
    import torch

    with torch.serialization.safe_globals(_numpy_globals()):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise UnsafeCheckpoint(
                "%s holds something other than tensors and plain data, and loading it "
                "would run code from the checkpoint, so it is not loaded (GPU-538). "
                "torch says: %s" % (path, str(exc).splitlines()[0] if str(exc) else exc)
            ) from exc
