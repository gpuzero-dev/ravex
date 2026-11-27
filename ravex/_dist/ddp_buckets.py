"""DDP's gradient buckets after a resume, as the run that never stopped had them (GPU-542).

DDP all-reduces gradients bucket by bucket, and after the first backward it
regroups the buckets in the order the gradients became ready. A resumed process
builds a fresh ``DistributedDataParallel``, whose first step still reduces with
the initial layout. With more than two ranks that changes the result: a ring
all-reduce starts each chunk of a bucket at a different rank, so where a
gradient sits in its bucket decides the order its ranks' values are added in,
and ``(a + b) + c`` is not ``(b + c) + a`` in floating point. Measured on
2026-11-27 with three gloo processes on a CPU: the same sum came out ``0.0`` or
``1.0`` depending on the offset, and a resume diverged from the uninterrupted
run from its first step. With two ranks ``a + b == b + a``, which is why the
two-process test never saw it.

So a checkpoint records the rebuilt layout - DDP reports it in
``_get_ddp_logging_data()`` - and a resume rebuilds the same one before the
first forward: one backward per parameter, of zero, in the recorded order,
which is exactly what DDP watches to regroup. It all-reduces zeros once, on
every rank together, and puts each parameter's gradient back as it was.

It is skipped where it would not be the same rebuild or would disturb state of
its own: a comm hook (PowerSGD keeps error feedback across steps), a static
graph or ``find_unused_parameters`` (bucketed differently, or never rebuilt),
mixed precision (its own path to the reducer), and a wrapper that has already
run a forward - its first backward has recorded an order of its own, and
recording a second one on top would corrupt the rebuild.
"""

from __future__ import annotations

from typing import Any, List, Optional


def _ddp(model: Any) -> Optional[Any]:
    try:
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:  # pragma: no cover - torch without distributed
        return None
    return model if isinstance(model, DistributedDataParallel) else None


def layout(model: Any) -> Optional[List[List[int]]]:
    """The rebuilt buckets of a DDP ``model`` as lists of parameter indices,
    or None when it is not DDP or has not rebuilt them yet."""
    ddp = _ddp(model)
    if ddp is None:
        return None
    data = ddp._get_ddp_logging_data()
    if not data.get("has_rebuilt_buckets"):
        return None
    indices = data.get("rebuilt_per_bucket_param_indices") or ""
    return [[int(i) for i in bucket.split()] for bucket in indices.split(",") if bucket.strip()]


def rebuild(model: Any, buckets: Any) -> bool:
    """Regroup a fresh DDP ``model``'s buckets as ``buckets`` before its first
    forward. Returns whether it did; see the module docstring for when not."""
    import torch

    ddp = _ddp(model)
    if ddp is None or not buckets:
        return False
    if (
        getattr(ddp, "_lazy_init_ran", True)
        or getattr(ddp, "_has_rebuilt_buckets", True)
        or getattr(ddp, "_comm_hooks", None)
        or getattr(ddp, "static_graph", True)
        or getattr(ddp, "find_unused_parameters", True)
        or getattr(ddp, "mixed_precision", None) is not None
        or getattr(ddp, "_use_python_reducer", False)
        or getattr(ddp, "_delay_all_reduce_all_params", False)
    ):
        return False

    params, _ = ddp._build_params_for_reducer()
    order = [int(i) for bucket in buckets for i in bucket]
    # Another model, or another set of trainable parameters: the recorded
    # order says nothing about this one.
    if sorted(order) != list(range(len(params))):
        return False

    held = [p.grad for p in params]
    ddp.reducer.prepare_for_backward([])
    for index in order:
        torch.autograd.backward(params[index].sum() * 0)
    for param, grad in zip(params, held):
        param.grad = grad
    return True
