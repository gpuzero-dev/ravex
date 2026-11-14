"""Dataset position tracking.

``TrackedSampler`` wraps whatever index source a ``DataLoader`` uses (its
``batch_sampler``, or its ``sampler`` when auto-collation is off) and records
how far into the current epoch iteration has got.

Fast-forward happens at the index level: on resume we consume — and throw away
— the first *N* index batches. Those are lists of integers produced in the main
process, so no sample is ever fetched, collated, or sent to a worker. Skipping
50k batches costs milliseconds, not a data-loading pass.

For the replayed order to match the original run, the shuffling RNG must be
back where it was when the epoch started. That is why we capture the sampler's
generator state at ``__iter__`` time rather than at checkpoint time, and why an
unseeded ``RandomSampler`` gets an explicit generator attached (see
``_ensure_generator``): without one, PyTorch draws a fresh seed from the global
RNG on every epoch and the order is unreproducible after a restart.

In a run on several nodes the index sampler is first cut to this node's part
(:class:`NodeShare`, GPU-417), so what is tracked and replayed is the part.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("ravex")


def _inner_sampler(obj: Any) -> Any:
    """Reach the object that actually owns the shuffling RNG.

    A ``BatchSampler`` delegates to ``.sampler``; a plain sampler is its own
    inner object.
    """
    return getattr(obj, "sampler", obj)


def _install_epoch_offset(inner: Any, tracked: "TrackedSampler"):
    """Make ``set_epoch()`` continue the run rather than restart it.

    ``DistributedSampler`` derives its shuffle purely from ``seed + epoch``, and
    the epoch comes from the user's own loop counter:

        for epoch in range(EPOCHS):
            sampler.set_epoch(epoch)

    A resumed script starts that counter at 0 again. Restoring the epoch once
    fixes only the first pass — from the second onwards the run would replay
    epochs it already did, in the wrong order relative to the run it is
    continuing.

    So the sampler's ``set_epoch`` is shifted by the epoch the checkpoint was
    taken in. The user keeps counting from zero and the data keeps moving
    forward.

    The shift is read from ``tracked`` on every call rather than captured once,
    because a rollover can consume an epoch the user's loop never counted; see
    :meth:`TrackedSampler.advance_epoch`.
    """
    original = getattr(inner, "set_epoch", None)
    if original is None or getattr(original, "_ravex_shifted", False):
        return original

    def set_epoch(epoch, *args, **kwargs):
        return original(epoch + tracked.epoch_offset, *args, **kwargs)

    # Through setattr: the marker is a dynamic attribute on a function object,
    # which is exactly what this is, and writing it as an assignment makes a
    # type checker read it as a declaration that does not exist.
    setattr(set_epoch, "_ravex_shifted", True)
    try:
        inner.set_epoch = set_epoch
    except AttributeError:  # pragma: no cover - exotic sampler
        logger.warning("Could not shift set_epoch on %r", inner)
    return original


def _ensure_generator(sampler: Any) -> None:
    """Give an unseeded shuffling sampler a generator we can snapshot."""
    if not hasattr(sampler, "generator"):
        return
    if sampler.generator is not None:
        return
    try:
        import torch

        generator = torch.Generator()
        # Seed from the global RNG so the run stays reproducible end to end
        # under torch.manual_seed(), instead of from entropy.
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(seed)
        sampler.generator = generator
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not attach a generator to %r: %s", sampler, exc)


class TrackedSampler:
    """Position-tracking proxy around a sampler or batch sampler."""

    def __init__(self, original: Any):
        self._original = original
        # Batches the *training loop* has taken, which is not the same as
        # batches the sampler has produced: with num_workers > 0 the sampler
        # runs ahead by the prefetch depth. The loop's count is the one that
        # describes where training actually is, so it is fed in from the
        # dataloader wrapper via note_consumed().
        self._consumed = 0
        self._epoch_index = 0
        self._epoch_generator_state = None
        self._pending_skip = 0
        self._pending_generator_state = None

        #: True when the last __iter__ fast-forwarded past the end of the
        #: epoch, so it will yield nothing. Read by the dataloader wrapper.
        self.skipped_whole_epoch = False

        #: How far the sampler's epoch runs ahead of the user's loop counter.
        self.epoch_offset = 0
        self._original_set_epoch = None

        _ensure_generator(_inner_sampler(original))

    # ─── iteration ──────────────────────────────────────────────────

    def __iter__(self):
        original = self._original
        inner = _inner_sampler(original)

        # A pending restore must be applied before the underlying sampler draws
        # its permutation, i.e. right here.
        if self._pending_generator_state is not None:
            self._set_generator_state(inner, self._pending_generator_state)
            self._pending_generator_state = None

        self._epoch_generator_state = self._generator_state(inner)

        skip = self._pending_skip
        self._pending_skip = 0
        self._consumed = skip

        iterator = iter(original)

        if skip > 0:
            logger.info("Fast-forwarding dataloader by %d batches", skip)
            skipped = 0
            for _ in iterator:
                skipped += 1
                if skipped >= skip:
                    break
            if skipped < skip:
                # The epoch was shorter than the recorded position (dataset
                # changed, or the sampler is not replayable). Carry on from
                # here rather than failing the run.
                logger.warning(
                    "Dataloader exhausted after %d of %d skipped batches; "
                    "continuing from the start of the next epoch",
                    skipped,
                    skip,
                )
                self._consumed = skipped

            length = self._safe_len()
            # The checkpoint fell on an epoch boundary, so the skip ate the
            # whole pass and this iteration will yield nothing. That is not
            # merely wasted work: an empty pass is a signal frameworks act on -
            # HuggingFace `Trainer` reads "no batches this epoch" as an
            # exhausted dataset and stops training outright. The dataloader
            # wrapper watches this flag and starts the next epoch instead.
            self.skipped_whole_epoch = length is not None and skipped >= length

        yield from iterator

        self._epoch_index += 1

    def __len__(self):
        return len(self._original)

    # ─── transparency ───────────────────────────────────────────────

    def __getattr__(self, name):
        # Only reached for attributes not found on the instance, so the
        # bookkeeping fields set in __init__ never come through here.
        if name.startswith("__") or name == "_original":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_original"), name)

    def __repr__(self):
        return f"TrackedSampler({self._original!r})"

    @property
    def original(self) -> Any:
        return self._original

    @property
    def position(self) -> int:
        """Batches the training loop has taken this epoch."""
        return self._consumed

    def note_consumed(self) -> None:
        """Record that the loop received one more batch."""
        self._consumed += 1

    def advance_epoch(self) -> None:
        """Account for an epoch the user's loop never counted.

        When a resume lands on an epoch boundary the whole pass is skipped and
        the dataloader wrapper rolls straight into the next epoch - inside a
        single turn of the user's ``for epoch in ...`` loop. For a sampler
        whose order comes from an epoch number rather than a generator
        (``DistributedSampler``), that pass has to be moved on explicitly, and
        the shift applied to later ``set_epoch`` calls has to grow with it, or
        the run replays the epoch it just skipped.
        """
        inner = _inner_sampler(self._original)
        setter = self._original_set_epoch or getattr(inner, "set_epoch", None)
        if setter is None:
            return  # order comes from the generator; draining already moved it

        current = getattr(inner, "epoch", None)
        self.epoch_offset += 1
        if current is None:
            return
        try:
            setter(int(current) + 1)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not advance the sampler epoch: %s", exc)

    # ─── state ──────────────────────────────────────────────────────

    @staticmethod
    def _generator_state(inner: Any):
        generator = getattr(inner, "generator", None)
        if generator is None:
            return None
        try:
            return generator.get_state().clone()
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def _set_generator_state(inner: Any, state) -> None:
        generator = getattr(inner, "generator", None)
        if generator is None or state is None:
            return
        try:
            generator.set_state(state)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not restore sampler generator state: %s", exc)

    def state(self) -> Dict[str, Any]:
        """Capture where in the epoch we are, and what to replay from.

        The pair recorded is always *(state the epoch was drawn from, how many
        batches were consumed since)* — never an attempt to predict where the
        next epoch will start. Predicting is tempting when the checkpoint lands
        on the last batch, but wrong: ``RandomSampler`` draws an extra
        permutation when its iterator is exhausted, which has not happened yet
        at that point. Replaying the epoch and discarding it costs microseconds
        and leaves the generator exactly where the original run left it.
        """
        inner = _inner_sampler(self._original)
        generator_state = self._epoch_generator_state
        if generator_state is None:
            # state() called before the first iteration: the generator is
            # already at the epoch start.
            generator_state = self._generator_state(inner)

        state: Dict[str, Any] = {
            "position": self._consumed,
            "epoch_index": self._epoch_index,
            "length": self._safe_len(),
        }
        if generator_state is not None:
            state["generator_state"] = generator_state
        if hasattr(inner, "epoch"):
            state["sampler_epoch"] = inner.epoch
        return state

    def restore(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        inner = _inner_sampler(self._original)

        self._epoch_index = int(state.get("epoch_index", 0))

        if "sampler_epoch" in state and hasattr(inner, "set_epoch"):
            epoch = int(state["sampler_epoch"])
            try:
                inner.set_epoch(epoch)  # before the shift is installed
                if epoch > 0:
                    self._original_set_epoch = _install_epoch_offset(inner, self)
                    self.epoch_offset = epoch
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("set_epoch failed during restore: %s", exc)

        # A position at (or past) the end of the epoch means the whole epoch is
        # skipped: the next pass through the loader yields nothing and the one
        # after it carries on with the right order.
        self._pending_skip = int(state.get("position", 0))
        self._pending_generator_state = state.get("generator_state")

    def _safe_len(self) -> Optional[int]:
        try:
            return len(self._original)
        except Exception:
            return None


# ─── each node its own data (GPU-417) ───────────────────────────────


class NodeShare:
    """The part of a sampler's order one node of a run reads.

    Every node of a run on several nodes runs the same script on the same
    data, so without this the round averages N copies of one node's work: the
    run pays for N GPUs and trains on one. The script is not told which node
    it is on, and is not meant to be (it trains, on one node or on several),
    so the split is made here, under the ``DataLoader`` Ravex already tracks.

    Node ``part`` of ``parts`` keeps the indices at positions ``part``,
    ``part + parts``, ... of the order the wrapped sampler draws. Every
    starting node draws the same order - the same generator, seeded from the
    run (:func:`share_between_nodes`) - so the parts are disjoint and together
    are the whole epoch, as ``DistributedSampler`` cuts one.

    It is an index sampler like the one it wraps, so the batch sampler above
    it, the position Ravex tracks and the fast-forward on resume all work on
    this node's part: a resumed node carries on in its own part. Everything
    else - the generator, ``set_epoch`` - is the wrapped sampler's.
    """

    def __init__(self, original: Any, part: int, parts: int):
        self._original = original
        self.part = part
        self.parts = parts

    def __iter__(self):
        for position, index in enumerate(self._original):
            if position % self.parts == self.part:
                yield index

    def __len__(self):
        total = len(self._original)
        return max(0, (total - self.part + self.parts - 1) // self.parts)

    def __getattr__(self, name):
        if name.startswith("__") or name == "_original":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_original"), name)

    def __repr__(self):
        return f"NodeShare({self._original!r}, part={self.part}, parts={self.parts})"


def _run_seed(job: str) -> int:
    """A seed every node of one run computes alike, from the run's job name."""
    import hashlib

    return int.from_bytes(hashlib.sha256(job.encode("utf-8")).digest()[:8], "little") & (2**63 - 1)


def node_place(config: Any):
    """``(part, parts)`` for this node's data, or ``(None, parts)`` for a
    node of a run on several nodes that has no place of its own, or None
    when the data is not split at all.

    Split only in a run of the outer loop that starts on two or more nodes.
    A starting node's place is its number in the run (``outer_member``,
    GPU-237), which the platform gives each one before the script runs; the
    number of parts is how many start it (``outer_min_nodes``). A node that
    joins later has no place there, and gets ``(None, parts)``.
    """
    if config is None or not getattr(config, "outer_loop", False):
        return None
    try:
        parts = int(getattr(config, "outer_min_nodes", 1) or 1)
    except (TypeError, ValueError):
        return None
    if parts < 2:
        return None
    member = getattr(config, "outer_member", None)
    try:
        member = int(member) if member is not None else None
    except (TypeError, ValueError):
        member = None
    if member is None or not 0 <= member < parts:
        return (None, parts)
    return (member, parts)


def share_between_nodes(sampler: Any, config: Any) -> Any:
    """The index sampler this node should read through, given the run.

    - **A starting node** reads its part (:class:`NodeShare`). The order the
      parts are cut from has to be the same on every node, so a shuffling
      sampler that came without a generator gets one seeded from the run's
      job, not from this process's RNG; one the script seeded itself keeps
      the script's seed, which a script run alike on every node gives alike.
    - **A node without a place** - one that joined later, or a run whose
      nodes were numbered by arrival - cannot know which part is free, so it
      reads the whole data in an order of its own: its generator is seeded
      afresh from the system's entropy. On a resume that seed does not
      matter, since the generator's state comes back from the checkpoint.
    - **One node, or no outer loop**: the sampler as it is.

    A ``DistributedSampler`` is the script's own split, among the processes
    of one node, and is left alone.
    """
    place = node_place(config)
    if place is None:
        return sampler
    try:
        import torch
        from torch.utils.data.distributed import DistributedSampler

        if isinstance(sampler, DistributedSampler):
            return sampler
    except Exception:  # pragma: no cover - defensive
        return sampler

    part, parts = place
    has_generator = hasattr(sampler, "generator")
    if part is None:
        if has_generator:
            import os

            generator = torch.Generator()
            generator.manual_seed(int.from_bytes(os.urandom(8), "little") & (2**63 - 1))
            sampler.generator = generator
            logger.info(
                "This node has no place among the run's starting nodes, so it "
                "reads the whole data in an order of its own."
            )
        else:
            logger.warning(
                "This node has no place among the run's starting nodes and the "
                "data is read in a fixed order, so it reads what the starting "
                "nodes read. Shuffle the DataLoader to have it read its own."
            )
        return sampler

    if has_generator and sampler.generator is None:
        generator = torch.Generator()
        generator.manual_seed(_run_seed(str(getattr(config, "outer_job", "default"))))
        sampler.generator = generator
    logger.info("This node reads part %d of %d of the data.", part + 1, parts)
    return NodeShare(sampler, part, parts)
