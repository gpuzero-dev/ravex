"""The saves between two checkpoints are rank 0's choice on every rank (GPU-520).

A sharded save is a collective, so the ranks have to save at the same steps.
Each rank's Moonclip times its own writer, and in a gathered run only rank 0
writes: left to itself rank 1 came down to a save at every step, entered a
checkpoint rank 0 never did, and the two hung until NCCL's watchdog killed them
(found on two RTX 5080, GPU-213).
"""

import ravex._runtime as runtime_module
from ravex._config import RavexConfig
from ravex._runtime import RavexRuntime


class _OwnMoonclip:
    """A rank's own Moonclip, which a rank other than 0 must not follow."""

    saves_every_step = True

    def __init__(self, every):
        self.every = every
        self.asked = []

    def wants(self, step):
        self.asked.append(step)
        return True

    def cadence(self):
        return self.every


def _rank(monkeypatch, rank, every=4, sharded=True):
    monkeypatch.setattr(runtime_module, "get_rank", lambda: rank)
    monkeypatch.setattr(runtime_module, "get_world_size", lambda: 2)
    config = RavexConfig()
    config.checkpoint_every = every
    runtime = RavexRuntime(config)
    monkeypatch.setattr(runtime.registry, "has_sharded_models", lambda: sharded)
    runtime._backend = _OwnMoonclip(every=1)
    return runtime


def _saved(runtime, steps):
    return [step for step in steps if runtime._save_due(step)]


def test_before_rank_zero_speaks_only_checkpoints_are_saved(monkeypatch):
    """A fresh Moonclip's ``every`` is the window, and so is the follower's."""
    runtime = _rank(monkeypatch, rank=1)
    assert _saved(runtime, range(1, 13)) == [4, 8, 12]
    assert runtime._backend.asked == [], "rank 1 asked its own Moonclip"


def test_rank_one_saves_where_rank_zero_said(monkeypatch):
    """Moonclip's rule with rank 0's number: a checkpoint, then every
    ``every`` steps since the last save, and the checkpoint resets the count."""
    runtime = _rank(monkeypatch, rank=1, every=10)
    assert _saved(runtime, range(1, 11)) == [10]
    runtime._rank_zero_every = 3
    assert _saved(runtime, range(11, 31)) == [13, 16, 19, 20, 23, 26, 29, 30]


def test_rank_zero_still_asks_its_moonclip(monkeypatch):
    runtime = _rank(monkeypatch, rank=0)
    assert _saved(runtime, [5, 6]) == [5, 6]
    assert runtime._backend.asked == [5, 6]


def test_replicated_ranks_keep_their_own_moonclip(monkeypatch):
    """Without a sharded model the ranks other than 0 do not enter a save at
    all, so there is nothing to agree on and nothing is sent."""
    runtime = _rank(monkeypatch, rank=1, sharded=False)
    assert _saved(runtime, [5]) == [5]


def test_the_number_comes_from_rank_zero(monkeypatch):
    sent = []

    def broadcast(dist, value, src, group=None):
        sent.append((value, src))
        return 7

    monkeypatch.setattr("ravex._dist.collectives._broadcast_object", broadcast)
    monkeypatch.setattr("ravex._dist.collectives._dist", lambda: None)

    zero = _rank(monkeypatch, rank=0)
    zero._backend.every = 7
    zero._share_cadence()
    assert sent == [(7, 0)]
    assert zero._rank_zero_every is None, "rank 0 follows its own Moonclip"

    one = _rank(monkeypatch, rank=1)
    one._share_cadence()
    assert sent[-1] == (None, 0), "rank 1 sent its own number"
    assert one._rank_zero_every == 7
