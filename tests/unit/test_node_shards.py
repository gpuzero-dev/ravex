"""A node of several processes in the outer loop: each one a shard (GPU-213, HSDP).

The run itself, two nodes of two FSDP2 processes, is in
``integration/test_hsdp.py``; here are the rules a process checks on its own.
"""

import pytest
import torch

import ravex._runtime as runtime_module
from ravex._config import RavexConfig
from ravex._dist.outer import trainable
from ravex._runtime import RavexRuntime


def _process(monkeypatch, rank=1, world=2, **settings):
    monkeypatch.setattr(runtime_module, "get_rank", lambda: rank)
    monkeypatch.setattr(runtime_module, "get_world_size", lambda: world)
    config = RavexConfig()
    config.outer_loop = True
    config.outer_rendezvous = "10.0.0.1:29400,10.0.0.2:29400"
    config.outer_member = 0
    config.outer_job = "run"
    for key, value in settings.items():
        setattr(config, key, value)
    return RavexRuntime(config)


def test_each_process_of_a_node_is_its_own_shard(monkeypatch):
    runtime = _process(monkeypatch, rank=1, outer_root="/tmp/rounds")
    assert runtime._node_shard() == (1, 2)
    assert runtime._outer_job() == "run.shard1"
    assert runtime._outer_root() == "/tmp/rounds.shard1"
    assert runtime._rounds_base().endswith(".rounds.shard1")


def test_one_process_per_node_is_unchanged(monkeypatch):
    runtime = _process(monkeypatch, rank=0, world=1)
    assert runtime._node_shard() is None
    assert runtime._outer_job() == "run"
    assert runtime._rounds_base().endswith(".rounds")


def test_without_a_rendezvous_the_process_group_is_the_run(monkeypatch):
    """Under torchrun across machines the ranks are the nodes, as before."""
    runtime = _process(monkeypatch, outer_rendezvous=None)
    assert runtime._node_shard() is None


@pytest.mark.parametrize(
    "setting, words",
    [
        ({"outer_round_seconds": 30.0}, "outer_round_seconds"),
        ({"outer_member": None}, "outer_member"),
    ],
)
def test_shards_that_could_not_close_rounds_together_are_refused(monkeypatch, setting, words):
    runtime = _process(monkeypatch, **setting)
    with pytest.raises(RuntimeError, match=words):
        runtime._check_node_shards()


def test_rounds_in_steps_with_a_node_number_pass(monkeypatch):
    _process(monkeypatch)._check_node_shards()
    _process(monkeypatch, outer_auto_steps=True)._check_node_shards()


class _Exchange:
    store = "own store"
    scope_base = b"own life"
    scope = b"own life/namespace"


class _ShardZeroJob:
    """Shard 0's job on the rendezvous, as another shard reaches it."""

    def __init__(self):
        self.written = []

    def check(self, keys):
        return True

    def get(self, key):
        return b"shard zero's life"

    def set(self, key, value):
        self.written.append(key)


def test_shard_zero_decides_h_on_its_own_job(monkeypatch):
    runtime = _process(monkeypatch, rank=0)
    assert runtime._inner_steps_road(_Exchange()) == ("own store", b"own life")


def test_the_other_shards_decide_h_on_shard_zero_s_job(monkeypatch):
    """Every shard proposes on the same key, so every process takes the same
    H, and no shard ever writes shard 0's incarnation."""
    from ravex._dist import rendezvous

    job = _ShardZeroJob()
    reached = []
    monkeypatch.setattr(
        rendezvous, "connect", lambda address, name, timeout: reached.append((address, name)) or job
    )
    runtime = _process(monkeypatch, rank=1)
    runtime._rendezvous_at = "10.0.0.1:29400"
    store, scope = runtime._inner_steps_road(_Exchange())
    assert store is job and scope == b"shard zero's life"
    assert reached == [("10.0.0.1:29400", "run.shard0")]
    assert job.written == []

    runtime._inner_steps_road(_Exchange())
    assert len(reached) == 1, "reached again for every round"
    runtime._rendezvous_at = "10.0.0.2:29400"
    runtime._inner_steps_road(_Exchange())
    assert reached[-1][0] == "10.0.0.2:29400", "a failover's new server was not followed"


class _Sharded(torch.nn.Parameter):
    """Stands in for a DTensor: what the loop must take is ``to_local()``."""

    local = None

    def to_local(self):
        return self.local


def test_the_loop_trains_a_sharded_parameter_through_its_local_shard():
    model = torch.nn.Linear(2, 1, bias=False)
    weight = _Sharded(torch.zeros(1, 2))
    weight.local = torch.ones(1, 1)
    model.weight = weight
    [(name, tensor)] = trainable(model)
    assert name == "weight" and tensor is weight.local
