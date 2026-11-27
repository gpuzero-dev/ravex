"""GPU-542: a DDP or FSDP resume ends on the same bits as a run never stopped, on 2 to 8 ranks.

Two ranks cannot show it: ``a + b == b + a``. With three, where a gradient sits
in its bucket decides the order the ring all-reduce adds the ranks' values in,
and DDP regroups its buckets after the first backward - so a resumed process,
whose fresh wrapper reduces its first step with the initial layout, drifted
from the uninterrupted run by a rounding error that every later step carried.

The model uses its layers in the reverse of the order it defines them, so the
regrouped layout is not the initial one, and the buckets are tiny so there are
several. The first run stops at step 6 and the second, in the same directory,
resumes there and goes to 12; its weights must equal those of a run of 12.

With ``RAVEX_TEST_CUDA=1`` on a machine with GPUs it runs the same over NCCL,
one GPU per rank, deterministic kernels on, and skips what needs more GPUs than
there are: that is how it is run on a rented box, since CI has no GPU.

FSDP is here too, on as many ranks: it does not regroup anything after the
first step - its reduce-scatter works on the flat shards ``fully_shard`` laid
out at construction, the same in both processes - so it needs nothing from
Ravex, and this is the check that it really doesn't.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="torchrun's rendezvous needs libuv, which Windows wheels lack"
)

CUDA = os.environ.get("RAVEX_TEST_CUDA") == "1"


def _gpus():
    import torch

    return torch.cuda.device_count()

SCRIPT = '''
import hashlib, os
import torch, torch.distributed as dist, torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import ravex


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 1)
        self.b = nn.Linear(64, 64)
        self.c = nn.Linear(32, 64)

    def forward(self, x):
        return self.a(torch.tanh(self.b(torch.tanh(self.c(x)))))


CUDA = os.environ.get("RAVEX_TEST_CUDA") == "1"
if CUDA:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.use_deterministic_algorithms(True)
DEVICE = "cuda" if CUDA else "cpu"


@ravex.train_loop(max_steps=int(os.environ["STEPS"]), checkpoint_every=3)
def train():
    torch.manual_seed(0)
    if os.environ["MODE"] == "fsdp":
        from torch.distributed.fsdp import fully_shard

        model = Net().to(DEVICE)
        for layer in (model.a, model.b, model.c):
            fully_shard(layer)
        fully_shard(model)
        inner = model
    else:
        model = DistributedDataParallel(Net().to(DEVICE), bucket_cap_mb=0.001)
        inner = model.module
    if os.environ["OPT"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4096, 32) * 10
    data = TensorDataset(x, x.sum(dim=1, keepdim=True))
    if os.environ.get("DATA") == "rng":
        # Every rank draws its own batches from its own seed, as random
        # augmentation or a per-rank sampler does: the resume has to give each
        # rank back its own generator, not rank 0's (GPU-548).
        torch.manual_seed(1000 + dist.get_rank())

        def batches():
            # A loop with no DataLoader stops on its own step count, as nanoGPT does.
            while ravex.step() < int(os.environ["STEPS"]):
                picks = torch.randint(len(data), (8,))
                yield x[picks], x[picks].sum(dim=1, keepdim=True)

        epochs = [batches()]
    else:
        sampler = DistributedSampler(data, seed=0)
        loader = DataLoader(data, batch_size=8, sampler=sampler)

        def epoch_of(epoch):
            sampler.set_epoch(epoch)
            return iter(loader)

        epochs = (epoch_of(epoch) for epoch in range(10))
    started = None
    for epoch_batches in epochs:
        for inputs, targets in epoch_batches:
            if started is None:
                started = ravex.step()
                if dist.get_rank() == 0:
                    print("start", started, flush=True)
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            loss = nn.functional.mse_loss(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    digest = hashlib.sha256()
    for p in inner.parameters():
        # Under FSDP a parameter is a shard; the whole one is a collective.
        whole = p.full_tensor() if hasattr(p, "full_tensor") else p
        # Through torch, not numpy: the CI job that runs this has no numpy.
        digest.update(bytes(whole.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    if dist.get_rank() == 0:
        print("weights", ravex.step(), digest.hexdigest(), flush=True)


if __name__ == "__main__":
    dist.init_process_group("nccl" if CUDA else "gloo")
    try:
        train()
    finally:
        dist.destroy_process_group()
'''


def run(cwd, steps, mode, ranks, optimizer, data="sampler"):
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(ranks), "train.py"
    ]
    env = dict(
        os.environ, STEPS=str(steps), MODE=mode, OPT=optimizer, DATA=data, MOONCLIP_FSYNC="0",
        OMP_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        # Passed on by hand: the conftest clears every RAVEX_* variable before
        # a test runs, so the children would otherwise fall back to gloo.
        RAVEX_TEST_CUDA="1" if CUDA else "0",
        # A rank once died at exit with "terminate called without an active
        # exception" (GPU-547): on a SIGABRT this prints every thread's Python
        # stack, which is what says whose thread was still running.
        PYTHONFAULTHANDLER="1",
    )
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    return [line for line in result.stdout.splitlines() if line.startswith(("start", "weights"))]


@pytest.mark.parametrize(
    "mode, ranks, optimizer",
    # 8 is a whole node, the box most of our users rent: the size a claim of
    # "exact on one node" has to be proven at.
    [("ddp", ranks, "sgd") for ranks in (2, 3, 4, 5, 8)]
    + [("ddp", 3, "adam")]
    # SGD without momentum has no tensor state, which FSDP's restore could
    # not take (GPU-543); Adam has some.
    + [("fsdp", ranks, optimizer) for ranks in (2, 3, 4, 8) for optimizer in ("sgd", "adam")],
)
def test_a_resume_reproduces_the_uninterrupted_run(tmp_path, mode, ranks, optimizer):
    if CUDA and ranks > _gpus():
        pytest.skip("%d ranks need %d GPUs, this machine has %d" % (ranks, ranks, _gpus()))
    whole, stopped = tmp_path / "whole", tmp_path / "stopped"
    for directory in (whole, stopped):
        directory.mkdir()
        (directory / "train.py").write_text(SCRIPT, encoding="utf-8")

    reference = run(whole, 12, mode, ranks, optimizer)
    assert run(stopped, 6, mode, ranks, optimizer)[-1].split()[1] == "6"
    resumed = run(stopped, 12, mode, ranks, optimizer)

    assert reference[-1].split()[1] == "12"
    # A second run that started over would end on the same weights too.
    assert int(resumed[0].split()[1]) > 0, resumed
    assert resumed[-1] == reference[-1]


@pytest.mark.parametrize("ranks", [3, 8])
def test_a_resume_gives_each_rank_its_own_random_state(tmp_path, ranks):
    """GPU-548: a DDP checkpoint is written by rank 0 alone, and it used to carry
    only rank 0's random state. Ranks that draw from their own seed then resumed
    with rank 0's generator, drew rank 0's batches, and the run was another."""
    if CUDA and ranks > _gpus():
        pytest.skip("%d ranks need %d GPUs, this machine has %d" % (ranks, ranks, _gpus()))
    whole, stopped = tmp_path / "whole", tmp_path / "stopped"
    for directory in (whole, stopped):
        directory.mkdir()
        (directory / "train.py").write_text(SCRIPT, encoding="utf-8")

    reference = run(whole, 12, "ddp", ranks, "sgd", data="rng")
    assert run(stopped, 6, "ddp", ranks, "sgd", data="rng")[-1].split()[1] == "6"
    resumed = run(stopped, 12, "ddp", ranks, "sgd", data="rng")

    assert int(resumed[0].split()[1]) > 0, resumed
    assert resumed[-1] == reference[-1]
