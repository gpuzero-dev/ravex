"""GPU-542: a DDP resume on three ranks ends on the same bits as a run never stopped.

Two ranks cannot show it: ``a + b == b + a``. With three, where a gradient sits
in its bucket decides the order the ring all-reduce adds the ranks' values in,
and DDP regroups its buckets after the first backward - so a resumed process,
whose fresh wrapper reduces its first step with the initial layout, drifted
from the uninterrupted run by a rounding error that every later step carried.

The model uses its layers in the reverse of the order it defines them, so the
regrouped layout is not the initial one, and the buckets are tiny so there are
several. The first run stops at step 6 and the second, in the same directory,
resumes there and goes to 12; its weights must equal those of a run of 12.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="torchrun's rendezvous needs libuv, which Windows wheels lack"
)

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


@ravex.train_loop(max_steps=int(os.environ["STEPS"]), checkpoint_every=3)
def train():
    torch.manual_seed(0)
    model = DistributedDataParallel(Net(), bucket_cap_mb=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(3 * 8 * 12, 32) * 10
    data = TensorDataset(x, x.sum(dim=1, keepdim=True))
    sampler = DistributedSampler(data, seed=0)
    loader = DataLoader(data, batch_size=8, sampler=sampler)
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for inputs, targets in loader:
            loss = nn.functional.mse_loss(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if dist.get_rank() == 0:
        digest = hashlib.sha256()
        for p in model.module.parameters():
            digest.update(p.detach().numpy().tobytes())
        print("weights", ravex.step(), digest.hexdigest(), flush=True)


if __name__ == "__main__":
    dist.init_process_group("gloo")
    try:
        train()
    finally:
        dist.destroy_process_group()
'''


def run(cwd, steps):
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "3", "train.py"]
    env = dict(os.environ, STEPS=str(steps), MOONCLIP_FSYNC="0")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    return [line for line in result.stdout.splitlines() if line.startswith("weights")]


def test_a_ddp_resume_on_three_ranks_reproduces_the_uninterrupted_run(tmp_path):
    whole, stopped = tmp_path / "whole", tmp_path / "stopped"
    for directory in (whole, stopped):
        directory.mkdir()
        (directory / "train.py").write_text(SCRIPT, encoding="utf-8")

    reference = run(whole, 12)
    assert run(stopped, 6)[-1].split()[1] == "6"
    resumed = run(stopped, 12)

    assert reference[-1].split()[1] == "12"
    assert resumed[-1] == reference[-1]
