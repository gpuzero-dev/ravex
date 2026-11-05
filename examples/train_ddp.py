"""Two processes training one model with DDP, on a CPU: stop it, run it again.

    torchrun --nproc-per-node 2 examples/train_ddp.py
    # Ctrl-C somewhere around step 200, or `kill` torchrun from another terminal
    torchrun --nproc-per-node 2 examples/train_ddp.py

The second run says which step it came back at and goes on to step 600, the
length set by `max_steps`. It is the same script a GPU node runs with "nccl"
instead of "gloo": nothing in it is about checkpoints but the decorator. Each
rank's place in its share of the data comes back too, because the
`DistributedSampler` is seeded and Ravex restores the position in the
dataloader along with the weights, the optimizer and the RNGs.

The process group is set up before the decorated function is called, the way
a script under `torchrun` usually does it, so that Ravex finds the ranks when
it starts. Only rank 0 prints and announces; every rank checkpoints its part.
The checkpoints go to ./checkpoints; delete it to start over (GPU-367).
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import ravex


@ravex.train_loop(max_steps=600, checkpoint_every=50)
def train():
    rank = dist.get_rank()
    torch.manual_seed(0)
    model = DistributedDataParallel(nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4096, 16)
    data = TensorDataset(x, x.sum(dim=1, keepdim=True))
    sampler = DistributedSampler(data, seed=0)
    loader = DataLoader(data, batch_size=32, sampler=sampler)
    for epoch in range(100):
        sampler.set_epoch(epoch)
        for inputs, targets in loader:
            loss = nn.functional.mse_loss(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if rank == 0 and ravex.step() % 100 == 0:
                print(f"step {ravex.step()}  loss {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    dist.init_process_group("gloo")
    try:
        train()
    finally:
        dist.destroy_process_group()
