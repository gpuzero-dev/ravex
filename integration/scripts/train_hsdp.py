"""HSDP with Ravex: FSDP2 across the processes of a node, the outer loop across nodes.

    torchrun --standalone --nproc_per_node=2 train_hsdp.py --steps 20

Run once per node, each with ``RAVEX_RENDEZVOUS`` naming every node's address
and ``RAVEX_OUTER_MEMBER`` its own place (GPU-213). Process ``r`` of every
node exchanges its shard of the model with process ``r`` of the others, in a
round of its own; nothing in this script knows that, which is the point.

Every node starts from a model of its own (``--seed`` defaults to the node's
number), so a run whose nodes end up holding the same model has had its seed
round put the start together from the same node's shards, and its rounds keep
them together. Each process writes what it holds to ``result-<rank>.json``.
"""

import argparse
import hashlib
import json
import os
import time

import ravex
import torch
import torch.distributed as dist
import torch.nn as nn


class Block(nn.Module):
    """``x + gelu(linear(x))``. Residual because a plain stack of these died
    on two RTX PRO 4500: by step 50 every output was exactly zero, and the
    loss was the data's own mean square, the same whatever the weights."""

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x + torch.nn.functional.gelu(self.linear(x))


def build_model(target_params, hidden):
    """Square blocks sized to a parameter count: the sharding is under test,
    not the architecture."""
    per_layer = hidden * hidden + hidden
    layers = max(2, round(target_params / per_layer))
    return nn.Sequential(*[Block(hidden) for _ in range(layers)])


@ravex.train_loop()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=float, default=2e4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=8)
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    node = int(os.environ.get("RAVEX_OUTER_MEMBER", "0"))
    torch.manual_seed(node if args.seed is None else args.seed)
    model = build_model(args.params, args.hidden)
    mesh = init_device_mesh("cuda" if cuda else "cpu", (world,))
    for layer in model:
        if any(True for _ in layer.parameters()):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # What there is to learn: a fixed linear map, the same on every node, so
    # the loss is an error the training can bring down.
    target = torch.randn(args.hidden, args.hidden, generator=torch.Generator().manual_seed(42))
    target = (target / args.hidden ** 0.5).to(device)

    # The data is a function of the step and the process, so each of them
    # trains on something of its own and nothing needs a dataset on disk.
    generator = torch.Generator().manual_seed(1000 * node + rank)
    started = time.monotonic()
    for _ in range(args.steps):
        x = torch.randn(args.batch, args.hidden, generator=generator).to(device)
        loss = (model(x) - x @ target).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started

    with torch.no_grad():
        # A round due at the last step closes at the next boundary, which in a
        # loop with no dataloader is the next forward: without it the model is
        # read with the last round still open, each node's own.
        model(x)
        shard = sum(p.to_local().double().sum().item() for p in model.parameters())
        whole = sum(p.full_tensor().double().sum().item() for p in model.parameters())
        # The same held-out batches on every process of every node: one
        # training batch's loss is too noisy to tell two runs apart.
        held_out = torch.Generator().manual_seed(999)
        evaluated = 0.0
        for _ in range(args.eval_batches):
            x = torch.randn(args.batch, args.hidden, generator=held_out).to(device)
            evaluated += float((model(x) - x @ target).pow(2).mean())
        digest = hashlib.sha256()
        for p in model.parameters():
            digest.update(p.full_tensor().detach().cpu().double().numpy().tobytes())
    result = json.dumps({
        "node": node, "rank": rank, "world": world, "steps": args.steps,
        "seconds": round(elapsed, 3), "loss": float(loss), "shard": shard, "whole": whole,
        "eval_loss": evaluated / max(args.eval_batches, 1), "model": digest.hexdigest()[:16],
    })
    # A file per process as well as the line: two processes printing at once
    # share a line of the launcher's output.
    with open("result-%d.json" % rank, "w") as handle:
        handle.write(result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
