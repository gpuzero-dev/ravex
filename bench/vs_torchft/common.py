"""What both arms of the benchmark share, so the only difference between them is the outer loop.

``train_ravex.py`` and ``train_torchft.py`` build the model, the inner
optimizer, the learning-rate schedule and the data from here, and run the same
inner step from here. Each arm adds one thing: how a round reaches the model -
Ravex's outer loop at the top of an iteration, torchft's ``DiLoCo`` inside the
optimizer's step - and how the arm learns that one landed.

**The model is this folder's GPT** (``model.py``), GPT-2 small on the GPUs
and a tiny one in the CPU tests, so the bench needs no other repository.

**The data is a shuffled ``DataLoader``, cut per node the way Ravex cuts
one** (GPU-417, GPU-213). A node of a run must read its own part, or
averaging the nodes' deltas averages copies. Under Ravex the script does not
cut anything: Ravex gives starting node ``k`` of ``n`` the positions ``k, k+n,
...`` of the epoch's shuffled order and a node that joined later the whole
data in an order of its own, under the loader. The torchft arm has no Ravex,
so :class:`Part` cuts its loader by the same rule - the starting nodes'
parts disjoint, a joiner's order its own - and the two arms read data that is
split the same way, if not in the same order.

**Everything a result is made from goes to ``events.jsonl``**, one JSON object
per line, flushed as it is written, so a node killed mid-run leaves every
event up to its last step and a copy taken while the run goes is as good as
one taken after. ``make_table.py`` builds the results table from these files
alone; nothing is read from a framework's own logs or stores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The settings both arms take, with the plan's values as defaults.

    The defaults are the primary comparison of the plan (section 3): AdamW at
    6e-4 with betas (0.9, 0.95), weight decay 0.1, 200 steps of warmup and a
    cosine to a tenth, clipping at 1.0; 4 x 1024 x 4 tokens per node-step;
    H = 250; an outer Nesterov SGD at lr 0.7 and momentum 0.9, DiLoCo's
    published setting. An arm that changes one of them says so in its name.
    """
    parser.add_argument("--model", default="124m", help="a preset of model.py: 124m, or tiny for the CPU tests")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="change one field of the model's config")
    parser.add_argument("--data", required=True, help="a directory with train.bin and val.bin from prepare.py")
    parser.add_argument("--steps", type=int, default=300, help="local steps this node trains")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--schedule-steps", type=int, default=None,
                        help="the length the cosine is drawn over; --steps by default, the run's length for a joiner")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--node", type=int, required=True, help="this node's number")
    parser.add_argument("--starting-nodes", type=int, required=True,
                        help="how many nodes the arm starts with; a node numbered from there on is a joiner")
    parser.add_argument("--inner-steps", type=int, default=250, help="H, local steps per round")
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--events", required=True, help="where this node writes events.jsonl")
    parser.add_argument("--arm", required=True, help="the arm's name in the results table")
    parser.add_argument("--pace", type=float, default=0.0,
                        help="seconds to sleep after each step, so a CPU test lasts long enough to inject a fault")


class Events:
    """``events.jsonl``: one line per event, flushed at once.

    Every line carries the wall clock, the arm and the node, so the files of
    every node of every arm can be concatenated and still be read.
    """

    def __init__(self, path: str, arm: str, node: int) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Appending, so a node restarted by the checkpoint-restart arm adds
        # its second life to the file of its first rather than erasing it.
        self._file = open(path, "a", encoding="utf-8")
        self.arm, self.node = arm, node

    def write(self, event: str, **fields: Any) -> None:
        record = {"t": time.time(), "arm": self.arm, "node": self.node, "event": event}
        record.update(fields)
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        overrides[key.strip()] = json.loads(value)
    return overrides


class Windows(torch.utils.data.Dataset):
    """The token file as windows of ``seq + 1`` tokens that do not overlap."""

    def __init__(self, tokens: np.ndarray, seq: int) -> None:
        self.tokens, self.seq = tokens, seq

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.seq

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.seq
        return torch.from_numpy(self.tokens[start : start + self.seq + 1].astype(np.int64))


class Part(torch.utils.data.Sampler):
    """The torchft arm's cut of the data: Ravex's rule, applied by the bench.

    Each epoch is a permutation of the windows seeded from ``seed`` and the
    epoch, the same on every node. Starting node ``k`` of ``n`` reads its
    positions ``k, k+n, ...``, so the starting nodes' parts are disjoint; a
    node numbered ``n`` or more joined later and reads every window in an
    order of its own. That is what Ravex does under the Ravex arm's loader
    (``docs/configuration.md``, GPU-417), so both arms split the data alike.
    """

    def __init__(self, length: int, node: int, starting: int, seed: int) -> None:
        self.length, self.node, self.starting, self.seed = length, node, starting, seed
        self.epoch = 0

    def _order(self) -> List[int]:
        joiner = self.node >= self.starting
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * (self.node + 1) * joiner + 7919 * self.epoch
        )
        order = torch.randperm(self.length, generator=generator).tolist()
        return order if joiner else order[self.node :: self.starting]

    def __iter__(self):
        order = self._order()
        self.epoch += 1
        return iter(order)

    def __len__(self) -> int:
        return len(self._order())


class Setup:
    """The model, the inner optimizer, the schedule and the data of one node.

    Built the same way in both arms, from a seed every node shares, so every
    node starts from the same parameters: torchft is told so
    (``init_sync=False``) and Ravex makes sure of it at its seed round anyway.

    ``cut`` says who splits the data among the nodes: False for the Ravex
    arm, where Ravex does under the loader; True where nothing else would -
    the torchft arm, and the VRAM check, a node of one.
    """

    def __init__(self, args: argparse.Namespace, cut: bool) -> None:
        from model import config_of

        self.args = args
        self.cut = cut
        overrides = parse_overrides(args.override)
        overrides["max_seq_len"] = args.seq
        self.config = config_of(args.model, **overrides)
        self.train = np.memmap(os.path.join(args.data, "train.bin"), dtype="<u4", mode="r")
        self.val = np.memmap(os.path.join(args.data, "val.bin"), dtype="<u4", mode="r")
        top = int(max(self.train.max(), self.val.max()))
        if top >= self.config.vocab_size:
            raise SystemExit("the data has token %d, past the model's vocabulary of %d" % (top, self.config.vocab_size))
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        # bf16 autocast on a GPU; full precision on a CPU, where only the
        # tests run.
        self.autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if self.cuda else torch.autocast("cpu", enabled=False)
        )
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
        self.tokens_per_step = args.batch * args.accum * args.seq
        self.schedule_steps = args.schedule_steps or args.steps
        self.parameters = 0
        self._batches = None

    def build(self):
        """The model, its AdamW, its schedule and its loader. Call inside the arm's own setup.

        Inside, because the Ravex arm's loader has to be made under
        ``@ravex.train_loop`` for Ravex to cut it.
        """
        from model import GPT, count_parameters

        args = self.args
        torch.manual_seed(args.seed)
        model = GPT(self.config).to(self.device)
        self.parameters = count_parameters(model)
        decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
        plain = [p for _, p in model.named_parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": plain, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=self.cuda,
        )
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: self.lr_at(s) / args.lr)
        if args.compile:
            for block in model.blocks:
                block.compile()
        windows = Windows(self.train, args.seq)
        if self.cut:
            sampler = Part(len(windows), args.node, args.starting_nodes, args.seed)
            self.loader = torch.utils.data.DataLoader(windows, batch_size=args.batch, sampler=sampler, drop_last=True)
        else:
            self.loader = torch.utils.data.DataLoader(windows, batch_size=args.batch, shuffle=True, drop_last=True)
        return model, optimizer, schedule

    def _next_batch(self) -> torch.Tensor:
        """The loader's next batch, starting a new epoch when one runs out."""
        if self._batches is None:
            self._batches = iter(self.loader)
        try:
            return next(self._batches)
        except StopIteration:
            self._batches = iter(self.loader)
            return next(self._batches)

    def lr_at(self, step: int) -> float:
        """Warmup then a cosine to a tenth, on this node's local step (plan, section 3)."""
        args = self.args
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup) / max(1, self.schedule_steps - args.warmup))
        floor = args.lr * args.min_lr_ratio
        return floor + (args.lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    def windows(self, source: np.ndarray, starts: List[int]):
        seq = self.args.seq
        window = np.stack([source[s : s + seq + 1] for s in starts]).astype(np.int64)
        window = torch.from_numpy(window).to(self.device, non_blocking=True)
        return window[:, :-1], window[:, 1:]

    def loss_of(self, model, x, y):
        with self.autocast:
            logits = model(x)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))

    def evaluate(self, model) -> float:
        """Held-out loss on the same fixed batches on every node of every arm."""
        args = self.args
        was_training = model.training
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(args.eval_batches):
                generator = torch.Generator().manual_seed(10_000_000 + i)
                starts = torch.randint(len(self.val) - args.seq - 1, (args.batch,), generator=generator).tolist()
                vx, vy = self.windows(self.val, starts)
                losses.append(self.loss_of(model, vx, vy).item())
        model.train(was_training)
        return sum(losses) / len(losses)

    def inner_step(self, model, optimizer, schedule, step: int) -> Dict[str, float]:
        """One local step, identical in both arms: the loader's micro-batches, clip, AdamW, schedule."""
        args = self.args
        model.train()
        total = 0.0
        for _ in range(args.accum):
            window = self._next_batch().to(self.device, non_blocking=True)
            loss = self.loss_of(model, window[:, :-1], window[:, 1:]) / args.accum
            loss.backward()
            total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        schedule.step()
        return {"loss": total, "grad_norm": float(grad_norm), "lr": schedule.get_last_lr()[0]}


def checksum(model: torch.nn.Module) -> float:
    """One number for the whole parameter vector: after a round, equal on every node of an arm."""
    return round(sum(p.detach().double().sum().item() for p in model.parameters()), 6)


def fingerprint(model: torch.nn.Module) -> float:
    """A cheap stand-in for :func:`checksum`, taken twice per step.

    An outer step moves every parameter, so the first and the last tensor are
    enough to see one happen; the whole vector would cost a pass over every
    parameter twice a step.
    """
    params = list(model.parameters())
    return params[0].detach().double().sum().item() + params[-1].detach().double().sum().item()


def train(
    setup: Setup,
    events: Events,
    model,
    optimizer,
    schedule,
    *,
    landed: Callable[[], List[Dict[str, Any]]],
    boundary: Optional[Callable[[], None]] = None,
    first_step: int = 0,
) -> None:
    """The loop both arms run.

    At the top of every iteration the arm's ``boundary`` runs - Ravex closes a
    due round there - and ``landed`` answers which rounds reached the model
    since the last iteration. Each is written down, and the model is
    evaluated right after it, when every node of the arm holds the same
    parameters: that is the loss the table compares, and the checksum that
    shows the round really averaged.

    After the last step the top of one more iteration is played, without its
    step, so the round the last step made due lands and is counted here
    instead of at the framework's teardown, where nothing would see it.
    """
    args = setup.args
    events.write(
        "start",
        host=socket.gethostname(),
        device=str(setup.device),
        parameters=setup.parameters,
        model=args.model,
        tokens_per_step=setup.tokens_per_step,
        inner_steps=args.inner_steps,
        starting_nodes=args.starting_nodes,
        steps=args.steps,
        first_step=first_step,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
    )
    if first_step == 0:
        events.write("eval", step=0, round=0, loss=setup.evaluate(model), checksum=checksum(model))

    def top(step: int) -> None:
        if boundary is not None:
            boundary()
        for report in landed():
            events.write("round", step=step, **report)
            if report.get("committed", True):
                events.write("eval", step=step, round=report.get("round"), loss=setup.evaluate(model),
                             checksum=checksum(model))

    step = first_step
    while step < args.steps:
        top(step)
        began = time.monotonic()
        stats = setup.inner_step(model, optimizer, schedule, step)
        if setup.cuda:
            torch.cuda.synchronize()
        seconds = time.monotonic() - began
        step += 1
        events.write("step", step=step, tokens=setup.tokens_per_step, seconds=seconds, **stats)
        if args.pace:
            time.sleep(args.pace)
    top(step)
    events.write("end", step=step)
