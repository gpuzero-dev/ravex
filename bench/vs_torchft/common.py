"""What both arms of the benchmark share, so the only difference between them is the outer loop.

``train_ravex.py`` and ``train_torchft.py`` build the model, the inner
optimizer, the learning-rate schedule and the data from here, and run the same
inner step from here. Each arm adds one thing: how a round reaches the model -
Ravex's outer loop at the top of an iteration, torchft's ``DiLoCo`` inside the
optimizer's step - and how the arm learns that one landed.

**The model and the data are deepseek-1b's**, imported from a checkout of
https://github.com/JHNMACHINE/deepseek-1b at the commit the README pins:
``model.py`` for the network and ``data.py`` for :func:`shard`, which gives
node ``k`` of a run its own disjoint windows at each step (GPU-204). Without
it every node of a multi-node run trains on the same batches and averaging
their deltas averages copies; both arms call it with the same arguments.

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
import sys
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
    parser.add_argument("--deepseek", default=os.environ.get("DEEPSEEK_1B", ""),
                        help="a checkout of deepseek-1b; DEEPSEEK_1B by default")
    parser.add_argument("--config", default="configs/1b.json", help="deepseek-1b model config, relative to --deepseek")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="change one field of the model config; the CPU tests shrink the vocabulary")
    parser.add_argument("--data", required=True, help="a directory with train.bin and val.bin from deepseek-1b's prepare.py")
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
    parser.add_argument("--balance-rate", type=float, default=0.001)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--node", type=int, required=True, help="this node's number and data shard")
    parser.add_argument("--shards", type=int, required=True,
                        help="disjoint data shards: the most nodes the arm will ever have, joiners included")
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


def import_deepseek(path: str):
    """deepseek-1b's ``model`` and ``data`` modules, from its checkout."""
    if not path or not os.path.isfile(os.path.join(path, "data.py")):
        raise SystemExit(
            "--deepseek (or DEEPSEEK_1B) must name a checkout of deepseek-1b that has "
            "data.py, the per-node shard (GPU-204); got %r" % path
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    import data as deepseek_data  # type: ignore[import-not-found]
    import model as deepseek_model  # type: ignore[import-not-found]

    return deepseek_model, deepseek_data


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        overrides[key.strip()] = json.loads(value)
    return overrides


class Setup:
    """The model, the inner optimizer, the schedule and the data of one node.

    Built the same way in both arms, from a seed every node shares, so every
    node starts from the same parameters: torchft is told so
    (``init_sync=False``) and Ravex makes sure of it at its seed round anyway.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        deepseek_model, deepseek_data = import_deepseek(args.deepseek)
        self.shard = deepseek_data.shard
        config = args.config if os.path.isabs(args.config) else os.path.join(args.deepseek, args.config)
        overrides = parse_overrides(args.override)
        overrides["max_seq_len"] = args.seq
        self.model_args = deepseek_model.ModelArgs.load(config, **overrides)
        self.counted = deepseek_model.count_parameters(self.model_args)
        self.train = np.memmap(os.path.join(args.data, "train.bin"), dtype="<u4", mode="r")
        self.val = np.memmap(os.path.join(args.data, "val.bin"), dtype="<u4", mode="r")
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        # bf16 autocast on a GPU, as deepseek-1b trains; full precision on a
        # CPU, where only the tests run.
        self.autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if self.cuda else torch.autocast("cpu", enabled=False)
        )
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
        self.tokens_per_step = args.batch * args.accum * args.seq
        self.schedule_steps = args.schedule_steps or args.steps
        # Fails now, before anything is built, if this node's shard does not
        # exist.
        self.shard(args.node, 0, length=len(self.train), seq=args.seq, batch=1, accum=1,
                   seed=args.seed, shards=args.shards)
        self._deepseek_model = deepseek_model

    def build(self):
        """The model, its AdamW and its schedule. Call inside the arm's own setup."""
        args = self.args
        torch.manual_seed(args.seed)
        model = self._deepseek_model.Transformer(self.model_args).to(self.device)
        decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
        plain = [p for _, p in model.named_parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": plain, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=self.cuda,
        )
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: self.lr_at(s) / args.lr)
        if args.compile:
            for block in model.layers:
                block.attn.compile()
                block.attn_norm.compile()
                block.ffn_norm.compile()
                if isinstance(block.ffn, self._deepseek_model.MoE):
                    block.ffn.shared_experts.compile()
                else:
                    block.ffn.compile()
        return model, optimizer, schedule

    def lr_at(self, step: int) -> float:
        """deepseek-1b's schedule, on this node's local step (plan, section 3)."""
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
        """One local step, identical in both arms: the shard's micro-batches, clip, AdamW, schedule, balance."""
        args = self.args
        model.train()
        total = 0.0
        for starts in self.shard(args.node, step, length=len(self.train), seq=args.seq, batch=args.batch,
                                 accum=args.accum, seed=args.seed, shards=args.shards):
            x, y = self.windows(self.train, starts)
            loss = self.loss_of(model, x, y) / args.accum
            loss.backward()
            total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        schedule.step()
        imbalance = model.balance(args.balance_rate)
        return {"loss": total, "grad_norm": float(grad_norm), "lr": schedule.get_last_lr()[0], "imbalance": imbalance}


def checksum(model: torch.nn.Module) -> float:
    """One number for the whole parameter vector: after a round, equal on every node of an arm."""
    return round(sum(p.detach().double().sum().item() for p in model.parameters()), 6)


def fingerprint(model: torch.nn.Module) -> float:
    """A cheap stand-in for :func:`checksum`, taken twice per step.

    An outer step moves every parameter, so the first and the last tensor are
    enough to see one happen; the whole vector of a 0.94B model would cost a
    pass over 3.8 GB at every step.
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
        parameters=setup.counted["total"],
        active=setup.counted["active"],
        tokens_per_step=setup.tokens_per_step,
        inner_steps=args.inner_steps,
        shards=args.shards,
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
