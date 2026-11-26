"""Goodput: how much of a run's wall-clock ended up in the model (GPU-522).

Throughput says how fast the GPU goes while it trains; goodput says how much
of the time paid for was training that counts. The difference is everything
around the steps: the start of each execution and its resume, the saves that
block the loop, the outer rounds the loop waits for, the batches the loop
waits for, and - after a crash - the steps between the last checkpoint and the
crash, which the next execution trains again.

The accounting is a handful of sums kept in the checkpoint, so they follow the
model: a resume restores the sums as they were at the step it resumes from,
and what happened after that step is gone from them, the way it is gone from
the model. What that tail cost in time is not: :class:`Goodput` writes where
it got to into a small file beside the store (``goodput.json``), and the next
execution on the same machine adds the time between that checkpoint and the
crash to the wall-clock and to the cost. A new machine does not have the file,
and then the tail is not counted.

Every number is this process's, as the throughput is: one GPU's tokens over
its own time. The cost is the node's: its hourly price over the tokens of the
GPUs on it.
"""

import json
import os
from typing import Any, Dict, Optional

#: The file beside the store where the last interval's sums are kept, read
#: by the next execution on the same machine to count a crash's lost tail.
FILE = "goodput.json"

GOODPUT = "perf/goodput"
GOODPUT_TOKENS_PER_SECOND = "perf/goodput_tokens_per_s"
TOKENS_TOTAL = "perf/tokens_total_per_gpu"
COST_PER_MILLION_TOKENS = "perf/usd_per_m_tokens"


class Goodput:
    """The run's wall-clock, its productive part, its tokens and its cost."""

    def __init__(self) -> None:
        self.wall = 0.0
        self.productive = 0.0
        self.tokens = 0
        self.cost = 0.0

    def add(self, seconds: float, lost: float, tokens: int, price_per_hour: Optional[float]) -> None:
        """One stretch of wall-clock, of which ``lost`` went to no step."""
        self.wall += seconds
        self.productive += max(0.0, seconds - lost)
        self.tokens += tokens
        if price_per_hour:
            self.cost += price_per_hour * seconds / 3600.0

    def metrics(self, gpus: int) -> Dict[str, float]:
        """The run's figures so far; the cost only once a price was seen."""
        if self.wall <= 0:
            return {}
        out = {
            GOODPUT: min(1.0, self.productive / self.wall),
            GOODPUT_TOKENS_PER_SECOND: self.tokens / self.wall,
            TOKENS_TOTAL: float(self.tokens),
        }
        if self.cost > 0 and self.tokens > 0:
            out[COST_PER_MILLION_TOKENS] = self.cost / (self.tokens * max(1, gpus)) * 1e6
        return out

    def state(self) -> Dict[str, Any]:
        return {"wall": self.wall, "productive": self.productive, "tokens": self.tokens, "cost": self.cost}

    def restore(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        self.wall = float(state.get("wall", 0.0))
        self.productive = float(state.get("productive", 0.0))
        self.tokens = int(state.get("tokens", 0))
        self.cost = float(state.get("cost", 0.0))

    def write(self, directory: str, step: int) -> None:
        """Where this execution got to, for the next one on this machine."""
        path = os.path.join(directory, FILE)
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump({"step": step, **self.state()}, handle)
        os.replace(path + ".tmp", path)

    def count_lost_tail(self, directory: str, resumed_at: int) -> float:
        """Add the time a crash threw away past ``resumed_at``; returns it.

        The file holds the sums of the last interval the previous execution
        recorded. Past the step this one resumed from, its wall-clock and its
        cost were spent on steps the model no longer has.
        """
        try:
            with open(os.path.join(directory, FILE), encoding="utf-8") as handle:
                last = json.load(handle)
        except (OSError, ValueError):
            return 0.0
        if int(last.get("step", 0)) <= resumed_at:
            return 0.0
        lost = max(0.0, float(last.get("wall", 0.0)) - self.wall)
        self.wall += lost
        self.cost += max(0.0, float(last.get("cost", 0.0)) - self.cost)
        return lost
