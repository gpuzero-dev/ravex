"""Gate a1 of phase (a): does the benchmark's step fit the GPU, and how fast is it?

    python vram_check.py --data /workspace/data --node 0 --shards 1 \\
        --events results/a1/events.jsonl --arm a1 --steps 50 [--compile]

One process, no outer loop and no framework: the inner step both arms run,
from ``common.py``, at the benchmark's micro-batch, for ``--steps`` steps.
It prints the peak memory torch allocated and the median tokens/s after the
first steps, and writes the steps to ``--events`` like an arm would. The plan
wants 2 GB or more of margin on the card (section 5); below that, rerun on the
48 GB fallback before renting anything else.
"""

from __future__ import annotations

import argparse

import torch

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_arguments(parser)
    args = parser.parse_args()
    setup = common.Setup(args)
    events = common.Events(args.events, args.arm, args.node)
    model, optimizer, schedule = setup.build()
    if setup.cuda:
        torch.cuda.reset_peak_memory_stats()
    common.train(setup, events, model, optimizer, schedule, landed=lambda: [])
    events.close()
    import json

    with open(args.events, encoding="utf-8") as handle:
        steps = [json.loads(line) for line in handle if '"event": "step"' in line]
    rates = sorted(s["tokens"] / s["seconds"] for s in steps[3:] or steps)
    print("tokens/s per node, median after the first 3 steps: %.0f" % rates[len(rates) // 2])
    if setup.cuda:
        # Reserved, not allocated: what the caching allocator holds is what
        # the card no longer has, fragmentation included.
        allocated = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print("peak reserved %.1f GiB (allocated %.1f) of %.1f GiB: margin %.1f GiB (the plan wants 2 or more)"
              % (reserved, allocated, total, total - reserved))


if __name__ == "__main__":
    main()
