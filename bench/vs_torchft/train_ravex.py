"""The Ravex arm: deepseek-1b under ``@ravex.train_loop`` with the outer loop on.

    python train_ravex.py --arm ravex --node 0 --shards 2 --data data \\
        --events results/ravex/node0/events.jsonl --steps 300 --inner-steps 100

The script knows nothing about the other nodes. Everything that makes it one
node of several comes from the launcher, as for any Ravex run: the rendezvous
and this node's place in it (``RAVEX_RENDEZVOUS``, ``RAVEX_OUTER_MEMBER``,
``RAVEX_OUTER_MIN_NODES``, ``RAVEX_OUTER_JOB``, ``RAVEX_JOB_TOKEN``), the
address peers reach it at (``RAVEX_EXCHANGE_ADDRESS``), its region
(``RAVEX_OUTER_REGION``) and its store (``RAVEX_STORAGE_PATH``). The rest of
the outer loop's settings come from the file ``RAVEX_CONFIG`` names
(``configs/ravex_primary.yaml``, ``configs/ckpt_restart.yaml``); only the
three numbers the comparison fixes for both arms - H, the outer learning rate
and the outer momentum - are passed from the command line, so the two arms
cannot be run with different ones by editing one file.

**A round is seen where it lands.** Ravex closes a due round at the top of an
iteration, and this loop hands it that moment explicitly with
``ravex.batch_boundary()`` before it asks what landed. What landed is read
from the report Ravex logs as it closes a round ("Outer round N closed over K
node(s), ..."), captured from its logger in this process: the round number
and how many nodes' deltas it combined, which is what the table counts as
committed. The parameters' fingerprint is compared across the boundary as
well, so a round the log claims and the model never received shows up as
``applied: false``.
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Any, Dict, List

import ravex

import common


class RoundReports(logging.Handler):
    """The reports Ravex logs as it closes outer rounds, kept until asked for.

    A round can close on the exchange's own thread (``outer_overlap``), so the
    list is guarded; it is read at the top of the loop's next iteration.
    """

    PREFIX = "Outer round %d closed over"

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._lock = threading.Lock()
        self._reports: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if not isinstance(record.msg, str) or not record.msg.startswith(self.PREFIX):
            return
        number, nodes, slowest, fastest = record.args  # type: ignore[misc]
        with self._lock:
            self._reports.append(
                {"round": int(number), "nodes": int(nodes), "slowest": int(slowest), "fastest": int(fastest)}
            )

    def take(self) -> List[Dict[str, Any]]:
        with self._lock:
            reports, self._reports = self._reports, []
        return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_arguments(parser)
    args = parser.parse_args()
    setup = common.Setup(args)

    @ravex.train_loop(
        outer_loop=True,
        outer_inner_steps=args.inner_steps,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
    )
    def run() -> None:
        # Attached after the decorator built Ravex's runtime, which replaces
        # the handlers of the "ravex" logger when it sets its own up.
        reports = RoundReports()
        logging.getLogger("ravex").addHandler(reports)
        events = common.Events(args.events, args.arm, args.node)
        model, optimizer, schedule = setup.build()
        # Asking for the step resumes a run that has a checkpoint: the
        # checkpoint-restart arm starts every node again from its store, and
        # the loop then goes on from where the run had got to.
        first = ravex.step()
        before = {"value": common.fingerprint(model)}

        def boundary() -> None:
            ravex.batch_boundary()

        def landed() -> List[Dict[str, Any]]:
            now = common.fingerprint(model)
            applied = now != before["value"]
            found = reports.take()
            for report in found:
                report["committed"] = True
                report["applied"] = applied
                # The log gives the slowest and the fastest node's steps, not
                # their sum. With H fixed, as in every arm of the plan, the
                # two are equal and this is exact; with a wall-clock round it
                # is the midpoint, an estimate.
                report["committed_steps"] = report["nodes"] * (report["slowest"] + report["fastest"]) // 2
            return found

        original = setup.inner_step

        def inner_step(model, optimizer, schedule, step):
            stats = original(model, optimizer, schedule, step)
            # The fingerprint after the step is what the next boundary is
            # compared with: a change between the two is the outer step.
            before["value"] = common.fingerprint(model)
            return stats

        setup.inner_step = inner_step  # type: ignore[method-assign]
        try:
            common.train(setup, events, model, optimizer, schedule, landed=landed, boundary=boundary, first_step=first)
        finally:
            events.close()

    run()


if __name__ == "__main__":
    main()
