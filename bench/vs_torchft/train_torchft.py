"""The torchft arm: deepseek-1b under torchft's ``Manager`` and ``DiLoCo``, without TorchTitan.

    torchft_lighthouse --bind [::]:29510 --min_replicas 1      # once, on node 0's host
    TORCHFT_LIGHTHOUSE=http://<node0>:29510 python train_torchft.py --arm torchft \\
        --node 0 --shards 2 --data data --events results/torchft/node0/events.jsonl \\
        --steps 300 --inner-steps 100 --advertise <this host's address>

This is the path torchft's own ``train_diloco.py`` takes for a plain
``nn.Module``: one ``Manager`` per replica group, here a group of one process
on one GPU, and ``DiLoCo`` wrapped around the inner optimizer. TorchTitan's
integration is the tuned path, but it wires up Llama 3 only, and deepseek-1b
is a mixture of experts; the README says so beside the results.

The settings are the plan's primary comparison (section 3), the same numbers
as the Ravex arm's: one fragment, ``sync_every`` = H, an outer Nesterov SGD at
lr 0.7 and momentum 0.9, no quantization, Gloo between replica groups over
TCP, healing over HTTP, ``min_replica_size`` 1 so the run goes on when a
replica dies. torchft's own defaults are not used where the plan fixes a
value.

**Every node starts from the same parameters**, built from the shared seed as
in the Ravex arm, so ``init_sync`` is off: on, the first quorum would
overwrite every replica's first H steps with replica 0's. A replica that joins
later is behind the others' step and torchft heals it from a peer at its
first quorum - its model and inner optimizer, plus DiLoCo's outer state.

**What the table needs comes from the manager.** A round is a sync at every
H local steps; it committed when ``Manager.current_step()`` moved, and
``num_participants()`` says over how many replica groups. A sync that did not
commit threw its window away, and its steps count as wasted.

**Reachability, which torchft needs and does not arrange** (plan, section 1).
Each replica advertises ``--advertise`` to its peers for its group's store,
its manager and its healing server, and they dial it on ports chosen at
random; nothing crosses NAT. On RunPod that is the pod's Global Networking
address. ``--advertise`` defaults to ``socket.gethostname()``, which is
torchft's default too, and is right only where every peer resolves that name.
"""

from __future__ import annotations

import argparse
import os
import socket
from datetime import timedelta
from typing import Any, Dict, List

import torch
from torch.distributed import TCPStore
from torchft.local_sgd import DiLoCo
from torchft.manager import Manager, WorldSizeMode
from torchft.process_group import ProcessGroupGloo

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_arguments(parser)
    parser.add_argument("--lighthouse", default=os.environ.get("TORCHFT_LIGHTHOUSE", ""),
                        help="the lighthouse's URL; TORCHFT_LIGHTHOUSE by default")
    parser.add_argument("--advertise", default=os.environ.get("BENCH_ADVERTISE") or socket.gethostname(),
                        help="the address peers reach this replica at")
    parser.add_argument("--replica-id", default=None, help="the replica group's id; node<N> by default")
    parser.add_argument("--min-replicas", type=int, default=1)
    # The defaults cover a WAN round of the plan's size (3.76 GB at 7-12 MB/s
    # is 5-9 minutes) and a joiner's pod starting; TORCHFT_TIMEOUT_SEC and
    # TORCHFT_QUORUM_TIMEOUT_SEC, which torchft reads itself, win over both.
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds for a collective, a commit or a heal")
    parser.add_argument("--quorum-timeout", type=float, default=1800.0)
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not args.lighthouse:
        raise SystemExit("--lighthouse or TORCHFT_LIGHTHOUSE must name the lighthouse")

    setup = common.Setup(args)
    events = common.Events(args.events, args.arm, args.node)
    model, optimizer, schedule = setup.build()

    def state_dict() -> Dict[str, Any]:
        return {"model": model.state_dict(), "optim": optimizer.state_dict()}

    def load_state_dict(state: Dict[str, Any]) -> None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])

    # A replica group of one process holds its own store; peers reach it at
    # the advertised address to reconfigure the process group between groups.
    store = TCPStore(host_name=args.advertise, port=0, is_master=True, wait_for_workers=False)
    manager = Manager(
        pg=ProcessGroupGloo(timeout=timedelta(seconds=args.timeout)),
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        min_replica_size=args.min_replicas,
        use_async_quorum=False,
        timeout=timedelta(seconds=args.timeout),
        quorum_timeout=timedelta(seconds=args.quorum_timeout),
        connect_timeout=timedelta(seconds=args.connect_timeout),
        rank=0,
        world_size=1,
        world_size_mode=WorldSizeMode.DYNAMIC,
        store_addr=args.advertise,
        store_port=store.port,
        lighthouse_addr=args.lighthouse,
        replica_id=args.replica_id or "node%d" % args.node,
        hostname=args.advertise,
        init_sync=False,
    )
    outer = torch.optim.SGD(model.parameters(), lr=args.outer_lr, momentum=args.outer_momentum, nesterov=True)
    state = {"manager_step": manager.current_step(), "seen": 0}

    def landed() -> List[Dict[str, Any]]:
        # DiLoCo syncs inside the optimizer step that completes every H-th
        # local step, so by the top of the next iteration that sync is over.
        step = state["seen"]
        if step == 0 or step % args.inner_steps != 0:
            return []
        state["seen"] = -1  # not again for the same step
        now = manager.current_step()
        committed = now > state["manager_step"]
        nodes = manager.num_participants()
        report = {
            "round": now if committed else now + 1,
            "nodes": nodes,
            "committed": committed,
            "committed_steps": nodes * args.inner_steps if committed else 0,
            "wasted_steps": 0 if committed else args.inner_steps,
            "participating": manager.is_participating(),
            # A joiner's manager step jumps to the group's when it heals.
            "healed": committed and now - state["manager_step"] > 1,
            "applied": committed,
        }
        state["manager_step"] = now
        return [report]

    original = setup.inner_step

    def inner_step(model, optimizer, schedule, step):
        stats = original(model, optimizer, schedule, step)
        state["seen"] = step + 1
        return stats

    setup.inner_step = inner_step  # type: ignore[method-assign]
    try:
        with DiLoCo(manager, [model], optimizer, outer, sync_every=args.inner_steps, should_quantize=False):
            common.train(setup, events, model, optimizer, schedule, landed=landed)
    finally:
        events.close()
        manager.shutdown(wait=False)


if __name__ == "__main__":
    main()
