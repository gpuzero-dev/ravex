"""HSDP over a slow link: what casting the shards' deltas and overlapping the
rounds buy (GPU-521).

Two Linux containers stand for two pods. Each runs ``torchrun`` with two
processes, FSDP2 over gloo inside it, and Ravex's outer loop between the two:
process ``r`` of one exchanges shard ``r`` with process ``r`` of the other.
The containers' outgoing traffic is shaped with ``tc`` (a rate and a one-way
delay), so every byte of the rendezvous and the exchange crosses a link with a
speed, as between two rented pods. ``tc`` needs ``NET_ADMIN`` and a Linux
kernel, which Docker provides on any host; ``round_link_cost.py`` has a relay
that needs neither, and drives the exchange by hand rather than the whole
runtime under torchrun, which is what HSDP needs.

The arms, each the same script, seeds and data:

* ``alone``: each container trains by itself, no outer loop. The compute the
  others are measured against, and the loss of a node that never averages.
* ``fp32-blocking``: rounds every ``--inner`` steps, the training stopped
  while they travel. What ran on two RTX PRO 4500 on 2026-11-25.
* ``fp32-overlap-auto``: the training goes on while a round travels, and H
  follows the link (`outer_overlap`, `outer_auto_steps`).
* ``bf16-feedback``, ``fp8-feedback``: the same, with each shard's delta cast
  on the wire and what the cast dropped carried to the next round
  (`outer_save_dtype`, `outer_error_feedback`).

    python bench/hsdp_link.py            # builds nothing: run after
    docker build -f integration/Dockerfile -t ravex-integration .

Writes ``bench/results/hsdp-link-<date>/`` with every log and a table.
"""

import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import time

IMAGE = "ravex-hsdp-bench"
NETWORK = "ravex-hsdp-bench"
SCRIPT = "/app/integration/scripts/train_hsdp.py"

ARMS = {
    "alone": None,
    "fp32-blocking": {},
    "fp32-overlap-auto": {"RAVEX_OUTER_OVERLAP": "true", "RAVEX_OUTER_AUTO_STEPS": "true"},
    "bf16-feedback": {
        "RAVEX_OUTER_OVERLAP": "true", "RAVEX_OUTER_AUTO_STEPS": "true",
        "RAVEX_OUTER_SAVE_DTYPE": "bf16", "RAVEX_OUTER_ERROR_FEEDBACK": "true",
    },
    "fp8-feedback": {
        "RAVEX_OUTER_OVERLAP": "true", "RAVEX_OUTER_AUTO_STEPS": "true",
        "RAVEX_OUTER_SAVE_DTYPE": "fp8_block", "RAVEX_OUTER_ERROR_FEEDBACK": "true",
    },
    # Blocking rounds with an H that keeps the network at a quarter of the
    # round: against the overlap arms, what a round late costs the model.
    "bf16-blocking-auto": {
        "RAVEX_OUTER_AUTO_STEPS": "true",
        "RAVEX_OUTER_SAVE_DTYPE": "bf16", "RAVEX_OUTER_ERROR_FEEDBACK": "true",
    },
}


def docker(*args, check=True, capture=False):
    result = subprocess.run(
        ["docker", *args], check=check, text=True,
        stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else result


def build_image():
    """The integration image with ``tc`` and Moonclip, the backend that ships."""
    recipe = (
        "FROM ravex-integration\n"
        "RUN apt-get update -qq && apt-get install -y -qq iproute2 >/dev/null "
        "&& rm -rf /var/lib/apt/lists/* && pip install -q 'moonclip>=0.4.1'\n"
    )
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, "-"], input=recipe, text=True, check=True,
                   stdout=subprocess.DEVNULL)


def node_command(args, node, outer, shaped):
    """What one container runs: the link, the rendezvous on node 0, the training."""
    lines = ["set -e", "mkdir -p /out/run && cd /out/run"]
    if shaped:
        # netem for the delay, tbf under it for the rate: every packet out of
        # this container waits its turn on a link of that speed.
        lines += [
            "tc qdisc add dev eth0 root handle 1: netem delay %dms" % args.delay_ms,
            "tc qdisc add dev eth0 parent 1: handle 2: tbf rate %s burst 64kbit latency 2s" % args.rate,
        ]
    env = {"RAVEX_STORAGE_PATH": "/out/run/store", "RAVEX_LOG_FILE": "/out/ravex.log",
           "RAVEX_OUTER_MEMBER": str(node), "OMP_NUM_THREADS": "1", "MOONCLIP_FSYNC": "0"}
    if outer is not None:
        env.update({
            "RAVEX_OUTER_LOOP": "true", "RAVEX_OUTER_INNER_STEPS": str(args.inner),
            "RAVEX_RENDEZVOUS": "hsdp0:29400,hsdp1:29400", "RAVEX_OUTER_MIN_NODES": "2",
            "RAVEX_OUTER_JOB": "bench", "RAVEX_RUN_ID": "bench", "RAVEX_JOB_TOKEN": "bench",
            "RAVEX_MAX_STEPS": str(args.steps), "RAVEX_OUTER_MIN_ROUNDS": str(args.min_rounds),
        })
        env.update(outer)
        lines.append('export RAVEX_EXCHANGE_ADDRESS="$(hostname -i | cut -d" " -f1)"')
        if node == 0:
            lines.append("(RAVEX_JOB_TOKEN=bench ravex rendezvous --port 29400 > /out/rendezvous.log 2>&1 &)")
            lines.append("sleep 2")
    lines.append(" ".join("%s=%s" % item for item in env.items()) + (
        " python -m torch.distributed.run --standalone --nproc_per_node=2 %s "
        "--params %s --hidden %d --steps %d --batch %d > /out/train.log 2>&1"
        % (SCRIPT, args.params, args.hidden, args.steps, args.batch)
    ))
    lines.append("du -sb /out/run/store.rounds.shard*/*/mine/* 2>/dev/null > /out/round-sizes.txt || true")
    # The checkpoints and round reports are measured; keeping them is gigabytes.
    lines.append("rm -rf /out/run/store /out/run/store.rounds*")
    return "\n".join(lines)


def run_arm(args, name, outer, where):
    """Both containers to the end. Returns their result directories."""
    dirs = []
    shaped = outer is not None
    for node in (0, 1):
        out = os.path.join(where, name, "node%d" % node)
        os.makedirs(out, exist_ok=True)
        docker("rm", "-f", "hsdp%d" % node, check=False, capture=True)
        docker(
            "run", "-d", "--name", "hsdp%d" % node, "--hostname", "hsdp%d" % node,
            "--network", NETWORK, "--cap-add", "NET_ADMIN", "--init",
            "-v", "%s:/out" % os.path.abspath(out), IMAGE,
            "sh", "-c", node_command(args, node, outer, shaped),
            capture=True,
        )
        dirs.append(out)
    for node in (0, 1):
        code = docker("wait", "hsdp%d" % node, capture=True)
        if code != "0":
            print("  node %d of %s exited %s; see %s" % (node, name, code, dirs[node]), file=sys.stderr)
        docker("rm", "-f", "hsdp%d" % node, check=False, capture=True)
    return dirs


ROUND_TOOK = re.compile(r"Outer round (\d+) took ([\d.]+)s over \d+ node\(s\): ([\d.]+)s network")
LANDED = re.compile(r"Outer round (\d+) landed (\d+) local step\(s\) after it was launched; the training waited ([\d.]+)s of its ([\d.]+)s")
H_MOVED = re.compile(r"Outer rounds go from (\d+) to (\d+) inner step")


def summarise(name, dirs, compute_seconds):
    """One arm's numbers, from what both containers left behind."""
    results, logs, sizes = [], [], []
    for out in dirs:
        for rank in (0, 1):
            path = os.path.join(out, "run", "result-%d.json" % rank)
            if os.path.exists(path):
                with open(path) as handle:
                    results.append(json.load(handle))
        log = os.path.join(out, "ravex.log")
        logs.append(open(log).read() if os.path.exists(log) else "")
        listing = os.path.join(out, "round-sizes.txt")
        if os.path.exists(listing):
            sizes += [int(line.split()[0]) for line in open(listing) if line.strip()]
    if len(results) < 4:
        return {"arm": name, "error": "%d of 4 processes left a result" % len(results)}
    text = "\n".join(logs)
    took = [(float(t), float(n)) for _, t, n in ROUND_TOOK.findall(text)]
    landed = [(int(k), float(w), float(f)) for _, k, w, f in LANDED.findall(text)]
    wall = statistics.mean(r["seconds"] for r in results)
    row = {
        "arm": name,
        "wall_seconds": round(wall, 1),
        "steps_per_second": round(results[0]["steps"] / wall, 2),
        "eval_loss": round(statistics.mean(r["eval_loss"] for r in results), 4),
        "eval_loss_by_node": [round(r["eval_loss"], 4) for r in results if r["rank"] == 0],
        "nodes_identical": len({r["model"] for r in results}) == 1,
        "whole_sum_gap": abs(results[0]["whole"] - results[-1]["whole"]) / max(abs(results[0]["whole"]), 1e-12),
    }
    if took:
        row["rounds_per_shard"] = len(took) / 4
        row["round_network_seconds"] = round(statistics.mean(n for _, n in took), 2)
    if landed:
        row["waited_per_round"] = round(statistics.mean(w for _, w, _ in landed), 2)
        row["flight_seconds"] = round(statistics.mean(f for _, _, f in landed), 2)
        row["landed_steps_late"] = round(statistics.mean(k for k, _, _ in landed), 1)
    moves = H_MOVED.findall(text)
    if moves:
        # Every process logs every move, so each appears once per process.
        path = [int(moves[0][0])]
        for _, to in moves:
            if int(to) != path[-1]:
                path.append(int(to))
        row["h_path"] = path
    if sizes:
        row["bytes_per_shard_round"] = int(statistics.mean(sizes))
    if compute_seconds:
        row["waiting_share"] = round(max(wall - compute_seconds, 0.0) / wall, 3)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--params", default="4e6")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--inner", type=int, default=40, help="H, and the start of an automatic H")
    parser.add_argument("--min-rounds", type=int, default=3)
    parser.add_argument("--rate", default="4mbit", help="each container's outgoing rate, as tc says it")
    parser.add_argument("--delay-ms", type=int, default=50, help="one way")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    where = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        "hsdp-link-%s" % datetime.date.today().isoformat(),
    )
    os.makedirs(where, exist_ok=True)
    build_image()
    docker("network", "create", NETWORK, check=False, capture=True)

    rows, compute = [], None
    for name in args.arms.split(","):
        started = time.monotonic()
        print("arm %s ..." % name, flush=True)
        dirs = run_arm(args, name, ARMS[name], where)
        row = summarise(name, dirs, compute)
        if name == "alone" and "wall_seconds" in row:
            compute = row["wall_seconds"]
        row["arm_seconds"] = round(time.monotonic() - started, 1)
        rows.append(row)
        print("  " + json.dumps(row), flush=True)

    with open(os.path.join(where, "results.json"), "w") as handle:
        json.dump({"args": vars(args), "rows": rows}, handle, indent=2)
    print("written to %s" % where)


if __name__ == "__main__":
    main()
