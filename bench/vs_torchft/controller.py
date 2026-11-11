"""Start the arms of a phase on the given hosts, inject its faults on schedule, bring the results home.

    python controller.py --hosts hosts.json --phase configs/phase_a.json

A phase file names the arms, the hosts each one's nodes run on, the training
arguments every node gets, and the faults: ``kill`` a node, or start a
``join``-er, when an arm reaches a local step. The same schedule applies to
every arm it names, at the same step, so every arm sees the same fault.

**What the controller does, in order.** For each arm it starts the
coordinator on the host of the arm's node 0 - ``ravex rendezvous`` for Ravex,
``torchft_lighthouse`` for torchft - and then every starting node at once, so
no node waits on a human and the meter runs on all of them together. Then it
polls each node's ``events.jsonl`` and copies what is new into the results
directory as it arrives: a node that dies, and a phase cut short, leave
everything they had written at home, not on a machine about to be destroyed.

**A fault is fired at a step, the step of the arm's lowest live starting
node**, which is never the node killed (the plan kills a non-coordinator
node). ``kill`` is SIGKILL to the node's process group - no SIGTERM, no final
checkpoint - followed by the phase's ``on_kill`` command, if it has one, run
on the operator's machine with ``{host}`` and ``{name}`` filled in: that is
where the provider's own "terminate this machine" goes, outside the harness.
``join`` starts one more node on a host kept for it, with the next node
number and the next data shard, for the steps the others have left.

**The checkpoint-restart arm** (kind ``ckpt_restart``) is a Ravex arm with
fixed membership whose faults are handled the classical way: on a kill, every
node of the arm is stopped, the controller waits ``provision_seconds`` (the
time a replacement machine takes to come up, measured on the provider), and
the whole arm starts again - a new rendezvous job, the dead node on its
replacement host - each node resuming from its last checkpoint.

Everything the controller did goes to ``controller.jsonl`` beside the nodes'
events, with the wall clock, so ``make_table.py`` can say what each fault
cost.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hosts as hostlib  # noqa: E402

KINDS = ("ravex", "torchft", "ckpt_restart")


# The schedule: pure, so it is tested without a process in sight.


@dataclass
class Fault:
    arm: str
    action: str  # "kill" or "join"
    at_step: int
    node: Optional[int] = None  # the node killed
    host: Optional[str] = None  # where a joiner, or a replacement, runs
    fired: bool = False


def parse_faults(raw: List[Dict[str, Any]], arms: List[str]) -> List[Fault]:
    faults = []
    for entry in raw:
        names = arms if entry.get("arm", "*") == "*" else [entry["arm"]]
        for arm in names:
            fault = Fault(arm=arm, action=entry["action"], at_step=int(entry["at_step"]),
                          node=entry.get("node"), host=entry.get("host"))
            if fault.action not in ("kill", "join"):
                raise ValueError("a fault is kill or join, not %r" % fault.action)
            if fault.action == "kill" and fault.node is None:
                raise ValueError("a kill names the node it kills")
            if fault.action == "join" and not fault.host:
                raise ValueError("a join names the host its node runs on")
            faults.append(fault)
    return faults


def due(faults: List[Fault], arm: str, watched_step: int) -> List[Fault]:
    """The faults of ``arm`` that the watched node's step has reached, in step order, each once."""
    ready = [f for f in faults if f.arm == arm and not f.fired and watched_step >= f.at_step]
    ready.sort(key=lambda f: f.at_step)
    for fault in ready:
        fault.fired = True
    return ready


def watched(steps: Dict[int, int], alive: Dict[int, bool], starting: List[int], doomed: List[int]) -> Optional[int]:
    """Which node's step the schedule follows: the lowest starting node that is alive and not to be killed.

    Never the node a fault will kill, so the schedule does not stop moving at
    the moment it fires; never a joiner, whose local steps count from its own
    start.
    """
    for node in sorted(starting):
        if alive.get(node) and node not in doomed:
            return node
    for node in sorted(starting):
        if alive.get(node):
            return node
    return None


def option(arguments: List[str], name: str, default: Optional[str] = None) -> Optional[str]:
    """The value of ``--name`` in an argument list, the last one given."""
    value = default
    for i, item in enumerate(arguments):
        if item == name and i + 1 < len(arguments):
            value = arguments[i + 1]
        elif item.startswith(name + "="):
            value = item.split("=", 1)[1]
    return value


def tool(host: hostlib.Host, name: str) -> str:
    """A console script installed beside the host's interpreter, or found on its PATH."""
    folder = os.path.dirname(host.python)
    return (folder + "/" + name) if folder else name


# The run.


@dataclass
class Node:
    arm: str
    number: int
    host: hostlib.Host
    joiner: bool
    process: Optional[hostlib.Process] = None
    offset: int = 0
    step: int = 0
    ended: bool = False
    killed: bool = False
    pending: bytes = b""
    #: The run's folder on the host, ``runs/<results folder's name>``: a
    #: second attempt at a phase, with results of its own, starts from empty
    #: stores and event files instead of resuming the first attempt's.
    tag: str = "runs/run"

    def file(self, name: str) -> str:
        return "%s/%s/node%d/%s" % (self.tag, self.arm, self.number, name)

    @property
    def events(self) -> str:
        return self.file("events.jsonl")

    @property
    def log(self) -> str:
        return self.file("stdout.log")


@dataclass
class Arm:
    name: str
    kind: str
    spec: Dict[str, Any]
    nodes: List[Node] = field(default_factory=list)
    coordinator: Optional[hostlib.Process] = None
    epoch: int = 0
    token: str = field(default_factory=lambda: secrets.token_hex(32))

    @property
    def starting(self) -> List[int]:
        return [n.number for n in self.nodes if not n.joiner]


class Controller:
    def __init__(self, phase: Dict[str, Any], hosts: Dict[str, hostlib.Host], results: str) -> None:
        self.phase, self.hosts, self.results = phase, hosts, results
        os.makedirs(results, exist_ok=True)
        self.tag = "runs/" + os.path.basename(os.path.normpath(results))
        self._log = open(os.path.join(results, "controller.jsonl"), "a", encoding="utf-8")
        self.arms: List[Arm] = []
        for spec in phase["arms"]:
            if spec["kind"] not in KINDS:
                raise ValueError("arm %s: kind is one of %s" % (spec["name"], ", ".join(KINDS)))
            arm = Arm(spec["name"], spec["kind"], spec)
            for number, name in enumerate(spec["nodes"]):
                arm.nodes.append(Node(arm.name, number, hosts[name], joiner=False, tag=self.tag))
            self.arms.append(arm)
        # Two arms on one GPU would each measure half a GPU, if they fit at
        # all: arms that share hosts are run one at a time, with --only.
        owners: Dict[str, str] = {}
        for spec in phase["arms"]:
            used = set(spec["nodes"]) | {f["host"] for f in phase.get("faults", [])
                                         if f.get("host") and f.get("arm") in (spec["name"], "*")}
            for name in used:
                if owners.setdefault(name, spec["name"]) != spec["name"]:
                    raise ValueError("arms %s and %s both use host %s: run them one after the other with --only"
                                     % (owners[name], spec["name"], name))
        self.faults = parse_faults(phase.get("faults", []), [a.name for a in self.arms])
        self.train_args = [str(a) for a in phase.get("train_args", [])]
        self.steps = int(option(self.train_args, "--steps", "300"))

    def note(self, event: str, **fields: Any) -> None:
        record = {"t": time.time(), "event": event}
        record.update(fields)
        self._log.write(json.dumps(record, sort_keys=True) + "\n")
        self._log.flush()
        print("controller: %s %s" % (event, json.dumps(fields, sort_keys=True)), flush=True)

    # Commands.

    def shards(self, arm: Arm) -> int:
        joins = sum(1 for f in self.faults if f.arm == arm.name and f.action == "join")
        return int(arm.spec.get("shards", len(arm.spec["nodes"]) + joins))

    def trainer(self, arm: Arm, node: Node, steps: Optional[int] = None) -> List[str]:
        script = "train_torchft.py" if arm.kind == "torchft" else "train_ravex.py"
        bench = self.phase.get("bench", HERE)
        arguments = list(self.train_args) + [str(a) for a in arm.spec.get("train_args", [])]
        if steps is not None:
            arguments += ["--steps", str(steps)]
        command = [node.host.python, os.path.join(bench, script).replace("\\", "/"),
                   "--arm", arm.name, "--node", str(node.number), "--shards", str(self.shards(arm)),
                   "--data", self.phase["data"], "--deepseek", self.phase["deepseek"],
                   "--events", node.host.path(node.events)] + arguments
        if arm.kind == "torchft":
            command += ["--advertise", node.host.address]
        return command

    def environment(self, arm: Arm, node: Node) -> Dict[str, str]:
        env = {str(k): str(v) for k, v in self.phase.get("env", {}).items()}
        env.update({str(k): str(v) for k, v in arm.spec.get("env", {}).items()})
        head = arm.nodes[0].host
        if arm.kind == "torchft":
            env["TORCHFT_LIGHTHOUSE"] = "http://%s:%d" % (head.address, int(arm.spec["port"]))
            return env
        port = int(arm.spec["port"])
        starting = [n for n in arm.nodes if not n.joiner]
        addresses = []
        for n in starting:
            if n.host.address not in [a for a, _ in addresses]:
                addresses.append((n.host.address, n))
        # A failover list only where every starting node has an address of
        # its own: on one machine, as in the CPU tests, every place would be
        # the same address and an election could not tell them apart.
        if len(addresses) == len(starting):
            env["RAVEX_RENDEZVOUS"] = ",".join("%s:%d" % (a, port) for a, _ in addresses)
            if not node.joiner:
                env["RAVEX_OUTER_MEMBER"] = str(node.number)
        else:
            env["RAVEX_RENDEZVOUS"] = "%s:%d" % (head.address, port)
        env["RAVEX_OUTER_MIN_NODES"] = str(len(starting))
        env["RAVEX_OUTER_JOB"] = "%s-%d" % (arm.name, arm.epoch)
        env["RAVEX_JOB_TOKEN"] = arm.token
        # One run id for every node of the arm, its restarts included, so the
        # nodes' stores say they are one run.
        env.setdefault("RAVEX_RUN_ID", "bench-%s" % arm.name)
        env["RAVEX_EXCHANGE_ADDRESS"] = node.host.address
        regions = {n.host.region for n in arm.nodes}
        if all(regions) and len(regions) > 1:
            env["RAVEX_OUTER_REGION"] = node.host.region
        env["RAVEX_STORAGE_PATH"] = node.host.path(node.file("store"))
        env["RAVEX_LOG_FILE"] = node.host.path(node.file("ravex.log"))
        config = arm.spec.get("ravex_config", "configs/ckpt_restart.yaml" if arm.kind == "ckpt_restart"
                              else "configs/ravex_primary.yaml")
        env["RAVEX_CONFIG"] = os.path.join(self.phase.get("bench", HERE), config).replace("\\", "/")
        return env

    def start_coordinator(self, arm: Arm) -> None:
        head = arm.nodes[0].host
        port = int(arm.spec["port"])
        if arm.kind == "torchft":
            command = [tool(head, "torchft_lighthouse"), "--bind", "[::]:%d" % port, "--min_replicas",
                       str(arm.spec.get("min_replicas", 1)), "--join_timeout_ms",
                       str(arm.spec.get("join_timeout_ms", 60000))]
            env: Dict[str, str] = {}
        else:
            command = [tool(head, "ravex"), "rendezvous", "--port", str(port)]
            env = {"RAVEX_JOB_TOKEN": arm.token}
        arm.coordinator = head.start(command, env, "%s/%s/coordinator-%d.log" % (self.tag, arm.name, arm.epoch))
        self.note("coordinator", arm=arm.name, host=head.name, epoch=arm.epoch)

    def start_node(self, arm: Arm, node: Node, steps: Optional[int] = None) -> None:
        node.process = node.host.start(self.trainer(arm, node, steps), self.environment(arm, node), node.log)
        node.ended = node.killed = False
        self.note("start", arm=arm.name, node=node.number, host=node.host.name, joiner=node.joiner,
                  epoch=arm.epoch, steps=steps or self.steps)

    # Watching.

    def poll(self, node: Node) -> None:
        data = node.host.read(node.events, node.offset)
        if not data:
            return
        node.offset += len(data)
        mirror = os.path.join(self.results, node.arm, "node%d" % node.number, "events.jsonl")
        os.makedirs(os.path.dirname(mirror), exist_ok=True)
        with open(mirror, "ab") as handle:
            handle.write(data)
        node.pending += data
        *lines, node.pending = node.pending.split(b"\n")
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") == "step":
                node.step = max(node.step, int(record["step"]))
            elif record.get("event") == "end":
                node.ended = True

    def alive(self, node: Node) -> bool:
        return node.process is not None and not node.killed and not node.ended and node.process.alive()

    # Faults.

    def fire(self, arm: Arm, fault: Fault, step: int) -> None:
        if fault.action == "join":
            node = Node(arm.name, max(n.number for n in arm.nodes) + 1, self.hosts[fault.host], joiner=True,
                        tag=self.tag)
            arm.nodes.append(node)
            self.note("fault", arm=arm.name, action="join", node=node.number, at_step=fault.at_step, step=step,
                      host=node.host.name)
            self.start_node(arm, node, steps=max(1, self.steps - step))
            return
        victim = next(n for n in arm.nodes if n.number == fault.node)
        victim.host.kill(victim.process)
        victim.killed = True
        self.note("fault", arm=arm.name, action="kill", node=victim.number, at_step=fault.at_step, step=step,
                  host=victim.host.name)
        hook = self.phase.get("on_kill")
        if hook:
            subprocess.run(hook.format(host=shlex.quote(victim.host.name), name=victim.host.name), shell=True)
        if arm.kind == "ckpt_restart":
            self.restart(arm, victim, fault)

    def restart(self, arm: Arm, victim: Node, fault: Fault) -> None:
        """Stop the whole arm, wait for the replacement, start everything again from the checkpoints."""
        for node in arm.nodes:
            if node is not victim and node.process is not None and node.process.alive():
                node.host.kill(node.process)
            node.killed = True
        if arm.coordinator is not None:
            arm.nodes[0].host.kill(arm.coordinator)
        self.note("stopped", arm=arm.name, reason="kill of node %d" % victim.number)
        time.sleep(float(self.phase.get("provision_seconds", 0)))
        if fault.host:
            victim.host = self.hosts[fault.host]
        arm.epoch += 1
        self.start_coordinator(arm)
        for node in arm.nodes:
            self.start_node(arm, node)
        self.note("restarted", arm=arm.name, epoch=arm.epoch)

    # The phase.

    def run(self, poll: float, timeout: float) -> bool:
        for arm in self.arms:
            self.start_coordinator(arm)
        # A coordinator a moment ahead of its nodes: both frameworks' nodes
        # wait for it, but a refused first dial is noise in every log.
        time.sleep(float(self.phase.get("coordinator_head_start", 2)))
        for arm in self.arms:
            for node in arm.nodes:
                self.start_node(arm, node)
        deadline = time.monotonic() + timeout
        finished = False
        try:
            while time.monotonic() < deadline:
                for arm in self.arms:
                    for node in list(arm.nodes):
                        self.poll(node)
                    doomed = [f.node for f in self.faults if f.arm == arm.name and f.action == "kill" and not f.fired]
                    alive = {n.number: self.alive(n) for n in arm.nodes}
                    which = watched({n.number: n.step for n in arm.nodes}, alive, arm.starting, doomed)
                    if which is not None:
                        step = next(n.step for n in arm.nodes if n.number == which)
                        for fault in due(self.faults, arm.name, step):
                            self.fire(arm, fault, step)
                if all(not self.alive(n) for arm in self.arms for n in arm.nodes):
                    finished = True
                    break
                time.sleep(poll)
        finally:
            for arm in self.arms:
                for node in arm.nodes:
                    if self.alive(node):
                        node.host.kill(node.process)
                    self.poll(node)
                if arm.coordinator is not None:
                    arm.nodes[0].host.kill(arm.coordinator)
            self.fetch()
            self.note("done", finished=finished,
                      ended={a.name: [n.number for n in a.nodes if n.ended] for a in self.arms})
        return finished

    def fetch(self) -> None:
        """Bring each node's logs home; the events are home already."""
        for arm in self.arms:
            for node in arm.nodes:
                base = os.path.join(self.results, arm.name, "node%d" % node.number)
                node.host.fetch(node.log, os.path.join(base, "stdout.log"))
                node.host.fetch(node.file("ravex.log"), os.path.join(base, "ravex.log"))
            for epoch in range(arm.epoch + 1):
                arm.nodes[0].host.fetch("%s/%s/coordinator-%d.log" % (self.tag, arm.name, epoch),
                                        os.path.join(self.results, arm.name, "coordinator-%d.log" % epoch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--results", default=None, help="the phase file's \"results\" by default")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between two reads of every node's events")
    parser.add_argument("--timeout", type=float, default=4 * 3600.0, help="the phase stops here, finished or not")
    parser.add_argument("--only", action="append", default=[], metavar="ARM",
                        help="run only this arm (repeatable): arms that share hosts run one after the other")
    args = parser.parse_args()
    with open(args.phase, encoding="utf-8") as handle:
        phase = json.load(handle)
    if args.only:
        unknown = set(args.only) - {a["name"] for a in phase["arms"]}
        if unknown:
            raise SystemExit("no arm named %s in %s" % (", ".join(sorted(unknown)), args.phase))
        phase["arms"] = [a for a in phase["arms"] if a["name"] in args.only]
        phase["faults"] = [f for f in phase.get("faults", []) if f.get("arm", "*") in ("*",) + tuple(args.only)]
    results = args.results or phase["results"]
    controller = Controller(phase, hostlib.load(args.hosts), results)
    finished = controller.run(args.poll, args.timeout)
    sys.exit(0 if finished else 1)


if __name__ == "__main__":
    main()
