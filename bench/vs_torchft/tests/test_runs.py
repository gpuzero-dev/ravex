"""Whole phases on one Linux machine: two Ravex nodes and two torchft nodes, then the faults.

Small enough for a CPU - deepseek-1b's tiny config with a 512-token
vocabulary, sequences of 32, H of 10 - and run the way a phase is run on
rented machines: ``controller.py`` with a hosts file, here naming this
machine, then ``make_table.py --checks``. What it proves is that the harness
measures what it says it does - rounds commit, every node of an arm holds the
same parameters after each, the shards differ, the faults land at their step
and the table is computed - not anything about either framework's speed.

Linux only, in ``Dockerfile.cpu``'s container: see ``conftest.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEPSEEK = os.environ.get("DEEPSEEK_1B", "")

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="multi-process runs are judged on Linux (conftest.py)"),
    pytest.mark.skipif(not os.path.isfile(os.path.join(DEEPSEEK, "data.py")),
                       reason="DEEPSEEK_1B must name a deepseek-1b checkout with data.py"),
]

VOCAB = 512


def data(folder) -> str:
    """Tokens a small model can learn in a few steps: each is the last plus 1, 2 or 3.

    Uniform random tokens would leave nothing to learn, and the check that
    the loss fell could not pass; this stream has an entropy of ln 3 against
    the ln 512 of a model that knows nothing.
    """
    rng = np.random.default_rng(0)
    os.makedirs(folder, exist_ok=True)
    for name, length in (("train.bin", 300_000), ("val.bin", 20_000)):
        stream = np.cumsum(rng.integers(1, 4, length)) % VOCAB
        stream.astype("<u4").tofile(os.path.join(folder, name))
    return str(folder)


def train_args(steps: int, pace: float = 0.0):
    return ["--config", "configs/tiny.json", "--override", "vocab_size=%d" % VOCAB, "--steps", str(steps),
            "--inner-steps", "10", "--batch", "2", "--accum", "2", "--seq", "32", "--lr", "3e-3",
            "--warmup", "5", "--eval-batches", "2", "--pace", str(pace)]


def run_phase(tmp_path, arms, faults=(), steps=30, pace=0.0, timeout=900, **extra):
    workdir = tmp_path / "work"
    workdir.mkdir()
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps({"hosts": [
        {"name": "here%d" % i, "local": True, "address": "127.0.0.1", "workdir": str(workdir),
         "python": sys.executable} for i in range(8)]}))
    phase = {"results": str(tmp_path / "results"), "bench": HERE, "deepseek": DEEPSEEK,
             "data": data(tmp_path / "data"), "train_args": train_args(steps, pace),
             "env": {"MOONCLIP_FSYNC": "0", "OMP_NUM_THREADS": "1"}, "arms": arms, "faults": list(faults),
             "coordinator_head_start": 3}
    phase.update(extra)
    phase_file = tmp_path / "phase.json"
    phase_file.write_text(json.dumps(phase))
    done = subprocess.run([sys.executable, os.path.join(HERE, "controller.py"), "--hosts", str(hosts),
                           "--phase", str(phase_file), "--poll", "0.5", "--timeout", str(timeout)],
                          capture_output=True, text=True, timeout=timeout + 120)
    results = phase["results"]
    return done, results


def table(results, expected_rounds, *extra):
    return subprocess.run([sys.executable, os.path.join(HERE, "make_table.py"), results, "--price", "0.34",
                           "--checks", "--expected-rounds", str(expected_rounds)] + list(extra),
                          capture_output=True, text=True, timeout=120)


def logs(results) -> str:
    out = []
    for root, _, files in os.walk(results):
        for name in files:
            if name.endswith(".log"):
                with open(os.path.join(root, name), errors="replace") as handle:
                    out.append("== %s\n%s" % (os.path.join(root, name), handle.read()[-4000:]))
    return "\n".join(out)


def events(results, arm):
    out = []
    for node in sorted(os.listdir(os.path.join(results, arm))):
        path = os.path.join(results, arm, node, "events.jsonl")
        if os.path.isfile(path):
            with open(path) as handle:
                out += [json.loads(line) for line in handle if line.strip()]
    return out


RAVEX = {"name": "ravex", "kind": "ravex", "nodes": ["here0", "here1"], "port": 29400,
         "ravex_config": "tests/ravex_cpu.yaml"}
TORCHFT = {"name": "torchft", "kind": "torchft", "nodes": ["here2", "here3"], "port": 29510,
           "join_timeout_ms": 5000, "env": {"TORCHFT_TIMEOUT_SEC": "60", "TORCHFT_QUORUM_TIMEOUT_SEC": "120"}}


def test_phase_a_in_small_two_nodes_per_system_three_rounds_each(tmp_path):
    done, results = run_phase(tmp_path, [RAVEX, TORCHFT])
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:] + logs(results)
    verdict = table(results, 3)
    print(verdict.stdout)
    # Every check but the speed one: on a CPU shared by four trainers, and
    # with Ravex's checkpoint at every step inside its step time while
    # torchft writes none, the two arms' tokens/s say nothing here.
    failed = [line for line in verdict.stdout.splitlines() if line.startswith("NO-GO") and "tokens/s" not in line]
    assert not failed and "GO " in verdict.stdout, verdict.stdout + logs(results)
    for arm in ("ravex", "torchft"):
        rounds = [e for e in events(results, arm) if e["event"] == "round"]
        # Both nodes report all three rounds, each over both nodes.
        assert sorted((e["node"], e["round"], e["nodes"]) for e in rounds) == [
            (n, r, 2) for n in (0, 1) for r in (1, 2, 3)], rounds


def test_a_join_then_a_kill_on_both_systems(tmp_path):
    # The join well before the kill, as the plan orders them: a Ravex joiner
    # is admitted two rounds after it asks (JOIN_MARGIN), and one that is
    # still being admitted when a member dies waits on that member's
    # acknowledgement first, which on a run this short can take its whole
    # remaining time.
    faults = []
    for arm, host in (("ravex", "here4"), ("torchft", "here5")):
        faults += [{"arm": arm, "action": "join", "at_step": 8, "host": host},
                   {"arm": arm, "action": "kill", "at_step": 45, "node": 1}]
    done, results = run_phase(tmp_path, [RAVEX, TORCHFT], faults, steps=80, pace=0.3)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:] + logs(results)
    verdict = table(results, 3)
    print(verdict.stdout)
    with open(os.path.join(results, "controller.jsonl")) as handle:
        notes = [json.loads(line) for line in handle]
    fired = [(n["arm"], n["action"], n["node"]) for n in notes if n["event"] == "fault"]
    assert sorted(fired) == [("ravex", "join", 2), ("ravex", "kill", 1), ("torchft", "join", 2), ("torchft", "kill", 1)]
    for arm in ("ravex", "torchft"):
        rounds = [e for e in events(results, arm) if e["event"] == "round" and e.get("committed")]
        # The run went on after the kill without anyone touching it, and the
        # joiner took part in a round.
        assert max(e["round"] for e in rounds if e["node"] == 0) >= 4, (arm, rounds, logs(results))
        assert any(e["node"] == 2 for e in rounds), (arm, rounds, logs(results))
    with open(os.path.join(results, "table.json")) as handle:
        summaries = {s["arm"]: s for s in json.load(handle)}
    for arm in ("ravex", "torchft"):
        # The table timed the join and the kill of each arm.
        assert summaries[arm]["join_seconds"][0] is not None, verdict.stdout
        assert summaries[arm]["lost_per_kill"][0] is not None, verdict.stdout


def test_the_checkpoint_restart_arm_stops_everyone_and_resumes(tmp_path):
    ckpt = {"name": "ckpt", "kind": "ckpt_restart", "nodes": ["here0", "here1"], "port": 29401,
            "ravex_config": "tests/ravex_cpu.yaml"}
    faults = [{"arm": "ckpt", "action": "kill", "at_step": 15, "node": 1}]
    done, results = run_phase(tmp_path, [ckpt], faults, steps=30, pace=0.2, provision_seconds=2)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:] + logs(results)
    with open(os.path.join(results, "controller.jsonl")) as handle:
        notes = [json.loads(line)["event"] for line in handle]
    assert "stopped" in notes and "restarted" in notes
    seen = events(results, "ckpt")
    starts = [e for e in seen if e["event"] == "start"]
    # Two lives per node, the second resumed from a checkpoint, not step 0.
    # The round grid restarts from the resume, so a restarted arm can end
    # with one round fewer than it would have had: that is the arm's cost, and
    # the table shows it.
    assert len(starts) == 4 and any(e["first_step"] > 0 for e in starts), (starts, logs(results))
    assert all(any(e["event"] == "end" and e["node"] == n for e in seen) for n in (0, 1)), logs(results)
    verdict = table(results, 3)
    print(verdict.stdout)
    assert "| ckpt |" in verdict.stdout


def test_the_vram_check_runs_the_inner_step_alone(tmp_path):
    done = subprocess.run([sys.executable, os.path.join(HERE, "vram_check.py"), "--deepseek", DEEPSEEK,
                           "--data", data(tmp_path / "data"), "--node", "0", "--shards", "1", "--arm", "a1",
                           "--events", str(tmp_path / "a1.jsonl")] + train_args(8),
                          capture_output=True, text=True, timeout=300, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "tokens/s per node" in done.stdout
