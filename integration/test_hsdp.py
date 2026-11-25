"""HSDP: two nodes of two processes each, FSDP2 inside a node, the outer loop
between them (GPU-213).

Process ``r`` of each node exchanges shard ``r`` with process ``r`` of the
other, in a run of its own on the same rendezvous. What has to come out is the
same model on both nodes: each node starts from its own random model, so that
holds only if the seed round put the start together from one node's shards
and every round then kept the shards of the two nodes in step.
"""

import collections
import json
import os
import re
import subprocess
import sys

import pytest

from conftest import SCRIPTS

pytest.importorskip("moonclip")

STEPS = 20
H = 5


def _node(tmp_path, number, rendezvous, **extra):
    directory = tmp_path / ("node%d" % number)
    directory.mkdir()
    env = {
        **os.environ,
        "RAVEX_OUTER_LOOP": "true",
        "RAVEX_OUTER_INNER_STEPS": str(H),
        "RAVEX_RENDEZVOUS": rendezvous,
        "RAVEX_OUTER_MEMBER": str(number),
        "RAVEX_OUTER_MIN_NODES": "2",
        "RAVEX_OUTER_JOB": "hsdp",
        "RAVEX_RUN_ID": "hsdp",
        "RAVEX_JOB_TOKEN": "hsdp-test",
        "RAVEX_EXCHANGE_ADDRESS": "127.0.0.1",
        "RAVEX_STORAGE_PATH": str(directory / "store"),
        "RAVEX_LOG_FILE": str(directory / "ravex.log"),
        "OMP_NUM_THREADS": "1",
        **extra,
    }
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", str(SCRIPTS / "train_hsdp.py"), "--steps", str(STEPS),
    ]
    return directory, subprocess.Popen(
        command, cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def _run(tmp_path, monkeypatch, **extra):
    """Two nodes of two processes to the end: their results and their logs."""
    from ravex._dist import rendezvous

    monkeypatch.setenv("RAVEX_JOB_TOKEN", "hsdp-test")
    # The test holds the server, on a port it was given: never a port probed
    # free and handed over.
    server = rendezvous.serve("127.0.0.1", 0)
    # One address per node, the platform's shape: `outer_member` is a node's
    # place in that list. The second is never served; it is where node 1
    # would host the run after a failover.
    rendezvous_at = "127.0.0.1:%d,127.0.0.1:1" % server.port

    nodes = [_node(tmp_path, number, rendezvous_at, **extra) for number in (0, 1)]
    results, logs = [], []
    for directory, process in nodes:
        out, err = process.communicate(timeout=600)
        assert process.returncode == 0, err[-3000:] + (directory / "ravex.log").read_text()[-3000:]
        results += [json.loads(path.read_text()) for path in sorted(directory.glob("result-*.json"))]
        logs.append((directory / "ravex.log").read_text())
    return results, logs


def test_two_nodes_of_two_shards_end_with_one_model(tmp_path, monkeypatch):
    results, logs = _run(tmp_path, monkeypatch)

    assert len(results) == 4, results
    by_place = {(r["node"], r["rank"]): r for r in results}
    for rank in (0, 1):
        assert by_place[(0, rank)]["shard"] == by_place[(1, rank)]["shard"], (
            "shard %d differs between the nodes" % rank
        )
    assert len({r["whole"] for r in results}) == 1, "the nodes hold different models"
    assert by_place[(0, 0)]["shard"] != by_place[(0, 1)]["shard"], "the shards are not shards"

    for log in logs:
        for shard in (0, 1):
            assert "job hsdp.shard%d" % shard in log, log
        closed = log.count("closed over 2 node(s)")
        assert closed == 2 * (STEPS // H), "each shard closes every round:\n" + log


def test_an_automatic_h_is_the_same_on_every_shard(tmp_path, monkeypatch):
    """H moved by the link, decided once for every process of every node on
    shard 0's job: each of a node's two processes takes the same H at the same
    round, or the node would hold shards of different rounds.

    Capped at 3 from a start of 5, so it has to move."""
    _, logs = _run(
        tmp_path, monkeypatch,
        RAVEX_OUTER_AUTO_STEPS="true", RAVEX_OUTER_MIN_INNER_STEPS="2", RAVEX_OUTER_MAX_INNER_STEPS="3",
    )
    changes = []
    for log in logs:
        moves = collections.Counter(re.findall(r"Outer rounds go from (\d+) to (\d+) inner step\(s\) [a-z ]+: round (\d+)", log))
        assert moves, "H never moved:\n" + log
        assert all(count == 2 for count in moves.values()), "the shards took different H: %s" % moves
        closed = log.count("closed over 2 node(s)")
        assert closed % 2 == 0, "one shard closed a round the other did not:\n" + log
        changes.append(sorted(moves))
    assert changes[0] == changes[1], "the nodes took different H"


def test_shards_overlap_their_rounds_with_the_training(tmp_path, monkeypatch):
    """With `outer_overlap` a round is in flight while the next one trains, and
    each process lands it at the first boundary after it came back, which need
    not be the same step on the two processes of a node: each lands its own
    shard, as Streaming DiLoCo lands fragments at different steps. What has to
    hold is that every round is launched by both shards at the same step, so
    each node launches and lands as many rounds on one shard as on the other."""
    _, logs = _run(
        tmp_path, monkeypatch,
        RAVEX_OUTER_OVERLAP="true", RAVEX_OUTER_AUTO_STEPS="true",
        RAVEX_OUTER_MIN_INNER_STEPS="2", RAVEX_OUTER_MAX_INNER_STEPS="3",
    )
    for log in logs:
        closed = log.count("closed over 2 node(s)")
        assert closed and closed % 2 == 0, "one shard landed a round the other did not:\n" + log
        moves = collections.Counter(re.findall(r"Outer rounds go from (\d+) to (\d+) inner step\(s\) [a-z ]+: round (\d+)", log))
        assert all(count == 2 for count in moves.values()), "the shards took different H: %s" % moves
