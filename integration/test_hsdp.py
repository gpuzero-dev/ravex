"""HSDP: two nodes of two processes each, FSDP2 inside a node, the outer loop
between them (GPU-213).

Process ``r`` of each node exchanges shard ``r`` with process ``r`` of the
other, in a run of its own on the same rendezvous. What has to come out is the
same model on both nodes: each node starts from its own random model, so that
holds only if the seed round put the start together from one node's shards
and every round then kept the shards of the two nodes in step.
"""

import json
import os
import subprocess
import sys

import pytest

from conftest import SCRIPTS

pytest.importorskip("moonclip")

STEPS = 20
H = 5


def _node(tmp_path, number, rendezvous):
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
    }
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", str(SCRIPTS / "train_hsdp.py"), "--steps", str(STEPS),
    ]
    return directory, subprocess.Popen(
        command, cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def test_two_nodes_of_two_shards_end_with_one_model(tmp_path, monkeypatch):
    from ravex._dist import rendezvous

    monkeypatch.setenv("RAVEX_JOB_TOKEN", "hsdp-test")
    # The test holds the server, on a port it was given: never a port probed
    # free and handed over.
    server = rendezvous.serve("127.0.0.1", 0)
    # One address per node, the platform's shape: `outer_member` is a node's
    # place in that list. The second is never served; it is where node 1
    # would host the run after a failover.
    rendezvous_at = "127.0.0.1:%d,127.0.0.1:1" % server.port

    nodes = [_node(tmp_path, number, rendezvous_at) for number in (0, 1)]
    results = []
    for directory, process in nodes:
        out, err = process.communicate(timeout=600)
        assert process.returncode == 0, err[-3000:] + (directory / "ravex.log").read_text()[-3000:]
        results += [json.loads(path.read_text()) for path in sorted(directory.glob("result-*.json"))]

    assert len(results) == 4, results
    by_place = {(r["node"], r["rank"]): r for r in results}
    for rank in (0, 1):
        assert by_place[(0, rank)]["shard"] == by_place[(1, rank)]["shard"], (
            "shard %d differs between the nodes" % rank
        )
    assert len({r["whole"] for r in results}) == 1, "the nodes hold different models"
    assert by_place[(0, 0)]["shard"] != by_place[(0, 1)]["shard"], "the shards are not shards"

    for directory, _ in nodes:
        log = (directory / "ravex.log").read_text()
        for shard in (0, 1):
            assert "job hsdp.shard%d" % shard in log, log
        closed = log.count("closed over 2 node(s)")
        assert closed == 2 * (STEPS // H), "each shard closes every round:\n" + log
