"""The controller's schedule and the commands it builds, with no process started."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import controller  # noqa: E402
import hosts as hostlib  # noqa: E402


def test_a_fault_for_every_arm_is_one_fault_per_arm():
    faults = controller.parse_faults([{"arm": "*", "action": "kill", "node": 3, "at_step": 750}], ["r", "t"])
    assert [(f.arm, f.node, f.at_step) for f in faults] == [("r", 3, 750), ("t", 3, 750)]


def test_a_kill_without_a_node_and_a_join_without_a_host_are_refused():
    with pytest.raises(ValueError):
        controller.parse_faults([{"arm": "r", "action": "kill", "at_step": 1}], ["r"])
    with pytest.raises(ValueError):
        controller.parse_faults([{"arm": "r", "action": "join", "at_step": 1}], ["r"])
    with pytest.raises(ValueError):
        controller.parse_faults([{"arm": "r", "action": "reboot", "at_step": 1}], ["r"])


def test_faults_fire_once_in_step_order_when_the_step_is_reached():
    faults = controller.parse_faults(
        [{"arm": "r", "action": "kill", "node": 1, "at_step": 750},
         {"arm": "r", "action": "join", "host": "spare", "at_step": 360},
         {"arm": "t", "action": "join", "host": "spare2", "at_step": 360}], ["r", "t"])
    assert controller.due(faults, "r", 359) == []
    # A poll that finds the node far ahead fires everything it passed, the
    # earlier fault first: the join must precede the kill (plan, section 4).
    fired = controller.due(faults, "r", 800)
    assert [f.action for f in fired] == ["join", "kill"]
    assert controller.due(faults, "r", 900) == []
    assert [f.arm for f in controller.due(faults, "t", 400)] == ["t"]


def test_the_schedule_follows_a_starting_node_that_will_not_be_killed():
    alive = {0: True, 1: True, 2: True}
    assert controller.watched({}, alive, [0, 1], doomed=[0]) == 1
    assert controller.watched({}, alive, [0, 1], doomed=[]) == 0
    # Node 2 is a joiner: its steps count from its own start.
    assert controller.watched({}, {0: False, 1: False, 2: True}, [0, 1], doomed=[]) is None
    assert controller.watched({}, {0: False, 1: True}, [0, 1], doomed=[1]) == 1


def test_option_reads_the_last_value_given():
    assert controller.option(["--steps", "300", "--x", "1", "--steps=1200"], "--steps") == "1200"
    assert controller.option([], "--steps", "7") == "7"


def phase(kind="ravex", hosts=("a", "b")):
    return {
        "bench": "/b", "deepseek": "/d", "data": "/data", "train_args": ["--steps", "300"],
        "arms": [{"name": "arm", "kind": kind, "nodes": list(hosts), "port": 29400}],
        "faults": [{"arm": "arm", "action": "join", "host": "c", "at_step": 100}],
    }


def make(tmp_path, kind="ravex", addresses=("10.0.0.1", "10.0.0.2", "10.0.0.3"), regions=("", "", "")):
    machines = {name: hostlib.Host(name=name, address=address, workdir="/w", region=region, ssh="root@x")
                for name, address, region in zip("abc", addresses, regions)}
    return controller.Controller(phase(kind), machines, str(tmp_path))


def test_ravex_nodes_get_a_failover_list_their_place_and_a_shard_for_the_joiner(tmp_path):
    c = make(tmp_path)
    arm = c.arms[0]
    env = c.environment(arm, arm.nodes[1])
    assert env["RAVEX_RENDEZVOUS"] == "10.0.0.1:29400,10.0.0.2:29400"
    assert env["RAVEX_OUTER_MEMBER"] == "1"
    assert env["RAVEX_OUTER_MIN_NODES"] == "2"
    assert env["RAVEX_EXCHANGE_ADDRESS"] == "10.0.0.2"
    assert env["RAVEX_CONFIG"] == "/b/configs/ravex_primary.yaml"
    assert "RAVEX_OUTER_REGION" not in env
    command = c.trainer(arm, arm.nodes[1])
    # Two starting nodes and one join: three disjoint shards.
    assert command[command.index("--shards") + 1] == "3"
    assert command[command.index("--node") + 1] == "1"


def test_nodes_on_one_machine_share_one_rendezvous_address(tmp_path):
    c = make(tmp_path, addresses=("127.0.0.1",) * 3)
    env = c.environment(c.arms[0], c.arms[0].nodes[1])
    assert env["RAVEX_RENDEZVOUS"] == "127.0.0.1:29400"
    assert "RAVEX_OUTER_MEMBER" not in env


def test_regions_are_set_only_when_every_node_has_one_and_there_are_several(tmp_path):
    c = make(tmp_path, regions=("eu", "us", "eu"))
    assert c.environment(c.arms[0], c.arms[0].nodes[1])["RAVEX_OUTER_REGION"] == "us"


def test_torchft_nodes_are_told_the_lighthouse_and_their_own_address(tmp_path):
    c = make(tmp_path, kind="torchft")
    arm = c.arms[0]
    assert c.environment(arm, arm.nodes[1])["TORCHFT_LIGHTHOUSE"] == "http://10.0.0.1:29400"
    command = c.trainer(arm, arm.nodes[1])
    assert command[1].endswith("train_torchft.py")
    assert command[command.index("--advertise") + 1] == "10.0.0.2"


def test_the_checkpoint_restart_arm_takes_its_own_config(tmp_path):
    c = make(tmp_path, kind="ckpt_restart")
    assert c.environment(c.arms[0], c.arms[0].nodes[0])["RAVEX_CONFIG"] == "/b/configs/ckpt_restart.yaml"


def test_a_host_without_ssh_or_local_is_refused(tmp_path):
    path = tmp_path / "hosts.json"
    path.write_text('{"hosts": [{"name": "x", "address": "1.2.3.4", "workdir": "/w"}]}')
    with pytest.raises(ValueError):
        hostlib.load(str(path))


def test_two_arms_on_one_host_are_refused_unless_run_one_at_a_time(tmp_path):
    shared = phase()
    shared["arms"].append({"name": "other", "kind": "torchft", "nodes": ["a", "b"], "port": 29510})
    machines = {name: hostlib.Host(name=name, address="10.0.0.%d" % i, workdir="/w", ssh="root@x")
                for i, name in enumerate("abc")}
    with pytest.raises(ValueError, match="one after the other"):
        controller.Controller(shared, machines, str(tmp_path))
    shared["arms"] = shared["arms"][:1]
    controller.Controller(shared, machines, str(tmp_path))


def test_a_second_attempt_at_a_phase_gets_folders_of_its_own_on_the_hosts(tmp_path):
    first = make(tmp_path / "phase-a")
    second = make(tmp_path / "phase-a-2")
    node = first.arms[0].nodes[0]
    assert node.events == "runs/phase-a/arm/node0/events.jsonl"
    assert second.arms[0].nodes[0].events == "runs/phase-a-2/arm/node0/events.jsonl"
    assert first.environment(first.arms[0], node)["RAVEX_STORAGE_PATH"] == "/w/runs/phase-a/arm/node0/store"
