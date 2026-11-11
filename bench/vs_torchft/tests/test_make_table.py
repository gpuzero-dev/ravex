"""The results table from hand-made events, where every number can be worked out by hand."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import make_table  # noqa: E402

TPS = 1000  # tokens per step


def start(t, node):
    return {"t": t, "arm": "a", "node": node, "event": "start", "tokens_per_step": TPS}


def step(t, node, number, loss=1.0, seconds=1.0):
    return {"t": t, "arm": "a", "node": node, "event": "step", "step": number, "tokens": TPS,
            "seconds": seconds, "loss": loss}


def round_(t, node, number, nodes, steps_each=10, committed=True):
    return {"t": t, "arm": "a", "node": node, "event": "round", "round": number, "nodes": nodes,
            "committed": committed, "committed_steps": nodes * steps_each if committed else 0}


def eval_(t, node, number, loss, checksum=5.0):
    return {"t": t, "arm": "a", "node": node, "event": "eval", "round": number, "loss": loss, "checksum": checksum}


def two_nodes_three_rounds():
    events = [start(0, 0), start(0, 1), eval_(0, 0, 0, 10.0), eval_(0, 1, 0, 10.0)]
    for node in (0, 1):
        events += [step(1 + s, node, s + 1, loss=5.0 + node) for s in range(30)]
    for r, t in ((1, 10), (2, 20), (3, 30)):
        for node in (0, 1):
            events += [round_(t, node, r, 2), eval_(t, node, r, 10.0 - r)]
    return sorted(events, key=lambda e: e["t"])


def test_each_round_counts_once_and_tokens_accumulate():
    committed = make_table.rounds(two_nodes_three_rounds(), TPS)
    assert [r["round"] for r in committed] == [1, 2, 3]
    assert [r["tokens"] for r in committed] == [20_000, 40_000, 60_000]


def test_a_summary_has_every_column_worked_out():
    s = make_table.summarize("a", two_nodes_three_rounds(), [], price=3.6, overhead_minutes=0, billed=None)
    assert s["committed_tokens"] == 60_000
    assert s["tokens_per_s"] == 60_000 / 30
    assert s["wasted_steps"] == 0
    # Two node lives of 30 s each, at $3.6 an hour: $0.06 for 0.06M tokens.
    assert abs(s["node_seconds"] - 60) < 1e-9
    assert abs(s["usd_per_mtok"] - 1.0) < 1e-9
    assert s["final_loss"] == 7.0 and s["initial_loss"] == 10.0
    assert make_table.loss_at(40_000, s["committed"], s["losses"]) == 8.0
    assert make_table.loss_at(70_000, s["committed"], s["losses"]) is None


def test_a_discarded_window_is_wasted_not_committed():
    events = two_nodes_three_rounds()
    events = [e for e in events if not (e["event"] == "round" and e["round"] == 2)]
    events += [round_(20, n, 2, 2, committed=False) for n in (0, 1)]
    s = make_table.summarize("a", sorted(events, key=lambda e: e["t"]), [], price=1, overhead_minutes=0, billed=None)
    assert s["committed_tokens"] == 40_000
    assert s["wasted_steps"] == 20


def test_a_round_trained_twice_after_a_restart_counts_once_at_its_later_time():
    events = two_nodes_three_rounds() + [start(40, 0), round_(50, 0, 3, 2), step(45, 0, 30)]
    committed = make_table.rounds(sorted(events, key=lambda e: e["t"]), TPS)
    assert [r["round"] for r in committed] == [1, 2, 3]
    assert committed[-1]["t"] == 50 and committed[-1]["tokens"] == 60_000
    # The second life is billed too.
    assert len(make_table.lives(sorted(events, key=lambda e: e["t"]))) == 3


def test_lost_time_is_against_the_healthy_tail_after_the_kill():
    # Rounds every 10 s commit 1000 tokens each; a kill at t=25 stalls the
    # arm for 30 s, then it goes on at the same rate.
    committed = [{"round": i, "t": t, "tokens": 1000 * i, "nodes": 2}
                 for i, t in enumerate([10, 20, 60, 70, 80], start=1)]
    assert make_table.lost_per_kill(committed, [25]) == [30.0]
    # One round after the kill is not a tail to measure against.
    assert make_table.lost_per_kill(committed[:3], [25]) == [None]


def test_join_is_timed_to_the_first_round_the_joiner_takes_part_in():
    events = [round_(20, 0, 2, 2), round_(25, 2, 2, 2, committed=False), round_(30, 2, 3, 3), round_(40, 2, 4, 3)]
    assert make_table.join_seconds(events, [(12, 2)]) == [18]
    assert make_table.join_seconds(events, [(12, 5)]) == [None]


def test_the_checks_pass_on_a_healthy_pair_and_fail_on_shared_data(tmp_path):
    good = make_table.summarize("a", two_nodes_three_rounds(), [], price=1, overhead_minutes=0, billed=None)
    assert all(ok for ok, _ in make_table.checks([good], expected_rounds=3))
    same = [e for e in two_nodes_three_rounds() if not (e["event"] == "step" and e["node"] == 1)]
    same += [step(1 + s, 1, s + 1, loss=5.0) for s in range(30)]
    same = sorted(same, key=lambda e: e["t"])
    for e in same:
        if e["event"] == "step" and e["node"] == 0:
            e["loss"] = 5.0
    bad = make_table.summarize("a", same, [], price=1, overhead_minutes=0, billed=None)
    verdicts = make_table.checks([bad], expected_rounds=3)
    assert not all(ok for ok, line in verdicts if "shards" in line)


def test_the_table_is_written_from_a_results_directory(tmp_path):
    folder = tmp_path / "a" / "node0"
    folder.mkdir(parents=True)
    events = two_nodes_three_rounds()
    for node in (0, 1):
        path = tmp_path / "a" / ("node%d" % node)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "events.jsonl", "w") as handle:
            for e in events:
                if e["node"] == node:
                    handle.write(json.dumps(e) + "\n")
            handle.write('{"t": 99, "cut by the kill')
    arms, controller = make_table.load(str(tmp_path))
    s = make_table.summarize("a", arms["a"], controller, price=1, overhead_minutes=0, billed=None)
    text = make_table.table([s], [30_000, 60_000], "header")
    assert "| a | 60000 | 8.000 | 7.000 |" in text
