"""Turn a phase's raw events into the results table.

    python make_table.py results/phase-a --price 0.34 [--milestones 65e6,131e6,262e6] [--checks]

It reads ``<results>/<arm>/node*/events.jsonl``, which every node wrote, and
``<results>/controller.jsonl``, which says when each fault was injected, and
nothing else: no framework's own log, store or dashboard. A second person
regenerates the table from the same files with the same command.

The columns, and what each one means exactly:

- **committed tokens**: the tokens of the steps whose work reached an outer
  step, summed over the nodes that contributed to each round, each round
  counted once (a round a checkpoint-restart arm trains twice counts once).
  A step whose delta was thrown away - a torchft window that did not commit,
  a Ravex report that missed its round, a step redone after a restart - is
  **wasted**, not a token.
- **eval loss @ T**: held-out loss of the arm's model at the first round
  whose committed tokens reach ``T``, on the same 32 fixed batches in every
  arm. The milestones default to a quarter, a half and all of the committed
  tokens of the arm that committed the fewest, so every arm has a value in
  every column.
- **tokens/s (cluster)**: committed tokens over the wall clock from the first
  node's start to the last round.
- **wasted steps**: every step any node trained, minus the committed ones.
- **lost s per kill**: from the last round before a kill to the arm's last
  round, the wall clock the arm took minus the time it would have taken to
  commit the same tokens at the rate of its last two rounds - its healthy
  tail after the fault. The excess is what the fault cost. It needs two
  rounds after the fault; with fewer it is not given.
- **join s to contribute**: from the moment the joiner was started to the
  first committed round the joiner itself reports taking part in. Not the
  first round wider than before the join: with a kill soon after, the joiner
  may only ever take the dead node's place.
- **$ per 1M tokens**: node-seconds times the price per node-hour, over the
  committed tokens. A node's seconds run from its first event to its last
  (each life of a restarted node counted), plus ``--overhead-minutes`` per
  node life for what the events cannot see (boot, data fetch, copy-off);
  ``--billed`` replaces them with the provider's billed seconds when known.

``--checks`` prints the go/no-go checks of the plan's phase (a) beside the
table.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    # A line cut by the kill, or by a copy taken mid-write.
                    continue
    except FileNotFoundError:
        pass
    return records


def load(results: str) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Every arm's events, by arm name, and the controller's."""
    arms: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results, "*", "node*", "events.jsonl"))):
        arm = os.path.basename(os.path.dirname(os.path.dirname(path)))
        arms[arm].extend(read_jsonl(path))
    for events in arms.values():
        events.sort(key=lambda e: e["t"])
    return dict(arms), read_jsonl(os.path.join(results, "controller.jsonl"))


def lives(events: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
    """Each node's lives, as (node, first event, last event); a restarted node has one per start."""
    spans: List[Tuple[int, float, float]] = []
    current: Dict[int, List[float]] = {}
    for e in events:
        node = int(e["node"])
        if e["event"] == "start" and node in current:
            spans.append((node, current[node][0], current[node][1]))
            del current[node]
        span = current.setdefault(node, [e["t"], e["t"]])
        span[1] = e["t"]
    spans.extend((node, a, b) for node, (a, b) in current.items())
    return spans


def rounds(events: List[Dict[str, Any]], tokens_per_step: int) -> List[Dict[str, Any]]:
    """The arm's committed rounds, each once, in the order they landed, with cumulative tokens.

    Each node reports every round it took part in, so a round is the earliest
    report of its number. A number reported again later - a restarted arm
    training a round a second time - is the same round, and keeps the
    *later* report's time, since that is when its work finally reached the
    model, but its tokens count once.
    """
    by_number: Dict[Any, Dict[str, Any]] = {}
    for e in events:
        if e["event"] != "round" or not e.get("committed", True):
            continue
        number = e["round"]
        seen = by_number.get(number)
        if seen is None:
            by_number[number] = {"round": number, "t": e["t"], "nodes": e["nodes"],
                                 "committed_steps": e["committed_steps"], "reporters": {e["node"]}}
        elif e["node"] in seen["reporters"]:
            # The same node reports the number again: a second life of it.
            seen.update(t=e["t"], nodes=e["nodes"], committed_steps=e["committed_steps"], reporters={e["node"]})
        else:
            seen["reporters"].add(e["node"])
    ordered = sorted(by_number.values(), key=lambda r: r["t"])
    total = 0
    for r in ordered:
        total += r["committed_steps"] * tokens_per_step
        r["tokens"] = total
        del r["reporters"]
    return ordered


def evals(events: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """The eval after each round, from the lowest node that reported it, and the spread of checksums."""
    by_round: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        if e["event"] == "eval":
            by_round[e["round"]].append(e)
    out = {}
    for number, items in by_round.items():
        items.sort(key=lambda e: (e["node"], e["t"]))
        sums = [e["checksum"] for e in items]
        out[number] = {"loss": items[0]["loss"], "spread": max(sums) - min(sums), "nodes": len(items),
                       "scale": max(abs(s) for s in sums) or 1.0}
    return out


def loss_at(milestone: float, committed: List[Dict[str, Any]], losses: Dict[Any, Dict[str, Any]]) -> Optional[float]:
    for r in committed:
        if r["tokens"] >= milestone and r["round"] in losses:
            return losses[r["round"]]["loss"]
    return None


def lost_per_kill(committed: List[Dict[str, Any]], kills: List[float]) -> List[Optional[float]]:
    """The wall clock each kill cost, against the arm's healthy tail after it (see the module's docstring)."""
    out: List[Optional[float]] = []
    for i, t_kill in enumerate(kills):
        t_next = kills[i + 1] if i + 1 < len(kills) else math.inf
        before = [r for r in committed if r["t"] <= t_kill]
        after = [r for r in committed if t_kill < r["t"] < t_next]
        if not before or len(after) < 2:
            out.append(None)
            continue
        anchor, last, prev = before[-1], after[-1], after[-2]
        if last["t"] <= prev["t"] or last["tokens"] <= prev["tokens"]:
            out.append(None)
            continue
        rate = (last["tokens"] - prev["tokens"]) / (last["t"] - prev["t"])
        took = last["t"] - anchor["t"]
        would = (last["tokens"] - anchor["tokens"]) / rate
        out.append(max(0.0, took - would))
    return out


def join_seconds(events: List[Dict[str, Any]], joins: List[Tuple[float, int]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for t_join, node in joins:
        mine = [e for e in events if e["event"] == "round" and e["node"] == node and e["t"] > t_join
                and e.get("committed", True) and e.get("participating", True)]
        out.append(min(e["t"] for e in mine) - t_join if mine else None)
    return out


def summarize(arm: str, events: List[Dict[str, Any]], controller: List[Dict[str, Any]], *, price: float,
              overhead_minutes: float, billed: Optional[Dict[str, float]]) -> Dict[str, Any]:
    starts = [e for e in events if e["event"] == "start"]
    if not starts:
        return {"arm": arm, "error": "no start event"}
    tokens_per_step = int(starts[0]["tokens_per_step"])
    committed = rounds(events, tokens_per_step)
    losses = evals(events)
    steps = [e for e in events if e["event"] == "step"]
    total_tokens = committed[-1]["tokens"] if committed else 0
    t0 = min(e["t"] for e in starts)
    t_end = committed[-1]["t"] if committed else max(e["t"] for e in events)
    committed_steps = total_tokens // tokens_per_step
    faults = [c for c in controller if c.get("event") == "fault" and c.get("arm") == arm]
    kills = [c["t"] for c in faults if c["action"] == "kill"]
    joins = [(c["t"], int(c["node"])) for c in faults if c["action"] == "join"]
    spans = lives(events)
    if billed:
        node_seconds = sum(float(v) for v in billed.values())
    else:
        node_seconds = sum(b - a for _, a, b in spans) + overhead_minutes * 60 * len(spans)
    per_node_rate = defaultdict(list)
    for e in steps:
        if e.get("seconds"):
            per_node_rate[e["node"]].append(e["tokens"] / e["seconds"])
    node_tps = {}
    for node, rates in per_node_rate.items():
        # The first steps carry warm-up (allocation, compile, the first
        # round's setup); the median of the rest is the node's speed.
        rest = sorted(rates[3:] or rates)
        node_tps[node] = rest[len(rest) // 2]
    initial = losses.get(0, {}).get("loss")
    return {
        "arm": arm,
        "tokens_per_step": tokens_per_step,
        "rounds": len(committed),
        "committed_tokens": total_tokens,
        "committed": committed,
        "losses": losses,
        "initial_loss": initial,
        "final_loss": losses[committed[-1]["round"]]["loss"] if committed and committed[-1]["round"] in losses else None,
        "tokens_per_s": total_tokens / (t_end - t0) if t_end > t0 else 0.0,
        "node_tokens_per_s": node_tps,
        "wasted_steps": max(0, len(steps) - committed_steps),
        "lost_per_kill": lost_per_kill(committed, kills),
        "join_seconds": join_seconds(events, joins),
        "node_seconds": node_seconds,
        "cost": node_seconds / 3600 * price,
        "usd_per_mtok": (node_seconds / 3600 * price) / (total_tokens / 1e6) if total_tokens else None,
        "checksum_spread": max((v["spread"] / v["scale"] for v in losses.values() if v["nodes"] > 1), default=0.0),
        "first_losses": {e["node"]: e["loss"] for e in steps if e["step"] == 1},
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return ("%." + str(digits) + "f") % value
    return str(value)


def tokens(n: float) -> str:
    return "%.1fM" % (n / 1e6) if n >= 1e5 else "%.0f" % n


def table(summaries: List[Dict[str, Any]], milestones: List[float], header: str) -> str:
    lines = [header, ""] if header else []
    columns = ["arm", "committed tokens"] + ["eval loss @ %s" % tokens(m) for m in milestones] + [
        "tokens/s (cluster)", "wasted steps", "lost s per kill", "join s to contribute", "$ per 1M tokens"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for s in summaries:
        if "error" in s:
            lines.append("| %s | %s |" % (s["arm"], s["error"]) + " |" * (len(columns) - 2))
            continue
        row = [s["arm"], tokens(s["committed_tokens"])]
        row += [fmt(loss_at(m, s["committed"], s["losses"])) for m in milestones]
        row += [fmt(s["tokens_per_s"], 0), str(s["wasted_steps"]),
                ", ".join(fmt(x, 0) for x in s["lost_per_kill"]) or "none",
                ", ".join(fmt(x, 0) for x in s["join_seconds"]) or "none",
                fmt(s["usd_per_mtok"], 3)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def checks(summaries: List[Dict[str, Any]], expected_rounds: Optional[int]) -> List[Tuple[bool, str]]:
    """The go/no-go checks of phase (a) the events can answer (plan, section 5)."""
    out: List[Tuple[bool, str]] = []
    good = [s for s in summaries if "error" not in s]
    for s in summaries:
        if "error" in s:
            out.append((False, "%s: %s" % (s["arm"], s["error"])))
    for s in good:
        if expected_rounds is not None:
            out.append((s["rounds"] >= expected_rounds, "%s completed %d of %d rounds" % (s["arm"], s["rounds"], expected_rounds)))
        out.append((s["checksum_spread"] < 1e-9,
                    "%s: after every round its nodes hold the same parameters (relative checksum spread %.2g)"
                    % (s["arm"], s["checksum_spread"])))
        if s["initial_loss"] is not None and s["final_loss"] is not None:
            out.append((s["final_loss"] < s["initial_loss"],
                        "%s: eval loss fell, %.3f -> %.3f" % (s["arm"], s["initial_loss"], s["final_loss"])))
        firsts = list(s["first_losses"].values())
        out.append((len(firsts) > 1 and len(set(round(x, 6) for x in firsts)) == len(firsts),
                    "%s: the nodes' first-step losses differ, so their data shards do (%s)"
                    % (s["arm"], ", ".join("%.4f" % x for x in firsts))))
    if len(good) >= 2:
        finals = [s["final_loss"] for s in good if s["final_loss"] is not None]
        if len(finals) == len(good):
            out.append((max(finals) - min(finals) <= 0.1, "final eval losses within 0.1 of each other (%s)"
                        % ", ".join("%s %.3f" % (s["arm"], s["final_loss"]) for s in good)))
        speeds = [sum(s["node_tokens_per_s"].values()) / max(1, len(s["node_tokens_per_s"])) for s in good]
        if all(speeds):
            out.append((max(speeds) / min(speeds) <= 1.10, "per-node tokens/s within 10%% (%s)"
                        % ", ".join("%s %.0f" % (s["arm"], v) for s, v in zip(good, speeds))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results")
    parser.add_argument("--price", type=float, required=True, help="USD per node-hour")
    parser.add_argument("--overhead-minutes", type=float, default=0.0)
    parser.add_argument("--billed", default=None, help="JSON {arm: {node: billed seconds}}")
    parser.add_argument("--milestones", default=None, help="comma-separated token counts")
    parser.add_argument("--header", default="", help="a line above the table: the network, the GPUs, the prices")
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--expected-rounds", type=int, default=None)
    args = parser.parse_args()

    arms, controller = load(args.results)
    billed = json.load(open(args.billed, encoding="utf-8")) if args.billed else {}
    summaries = [summarize(arm, events, controller, price=args.price, overhead_minutes=args.overhead_minutes,
                           billed=billed.get(arm)) for arm, events in sorted(arms.items())]
    if args.milestones:
        milestones = [float(x) for x in args.milestones.split(",")]
    else:
        least = min((s["committed_tokens"] for s in summaries if "error" not in s), default=0)
        milestones = [least / 4, least / 2, least] if least else []
    text = table(summaries, milestones, args.header)
    with open(os.path.join(args.results, "table.md"), "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(os.path.join(args.results, "table.json"), "w", encoding="utf-8") as handle:
        json.dump([{k: v for k, v in s.items() if k not in ("committed", "losses")} for s in summaries],
                  handle, indent=2, default=str)
    print(text)
    if args.checks:
        verdicts = checks(summaries, args.expected_rounds)
        for ok, line in verdicts:
            print("%s  %s" % ("GO   " if ok else "NO-GO", line))
        raise SystemExit(0 if all(ok for ok, _ in verdicts) else 2)


if __name__ == "__main__":
    main()
