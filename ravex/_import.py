"""Runs logged somewhere else, written as Ravex stores (GPU-314).

Somebody whose history is in W&B, or in a file exported from Neptune or
MLflow, should not have to rewrite their training loop before they can see
an old run beside a new one. ``ravex import`` writes such a run as a store
like any other - ``run.json``, ``status.json`` and one metrics segment - so
everything that reads a store reads it: :func:`ravex.metrics.read`, and
``ravex ship``, which sends it to a backend as if the run had just finished.

Only the scalars come across: a number per metric per step. Media, tables
and histograms of another tool have no form here that would not be a guess,
and checkpoints are not metrics: a run imported can be looked at and
compared, not resumed.

Nothing here talks to a network but the W&B reader, and that only through
the ``wandb`` package the person already has; Ravex does not depend on it.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from ravex._metrics import HEADER_CHUNK, METRICS_DIR, _number, chunk_name, segment_id
from ravex._runs import RUN_FILE, ensure_run, write_json, write_status

#: Records per chunk: a long run becomes several files, each small enough to
#: send in one request.
CHUNK_RECORDS = 5000

#: The format of a segment's header, as the writer of a live run puts it.
FORMAT_VERSION = 1

Record = Tuple[int, float, Dict[str, Any]]


def _scalar(value: Any) -> Optional[Any]:
    """A number as a chunk holds it, or None for anything that is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _number(float(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("nan", "inf", "-inf", "infinity", "-infinity"):
            return _number(float(text))
        try:
            return _number(float(text))
        except ValueError:
            return None
    return None


def write_store(
    storage: str,
    records: Iterable[Record],
    *,
    name: str,
    config: Optional[Mapping[str, Any]] = None,
    created_at: Optional[float] = None,
    source: str,
) -> Dict[str, Any]:
    """Write ``records`` - (step, time, values), in step order - as a new
    store at ``storage``. Returns what was written: the run document, the
    steps and the metric names.

    Refuses a directory that already holds a run: an import never adds to,
    or replaces, a run that exists.
    """
    if os.path.exists(os.path.join(storage, RUN_FILE)):
        raise FileExistsError("%s already holds a run; import into a new directory" % storage)
    os.makedirs(storage, exist_ok=True)
    run = ensure_run(storage, name=name, config=dict(config or {}), version="import")
    # The run's own birth, not the import's, and where it came from: run.json
    # is written once, here, so this is the only moment to say either.
    if created_at is not None:
        run["created_at"] = created_at
    run["imported_from"] = source
    write_json(os.path.join(storage, RUN_FILE), run)

    started = created_at if created_at is not None else time.time()
    segment = segment_id(started, "import", 0, 0)
    directory = os.path.join(storage, METRICS_DIR, segment)
    os.makedirs(directory)
    header = {
        "segment": segment,
        "format": FORMAT_VERSION,
        "rank": 0,
        "host": "import",
        "pid": 0,
        "start_step": 0,
        "time": started,
    }
    with open(os.path.join(directory, HEADER_CHUNK), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")

    names = set()
    last_step = 0
    count = 0
    lines: List[str] = []
    sequence = 0

    def flush() -> None:
        nonlocal sequence, lines
        if lines:
            sequence += 1
            with open(os.path.join(directory, chunk_name(sequence)), "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            lines = []

    for step, when, values in records:
        kept = {key: value for key, value in ((k, _scalar(v)) for k, v in values.items()) if value is not None}
        if not kept:
            continue
        names.update(kept)
        last_step = max(last_step, int(step))
        count += 1
        lines.append(json.dumps({"step": int(step), "time": float(when), "values": kept}))
        if len(lines) >= CHUNK_RECORDS:
            flush()
    flush()
    write_status(storage, state="finished", step=last_step, run_id=run["run_id"])
    return {"run": run, "records": count, "last_step": last_step, "metrics": sorted(names)}


# ─── Weights & Biases ───────────────────────────────────────────────


def _timestamp(text: Any) -> Optional[float]:
    if isinstance(text, (int, float)):
        return float(text)
    if isinstance(text, str) and text.strip():
        # Seconds or milliseconds as a CSV writes them, or ISO 8601.
        try:
            return float(text)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def from_wandb(path: str, api: Any = None) -> Tuple[Dict[str, Any], Iterator[Record]]:
    """A W&B run, ``entity/project/run_id``, as (what describes it, its
    records). ``api`` is a ``wandb.Api()``, made here when not given.

    Every row of the run's full history, not the sampled one: ``scan_history``
    pages through it. W&B's own columns (``_step``, ``_timestamp``,
    ``_runtime``...) give the step and the time and are not metrics.
    """
    if api is None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            raise RuntimeError("importing from W&B needs the wandb package: pip install wandb") from None
        api = wandb.Api()
    run = api.run(path)
    created = _timestamp(getattr(run, "created_at", None))
    about = {
        "name": getattr(run, "name", None) or path.rsplit("/", 1)[-1],
        "config": {k: v for k, v in dict(getattr(run, "config", {}) or {}).items() if not k.startswith("_")},
        "created_at": created,
        "source": "wandb:" + path,
    }

    def records() -> Iterator[Record]:
        for index, row in enumerate(run.scan_history()):
            step = row.get("_step", index)
            when = row.get("_timestamp") or (created or 0.0) + float(row.get("_runtime") or 0.0)
            yield int(step), float(when), {k: v for k, v in row.items() if not k.startswith("_")}

    return about, records()


# ─── a table: Neptune's or MLflow's export, or anything else ────────

#: Column names taken for the step and the time, in a wide table or a long one.
STEP_COLUMNS = ("step", "_step", "iteration", "epoch")
TIME_COLUMNS = ("time", "timestamp", "_timestamp", "wall_time")
#: In a long table, one value per row: the metric's name and its value.
NAME_COLUMNS = ("metric", "attribute_path", "path", "key", "name")
VALUE_COLUMNS = ("value",)


def _column(columns: Iterable[str], wanted: Tuple[str, ...]) -> Optional[str]:
    lowered = {column.strip().lower(): column for column in columns}
    for name in wanted:
        if name in lowered:
            return lowered[name]
    return None


def _rows(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    if path.endswith((".jsonl", ".ndjson")):
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row.get("values"), dict):
                        # Ravex's own record form: step, time, values.
                        flat = {k: v for k, v in row.items() if k != "values"}
                        flat.update(row["values"])
                        row = flat
                    rows.append(row)
        columns: List[str] = []
        for row in rows:
            columns.extend(key for key in row if key not in columns)
        return columns, rows
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def from_table(path: str) -> Iterator[Record]:
    """A CSV or JSONL file of metrics, as records in step order.

    Two shapes, told apart by their columns. **Wide**: a row per step, a
    ``step`` column and one column per metric. **Long**: a row per value,
    with the metric's name in ``metric`` (or ``attribute_path``, as Neptune
    names it) and the number in ``value``, beside ``step``. A ``time`` or
    ``timestamp`` column, in seconds or ISO 8601, is the time; without one
    the steps are a second apart, which keeps the order and claims nothing.
    """
    columns, rows = _rows(path)
    step_column = _column(columns, STEP_COLUMNS)
    if step_column is None:
        raise ValueError("%s has no step column (one of %s)" % (path, ", ".join(STEP_COLUMNS)))
    time_column = _column(columns, TIME_COLUMNS)
    name_column = _column(columns, NAME_COLUMNS)
    value_column = _column(columns, VALUE_COLUMNS)
    by_step: Dict[int, Tuple[float, Dict[str, Any]]] = {}
    for row in rows:
        raw_step = _scalar(row.get(step_column))
        if raw_step is None or isinstance(raw_step, str):
            continue
        step = int(raw_step)
        when = _timestamp(row.get(time_column)) if time_column else None
        if when is not None and when > 1e11:
            when = when / 1000.0  # milliseconds, as Neptune and MLflow write them
        when = float(step) if when is None else when
        if name_column and value_column:
            values = {str(row.get(name_column)): row.get(value_column)}
        else:
            values = {k: v for k, v in row.items() if k not in (step_column, time_column)}
        if step in by_step:
            by_step[step][1].update(values)
        else:
            by_step[step] = (when, dict(values))
    for step in sorted(by_step):
        when, values = by_step[step]
        yield step, when, values
