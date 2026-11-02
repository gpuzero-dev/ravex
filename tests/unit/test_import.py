"""Runs logged elsewhere, written as Ravex stores (GPU-314).

What an import has to get right is that the store it writes is one like any
other: :func:`ravex.metrics.read` draws it, and ``ravex ship`` sends it.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import ravex.metrics
from ravex import _cli
from ravex import _import as importing
from tests.unit.test_ship import FakeBackend


class FakeRun:
    name = "baseline-lr3e-4"
    config = {"lr": 3e-4, "_wandb": {"internal": True}}
    created_at = "2026-03-01T10:00:00Z"

    def scan_history(self):
        yield {"_step": 0, "_timestamp": 1772359200.0, "train/loss": 2.5, "lr": 3e-4}
        yield {"_step": 1, "_timestamp": 1772359201.0, "train/loss": 2.1, "media": {"_type": "image"}}
        yield {"_step": 2, "_timestamp": 1772359202.0, "train/loss": float("nan"), "eval/acc": "0.5"}


class FakeApi:
    def __init__(self):
        self.asked = None

    def run(self, path):
        self.asked = path
        return FakeRun()


def test_a_wandb_run_becomes_a_store_that_reads_back(tmp_path):
    api = FakeApi()
    about, records = importing.from_wandb("team/proj/abc123", api=api)
    store = str(tmp_path / "run")
    written = importing.write_store(store, records, name=about["name"], config=about["config"],
                                    created_at=about["created_at"], source=about["source"])

    assert api.asked == "team/proj/abc123"
    assert written["last_step"] == 2
    run = json.load(open(os.path.join(store, "run.json")))
    assert run["name"] == "baseline-lr3e-4"
    assert run["config"] == {"lr": 3e-4}, "W&B's own keys are not the run's configuration"
    assert run["imported_from"] == "wandb:team/proj/abc123"
    assert run["created_at"] == pytest.approx(1772359200.0)

    history = ravex.metrics.read(store)
    loss = history["scalars"]["train/loss"]
    assert loss["step"] == [0, 1, 2]
    assert loss["value"][:2] == [2.5, 2.1]
    assert loss["value"][2] != loss["value"][2], "a NaN comes back a NaN"
    assert history["scalars"]["eval/acc"]["value"] == [0.5]
    assert "media" not in history["scalars"], "only numbers come across"
    assert json.load(open(os.path.join(store, "status.json")))["state"] == "finished"


def test_a_wide_csv_and_a_long_one_give_the_same_series(tmp_path):
    wide = tmp_path / "wide.csv"
    wide.write_text("step,timestamp,loss,acc\n0,1772359200,1.0,0.1\n1,1772359201,0.5,0.2\n")
    # Neptune's shape: a row per value, the metric's path, milliseconds.
    long = tmp_path / "long.csv"
    long.write_text(
        "attribute_path,step,timestamp,value\n"
        "loss,0,1772359200000,1.0\nacc,0,1772359200000,0.1\n"
        "loss,1,1772359201000,0.5\nacc,1,1772359201000,0.2\n"
    )
    stores = []
    for source in (wide, long):
        store = str(tmp_path / ("store-" + source.stem))
        importing.write_store(store, importing.from_table(str(source)), name=source.stem, source=source.name)
        stores.append(ravex.metrics.read(store)["scalars"])
    assert stores[0] == stores[1]
    assert stores[0]["loss"]["value"] == [1.0, 0.5]
    assert stores[0]["acc"]["time"] == [1772359200.0, 1772359201.0]


def test_ravex_jsonl_records_import_too(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps({"step": 5, "time": 10.0, "values": {"loss": 0.3}}) + "\n"
        + json.dumps({"step": 6, "time": 11.0, "values": {"loss": 0.2}}) + "\n"
    )
    store = str(tmp_path / "run")
    importing.write_store(store, importing.from_table(str(source)), name="r", source="records.jsonl")
    assert ravex.metrics.read(store)["scalars"]["loss"]["step"] == [5, 6]


def test_an_import_never_writes_into_an_existing_run(tmp_path):
    source = tmp_path / "m.csv"
    source.write_text("step,loss\n0,1\n")
    store = str(tmp_path / "run")
    importing.write_store(store, importing.from_table(str(source)), name="a", source="m.csv")
    with pytest.raises(FileExistsError):
        importing.write_store(store, importing.from_table(str(source)), name="b", source="m.csv")


def test_a_table_without_a_step_is_refused(tmp_path):
    source = tmp_path / "m.csv"
    source.write_text("loss\n1\n")
    with pytest.raises(ValueError, match="no step column"):
        list(importing.from_table(str(source)))


# Rows as pyarrow reads them from neptune-exporter's Parquet (GPU-355): the
# step a decimal, the times aware datetimes, one value column per type.
NEPTUNE_START = datetime(2026, 2, 1, 9, 0, tzinfo=timezone.utc)


def neptune_rows(run_id="SAN-7"):
    def parameter(path, kind, value):
        return {"run_id": run_id, "attribute_path": path, "attribute_type": kind, kind + "_value": value, "step": None}

    def point(path, step, value):
        return {
            "run_id": run_id, "attribute_path": path, "attribute_type": "float_series",
            "step": Decimal(step).quantize(Decimal("1.000000")),
            "timestamp": NEPTUNE_START + timedelta(seconds=step), "float_value": value,
        }

    return [
        parameter("sys/name", "string", "shakespeare-small"),
        parameter("sys/creation_time", "datetime", NEPTUNE_START),
        parameter("parameters/lr", "float", 3e-4),
        parameter("parameters/batch", "int", 64),
        parameter("sys/tags", "string_set", ["baseline"]),
        {"run_id": run_id, "attribute_path": "logs/stdout", "attribute_type": "string_series", "step": Decimal(0)},
        point("train/loss", 0, 2.5), point("train/loss", 1, 2.1), point("train/loss", 2, 1.9),
        point("eval/acc", 2, 0.4),
    ]


def test_a_neptune_export_becomes_a_store_with_its_name_parameters_and_series(tmp_path):
    about, records = importing.from_neptune("export", rows=neptune_rows())
    assert about["name"] == "shakespeare-small"
    assert about["created_at"] == NEPTUNE_START.timestamp()
    assert about["config"] == {"parameters/lr": 3e-4, "parameters/batch": 64}
    assert about["source"] == "neptune:SAN-7"
    records = list(records)
    assert [step for step, _, _ in records] == [0, 1, 2]
    assert records[2] == (2, NEPTUNE_START.timestamp() + 2, {"train/loss": 1.9, "eval/acc": 0.4})

    store = str(tmp_path / "run")
    written = importing.write_store(store, iter(records), name=about["name"], config=about["config"],
                                    created_at=about["created_at"], source=about["source"])
    assert written["metrics"] == ["eval/acc", "train/loss"]
    assert ravex.metrics.read(store)["scalars"]["train/loss"]["step"] == [0, 1, 2]


def test_an_export_of_several_runs_asks_which(tmp_path):
    rows = neptune_rows("SAN-7") + neptune_rows("SAN-8")
    with pytest.raises(ValueError, match="SAN-7, SAN-8"):
        importing.from_neptune("export", rows=rows)
    about, records = importing.from_neptune("export", run="SAN-8", rows=rows)
    assert about["source"] == "neptune:SAN-8" and len(list(records)) == 3
    with pytest.raises(ValueError, match="no run SAN-9"):
        importing.from_neptune("export", run="SAN-9", rows=rows)


def test_fractional_neptune_steps_are_numbered_in_order_not_folded(capsys, tmp_path):
    # As neptune-exporter's own tests log them: 0.0, 0.1, 0.2..., where
    # rounding would fold every value into step 0.
    rows = [row for row in neptune_rows() if row["attribute_type"] != "float_series"]
    for k, (loss, acc) in enumerate([(2.5, None), (2.1, 0.3), (1.9, None), (1.7, 0.4)]):
        step = Decimal(k) / 10
        rows.append({"run_id": "SAN-7", "attribute_path": "train/loss", "attribute_type": "float_series",
                     "step": step, "timestamp": NEPTUNE_START + timedelta(seconds=k), "float_value": loss})
        if acc is not None:
            rows.append({"run_id": "SAN-7", "attribute_path": "eval/acc", "attribute_type": "float_series",
                         "step": step, "timestamp": NEPTUNE_START + timedelta(seconds=k), "float_value": acc})
    about, records = importing.from_neptune("export", rows=rows)
    assert about["renumbered"]
    assert [(step, values) for step, _, values in records] == [
        (0, {"train/loss": 2.5}),
        (1, {"train/loss": 2.1, "eval/acc": 0.3}),
        (2, {"train/loss": 1.9}),
        (3, {"train/loss": 1.7, "eval/acc": 0.4}),
    ]
    # Whole steps keep their numbers: gaps stay gaps.
    about, records = importing.from_neptune("export", rows=neptune_rows())
    assert not about["renumbered"]


def test_the_command_says_when_it_renumbered(tmp_path, capsys, monkeypatch):
    rows = neptune_rows()
    rows[-1]["step"] = Decimal("2.500000")
    monkeypatch.setattr(importing, "_parquet_rows", lambda path: rows)
    assert _cli.main(["import", "neptune", "export", "--storage", str(tmp_path / "run")]) == 0
    assert "fractional steps" in capsys.readouterr().err


def test_the_parquet_neptune_exporter_writes_is_read(tmp_path):
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    # neptune_exporter.model.SCHEMA, as the exporter writes it.
    schema = pa.schema([
        ("project_id", pa.string()), ("run_id", pa.string()), ("attribute_path", pa.string()),
        ("attribute_type", pa.string()), ("step", pa.decimal128(18, 6)),
        ("timestamp", pa.timestamp("ms", tz="UTC")), ("int_value", pa.int64()), ("float_value", pa.float64()),
        ("string_value", pa.string()), ("bool_value", pa.bool_()),
        ("datetime_value", pa.timestamp("ms", tz="UTC")), ("string_set_value", pa.list_(pa.string())),
    ])
    rows = [{field.name: row.get(field.name) for field in schema} for row in neptune_rows()]
    for row in rows:
        row["project_id"] = "workspace/project"
    directory = tmp_path / "data" / "workspace_project"
    directory.mkdir(parents=True)
    parquet.write_table(pa.Table.from_pylist(rows, schema=schema), directory / "SAN-7_part_0.parquet")

    store = str(tmp_path / "run")
    assert _cli.main(["import", "neptune", str(tmp_path / "data"), "--storage", store]) == 0
    assert json.load(open(os.path.join(store, "run.json")))["name"] == "shakespeare-small"
    assert ravex.metrics.read(store)["scalars"]["train/loss"]["step"] == [0, 1, 2]


def test_the_command_writes_the_store_and_ravex_ship_sends_it(tmp_path, capsys):
    source = tmp_path / "neptune.csv"
    source.write_text("step,loss\n0,1.0\n1,0.8\n2,0.6\n")
    store = str(tmp_path / "run")
    assert _cli.main(["import", "table", str(source), "--storage", store, "--name", "from-neptune"]) == 0
    assert "3 steps up to 2" in capsys.readouterr().out

    backend = FakeBackend()
    try:
        assert _cli.main(["ship", "--storage", store, "--endpoint", backend.url]) == 0
        (run_id, document), = backend.runs.items()
        assert document["name"] == "from-neptune"
        assert backend.points(run_id, "loss") == [0, 1, 2]
    finally:
        backend.close()
