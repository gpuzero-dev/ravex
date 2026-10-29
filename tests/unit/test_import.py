"""Runs logged elsewhere, written as Ravex stores (GPU-314).

What an import has to get right is that the store it writes is one like any
other: :func:`ravex.metrics.read` draws it, and ``ravex ship`` sends it.
"""

import json
import os

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
