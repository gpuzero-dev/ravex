"""GPU-538: a resume reads a checkpoint without running code from it.

Whoever can write to the checkpoint directory or bucket must not be able to
run code in the process that resumes. These pin that for the ``torch_save``
backend and for the files of other tools Ravex resumes from, which all go
through ``ravex._safe_load.load``; Moonclip's own reader has the same test.
"""

import random

import pytest
import torch

from ravex._backends import TorchSaveBackend
from ravex._safe_load import UnsafeCheckpoint, load
from test_backends import make_config, sample_state


class _Planted:
    """What a writer of the store could plant: unpickling it calls ``open``
    and creates a file."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_a_planted_object_in_a_torch_save_file_never_runs(tmp_path):
    marker = tmp_path / "ran"
    path = tmp_path / "evil.pt"
    torch.save({"weight": torch.ones(2), "extra": _Planted(str(marker))}, path)

    with pytest.raises(UnsafeCheckpoint, match="would run code"):
        load(path)
    assert not marker.exists()


def test_the_torch_save_backend_refuses_a_planted_checkpoint(tmp_path):
    marker = tmp_path / "ran"
    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    state = sample_state(3)
    state["planted"] = _Planted(str(marker))
    # Written as an attacker with the bucket would: straight to the file.
    torch.save(state, f"{backend.directory}/step_{3:012d}.pt")

    with pytest.raises(UnsafeCheckpoint):
        backend.load_latest()
    with pytest.raises(UnsafeCheckpoint):
        backend.load_step(3)
    assert not marker.exists()


def test_what_a_resume_needs_still_round_trips(tmp_path):
    """Optimizer, LambdaLR and OneCycleLR, a GradScaler, and Python's and
    NumPy's RNG state, through the backend and back."""
    np = pytest.importorskip("numpy")

    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lam = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)
    one = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=10)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    model(torch.randn(2, 8)).sum().backward()
    optimizer.step()
    lam.step()
    one.step()

    state = sample_state(5)
    state["optimizers"] = {"opt": optimizer.state_dict()}
    state["schedulers"] = {"lambda": lam.state_dict(), "onecycle": one.state_dict()}
    state["scalers"] = {"scaler": scaler.state_dict()}
    state["rng"] = {"python": random.getstate(), "numpy": np.random.get_state()}

    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    backend.save(5, state, {"step": "5"})
    backend.flush()
    loaded = backend.load_latest()

    assert loaded["schedulers"]["lambda"]["last_epoch"] == lam.state_dict()["last_epoch"]
    assert loaded["schedulers"]["onecycle"]["total_steps"] == 10
    assert loaded["scalers"]["scaler"]["scale"] == scaler.state_dict()["scale"]
    assert loaded["rng"]["python"] == random.getstate()
    assert np.array_equal(loaded["rng"]["numpy"][1], np.random.get_state()[1])
    restored = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restored.load_state_dict(loaded["optimizers"]["opt"])
    for name, value in optimizer.state_dict()["state"][0].items():
        assert torch.equal(restored.state_dict()["state"][0][name], value), name
