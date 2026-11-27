"""GPU-540: state the script declares with ``ravex.track(state=)`` survives a resume.

An EMA that keeps its shadow weights in a dict, the best metric so far and an
early-stopping counter are not torch objects, so Ravex cannot notice them on
its own; before this a resumed run started the average over from the loaded
weights and forgot its best. The scenario is ``test_e2e_vanilla``'s: the same
script run straight through, then killed and run again, must end with the
same EMA and the same counters.
"""

import importlib.util

import pytest
import torch
import torch.nn as nn

import ravex

HAVE_MOONCLIP = importlib.util.find_spec("moonclip") is not None

TOTAL_STEPS = 24
CRASH_AT = 14


class Killed(RuntimeError):
    pass


class Counter:
    """A loop value kept on an object of the script's own, the other shape
    ``track(state=)`` takes."""

    def __init__(self):
        self.seen = 0

    def state_dict(self):
        return {"seen": self.seen}

    def load_state_dict(self, state):
        self.seen = state["seen"]


def train(die_at=None):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}
    stats = {"best": float("inf"), "since_best": 0}
    counter = Counter()
    ravex.track(state=shadow)
    ravex.track(state=stats)
    ravex.track(state=counter)

    data = torch.randn(TOTAL_STEPS, 8, 4)
    try:
        while ravex.step() < TOTAL_STEPS:
            x = data[ravex.step()]
            loss = ((model(x) - x.sum(dim=1, keepdim=True)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, value in model.state_dict().items():
                    shadow[name].mul_(0.9).add_(value, alpha=0.1)
            if loss.item() < stats["best"]:
                stats["best"], stats["since_best"] = loss.item(), 0
            else:
                stats["since_best"] += 1
            counter.seen += 1

            if die_at is not None and ravex.step() >= die_at:
                raise Killed
    except Killed:
        pass
    return shadow, stats, counter


@pytest.mark.parametrize(
    "backend",
    [
        "torch_save",
        pytest.param(
            "moonclip",
            marks=pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed"),
        ),
    ],
)
def test_tracked_state_comes_back_on_resume(tmp_path, monkeypatch, backend):
    options = dict(backend=backend, checkpoint_every=4, max_steps=TOTAL_STEPS)

    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "reference"))
    ravex._activate(**options)
    shadow, stats, counter = train()
    ravex.deactivate()

    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path / "interrupted"))
    ravex._activate(checkpoint_on_exit=False, **options)
    train(die_at=CRASH_AT)
    ravex.deactivate()

    ravex._activate(**options)
    resumed_shadow, resumed_stats, resumed_counter = train()
    ravex.deactivate()

    for name, value in shadow.items():
        assert torch.equal(resumed_shadow[name], value), name
    assert resumed_stats == stats
    assert resumed_counter.seen == counter.seen == TOTAL_STEPS


def test_state_that_cannot_be_saved_is_refused_when_tracked(tmp_path, monkeypatch):
    monkeypatch.setenv("RAVEX_STORAGE_PATH", str(tmp_path))
    ravex._activate(backend="torch_save", checkpoint_every=4, max_steps=4)
    try:
        with pytest.raises(TypeError, match="state_dict"):
            ravex.track(state=0.5)
    finally:
        ravex.deactivate()
