"""The allocator's threshold held for the run (GPU-443): see ravex._memory."""

import sys

import pytest

from ravex import _memory


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(_memory, "_held", False)
    monkeypatch.delenv("MALLOC_MMAP_THRESHOLD_", raising=False)
    monkeypatch.delenv("MALLOC_TRIM_THRESHOLD_", raising=False)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc's mallopt")
def test_held_on_linux_once():
    assert _memory.hold_the_threshold() is True
    assert _memory.hold_the_threshold() is True


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="anything but Linux is left alone")
def test_left_alone_elsewhere():
    assert _memory.hold_the_threshold() is False


def test_a_threshold_the_person_set_is_theirs(monkeypatch):
    monkeypatch.setenv("MALLOC_MMAP_THRESHOLD_", "1048576")
    assert _memory.hold_the_threshold() is False


def test_activation_holds_it(monkeypatch, storage):
    """The first thing a run does that touches memory, before any save."""
    import ravex

    calls = []
    monkeypatch.setattr(_memory, "hold_the_threshold", lambda: calls.append(1) or True)

    @ravex.train_loop(backend="torch_save", checkpoint_every=10**6)
    def train():
        pass

    train()
    assert calls == [1]
