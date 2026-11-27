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


@pytest.mark.parametrize("cuda", [True, False])
def test_activation_holds_it_only_with_cuda(monkeypatch, storage, cuda):
    """The first thing a run does that touches memory, before any save. Not
    on a machine without CUDA, where activations are malloc's and holding
    the threshold cost 8 DDP ranks on the CPU 40% of their speed."""
    import torch

    import ravex

    calls = []
    monkeypatch.setattr(_memory, "hold_the_threshold", lambda: calls.append(1) or True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

    @ravex.train_loop(backend="torch_save", checkpoint_every=10**6)
    def train():
        pass

    train()
    assert calls == ([1] if cuda else [])
