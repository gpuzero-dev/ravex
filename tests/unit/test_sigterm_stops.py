"""SIGTERM stops a run that the coordinated emergency path does not cover (GPU-294).

Until this the handler raised a flag that only `_check_emergency_signal` read,
and that runs only for a sharded model across several machines. One process,
DDP, a sharded model on one machine: SIGTERM was caught and then ignored, the
run trained on to its end and exited 0, and `scancel`, `docker stop` and
`timeout` no longer stopped it. The tests here run in this interpreter, with
the process's own death replaced by a record of it; the one that sends a real
SIGTERM to a real child and reads its exit status is in
``tests/distributed/test_sigterm_process.py``.
"""

import os
import signal

import pytest
import torch

import ravex
import ravex.runs
from ravex._backends import TorchSaveBackend
from ravex._config import RavexConfig
from ravex._runtime import Preempted, RavexRuntime

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Ravex does not install a SIGTERM handler on Windows"
)


@pytest.fixture
def deaths(monkeypatch):
    """What `train_loop` does last after a stop, recorded instead of done."""
    calls = []
    monkeypatch.setattr(ravex, "_die_of_sigterm", lambda: calls.append(True))
    return calls


def backend_at(storage):
    config = RavexConfig()
    config.storage.path = str(storage)
    config.backend = "torch_save"
    config._normalize()
    return TorchSaveBackend(config)


def run(steps, signal_at, boundary=True, catch_everything=False):
    """Train up to `steps`, sending this process SIGTERM after `signal_at`.

    Returns the steps the loop got through.
    """
    done = []

    @ravex.train_loop(backend="torch_save", checkpoint_every=1000)
    def train():
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ravex.track(model=model, optimizer=optimizer)
        for _ in range(steps):
            try:
                if boundary:
                    ravex.batch_boundary()
                model(torch.randn(2, 4)).sum().backward()
                optimizer.step()
                optimizer.zero_grad()
            except Exception:
                # A script that guards its step this way must still stop:
                # the stop is not an `Exception`.
                if not catch_everything:
                    raise
            done.append(ravex.step())
            if ravex.step() == signal_at:
                os.kill(os.getpid(), signal.SIGTERM)

    train()
    return done


def test_the_run_stops_at_the_next_iteration_with_a_checkpoint_of_that_step(
    storage, deaths
):
    done = run(steps=50, signal_at=5)

    assert done == [1, 2, 3, 4, 5], "the run trained on after SIGTERM"
    assert backend_at(storage).load_step(5) is not None, "no checkpoint of the step it stopped at"
    status = ravex.runs.describe(str(storage))["status"]
    assert status["state"] == "interrupted" and status["step"] == 5
    assert deaths == [True], "the process has to die of the signal it was sent"


def test_without_batch_boundary_the_next_forward_is_where_it_stops(storage, deaths):
    # No dataloader and no `batch_boundary()`: the first forward after the
    # step is the iteration's top (GPU-276), and the stop happens there,
    # before that iteration trains on anything.
    done = run(steps=50, signal_at=5, boundary=False)

    assert done == [1, 2, 3, 4, 5]
    status = ravex.runs.describe(str(storage))["status"]
    assert status["state"] == "interrupted" and status["step"] == 5
    assert backend_at(storage).load_step(5) is not None
    assert deaths == [True]


def test_with_no_iteration_top_at_all_it_stops_at_the_next_step(storage, deaths):
    # Nothing delivers the top: not a dataloader, not the script, not a
    # forward. The stop then has no better place than the step after the
    # signal, the same hatch the checkpoint uses.
    seen = []

    @ravex.train_loop(backend="torch_save", checkpoint_every=1000)
    def train():
        torch.manual_seed(0)
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        ravex.track(optimizer=optimizer)
        for _ in range(50):
            (parameter * torch.randn(2)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            seen.append(ravex.step())
            if ravex.step() == 5:
                os.kill(os.getpid(), signal.SIGTERM)

    train()

    assert seen == [1, 2, 3, 4, 5]
    status = ravex.runs.describe(str(storage))["status"]
    assert status["state"] == "interrupted" and status["step"] == 6
    assert deaths == [True]


def test_a_script_that_catches_exception_around_its_step_still_stops(storage, deaths):
    done = run(steps=50, signal_at=5, boundary=False, catch_everything=True)

    assert len(done) == 5
    assert deaths == [True]


def test_a_run_that_finishes_gives_sigterm_back(storage, deaths):
    seen = []

    for _ in range(2):

        @ravex.train_loop(backend="torch_save", checkpoint_every=1000)
        def train():
            seen.append(signal.getsignal(signal.SIGTERM))

        train()

    assert seen[0] is not signal.SIG_DFL and seen[1] is not signal.SIG_DFL, (
        "the second run in the same process found SIGTERM still claimed by "
        "the first and installed nothing"
    )
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, (
        "after the run SIGTERM has to stop the process again"
    )
    assert deaths == []


def test_a_handler_the_script_installed_is_left_alone(storage):
    def mine(signum, frame):
        pass

    @ravex.train_loop(backend="torch_save", checkpoint_every=1000)
    def train():
        signal.signal(signal.SIGTERM, mine)

    try:
        train()
        assert signal.getsignal(signal.SIGTERM) is mine
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_the_coordinated_path_keeps_the_flag_for_itself():
    # A sharded model across machines: `_check_emergency_signal` reads the
    # flag at a step every rank agreed on, and a rank stopping on its own
    # would leave the others in the collective save.
    runtime = RavexRuntime.__new__(RavexRuntime)
    runtime._emergency_requested = True
    runtime._emergency_active = True

    runtime._stop_if_preempted()

    runtime._emergency_active = False
    runtime.registry = type("R", (), {"step_count": 7})()
    with pytest.raises(Preempted):
        runtime._stop_if_preempted()
