"""SIGTERM to a real training process: it stops, saves and exits 143 (GPU-294).

The unit half, in ``tests/unit/test_sigterm_stops.py``, replaces the process's
death with a record of it. This is the part only a child process can show:
the exit status a scheduler reads, how long after the signal it comes, and a
second run of the same command picking up at the step the first one stopped.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

import ravex.runs

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Ravex does not install a SIGTERM handler on Windows"
)

SCRIPT = r"""
import time, torch, ravex

@ravex.train_loop(backend="torch_save", checkpoint_every=1000)
def train():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ravex.track(model=model, optimizer=optimizer)
    ravex.batch_boundary()
    print("start", ravex.step(), flush=True)
    while ravex.step() < 2000:
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        print("step", ravex.step(), flush=True)
        time.sleep(0.02)
        ravex.batch_boundary()

train()
print("returned", flush=True)
"""


def start(storage):
    env = dict(os.environ, RAVEX_STORAGE_PATH=str(storage))
    return subprocess.Popen(
        [sys.executable, "-c", SCRIPT], stdout=subprocess.PIPE, text=True, env=env
    )


def test_sigterm_ends_the_process_with_143_and_the_next_run_resumes(tmp_path):
    storage = tmp_path / "checkpoints"
    process = start(storage)
    try:
        for line in process.stdout:
            if line.split() == ["step", "10"]:
                break
        else:
            raise AssertionError("the run never reached step 10")
        sent = time.monotonic()
        process.send_signal(signal.SIGTERM)
        rest = process.stdout.read()
        code = process.wait(timeout=60)
        took = time.monotonic() - sent
    finally:
        if process.poll() is None:
            process.kill()

    assert code == -signal.SIGTERM, "exit status %s, output after the signal: %r" % (code, rest[-300:])
    assert "returned" not in rest, "the training function ran to its end after SIGTERM"
    assert took < 30, "the process took %.1fs to stop" % took

    status = ravex.runs.describe(str(storage))["status"]
    assert status["state"] == "interrupted"
    stopped = status["step"]
    # The signal can land while the child is between printing step 10 and
    # the next iteration's top, so a step or two later is the same answer.
    assert 10 <= stopped <= 13

    again = start(storage)
    try:
        first = again.stdout.readline().split()
    finally:
        again.kill()
        again.wait(timeout=60)
    assert first == ["start", str(stopped)], "the second run did not resume where the first stopped"
