import json
import re
import subprocess
import sys
import os
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent / "scripts"

__all__ = ["BACKEND", "SCRIPTS", "latest_checkpoint", "read_trace", "resume_from", "run_training"]

# RAVEX_INTEGRATION_BACKEND=moonclip runs every scenario through Moonclip, the
# backend that ships (GPU-213); torch_save, the default, needs nothing but torch.
BACKEND = os.environ.get("RAVEX_INTEGRATION_BACKEND", "torch_save")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="needs SIGKILL and torchrun; run in the container"
)


@pytest.fixture(scope="session", autouse=True)
def ravex_is_installed():
    """A real install of ravex.

    This replaced a check that the ``.pth`` autoloader was in site-packages,
    which was the thing that made these tests mean anything until GPU-108. What
    makes them mean something now is narrower and easier to state: the training
    scripts import ravex and are decorated, so the suite needs an importable
    ravex. From GPU-105 to GPU-267 it also needed one whose compiled core had
    built; Ravex is Python only again.
    """
    try:
        import ravex  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"ravex is not importable in this interpreter: {exc}")


@pytest.fixture
def workspace(tmp_path):
    """A directory that behaves like a user's project.

    The ravex.yaml here is the only thing that turns Ravex on for processes
    started with this as their working directory.
    """

    def make(name, **overrides):
        directory = tmp_path / name
        directory.mkdir()
        settings = {
            "checkpoint_every": 4,
            "max_steps": 40,
            "backend": BACKEND,
            "keep_last": 5,
        }
        settings.update(overrides)

        lines = [f"{key}: {_yaml(value)}" for key, value in settings.items()]
        (directory / "ravex.yaml").write_text(
            "\n".join(lines)
            + textwrap.dedent(
                """
                storage:
                  type: local
                  path: ./checkpoints
                log_file: ./ravex.log
                log_level: INFO
                """
            ),
            encoding="utf-8",
        )
        return directory

    return make


def _yaml(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def run_training(directory, trace_name="trace.jsonl", **script_args):
    """Run the training script as a real, separate process."""
    command = [sys.executable, str(SCRIPTS / "train_vanilla.py"), "--trace", trace_name]
    for key, value in script_args.items():
        command += [f"--{key.replace('_', '-')}", str(value)]

    return subprocess.run(
        command,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=600,
    )


def latest_checkpoint(directory, rank=None):
    """The newest checkpoint a run in ``directory`` wrote, or None.

    Read through Ravex, the way ``ravex export`` reads a store, so the same
    assertions hold for every backend: globbing for ``step_*.pt`` only ever
    found what torch_save writes. ``rank`` reads one rank's own store of a
    per-rank sharded run.
    """
    from ravex._interop.export import CannotConvert, load_store

    path = Path(directory) / "checkpoints"
    if rank is not None:
        path = path / f"rank_{rank}"
    try:
        return load_store(str(path), backend=BACKEND)
    except CannotConvert:
        return None


def resume_from(directory):
    """The first step the restarted run in ``directory`` trained.

    Read from what Ravex logged, because it depends on the backend: torch_save
    resumes at the last ``checkpoint_every`` boundary before the kill (16, so
    step 17), Moonclip keeps a delta every step and resumes nearer the kill.
    """
    found = re.findall(r"Resumed at step (\d+)", (Path(directory) / "ravex.log").read_text())
    assert found, "the restarted run did not resume"
    return int(found[-1]) + 1


def read_trace(directory, name="trace.jsonl"):
    path = Path(directory) / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
