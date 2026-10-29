"""The README's quickstart and the examples, run the way a reader runs them (GPU-295).

Each is a script started as its own process from an empty directory, stopped
part way with Ctrl-C and started again, which is what the README tells a new
user to do. Before this nothing ran them: the README's snippet raised
`NameError` on names it never defined, and `examples/train_mlp.py` relied on a
`ravex.yaml` autoload removed in 0.0.5, so Ravex never switched on and the
"resume" it promised was a fresh start. A change that breaks either now breaks
here first.
"""

import os
import re
import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="stopped with SIGINT, which Windows does not deliver this way"
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(ROOT, "examples")


def quickstart() -> str:
    """The first Python block of the README's "Use" section, as printed."""
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as handle:
        readme = handle.read()
    use = readme.split("## Use", 1)[1]
    match = re.search(r"```python\n(.*?)```", use, re.S)
    assert match, "the README's Use section has no Python block"
    return match.group(1)


def stop_at(command, cwd, needle, env=None):
    """Run `command` until it prints a line holding `needle`, then Ctrl-C it."""
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env
    )
    seen = []
    try:
        for line in process.stdout:
            seen.append(line.strip())
            if needle in line:
                process.send_signal(signal.SIGINT)
                break
        else:
            raise AssertionError("never printed %r; got %r" % (needle, seen[-5:]))
        process.stdout.read()
        process.wait(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()
    return seen


def run(command, cwd, env=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()


def test_the_readme_quickstart_runs_and_resumes(tmp_path):
    (tmp_path / "train.py").write_text(quickstart(), encoding="utf-8")
    command = [sys.executable, "train.py"]

    first = stop_at(command, tmp_path, "step 600 ")
    second = run(command, tmp_path)

    assert first[0].startswith("step 200 ")
    step = int(second[0].split()[1])
    assert step > 600, "the second run started over: %r" % second[:2]
    assert second[-1].startswith("step 2000 "), "max_steps did not end it at 2000: %r" % second[-2:]


def test_train_mlp_resumes_with_the_losses_of_a_run_never_stopped(tmp_path):
    script = os.path.join(EXAMPLES, "train_mlp.py")
    whole_dir, stopped_dir = tmp_path / "whole", tmp_path / "stopped"
    whole_dir.mkdir()
    stopped_dir.mkdir()
    # torch_save, so the comparison holds whether or not Moonclip is
    # installed: the loss line is what is compared, not the store.
    env = dict(os.environ, RAVEX_BACKEND="torch_save")

    whole = run([sys.executable, script], whole_dir, env)
    stop_at([sys.executable, script], stopped_dir, "step   300 ", env)
    resumed = run([sys.executable, script], stopped_dir, env)

    assert whole[0] == "starting at step 0"
    assert whole[-1] == "done at step 1000"
    start = int(resumed[0].rsplit(" ", 1)[1])
    assert start >= 300, "the second run started over: %r" % resumed[0]
    assert resumed[-1] == "done at step 1000"

    def losses(lines):
        return [line for line in lines if line.startswith("epoch")]

    after = [line for line in losses(whole) if int(line.split()[3]) > start]
    assert after and after == losses(resumed), "the resumed run's losses differ from the uninterrupted run's"


def test_spiral_runs(tmp_path):
    env = dict(os.environ, RAVEX_STORAGE_PATH=str(tmp_path / "runs" / "mine"))
    lines = run([sys.executable, os.path.join(EXAMPLES, "spiral.py"), "--steps", "200"], tmp_path, env)
    assert lines[-1].startswith("step 200 ")
