"""The lines Ravex says on stderr whatever the log level (GPU-308)."""

from ravex._config import RavexConfig
from ravex._runtime import RavexRuntime


def _runtime(**options):
    config = RavexConfig()
    for name, value in options.items():
        setattr(config, name, value)
    return RavexRuntime(config)


def test_a_resume_says_the_step_and_where_it_stops(capsys):
    runtime = _runtime(max_steps=2000)
    runtime._announce_resume("resumed from step 1400")
    err = capsys.readouterr().err
    assert "ravex: resumed from step 1400 in" in err
    assert "it stops at step 2000 (max_steps)" in err


def test_without_max_steps_the_resume_says_the_loop_decides(capsys):
    """A restarted `for epoch in range(100)` trains a hundred more epochs from
    the step it came back at, which reads as a resume that did not work
    unless something said why."""
    runtime = _runtime(max_steps=None)
    runtime._announce_resume("resumed from step 640")
    err = capsys.readouterr().err
    assert "no max_steps is set" in err


def test_announce_false_says_nothing(capsys):
    runtime = _runtime(announce=False)
    runtime._announce_resume("resumed from step 10")
    runtime._announce("first checkpoint at step 100")
    assert capsys.readouterr().err == ""


def test_the_environment_turns_it_off(monkeypatch):
    monkeypatch.setenv("RAVEX_ANNOUNCE", "0")
    config = RavexConfig()
    config._apply_env()
    assert config.announce is False
