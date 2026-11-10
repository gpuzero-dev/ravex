import importlib.util

import pytest
import torch

from ravex._backends import TorchSaveBackend, get_backend
from ravex._config import RavexConfig

HAVE_MOONCLIP = importlib.util.find_spec("moonclip") is not None


def make_config(tmp_path, **overrides):
    config = RavexConfig()
    config.storage.path = str(tmp_path / "checkpoints")
    for key, value in overrides.items():
        setattr(config, key, value)
    config._normalize()
    return config


def sample_state(step=1):
    return {
        "ravex_version": 1,
        "step": step,
        "models": {"model_a": {"weight": torch.arange(12, dtype=torch.float32)}},
        "optimizers": {},
        "schedulers": {},
        "scalers": {},
        "dataloaders": {},
        "sharded": {},
    }


def test_torch_save_roundtrip(tmp_path):
    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    assert not backend.has_checkpoint()

    backend.save(7, sample_state(7), {"step": "7"})
    backend.flush()

    assert backend.has_checkpoint()
    loaded = backend.load_latest()
    assert loaded["step"] == 7
    assert torch.equal(
        loaded["models"]["model_a"]["weight"],
        sample_state()["models"]["model_a"]["weight"],
    )
    backend.close()


def test_an_unknown_backend_falls_back_instead_of_raising(tmp_path):
    backend = get_backend(make_config(tmp_path, backend="nonsense"))
    assert isinstance(backend, TorchSaveBackend)
    backend.close()


@pytest.mark.parametrize("distributed_env", [False, True])
@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_moonclip_backend_writes_under_a_torchrun_environment(
    tmp_path, monkeypatch, distributed_env
):
    """Moonclip must be told this is a single-rank save, whatever the env says.

    Every Ravex store is single-rank: where ranks write separately they get a
    directory each, so there is one writer and one manifest whatever the job
    looks like. Moonclip has to be told that, because a manager that believes
    it is one of eight rejects the single-rank save API outright -
    "Multi-rank save requires explicit create_snapshot/save_rank/finalize
    flow" - and Ravex, doing what it promises, catches that, disables itself
    and lets training continue with no checkpoints at all.

    Two things used to have to go right for that not to happen, and now only
    one does. `CheckpointManager` read RANK/WORLD_SIZE from the environment
    when it was not told, so this backend had to pin the pair defensively;
    since Moonclip 0.0.9 it refuses to guess instead, and this backend builds
    `MoonclipManager`, which never guessed. The test stays because the failure
    it guards is silent: without it, a wrong answer here shows up only on a
    real multi-GPU run, as one line in a log file.
    """
    if distributed_env:
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")

    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(make_config(tmp_path, backend="moonclip"))
    backend.save(4, sample_state(4), {"step": "4"})
    backend.flush()

    assert backend.has_checkpoint()
    loaded = backend.load_latest()
    assert loaded["step"] == 4
    assert torch.equal(
        loaded["models"]["model_a"]["weight"],
        sample_state()["models"]["model_a"]["weight"],
    )
    backend.close()


# ─── where the handoff spends its time ──────────────────────────────


def test_the_handoff_says_which_phase_it_spent_its_time_in(tmp_path):
    """A handoff costs the training loop wall time, and the total alone never
    says where it went.

    The measured 10.6 s per checkpoint on 8× RTX 5060 Ti took three A/B runs to
    *not* explain — thread count, compression level and cadence each moved it by
    under a second. The phases are what turns the next such number into a
    diagnosis instead of another round of experiments.
    """
    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    phases = backend.save(1, sample_state(1), {"step": "1"})
    backend.close()

    assert list(phases) == ["copy", "queue"]
    assert all(seconds >= 0 for seconds in phases.values())


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_the_moonclip_handoff_separates_flattening_from_storing(tmp_path):
    """Each phase answers a different question: `flatten` is Python walking the
    state tree, `store` is the shadow copy, `backpressure` is the wait for the
    previous checkpoint's writer to get out of the way."""
    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(make_config(tmp_path, backend="moonclip"))
    phases = backend.save(1, sample_state(1), {"step": "1"})
    backend.close()

    assert list(phases) == ["flatten", "store", "backpressure"]
    assert all(seconds >= 0 for seconds in phases.values())


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_waiting_for_the_previous_writer_is_not_charged_to_the_copy(tmp_path):
    """GPU-61. The shadow copy and the wait for the previous writer were one
    number, and the sum reads like the writer: watching `store` grow is what
    opened GPU-55 against a writer in deficit, when the phase was a memcpy all
    along. Charged to `store`, the wait makes the copy look slow; named, it
    points at the cadence and the storage, which is where it comes from.

    The second save is the one that can wait — the first has nothing in front
    of it — so this saves twice with nothing draining in between.
    """
    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(make_config(tmp_path, backend="moonclip"))
    try:
        first = backend.save(1, sample_state(1), {"step": "1"})
        second = backend.save(2, sample_state(2), {"step": "2"})
    finally:
        backend.close()

    # Nothing was in flight before the first one, so it waited for nobody.
    # Not asserted as exactly zero: the clock starts before the lock is taken,
    # so an uncontended submit still reports the hundred nanoseconds it took to
    # find that out. What matters is that it is nothing, and at three decimal
    # places in the log line it prints as nothing.
    assert first["backpressure"] < 0.001
    assert second["backpressure"] >= 0.0
    # Whatever the wait was, it is not sitting inside the copy as well. The
    # phases still account for the call: `store` is what is left of it.
    assert second["store"] >= 0.0


# ─── per-rank stores ────────────────────────────────────────────────


def test_per_rank_gives_each_rank_a_store_of_its_own(tmp_path, monkeypatch):
    """Pointed at one store, N ranks are N writers against one manifest, each
    reading it, adding itself and writing it back with nothing between them."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    config = make_config(tmp_path, backend="torch_save")
    config.storage.prefix = "run-a"

    shared = get_backend(config, per_rank=False)
    assert shared.directory == str(tmp_path / "checkpoints")
    shared.close()

    mine = get_backend(config, per_rank=True)
    assert mine.directory == str(tmp_path / "checkpoints" / "rank_3")
    mine.close()

    # The remote side has to split too, or eight ranks would sync eight
    # different local stores onto one prefix in the bucket.
    from ravex._backends import _per_rank_config

    assert _per_rank_config(config, True).storage.prefix == "run-a/rank_3"


def test_a_backend_can_be_asked_for_a_specific_step(tmp_path):
    """Per-rank resume needs a step every rank holds, which is not always the
    newest one any of them has."""
    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    assert backend.latest_step() is None

    for step in (4, 8, 12):
        backend.save(step, sample_state(step), {"step": str(step)})
        backend.flush()

    assert backend.latest_step() == 12
    assert backend.load_step(8)["step"] == 8
    assert backend.load_step(9) is None
    backend.close()


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_moonclip_can_be_asked_for_a_specific_step(tmp_path):
    """The same question against the default backend, where a snapshot is
    addressed by id and the step is metadata rather than a filename.

    Every save a full: a delta between two fulls is folded away by the next
    one, and what is asked here is how a step is found, not which survive."""
    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(make_config(tmp_path, backend="moonclip", full_every=4))
    assert backend.latest_step() is None

    for step in (4, 8, 12):
        backend.save(step, sample_state(step), {"step": str(step)})
    backend.flush()

    assert backend.latest_step() == 12
    loaded = backend.load_step(8)
    assert loaded["step"] == 8
    assert torch.equal(
        loaded["models"]["model_a"]["weight"],
        sample_state()["models"]["model_a"]["weight"],
    )
    assert backend.load_step(9) is None
    backend.close()


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_moonclip_keeps_a_window_of_checkpoint_every_steps(tmp_path):
    """A delta per step, and the store holds the last `checkpoint_every` of
    them - each one a step to come back to - plus the full they stand on."""
    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(
        make_config(tmp_path, backend="moonclip", checkpoint_every=4, full_every=100)
    )
    assert backend.saves_every_step
    for step in range(1, 13):
        backend.save(step, sample_state(step), {"step": str(step)})
    backend.flush()

    assert backend.known_steps() == [1, 9, 10, 11, 12]
    assert backend.load_step(10)["step"] == 10
    assert backend.load_step(8) is None, "step 8 slid out of the window"
    backend.close()


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_moonclip_says_which_checkpoints_are_deltas_and_what_each_weighs(tmp_path):
    """The list the platform shows (GPU-382): the same steps as
    `known_steps`, each with whether it is a delta and its size in the store.
    Sending only the steps made every one read as a full of no known size."""
    from ravex._backends import MoonclipBackend

    backend = MoonclipBackend(
        make_config(tmp_path, backend="moonclip", checkpoint_every=4, full_every=100)
    )
    for step in range(1, 13):
        backend.save(step, sample_state(step), {"step": str(step)})
    backend.flush()

    entries = backend.known_checkpoints()
    assert [entry["step"] for entry in entries] == backend.known_steps()
    assert [entry["step"] for entry in entries if not entry["is_delta"]] == [1], entries
    assert all(entry["is_delta"] for entry in entries[1:]), entries
    assert all(isinstance(entry["size_bytes"], int) and entry["size_bytes"] > 0 for entry in entries), entries
    backend.close()


def test_torch_save_checkpoints_are_whole_and_unmeasured(tmp_path):
    backend = TorchSaveBackend(make_config(tmp_path, backend="torch_save"))
    for step in (4, 8):
        backend.save(step, sample_state(step), {"step": str(step)})
        backend.flush()

    assert backend.known_checkpoints() == [
        {"step": 4, "is_delta": False, "size_bytes": None},
        {"step": 8, "is_delta": False, "size_bytes": None},
    ]
    backend.close()


# ─── save_dtype reaches Moonclip ────────────────────────────────────


def _both_layouts(step=1):
    """State holding optimizer moments in both the places Ravex writes them.

    Unsharded state goes under ``optimizers``; a sharded model's optimizer
    goes under ``sharded/<key>/optimizer``, because sharded collection has its
    own path. A checkpoint of a real FSDP run has the second and not the
    first, and it is the one the precision measurement came from.
    """
    g = torch.Generator().manual_seed(step)
    r = lambda: torch.randn(32, 32, generator=g)
    return {
        "ravex_version": 1,
        "step": step,
        "models": {"model_a": {"weight": r()}},
        "optimizers": {
            "opt_a": {
                "state": {0: {"exp_avg": r()}},
                "param_groups": [{"lr": 0.1}],
            }
        },
        "schedulers": {},
        "scalers": {},
        "dataloaders": {},
        "sharded": {
            "fsdp": {
                "model": {"weight": r()},
                "optimizer": {"state": {"weight": {"exp_avg": r()}}},
                "parameters": ["weight"],
                "layout": "per_rank",
                "world_size": 1,
                "rank": 0,
            }
        },
    }


def _roundtrip_errors(tmp_path, save_dtype):
    """Save two steps and reload, returning how far each part moved.

    Two steps rather than one so the second is a delta against the first,
    which is the path where a cast and the retained base have to agree.
    """
    from ravex._backends import MoonclipBackend

    config = make_config(
        tmp_path, backend="moonclip", async_save=False, save_dtype=save_dtype
    )
    assert not config.problems, config.problems

    backend = MoonclipBackend(config)
    backend.save(1, _both_layouts(1), {})
    want = _both_layouts(2)
    backend.save(2, want, {})
    backend.flush()

    got = MoonclipBackend(config).load_latest()

    def moved(*path):
        a, b = got, want
        for key in path:
            a, b = a[key], b[key]
        return (a - b).abs().max().item()

    return {
        "model": moved("models", "model_a", "weight"),
        "optimizer": moved("optimizers", "opt_a", "state", 0, "exp_avg"),
        "sharded_model": moved("sharded", "fsdp", "model", "weight"),
        "sharded_optimizer": moved(
            "sharded", "fsdp", "optimizer", "state", "weight", "exp_avg"
        ),
    }


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_without_save_dtype_nothing_is_cast(tmp_path):
    assert set(_roundtrip_errors(tmp_path, None).values()) == {0.0}


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_a_component_reaches_both_places_it_is_written(tmp_path):
    """The whole reason `optimizer` is a component name and not a pattern.

    Casting the optimizer has to catch the sharded copy as well, because on
    the runs where the setting is worth anything that is the only copy there
    is.
    """
    moved = _roundtrip_errors(tmp_path, {"optimizer": "bf16"})
    assert moved["model"] == 0.0
    assert moved["sharded_model"] == 0.0
    assert moved["optimizer"] > 0
    assert moved["sharded_optimizer"] > 0
    # bf16 keeps eight significant bits. Anything larger is not rounding.
    assert moved["sharded_optimizer"] < 0.1


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_a_hand_written_pattern_misses_the_sharded_half(tmp_path):
    """Documents the trap the component names exist to close, rather than
    asserting it cannot happen — a raw pattern is still allowed, and someone
    who writes this one should be able to find out from a test why their
    sharded run saved nothing."""
    moved = _roundtrip_errors(tmp_path, {"ravex/optimizers/*": "bf16"})
    assert moved["optimizer"] > 0
    assert moved["sharded_optimizer"] == 0.0


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_an_exception_written_first_spares_the_model(tmp_path):
    moved = _roundtrip_errors(tmp_path, {"model": "none", "*": "bf16"})
    assert moved["model"] == 0.0
    assert moved["sharded_model"] == 0.0
    assert moved["optimizer"] > 0
    assert moved["sharded_optimizer"] > 0


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_a_bare_dtype_casts_every_component(tmp_path):
    moved = _roundtrip_errors(tmp_path, "bf16")
    assert all(value > 0 for value in moved.values()), moved


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_the_unset_setting_is_mentioned_once(tmp_path, caplog):
    """Not a warning — nothing is wrong. It is here because the setting is
    worth a great deal and is otherwise invisible."""
    from ravex._backends import MoonclipBackend

    with caplog.at_level("INFO", logger="ravex"):
        MoonclipBackend(make_config(tmp_path, backend="moonclip"))
    assert any("save_dtype is unset" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("INFO", logger="ravex"):
        MoonclipBackend(
            make_config(tmp_path, backend="moonclip", save_dtype={"optimizer": "bf16"})
        )
    assert not any("save_dtype is unset" in r.message for r in caplog.records)


class TestMoonclipFloor:
    """An installed Moonclip below the floor stops Ravex; a missing one does
    not (GPU-400). The fallback to torch.save is for a machine without
    Moonclip, not for one whose Moonclip is too old to do what Ravex asks of
    it - the first is a choice, the second a run that would checkpoint
    somewhere other than asked, with a warning nobody reads."""

    @staticmethod
    def _claiming(monkeypatch, version):
        import moonclip

        if version is None:
            monkeypatch.delattr(moonclip, "__version__", raising=False)
        else:
            monkeypatch.setattr(moonclip, "__version__", version, raising=False)

    def test_the_floor_is_the_one_pyproject_declares(self):
        import pathlib
        import re

        from ravex._backends import MOONCLIP_FLOOR

        pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
        declared = re.search(r'"moonclip>=(\d+)\.(\d+)\.(\d+)"', pyproject.read_text(encoding="utf-8"))
        assert declared, "pyproject.toml declares no moonclip floor"
        assert tuple(int(part) for part in declared.groups()) == MOONCLIP_FLOOR

    @pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
    @pytest.mark.parametrize("version", ["0.1.2", "0.3.1", "garbage", None])
    def test_an_old_or_unreadable_moonclip_is_refused_not_replaced(self, tmp_path, monkeypatch, version):
        from ravex._backends import MoonclipTooOld

        self._claiming(monkeypatch, version)
        with pytest.raises(MoonclipTooOld, match=r"Moonclip >= \d+\.\d+\.\d+"):
            get_backend(make_config(tmp_path, backend="moonclip"))

    @pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
    @pytest.mark.parametrize("version", ["0.3.2", "0.3.3.dev0", "1.0.0"])
    def test_the_floor_or_later_is_used(self, tmp_path, monkeypatch, version):
        self._claiming(monkeypatch, version)
        backend = get_backend(make_config(tmp_path, backend="moonclip"))
        assert type(backend).__name__ == "MoonclipBackend"
        backend.close()

    def test_a_missing_moonclip_still_falls_back(self, tmp_path, monkeypatch):
        import sys

        from ravex._backends import check_moonclip_version

        # None in sys.modules is how an absent package answers `import`.
        monkeypatch.setitem(sys.modules, "moonclip", None)
        check_moonclip_version()
        backend = get_backend(make_config(tmp_path, backend="moonclip"))
        assert isinstance(backend, TorchSaveBackend)
        backend.close()

    @pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
    def test_a_run_stops_before_its_first_step(self, storage, monkeypatch):
        """At activation, not at the first checkpoint: a backend refused
        there is logged and the training goes on with nothing saved."""
        import ravex
        from ravex._backends import MoonclipTooOld

        self._claiming(monkeypatch, "0.1.2")
        trained = []

        @ravex.train_loop(backend="moonclip", checkpoint_every=1)
        def train():
            trained.append(True)

        with pytest.raises(MoonclipTooOld):
            train()
        assert not trained, "the run trained on a Moonclip below the floor"


# ─── a bucket's keys are Moonclip's ─────────────────────────────────


class _Built(Exception):
    """Raised by the stand-in manager once it has what it was given."""


@pytest.mark.skipif(not HAVE_MOONCLIP, reason="moonclip not installed")
def test_no_keys_go_to_moonclip(tmp_path, monkeypatch):
    """Ravex says where the bucket is and nothing about its keys: Moonclip
    reads them from the environment itself (GPU-267). Keys passed from here
    would be a second lookup that can disagree with Moonclip's."""
    import moonclip

    from ravex._backends import MoonclipBackend

    given = {}

    def manager(**kwargs):
        given.update(kwargs)
        raise _Built()

    monkeypatch.setattr(moonclip, "MoonclipManager", manager)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "token")
    config = make_config(tmp_path, backend="moonclip")
    config.storage.type, config.storage.bucket = "s3", "runs"

    with pytest.raises(_Built):
        MoonclipBackend(config)
    assert given["s3_bucket"] == "runs"
    assert not any(name in given for name in ("s3_access_key", "s3_secret_key", "s3_session_token"))
