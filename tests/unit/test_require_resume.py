"""A resume that must resume, and checkpoints that reach the bucket as they
are written (2026-10-08).

A node the provider took back had written checkpoints to step 1500, and the
bucket held none of them: Moonclip synced every hundred saves, which at one
save every 250 steps meant only at close. The resume on a new node found
nothing, started from step 0 under the run's own name and overwrote its
``run.json``. Both halves are pinned here: the sync at every save, and a start
that must resume stopping instead of training from scratch.
"""

import pytest
import torch

import ravex
from ravex._config import RavexConfig
from ravex._resume import NothingToResume


def script(total, seen):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ravex.track(model=model, optimizer=optimizer)
    while ravex.step() < total:
        ravex.batch_boundary()
        model(torch.randn(2, 4)).sum().backward()
        # Ravex resumes inside the first optimizer step, so the step this
        # start began at is read after it.
        optimizer.step()
        seen.setdefault("first", ravex.step() - 1)
        optimizer.zero_grad()


class TestRequireResume:
    def test_an_empty_store_stops_the_run_before_any_step(self, storage):
        seen = {}
        loop = ravex.train_loop(backend="torch_save", checkpoint_every=2, require_resume=True)
        with pytest.raises(NothingToResume, match="must resume"):
            loop(lambda: script(4, seen))()
        assert seen == {}, "a run that must resume trained from scratch"
        # Nor did it give the empty store an identity: uploaded, a new
        # run.json would land over the one the bucket holds for the run.
        assert not (storage / "run.json").exists()
        assert not (storage / "status.json").exists()

    def test_a_store_with_a_checkpoint_carries_on(self, storage):
        ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: script(4, {}))()
        seen = {}
        ravex.train_loop(backend="torch_save", checkpoint_every=2, require_resume=True)(
            lambda: script(6, seen)
        )()
        assert seen["first"] == 4

    def test_the_environment_sets_it(self, storage, monkeypatch):
        monkeypatch.setenv("RAVEX_REQUIRE_RESUME", "1")
        with pytest.raises(NothingToResume):
            ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: script(4, {}))()

    def test_without_it_an_empty_store_starts_from_scratch(self, storage):
        seen = {}
        ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: script(2, seen))()
        assert seen["first"] == 0


class TestCheckpointsReachTheBucketAsTheyAreWritten:
    @staticmethod
    def manager_arguments(tmp_path, monkeypatch, **settings):
        """What the backend hands Moonclip for a store in a bucket."""
        moonclip = pytest.importorskip("moonclip")
        from ravex import _backends

        seen = {}

        class Recording:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        monkeypatch.setattr(moonclip, "MoonclipManager", Recording)
        monkeypatch.setattr(_backends, "moonclip", moonclip, raising=False)
        config = RavexConfig()
        config.backend = "moonclip"
        config.storage.path = str(tmp_path / "store")
        config.storage.type = "s3"
        config.storage.bucket = "runs"
        config.storage.endpoint = "http://127.0.0.1:1"
        for name, value in settings.items():
            setattr(config, name, value)
        config._normalize()

        backend = _backends.MoonclipBackend(config)
        return seen, backend

    def test_with_a_delta_per_step_the_bucket_gets_one_every_checkpoint(
        self, tmp_path, monkeypatch
    ):
        seen, backend = self.manager_arguments(
            tmp_path, monkeypatch, checkpoint_every=50, full_every=400
        )
        assert backend.saves_every_step
        assert seen.get("sync_every_n_saves") == 50, (
            "a save per step: the sync every checkpoint_every is that many saves"
        )
        assert seen.get("full_every_steps") == 400
        assert seen.get("max_deltas_per_full") == 400, (
            "Moonclip's default of ten would force a full every ten steps"
        )
        assert seen.get("max_total_snapshots") == 51, (
            "the window: the last checkpoint_every deltas and their full"
        )
        assert not seen.get("merge_stride"), (
            "the merger deletes deltas the sync may not have sent yet"
        )

    def test_with_a_full_per_save_every_save_is_synced(self, tmp_path, monkeypatch):
        seen, backend = self.manager_arguments(
            tmp_path, monkeypatch, checkpoint_every=50, delta=False
        )
        assert not backend.saves_every_step
        assert seen.get("sync_every_n_saves") == 1, (
            "Moonclip's default syncs every hundred saves: a node lost before "
            "then leaves no checkpoint in the bucket"
        )

    def test_with_a_full_per_save_keep_last_is_every_snapshot_kept(
        self, tmp_path, monkeypatch
    ):
        seen, _ = self.manager_arguments(tmp_path, monkeypatch, delta=False, keep_last=12)
        assert seen.get("max_total_snapshots") == 12
        assert seen.get("max_full_snapshots") == 12, (
            "every snapshot is a full one, and Moonclip's own cap on fulls is five"
        )


def loader_script(epochs, seen, warm_up=False):
    """The loop most scripts have: ``for batch in loader`` and nothing of
    Ravex's before it, so the resume and the checkpoints happen inside the
    dataloader's wrappers. Four steps an epoch."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 4)), batch_size=2)
    if warm_up:
        # A pass over the data before the optimizer exists, with the model
        # already known to Ravex.
        ravex.track(model=model)
        with torch.no_grad():
            for (batch,) in data:
                model(batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for _ in range(epochs):
        for (batch,) in data:
            model(batch).sum().backward()
            optimizer.step()
            seen.setdefault("first", ravex.step() - 1)
            optimizer.zero_grad()


class TestWithADataLoader:
    """GPU-458: the wrappers around the dataloader caught everything, so the
    errors that stop a run were a warning there, and the run trained from
    step 0 over its own history."""

    def test_an_empty_store_stops_the_run_before_any_step(self, storage):
        seen = {}
        loop = ravex.train_loop(backend="torch_save", checkpoint_every=2, require_resume=True)
        with pytest.raises(NothingToResume, match="must resume"):
            loop(lambda: loader_script(1, seen))()
        assert "first" not in seen, "a run that must resume trained from scratch"

    def test_a_store_with_a_checkpoint_carries_on(self, storage):
        ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: loader_script(1, {}))()
        seen = {}
        ravex.train_loop(backend="torch_save", checkpoint_every=2, require_resume=True)(
            lambda: loader_script(1, seen)
        )()
        assert seen["first"] == 4

    def test_a_pass_before_the_optimizer_exists_does_not_resume_the_model_alone(self, storage):
        ravex.train_loop(backend="torch_save", checkpoint_every=2)(lambda: loader_script(1, {}))()
        seen = {}
        resumed = {}

        def script():
            loader_script(1, seen, warm_up=True)

        import ravex._runtime as runtime_module

        original = runtime_module.RavexRuntime._try_resume

        def spy(self, *args, **kwargs):
            resumed["optimizers"] = len(self.registry.optimizers)
            return original(self, *args, **kwargs)

        runtime_module.RavexRuntime._try_resume = spy
        try:
            ravex.train_loop(backend="torch_save", checkpoint_every=2, require_resume=True)(script)()
        finally:
            runtime_module.RavexRuntime._try_resume = original
        assert resumed["optimizers"] == 1, "the resume ran before there was an optimizer to restore"
        assert seen["first"] == 4
