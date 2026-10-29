# Ravex

Checkpoint and resume for PyTorch training, from one decorator on the function
that trains. Inside that function nothing changes: Ravex notices the model, the
optimizer, the LR scheduler, the AMP scaler and the dataloader as they are
built, counts every `optimizer.step()`, and writes a checkpoint on the cadence
you configured. If the process dies, running the same command again picks up
where it stopped.

## Install

```bash
pip install ravex
```

Ravex is pure Python and needs Python 3.11 or later. It does not depend on
torch, so that it attaches to whatever build you already have (CUDA, ROCm, CPU)
instead of pulling another one in.

On Python 3.10 or older, `pip` may not refuse: the first releases, up to 0.0.4,
said they ran on 3.9, and pip installs one of those instead, which has none of
what this page describes and fails at `ravex.train_loop`. Check
`python --version` first; `pip install "ravex>=0.10"` turns that quiet
downgrade into an error that names the Python version.

Its default checkpoint engine is [Moonclip](https://github.com/gpuzero-dev/moonclip),
installed with `pip install "ravex[moonclip]"`. Moonclip publishes wheels only
for Linux x86_64; elsewhere it has to be built from source, and where it is not
installed Ravex falls back to `torch_save` on its own.

## Use

This runs as it is, on a CPU, in a few seconds. Save it as `train.py`:

```python
import torch
import ravex

@ravex.train_loop(max_steps=2000)
def train():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4096, 16)
    data = torch.utils.data.TensorDataset(x, x.sum(dim=1, keepdim=True))
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    for epoch in range(100):
        for x, y in loader:
            loss = torch.nn.functional.mse_loss(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ravex.step() % 200 == 0:
                print(f"step {ravex.step()}  loss {loss.item():.4f}")

train()
```

```bash
python train.py      # press Ctrl-C after a few lines
python train.py      # the first line is past where you stopped, not step 200
```

The checkpoints are in `./checkpoints`; delete it to start over. Ravex says
two things on stderr, whatever the log level: where the first checkpoint went,
and, on a restart, the step it came back at:

```text
ravex: resumed from step 1400 in ./checkpoints; it stops at step 2000 (max_steps)
```

`announce=False` silences them; nothing is sent anywhere.

**Set `max_steps`.** It is the length of the whole training across restarts,
and it is what makes a resumed run end where the uninterrupted one would have:
without it, the resumed script runs its own `for epoch in range(100)` from the
top again, from the step it came back at, and trains more than the run that
never stopped. For a longer example, see
[`examples/train_mlp.py`](examples/train_mlp.py).

A resumed run continues rather than restarting from a good place, because
Ravex restores everything around the weights too: optimizer state, LR
scheduler, AMP loss scale, the torch, CUDA, Python and NumPy RNGs, and the
position in the dataset. The test suite holds it to the strong form of that: a
run killed at step 20 and resumed produces the same losses, step by step, and
the same final weights as the run that was never interrupted.

On SIGTERM (a spot preemption, `scancel`, `docker stop`) the training stops at
the next iteration, writes a final checkpoint and exits with status 143; run
the same command again and it resumes there. That is on by default, and
`preemption_handler=False` turns it off. Any other keyword argument to `train_loop` is a
configuration override, and an unknown name raises `TypeError` rather than
being ignored.

The decorator is the only way in: nothing is ever checkpointed by code that
does not say so. Importing Ravex does nothing until a decorated function is
called, and its patches on PyTorch come off when that function returns or
raises.

When "noticed on its own" is a guess, such as two models of which only one
trains, or an optimizer built before the function was entered, name it with
`ravex.track(model, optimizer)`. The one moment a checkpoint is consistent,
the top of an iteration, Ravex takes from the `DataLoader` iterator or, in a
loop without one, from the first forward of the trained model after each
step; `ravex.batch_boundary()` hands it over by hand where neither is seen.

## Configuration

`ravex.yaml`, at or above the working directory:

```yaml
checkpoint_every: 500        # optimizer steps between checkpoints (default 100); see below for Moonclip
backend: moonclip            # moonclip | torch_save
storage:
  type: local                # local | s3 | r2
  path: ./checkpoints
keep_last: 5
max_steps: null              # stop at this step however many times the process restarted
sharded_checkpoints: gather  # gather | per_rank, for FSDP
```

Every option is also read from a `RAVEX_*` environment variable, which wins
over the file, and the keyword arguments to `train_loop` win over both. A
resume is the default (`resume: true`); `RAVEX_ENABLED=0` turns Ravex off for
one run without touching the code.

**How often it saves.** With `torch_save`, a checkpoint every
`checkpoint_every` steps, and `keep_last` of them are kept. With Moonclip, the
default, Ravex saves a small delta against the last full snapshot at *every*
step, so a crash costs at most one step; `checkpoint_every` is then how often
the store is synced to a bucket and the window it keeps (the last that many
steps and the full they stand on), and `keep_last` is not used. A run of 147
steps holds about a hundred snapshots: that is the design, not a leak.
`delta: false` goes back to one full save every `checkpoint_every`
([docs](docs/configuration.md#a-delta-at-every-step)).

**Precision.** `save_dtype` (Moonclip only) stores parts of the checkpoint at
a lower precision to make it smaller. It is lossy: a run resumed from a
`bf16` optimizer state is close to, not identical with, the run that never
stopped. Leave it unset when a resume must be exact
([docs](docs/configuration.md#precision-per-component)).

**Storage.** Ravex decides when to save and what, and puts it back into the
live objects; Moonclip decides how the bytes are written and where, bucket
included. So `storage` says where the bucket is, and the credentials are never
read by Ravex or from the config file: Moonclip reads them from
`MOONCLIP_S3_ACCESS_KEY` / `MOONCLIP_S3_SECRET_KEY`, the usual `AWS_*` pair, a
profile in `~/.aws/credentials`, or the machine's IAM role
([Moonclip's README](https://github.com/gpuzero-dev/moonclip#s3)). Restoring
unpickles part of a checkpoint, so give write access to the bucket only to
what you would let run code on your training machines.

**More than one GPU or machine.** DDP and FSDP resume on every rank. With FSDP,
`gather` writes one checkpoint that loads at any world size, bounded by what
rank 0 can hold; `per_rank` has each rank write its own shard, which is much
cheaper and resumes only at the same world size (`reshard_on_resume` handles a
different one after a restart). Across machines without a bucket or a shared
filesystem, `replicate_every` copies each rank's store to a peer on another
machine, so losing one machine costs at most that many checkpoints; with a
bucket, a rank that comes up with an empty disk pulls its store back. Ravex
also trains across slow links between distant nodes with an outer loop that
exchanges parameter differences every `outer_inner_steps` instead of gradients
every step.

The full option list, with the measurements behind the defaults:
[docs/configuration.md](docs/configuration.md). How the patches, the step
count and the resume work: [docs/how-it-works.md](docs/how-it-works.md).

## Limits

- Under HuggingFace `Trainer` and Lightning the state comes back exactly, but
  with shuffling and dropout on, the resumed run sees a different random draw,
  because the framework drives the dataloader and the RNG on its own schedule.
  Plain loops, DDP and FSDP replay exactly.
- An `IterableDataset` has no index sampler, so its position cannot be
  replayed; everything else is still restored.
- A sharded model gets no final checkpoint at exit, since collecting one is a
  collective and ranks stop being in lockstep at shutdown; the newest
  checkpoint is the last periodic one.
- A failed checkpoint is lost, logged, and tried again at the next one;
  a failure inside Ravex never stops the training (`fallback_on_error`).

What changed between versions is in the [CHANGELOG](CHANGELOG.md).

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit
```

`tests/unit` runs in one process on any OS. `tests/distributed` and
`integration/` start several processes, use `torchrun` or kill a process with
SIGKILL, and need Linux. The integration tests run in their own image, with
`--init` so that the SIGKILL they send is not a no-op:

```bash
docker build -f integration/Dockerfile -t ravex-integration .
docker run --rm --init ravex-integration
```

## Licence

Apache-2.0 — [GPU Zero](https://gpuzero.dev)
