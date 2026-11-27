# Ravex's outer loop against torchft's DiLoCo, on the same model

A benchmark that a third party can rerun: the same model, the same data, the
same inner loop and the same outer optimizer, trained across machines by
Ravex's outer loop in one arm and by torchft's `DiLoCo` in the other, with
the same faults injected into both at the same step.

Nothing here rents a machine. The harness is given machines that are already
up and reachable, starts the runs on them, injects the faults, copies the
results home as they are written, and turns them into a table. Renting and
destroying the machines is the operator's, by hand or with the provider's
own tools.

## What is compared

| | both arms |
|---|---|
| model | `model.py`'s `124m`, a GPT of GPT-2 small's shape: 124.5M parameters, embeddings tied; fp32 parameters, bf16 autocast. A round is 0.5 GB in fp32. Chosen over deepseek-1b on 2026-11-25: the bench measures the outer loop, and a small model makes the phases affordable |
| data | FineWeb-Edu `sample-10BT` in GPT-2's tokens, written once by `prepare.py`, read through a shuffled `DataLoader` of 1024-token windows. Split among the nodes by Ravex's rule (GPU-417): starting node `k` of `n` reads positions `k, k+n, ...` of each epoch's order, a joiner the whole data in an order of its own. Ravex applies it under the Ravex arm's loader; the bench's `Part` sampler applies it to torchft's |
| inner step | AdamW lr 6e-4, betas (0.9, 0.95), weight decay 0.1, 200 warmup steps, cosine to 0.1x on the node's local step, clip 1.0; 4 sequences x 1024 tokens x 4 micro-batches = 16,384 tokens per node-step |
| outer step | Nesterov SGD, lr 0.7, momentum 0.9 (DiLoCo's published setting), every H local steps; fp32 on the wire |
| eval | the same 32 fixed held-out batches, right after every round, when every node of the arm holds the same parameters |

The two arms import the same `common.py` and differ only in how a round
reaches the model:

- `train_ravex.py`: the loop under `@ravex.train_loop` with `outer_loop` on.
  Ravex closes a round at the top of an iteration, pulling every node's delta
  over its own sockets, and the round closes over whoever answered.
- `train_torchft.py`: the loop under torchft's `Manager` and `DiLoCo`, without
  TorchTitan, the path torchft's `train_diloco.py` takes for a plain module.
  One fragment, `sync_every` = H, Gloo between replica groups,
  `min_replica_size` 1, `init_sync` off since every node builds the model from
  the same seed.
- The checkpoint-restart reference (kind `ckpt_restart`): Ravex's outer loop
  with fixed membership, whose faults the controller handles the classical
  way - stop every node, wait for a replacement, start everything again from
  the last checkpoint.

Two more rows put the systems at equal guarantees (GPU-213), since one keeps
its training on disk and the other does not:

- `ravex-lean`: the Ravex arm with no checkpoint inside the run and its
  rounds' reports on a tmpfs (`configs/ravex_lean.yaml`, and the arm's
  `rounds_root`): the outer loop alone, as torchft's is. Mind the tmpfs's
  size, a round per node of the model in fp32: Docker's `/dev/shm` is 64 MB.
- `torchft-ckpt`: the torchft arm with a synchronous DCP checkpoint of the
  model and the optimizer every `checkpoint_every` steps (the arm's
  setting), the last two kept: what Ravex's primary arm has, a run that
  survives every node going at once.

Phase (b) adds two rows for the case Ravex exists for, nodes that are not
alike (GPU-545). They run one after the other, with `--only`, on four pods of
their own (`h-eu0` ... `h-us1`), two regions and **at least two GPU models**
among them, so that one node is clearly slower:

- `ravex-clock`: the primary arm with rounds closed on the clock,
  `outer_round_seconds` 60 and an H too large to be reached
  (`configs/ravex_clock.yaml`): every node sends what it did in the minute,
  and the slow one costs only its share of the steps.
- `torchft-hetero`: the torchft arm at H = 250 on the same pods, so its syncs
  wait for the slowest replica, as torchft does.

With a round on the clock the nodes' step counts differ, and the table's
committed steps for those rounds are the midpoint of the slowest and the
fastest node's (`train_ravex.py`), an estimate: read the tokens of the
`ravex-clock` row from each node's events, not from that column.

**The clock starts at the end of each arm's first step** (`make_table.py`).
torchft connects its `Manager` before the arm's `start` event, while Ravex
builds its outer loop - the rendezvous, the starting state - inside the first
optimizer step; counted from `start`, the table charged Ravex's coordination
and none of torchft's. Both arms are given their first step.

**Where the comparison is not like for like**, said here because a reader
will ask:

- torchft's tuned path is TorchTitan, which wires up Llama 3 only; the GPT
  here is a plain module, so torchft runs through its lower-level API.
- torchft's lighthouse is a single point of failure; Ravex's rendezvous fails
  over. The node killed in a fault is never the coordinator's host.
- Ravex's primary arm checkpoints, and Moonclip chooses saves between two
  checkpoints, all inside its tokens/s; torchft writes nothing persistent.
  The `ravex-lean` and `torchft-ckpt` rows above compare them without and
  with that.
- The two arms split the data by the same rule but not in the same order:
  Ravex seeds its order from the job, the bench's sampler from `--seed`.
- On one machine, as in the CPU tests, Ravex's nodes have no place
  (`RAVEX_OUTER_MEMBER` is set only when every starting node has an address
  of its own), so each reads the whole data in its own order; on rented
  machines each has its place and its part.

## Files

```
model.py           the GPT: 124m, and tiny for the CPU tests
prepare.py         FineWeb-Edu in GPT-2's tokens: train.bin, val.bin, meta.json
common.py          optimizer, schedule, the loader and its cut, eval and the loop both arms run; events.jsonl
train_ravex.py     the Ravex arm
train_torchft.py   the torchft arm
vram_check.py      gate a1: one process, the inner step alone, peak memory and tokens/s
controller.py      start a phase's arms on the hosts, inject its faults, mirror the events home
hosts.py           the hosts: local, or over SSH; start, kill (SIGKILL), read, fetch
make_table.py      results/<phase>/ -> table.md, table.json, and phase (a)'s go/no-go
configs/           Ravex settings per arm, the phases (phase_a.json, phase_b.json), hosts.example.json
remote/setup.sh    prepare one rented machine from two archives
Dockerfile.cpu     the Linux container the tests run in
tests/             the controller and the collector anywhere; whole phases on one CPU in the container
```

## Versions

Pinned by what a phase runs, and written into its results folder by the
operator alongside the table:

- ravex: this checkout, installed from it on every machine (`remote/setup.sh`),
  so the code that ran is the code at the commit the results name;
- the data: `prepare.py`'s `meta.json` and the files' hashes, which `setup.sh` prints;
- torchft 0.3.1 (`requirements.txt`), its Linux wheel, which needs torch 2.7
  or later;
- torch: the machine image's, at least 2.7; the CPU tests run 2.15.0;
- moonclip 0.4.1 or later.

## Running the tests

On a CPU, in the Linux container: the multi-process runs are judged on Linux
only, as every multi-process test of Ravex's.

```sh
docker build -t ravex-bench-cpu -f bench/vs_torchft/Dockerfile.cpu bench/vs_torchft
docker run --rm --init -e MOONCLIP_FSYNC=0 -v <ravex checkout>:/mnt/ravex:ro ravex-bench-cpu
```

`tests/test_runs.py` runs whole phases on one machine with the tiny GPT
and a 512-token vocabulary: two Ravex nodes and two torchft nodes for
three rounds each, then a join and a kill on both, then the
checkpoint-restart arm's stop and resume, each followed by
`make_table.py --checks`. It shows that the harness measures what it says -
rounds commit, every node of an arm holds the same parameters after each, the
shards differ, the faults land at their step - and nothing about either
system's speed. `test_schedule.py` and `test_make_table.py` need no process
and run anywhere.

## Running a phase

1. Rent the machines, with a private network between them that every node
   can reach every other on, on any port: torchft dials random ports and
   crosses no NAT. On RunPod that is Global Networking, chosen at creation;
   the address is the `10.x` on `podnet1` (`hostname -I`).
2. Copy the code and the data to each, and prepare it:

   ```sh
   git -C ravex archive -o ravex.tar HEAD
   tar -cf data.tar -C data .               # train.bin, val.bin, meta.json from prepare.py
   scp -P <port> ravex.tar data.tar root@<host>:/workspace/
   ssh -p <port> root@<host> "mkdir -p /workspace/ravex && tar -xf /workspace/ravex.tar -C /workspace/ravex && sh /workspace/ravex/bench/vs_torchft/remote/setup.sh"
   ```

   `setup.sh` prints the private address, the GPU, the versions and the
   data's hashes, which must match on every machine.
3. Write `hosts.json` from `configs/hosts.example.json`: each machine's SSH
   endpoint, its private address, its region (empty in one data center).
4. Run the phase from the operator's machine. It needs only Python and `ssh`:

   ```sh
   python controller.py --hosts hosts.json --phase configs/phase_a.json --only ravex --poll 10
   python make_table.py results/phase-a --price <USD per node-hour> --checks --expected-rounds 3
   ```

   Every node's `events.jsonl` is mirrored into `results/<phase>/<arm>/node<k>/`
   at every poll, so a machine lost mid-run, or destroyed as soon as its
   phase ends, has already sent everything it wrote. The logs follow at the
   end of each arm.

   An arm launched again into the same results - a retry after a hang -
   starts clean: the earlier attempt's folder and its lines of
   `controller.jsonl` move to `results/<phase>/attempts/<arm>-<n>/`, which
   `make_table.py` does not read, and on the hosts the new attempt writes
   under `runs/<phase>/attempt-<n>/`, so it neither appends to the old events
   nor resumes from the old store (GPU-545). Nothing is deleted.

## Phase (a): the runbook

Phase (a) is the smoke run, and the only one planned soon. Its budget is
**$3.50 on RunPod**, with gate a1 on vast.ai. As the plan wrote it - four pods
in one data center, the two systems concurrently - it does not fit with any
margin at Secure prices (4 pods x about 1.25 h x $0.89 is $4.45, before a
single retry), so it is **shrunk to two pods**, with the systems one after the
other on them.

### The machines

| | count | GPU | where | cloud |
|---|---|---|---|---|
| a1 | 1 | RTX 4090 24 GB | vast.ai, any | interruptible bid, about $0.25/h |
| a2 | 2 | RTX 4090 24 GB, one each | RunPod, **one** data center with Global Networking and 4090 stock on the day (EU-NL-1, EU-FR-1, US-TX-3 or US-CA-2) | Secure, unless Global Networking shows up for Community in the console, then Community |

Both a2 pods are created together, with Global Networking on, SSH exposed and
a PyTorch image with torch 2.7 or later; the container or volume disk needs
about 20 GB (the image, two full Ravex checkpoints of about 1.5 GB with
AdamW's moments, the data). The GPT fits a 4090 with room; a1 measures by how
much, and the step time.

### The commands

On the operator's machine, once:

```powershell
git -C ravex archive -o ravex.tar HEAD
pip install datasets tiktoken
python ravex\bench\vs_torchft\prepare.py --tokens 110000000 --out data-110M
tar -cf data.tar -C data-110M .
```

110M tokens cover phase (a) with room: 3000 steps x 16,384 tokens is 49M per
node, and each node owns half the file.

**a1, on vast.ai, before anything on RunPod** (gate: the step fits with 2 GB
or more to spare):

```powershell
scp -P <port> ravex.tar data.tar root@<vast host>:/workspace/
ssh -p <port> root@<vast host> "mkdir -p /workspace/ravex && tar -xf /workspace/ravex.tar -C /workspace/ravex && sh /workspace/ravex/bench/vs_torchft/remote/setup.sh"
ssh -p <port> root@<vast host> "cd /workspace/ravex/bench/vs_torchft && /workspace/venv/bin/python vram_check.py --data /workspace/data --node 0 --starting-nodes 1 --steps 200 --arm a1 --events /workspace/a1/off.jsonl && /workspace/venv/bin/python vram_check.py --data /workspace/data --node 0 --starting-nodes 1 --steps 200 --compile --arm a1c --events /workspace/a1/on.jsonl"
scp -P <port> -r root@<vast host>:/workspace/a1 results/a1
```

Then destroy the vast.ai machine. Write down the step time: it decides
`--steps` below.

**a2, on RunPod**: create the two pods, then for each of them:

```powershell
scp -P <port> ravex.tar data.tar root@<pod>:/workspace/
ssh -p <port> root@<pod> "mkdir -p /workspace/ravex && tar -xf /workspace/ravex.tar -C /workspace/ravex && sh /workspace/ravex/bench/vs_torchft/remote/setup.sh"
```

Write `hosts.json` with `pod0` and `pod1` (their SSH endpoints and `10.x`
addresses), check that each reaches the other, then run the two systems one
after the other and the table:

```powershell
ssh -p <port1> root@<pod1> "timeout 5 bash -c 'echo > /dev/tcp/<pod0 10.x>/22' && echo reachable"
cd ravex\bench\vs_torchft
python controller.py --hosts hosts.json --phase configs\phase_a.json --only ravex --poll 10 --timeout 1800
python controller.py --hosts hosts.json --phase configs\phase_a.json --only torchft --poll 10 --timeout 1800
python make_table.py results\phase-a --price 0.89 --overhead-minutes 5 --checks --expected-rounds 6 --header "phase (a): 2 x RTX 4090, RunPod <data center>, Global Networking, <date>"
```

Then destroy both pods. If a1 measured more than 0.4 s per step, run both
systems with `--steps 1500` instead (edit `train_args` in
`configs/phase_a.json`, and `--expected-rounds 3`): three rounds, the same
checks, a cost that still fits.

### What it costs

Assumed, and unmeasured until a1: 0.15 s per step on a 4090 (16,384 tokens of
GPT-2 small in bf16), a round of 0.5 GB in fp32 at Global Networking's
documented 100 Mbps (40 s; 71 s at the 7 MB/s measured in 2026), 10 minutes
per pod for boot, setup and the upload.

| | per pod | node-hours | Secure $0.89/h | Community $0.34/h |
|---|---|---|---|---|
| setup and upload | 10 min | | | |
| Ravex: load, 3000 steps (7.5 min), 6 rounds (4-7 min), evals and checkpoints | 13-16 min | | | |
| torchft: the same, without the checkpoints | 12-15 min | | | |
| copy-off of the logs and destroy | 2 min | | | |
| **total, two pods** | **37-43 min** | **1.2-1.4** | **$1.10-1.28** | **$0.42-0.49** |
| ceiling: setup given up at 15 min, each run stopped by `--timeout 1800`, 5 min to destroy | 80 min | 2.7 | $2.37 | $0.91 |

The `ravex-lean` and `torchft-ckpt` rows (GPU-213) run on the same two pods
with `--only ravex-lean` and `--only torchft-ckpt`, after the two above: about
25-30 minutes more per pod, $0.75-0.90 on Secure, which still fits the budget
without the retry. Run them only once the first two have passed their checks.

a1 on vast.ai is 20-30 minutes at about $0.25/h, about $0.10 of the vast.ai
balance. The phase fits under $3.50 at its timeouts' ceiling on Secure 4090s
with room for one retry; a failed check is still fixed off the meter, in the
CPU container, before anything is rented again.

### Go / no-go

Before starting the runs, from `setup.sh`'s output on each pod:

- a `10.x` address, and each pod reaches the other's;
- `torch ... cuda True`, the same data hashes on both;
- a1's margin was 2 GB or more.

After both runs, `make_table.py --checks` answers the rest:

- each system completed 6 rounds (3 with `--steps 1500`);
- after every round, both nodes of a system hold the same parameters
  (checksum spread 0);
- each system's eval loss fell from its initial value;
- the two nodes' first-step losses differ: they read different parts of the data;
- the two systems' final eval losses are within 0.1 of each other;
- their per-node tokens/s are within 10% of each other.

And by hand: torchft's lighthouse log (`results/phase-a/torchft/coordinator-0.log`)
shows a quorum of 2; no `RegionUnreachable` in either Ravex node's
`ravex.log`; `results/phase-a/<arm>/node*/events.jsonl` grew during the run,
not only at its end.

A no-go on any of them is fixed off the meter, in the container, and phase (a)
is run again. Phase (b) is not rented on a no-go.

### What the shrunk phase can and cannot show

It can show that the step fits and how fast it is, on both systems on the same
two GPUs; that the rounds close and leave both nodes with the same
parameters; that the shards differ and the loss falls; that torchft's
lighthouse and quorum and Ravex's rendezvous work over Global Networking; and
what a full-size fp32 round costs on that link, in each system - the first
measurement of it.

It cannot show anything about faults or joins (it has none), anything across
regions (one data center), anything about scaling (with two nodes Ravex's
pull and torchft's ring move the same bytes), or the same-hour network
comparison the plan wants: the two systems run one after the other, so a link
that changed between them changes their round times. Ravex runs it with a full
checkpoint every 100 steps instead of letting Moonclip choose its saves
(`configs/ravex_phase_a.yaml`), so its tokens/s is the outer loop's alone; the
cost of the saves is (b)'s row. At 3000 steps of GPT-2 small the loss has
moved well off its start, but the 0.1 agreement is still a sanity check, not a
result.
