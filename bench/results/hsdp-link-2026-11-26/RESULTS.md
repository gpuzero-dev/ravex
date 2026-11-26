# HSDP over a slow link: results (GPU-521)

`bench/hsdp_link.py` with its defaults, 2026-11-26: two containers as two pods,
2 FSDP2 processes each (gloo, CPU), 4.2M parameters, 400 steps of batch 16,
H starting at 40, each container's outgoing link 4 Mbit/s with 50 ms one way.
Expectations and targets were written before the run, in `EXPECTED.md`.

"Before" is ravex `64bb887`. "After" adds the fix found by this run: with
`outer_overlap`, the wait for the previous round to land counted as training,
so an automatic H swung between 40 and 80 and never hid the network
(`OuterLoop.trained_seconds`). Blocking and `alone` do not go through it and
were not rerun; `alone` was, as the reference, and matched (71.6 s against
72.5 s, the same losses).

## Every metric

| Arm | Wall s | Steps/s | vs alone | Waiting share | Rounds per shard | Network s/round | Waited s/round | Bytes/shard/round | Held-out loss | H path |
|---|---|---|---|---|---|---|---|---|---|---|
| alone | 71.6 | 5.58 | 1.00× | — | — | — | — | — | 1.888 | — |
| fp32 blocking | 449.1 | 0.89 | 0.16× | 83.9% | 10 | 36.8 | (all) | 7.23 MB | **1.615** | 40 |
| fp32 overlap + auto, before | 303.2 | 1.32 | 0.24× | 76.1% | 7 | 42.8 | 28.5 | 7.28 MB | 1.709 | 40↔80 |
| fp32 overlap + auto, after | 182.7 | 2.19 | 0.39× | 60.8% | 4 | 41.0 | 23.5 | 7.33 MB | 1.836 | 40→80→160→124 |
| bf16 + feedback, before | 162.4 | 2.46 | 0.45× | 55.3% | 7 | 21.4 | 12.3 | 3.28 MB | 1.709 | 40↔80 |
| bf16 + feedback, after | 131.3 | 3.05 | 0.55× | 45.5% | 6 | 21.6 | 11.1 | 3.29 MB | 1.723 | 40→80→111→93→36 |
| fp8_block + feedback, before | 124.6 | 3.21 | 0.58× | 41.8% | 7 | 13.3 | 5.8 | 1.76 MB | 1.712 | 40↔80 |
| fp8_block + feedback, after | 113.0 | 3.54 | 0.63× | 36.7% | 6 | 14.4 | 5.5 | 1.76 MB | 1.738 | 40→80 |

Waiting share is the wall time beyond `alone`'s over the arm's wall time:
everything a node spent not training, the seed round at the start included.
Same model on both nodes: identical in the blocking arm, as expected; the
overlap arms differ by at most 0.2% of the parameters' sum, the last round
still in flight when the training ended.

## Expected and target against measured

| Metric | Expected | Target | Measured | |
|---|---|---|---|---|
| bf16 bytes vs fp32 | 0.50× | ≤ 0.55× | 0.45× | met |
| fp8_block bytes vs fp32 | 0.25× | ≤ 0.30× | 0.24× | met |
| bf16 network per round vs fp32 (same mode) | 0.50× | ≤ 0.55× | 0.53× | met |
| fp8_block network per round vs fp32 | 0.27× | ≤ 0.30× | 0.35× | missed: ~2–3 s per round do not shrink with the bytes (latency, the round's decision, the publish) |
| Waiting share, fp32 blocking | ~80% | — | 83.9% | as expected |
| Waiting share, fp32 overlap + auto | ~40% | ≤ 25% | 60.8% (76.1% before the fix) | missed |
| Waiting share, bf16 + feedback | ≤ 20% | ≤ 25% | 45.5% (55.3%) | missed |
| Waiting share, fp8_block + feedback | ~5% | ≤ 25% | 36.7% (41.8%) | missed |
| Throughput, overlap arms vs alone | 0.6–0.95× | ≥ 0.7× | 0.39–0.63× | missed |
| Loss, HSDP vs alone | lower | lower | 1.615–1.836 vs 1.888 | met, every arm |
| Loss, bf16 / fp8 + feedback vs fp32, same rounds | within 1% | within 1% | +0.04% / +0.19% (before: 7 rounds each) | met |
| Loss, overlap vs blocking | slightly higher | within 3% | +5.8% (before), +6.7–13.7% (after) | missed |
| Same model on both nodes, blocking | identical | identical | identical | met |

## What it says

**The casts do what was claimed, and lose nothing.** bf16 sends 0.45× the
bytes and fp8_block 0.24×, and with error feedback the loss after the same
seven rounds is within 0.2% of fp32. On this link that halves (bf16) or
thirds (fp8) the network per round. Turning `outer_save_dtype: bf16` with
`outer_error_feedback` on is supported by this and by the unit test that the
sum of the rounds loses 1e-7.

**Overlap with an automatic H was broken, and is not enough even fixed.** The
bug made H swing between 40 and 80. Fixed, H climbs, the waiting drops by
10–16 points and the throughput rises by 10–65%, but no arm gets near 25%:

* 400 steps are too few for H to settle. `outer_min_rounds` keeps at least 3
  rounds, so H is held down and pulled back near the end (93, then 36).
* Every raise of H costs rounds, and rounds are what the loss is made of:
  fp32 overlap after the fix did 4 rounds and lost most of HSDP's gain (1.836
  against 1.615 blocking with 10). A round late on top of that, overlap is
  5–14% worse than blocking here.
* The seed round, about half a minute on this link, is in every arm's wall
  time and a large share of a two-minute run.

**For the pods**: bf16 or fp8_block with error feedback, yes. Overlap with an
automatic H only on a run long enough for H to settle, thousands of steps,
judged on loss per hour rather than steps per second, since on this link
blocking rounds made the better model.
