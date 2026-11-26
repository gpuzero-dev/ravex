# HSDP over a slow link, 2,000 steps: results (GPU-521)

`bench/hsdp_link.py --steps 2000` with ravex `6aad15e`, 2026-11-26. Setup and
expectations in `EXPECTED.md`, written before the run.

## Every metric

| Arm | Wall s | Steps/s | vs alone | Waiting share | Rounds per shard | H it settled around | Network s/round | Waited s/round | Bytes/shard/round | Held-out loss | vs alone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alone | 368.2 | 5.43 | 1.00× | — | — | — | — | — | — | 0.908 | — |
| fp32 overlap + auto | 619.6 | 3.23 | 0.59× | 40.6% | 14 | 120-210 | 38.5 | 11.3 | 7.35 MB | 0.685 | −24.6% |
| bf16 + feedback, overlap + auto | 497.5 | 4.02 | 0.74× | 26.0% | 23 | 81-113 | 19.8 | 4.6 | 3.30 MB | 0.658 | −27.5% |
| fp8_block + feedback, overlap + auto | 431.1 | 4.64 | **0.85×** | **14.6%** | 27 | 70-80 | 13.2 | 1.6 | 1.76 MB | **0.654** | **−28.0%** |
| bf16 + feedback, blocking + auto | 567.0 | 3.53 | 0.65× | 35.1% | 9 | 160-320 | 21.7 | (all) | 3.29 MB | 0.686 | −24.4% |

Same model on both nodes: identical with blocking rounds; with overlap the
nodes differ by 0.03-5.8% of the parameters' sum at the end, the last round
still in flight (fp32's is the widest because its rounds are the longest).
Every waiting share includes the seed round, about 35 s of each arm.

## Expected and target against measured

| Metric | Arm | Expected | Target | Measured | |
|---|---|---|---|---|---|
| H it settles at | fp32 overlap | 230-320 | — | 120-210 | lower |
| | bf16 overlap | 120-160 | — | 81-113 | lower |
| | fp8 overlap | 80-110 | — | 70-80 | lower |
| | bf16 blocking | 350-480 | — | 160-320 | lower |
| Rounds per shard | fp32 / bf16 / fp8 overlap / bf16 blocking | 8 / 13-15 / 18-22 / 5-6 | — | 14 / 23 / 27 / 9 | more, from the lower H |
| Waiting share | fp32 overlap | ~25% | ≤ 25% | 40.6% | missed |
| | bf16 overlap | ~13% | ≤ 25% | 26.0% | missed by 1 point |
| | fp8 overlap | ~10% | ≤ 25% | 14.6% | **met** |
| | bf16 blocking + auto | ~28% | ≤ 30% | 35.1% | missed |
| Throughput vs alone | overlap arms | 0.75-0.9× | ≥ 0.7× | 0.59× / 0.74× / 0.85× | met for bf16 and fp8 |
| | bf16 blocking + auto | ~0.72× | ≥ 0.7× | 0.65× | missed |
| Loss vs alone | every HSDP arm | lower | ≥ 5% lower | 24-28% lower | **met** |
| Loss, overlap vs blocking (bf16) | | within 3% | within 3% | overlap 4% *better* | met, and better than expected |
| Same model on both nodes | blocking | identical | identical | identical | met |

## What it says

**On a long run, cast + error feedback + overlap + automatic H is the best
configuration on every axis.** fp8_block with feedback trains at 0.85× the
speed of a node alone and ends 28% below its loss; bf16 at 0.74× and 27.5%.
The smaller the round, the more rounds fit, and the more rounds, the better
the model: fp32 did 14 rounds, bf16 23, fp8 27, and the loss follows.

**Overlap beats blocking here**, which the 400-step run did not show: on 2,000
steps bf16 with overlap ends 4% below bf16 blocking, in 12% less time. The
round arriving late costs less than the rounds blocking gives up.

**What still misses:**

* An automatic H settles lower than the arithmetic says (bf16 81-113 against
  120-160) and keeps moving by ±15% round to round, so some rounds still wait
  (4.6 s on average in bf16). Not the swing the fix removed, a smaller one;
  probably the exchange competing with the training for the CPU, which
  makes a step during a flight slower than one alone. Worth measuring on GPUs,
  where the two do not share cores.
* fp32 does not fit this link: 7.3 MB per shard per round needs more compute
  per round than `outer_min_rounds` and the run's length allow.
* The seed round, about 35 s, is a fixed cost in each of these runs, and
  dominates any run of a few minutes.

**For the pods**: fp8_block (or bf16) with `outer_error_feedback`,
`outer_overlap` and `outer_auto_steps`, on a run of thousands of steps.
