# HSDP over a slow link, 2,000 steps: expected and target, written before the run (GPU-521)

Same setup as `../hsdp-link-2026-11-26` (4.2M parameters, 2 containers × 2
FSDP2 processes, 4 Mbit/s with 50 ms one way, H starting at 40), with ravex
`6aad15e` (an automatic H with overlap counts only the training) and 2,000
steps, so H has room to settle. The fixed-H blocking arm is left out: at H=40
it is 50 rounds of 37 s. In its place, bf16 with blocking rounds and an
automatic H.

Numbers the expectations come from, measured on the 400-step run: 0.18 s per
step; network per round 41 s in fp32, 21.6 s in bf16, 14.4 s in fp8_block;
the seed round about 30 s.

| Metric | Arm | Expected | Target | Why |
|---|---|---|---|---|
| H it settles at | fp32 overlap + auto | ~230-320 | — | hides 41 s at 0.18 s per step |
| | bf16 + feedback, overlap | ~120-160 | — | hides 21.6 s |
| | fp8_block + feedback, overlap | ~80-110 | — | hides 14.4 s |
| | bf16 + feedback, blocking + auto | ~350-480 | — | network at a quarter of the round |
| Rounds per shard | fp32 overlap | ~8 | — | |
| | bf16 overlap | ~13-15 | — | |
| | fp8 overlap | ~18-22 | — | |
| | bf16 blocking + auto | ~5-6 | — | |
| Waiting share of wall time | fp32 overlap | ~25% | ≤ 25% | the first rounds wait while H climbs, and the seed round |
| | bf16 overlap | ~13% | ≤ 25% | |
| | fp8 overlap | ~10% | ≤ 25% | |
| | bf16 blocking + auto | ~28% | ≤ 30% | a quarter by design, plus the seed round |
| Throughput vs alone | overlap arms | 0.75-0.9× | ≥ 0.7× | |
| | bf16 blocking + auto | ~0.72× | ≥ 0.7× | |
| Held-out loss | every HSDP arm vs alone | lower | ≥ 5% lower | each node learns from both nodes' data |
| | overlap vs blocking + auto, bf16 | within 3% | within 3% | more rounds against a round late |
| Same model on both nodes | blocking + auto | identical | identical | |
