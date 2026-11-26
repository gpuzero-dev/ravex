# HSDP over a slow link: expected and target, written before the run (GPU-521)

Setup: `bench/hsdp_link.py` with its defaults. Two containers as two pods, 2
processes each (FSDP2 over gloo), a 4.2M-parameter model (4 blocks of 1024),
400 steps of batch 16, H starting at 40, each container's outgoing link at
4 Mbit/s (0.5 MB/s) with 50 ms one way. Measured before: 170 ms per step on
CPU, so 40 steps are 6.8 s of compute.

Arithmetic the expectations come from: one shard is 2.1M parameters, 8.4 MB in
fp32; a pod sends its two shards to the other pod every round, 16.8 MB on a
0.5 MB/s uplink, so about 34 s of network per round in fp32.

| Metric | Arm | Expected | Target | Why |
|---|---|---|---|---|
| Bytes per shard per round | fp32 | ~8.4 MB | — | 2.1M params × 4 B |
| | bf16 + feedback | ~4.2 MB (0.50×) | ≤ 0.55× fp32 | 2 B per value; deltas barely compress |
| | fp8_block + feedback | ~2.1 MB (0.25×) | ≤ 0.30× fp32 | 8.06 bits per value |
| Network per round | fp32 | ~34 s | — | 16.8 MB at 0.5 MB/s |
| | bf16 + feedback | ~17 s | ≤ 0.55× fp32 | follows the bytes |
| | fp8_block + feedback | ~9 s | ≤ 0.30× fp32 | |
| Share of wall time waiting for the network | fp32 blocking | ~80% | — | 34 s of network for 6.8 s of compute per round |
| | fp32 overlap + auto H | ~40% | ≤ 25% | H grows toward the network, but at most 400/3 steps per round leaves fp32 short |
| | bf16 + feedback | ≤ 20% | ≤ 25% | 17 s of network against ~22 s of compute once H is ~130 |
| | fp8_block + feedback | ~5% | ≤ 25% | |
| Throughput vs alone | fp32 blocking | ~0.2× | — | |
| | overlap arms | 0.6–0.95× | ≥ 0.7× | |
| Held-out loss | HSDP arms vs alone | lower | lower | each node learns from both nodes' data |
| | bf16 / fp8 + feedback vs fp32 overlap | within 1% | within 1% relative | feedback loses nothing over the rounds (unit test: 1e-7) |
| | overlap vs blocking | slightly higher | within 3% relative | the outer update lands a round late |
| Same model on both nodes | blocking | identical | identical | every round applied at the same step |
| | overlap arms | may differ slightly | — | the last round can still be in flight at the end |
