# Phase (a), 2026-11-25

Two RTX PRO 4500 SE pods (Blackwell, 32 GB) on RunPod Secure, one data center,
Global Networking (about 13 MB/s between them). GPT 124M (`model.py`), 3000
steps per node of 16,384 tokens, H = 500, outer Nesterov SGD lr 0.7 momentum
0.9, fp32 rounds of 0.5 GB. Ravex at commit `5d5495f` (0.16.2 plus the join
and seed-poll changes), torchft 0.3.1, torch 2.8.0+cu129. Data: `prepare.py`,
110M tokens, `train.bin` sha256 `208ddf61...`, the same on both pods.

`table.md` and `table.json` are `make_table.py --checks` over the four arms;
every check passed.

What to read in it:

- Per node the systems are the same speed: 75.9k tokens/s under Ravex, 76.2k
  under torchft, 76.6k with neither (gate a1). A round costs 41 s under Ravex
  and 43 s under torchft on this link.
- At equal guarantees - a checkpoint every 100 steps on both - Ravex 104.0k
  tokens/s, torchft with synchronous DCP 104.5k; torchft without one 108.6k.
- Ravex's cluster figure carries 39 s at the start: the node that is not the
  source downloaded the starting state (0.5 GB) while the source trained, and
  the source waited for it at the end of round 1. `ea5bcd6` skips that
  download when the states match, but the outer loop is still built after the
  first step, so they never matched here (checked in `phase-a-seed`, a 1000-step
  run on the same pods); the order is the next fix.
- The torchft row's events had to be cut to its second attempt: the first hung
  at its first sync (Gloo on the container's address; `GLOO_SOCKET_IFNAME` is
  now in `phase_a.json`) and its events were appended to the same file.
