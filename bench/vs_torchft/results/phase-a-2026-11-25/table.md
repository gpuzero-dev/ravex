phase (a): 2 x RTX PRO 4500 SE, RunPod Secure, Global Networking, 2026-11-25; GPT 124M, H=500, 3000 steps per node

| arm | committed tokens | eval loss @ 24.6M | eval loss @ 49.2M | eval loss @ 98.3M | tokens/s (cluster) | wasted steps | lost s per kill | join s to contribute | $ per 1M tokens |
|---|---|---|---|---|---|---|---|---|---|
| ravex | 98.3M | 6.166 | 5.528 | 4.739 | 104016 | 0 | none | none | 0.005 |
| ravex-lean | 98.3M | 6.164 | 5.511 | 4.751 | 104500 | 0 | none | none | 0.005 |
| torchft | 98.3M | 6.189 | 5.496 | 4.738 | 108559 | 0 | none | none | 0.005 |
| torchft-ckpt | 98.3M | 6.171 | 5.497 | 4.733 | 104498 | 0 | none | none | 0.005 |
