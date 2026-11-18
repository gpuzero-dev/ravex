"""What the default pre-training metrics cost a 1B model's step on a GPU (GPU-436).

A decoder of about a billion parameters, trained with nothing but the
decorator and a logged ``train/loss`` - the script GPU-436 is about. Ravex
then charts on its own the throughput, the MFU, the perplexity, and every
``histograms_every`` steps the histograms per layer, the update-to-weight
ratios and the activations. The question is what such a histogram step adds,
and the acceptance is "a few milliseconds".

Each step is timed between two ``torch.cuda.synchronize()``: that serialises
the step with the host, which costs every step alike, and it is the only way
to see one step's own cost rather than an average that spreads it. The answer
is the median of the histogram steps minus the median of the others, each
taken after the warm-up.

It runs in the job's own store, so the run it leaves is the one the metrics
page shows: what a script with only the decorator gets, on a real GPU.

    python bench/pretraining_cost.py
    python bench/pretraining_cost.py --width 2048 --layers 16 --steps 300
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn as nn

import ravex


class Block(nn.Module):
    """A pre-norm decoder block, named as the common implementations name it."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.input_norm = nn.RMSNorm(width)
        self.self_attn = nn.ModuleDict({
            "qkv_proj": nn.Linear(width, 3 * width, bias=False),
            "o_proj": nn.Linear(width, width, bias=False),
        })
        self.post_norm = nn.RMSNorm(width)
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(width, 4 * width, bias=False),
            "down_proj": nn.Linear(4 * width, width, bias=False),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.self_attn["qkv_proj"](self.input_norm(x))
        q, k, v = qkv.view(batch, length, 3, self.heads, width // self.heads).unbind(2)
        attended = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.self_attn["o_proj"](attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp["down_proj"](nn.functional.gelu(self.mlp["gate_proj"](self.post_norm(x))))


class Decoder(nn.Module):
    def __init__(self, vocab: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.RMSNorm(width)
        self.lm_head = nn.Linear(width, vocab, bias=False)

        class Config:
            num_hidden_layers = layers
            hidden_size = width

        # How a Hugging Face model says its shape: the MFU's attention term.
        self.config = Config()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--histograms_every", type=int, default=25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this bench measures a GPU, and there is none")
    device = torch.device("cuda")
    times: list = []

    @ravex.train_loop(
        checkpoint_every=10**9,
        checkpoint_on_exit=False,
        metrics_every=10,
        histograms_every=args.histograms_every,
        backend="torch_save",
    )
    def train() -> None:
        torch.manual_seed(0)
        model = Decoder(args.vocab, args.width, args.layers, args.heads).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        parameters = sum(p.numel() for p in model.parameters())
        print("%s, %.2fB parameters, batch %d x %d tokens, histograms every %d steps"
              % (torch.cuda.get_device_name(0), parameters / 1e9, args.batch, args.length,
                 args.histograms_every), flush=True)
        for _ in range(args.steps):
            ids = torch.randint(0, args.vocab, (args.batch, args.length), device=device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids)
                loss = nn.functional.cross_entropy(logits.reshape(-1, args.vocab).float(), ids.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ravex.log_metrics({"train/loss": loss.detach()})
            torch.cuda.synchronize()
            times.append((ravex.step(), (time.perf_counter() - started) * 1000.0))

    train()
    measured = [(step, ms) for step, ms in times if step > args.warmup]
    charting = [ms for step, ms in measured if step % args.histograms_every == 0]
    plain = [ms for step, ms in measured if step % args.histograms_every != 0]
    if not charting or not plain:
        raise SystemExit("too few steps after the warm-up to compare")
    plain_ms, charting_ms = statistics.median(plain), statistics.median(charting)
    print("median step %.2f ms; histogram step %.2f ms; added %.2f ms (%d histogram steps, %d others)"
          % (plain_ms, charting_ms, charting_ms - plain_ms, len(charting), len(plain)), flush=True)
    print(json.dumps({
        "card": torch.cuda.get_device_name(0), "step_ms": plain_ms, "histogram_step_ms": charting_ms,
        "added_ms": charting_ms - plain_ms, "histogram_steps": len(charting), "other_steps": len(plain),
    }))


if __name__ == "__main__":
    main()
