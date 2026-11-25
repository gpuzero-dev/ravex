"""The benchmark's model: a GPT of GPT-2's shape, here so the bench needs no other repository.

``124m`` is GPT-2 small - 12 layers, 12 heads, 768 wide, a vocabulary of
50,304 (GPT-2's 50,257 rounded up to a multiple of 64) - about 124M
parameters with the input and output embeddings tied. ``tiny`` is the same
network small enough for the CPU tests. Chosen on 2026-11-25 over deepseek-1b
(GPU-213): the benchmark measures the outer loop, not the model, and a small
model makes a round's bytes and a phase's cost small enough to run the phases
at all; its 0.5 GB round in fp32 is still a real load on a link between
regions.

Plain on purpose: pre-norm blocks, GELU, ``scaled_dot_product_attention``
with its causal mask, no dropout. Nothing in it is specific to either arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    max_seq_len: int = 1024


PRESETS: Dict[str, Config] = {
    "124m": Config(),
    "tiny": Config(vocab_size=512, n_layer=2, n_head=2, n_embd=32, max_seq_len=64),
}


def config_of(preset: str, **overrides: Any) -> Config:
    """A preset with some fields changed; an unknown preset or field raises."""
    if preset not in PRESETS:
        raise SystemExit("--model must be one of %s, got %r" % (", ".join(sorted(PRESETS)), preset))
    return replace(PRESETS[preset], **overrides)


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.out = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln_1(x)).split(c, dim=2)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        return x + self.out(F.gelu(self.fc(self.ln_2(x))))


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tied, as GPT-2's are: the 124M counts one matrix of 50,304 x 768.
        self.head.weight = self.wte.weight
        self.apply(self._init)
        # GPT-2's scaled init of the projections into the residual stream.
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def count_parameters(model: nn.Module) -> int:
    """Each tensor once: the tied head is the embedding, not a second matrix."""
    return sum(p.numel() for p in model.parameters())
