"""A small GPT trained on characters, the shape of a nanoGPT-style training
script, on a CPU: stop it, run it again, and it carries on.

    python examples/train_gpt.py
    # Ctrl-C somewhere around step 200
    python examples/train_gpt.py

It is the loop most language-model scripts have: batches drawn at random
offsets of one long token stream with ``torch.randint``, AdamW, a warm-up and
cosine learning rate, gradient clipping, and a loss printed every so often.
Nothing of it is about checkpoints but the decorator. A resumed run draws the
same batches and follows the same learning rate as one that was never
stopped, because Ravex restores torch's RNG and the LR scheduler along with
the weights and the optimizer, so its losses are the uninterrupted run's.

The text is made here, so the example needs no download. The checkpoints go
to ./checkpoints; delete it to start over (GPU-367).
"""

import math

import torch
import torch.nn as nn

import ravex

CONTEXT = 32
STEPS = 400


def corpus() -> str:
    """A few thousand characters with structure a model can learn."""
    words = ["checkpoint", "resume", "step", "loss", "model", "train", "save", "node", "gpu", "run"]
    generator = torch.Generator().manual_seed(1)
    picks = torch.randint(len(words), (1500,), generator=generator).tolist()
    return " ".join(words[i] for i in picks) + "."


class GPT(nn.Module):
    def __init__(self, vocab: int, width: int = 64, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.tokens = nn.Embedding(vocab, width)
        self.positions = nn.Embedding(CONTEXT, width)
        block = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        length = idx.shape[1]
        h = self.tokens(idx) + self.positions(torch.arange(length))
        mask = nn.Transformer.generate_square_subsequent_mask(length)
        return self.head(self.norm(self.blocks(h, mask=mask, is_causal=True)))


@ravex.train_loop(max_steps=STEPS, checkpoint_every=50)
def train():
    torch.manual_seed(0)
    text = corpus()
    vocab = sorted(set(text))
    data = torch.tensor([vocab.index(c) for c in text])
    model = GPT(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    warmup = 50

    def rate(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (STEPS - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    # Counted by Ravex rather than by a `range`: a resumed run starts its loop
    # again, and ravex.step() is where the training really is.
    while ravex.step() < STEPS:
        starts = torch.randint(len(data) - CONTEXT - 1, (16,))
        x = torch.stack([data[i : i + CONTEXT] for i in starts])
        y = torch.stack([data[i + 1 : i + CONTEXT + 1] for i in starts])
        loss = nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if ravex.step() % 100 == 0:
            print(f"step {ravex.step()}  loss {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    train()
