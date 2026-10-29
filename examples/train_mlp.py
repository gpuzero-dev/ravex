"""Kill it, run it again, and it carries on: a small MLP on this machine's CPU.

    python examples/train_mlp.py
    # Ctrl-C somewhere around step 300, or `kill` it from another terminal
    python examples/train_mlp.py

The second run says which step it starts at and finishes the same 1000 steps
the first one would have, with the same losses, because Ravex restores the
optimizer, the LR scheduler, the RNGs and the position in the dataloader along
with the weights. Delete ./checkpoints to start over.

The only Ravex line is the decorator. Its arguments are the configuration a
`ravex.yaml` beside the working directory could hold instead (GPU-295):

- `max_steps` is the length of the whole training across restarts. Without
  it, a resumed script would run its own `for epoch in range(EPOCHS)` again
  from the top and train past where the first run would have stopped.
- `checkpoint_every` is how often a checkpoint is due. With Moonclip installed
  (`pip install "ravex[moonclip]"`) a small delta is saved at every step on
  top of that; without it, Ravex writes plain `torch.save` files.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ravex

SAMPLES = 2048
BATCH = 32
EPOCHS = 20


@ravex.train_loop(max_steps=1000, checkpoint_every=100, storage={"path": "./checkpoints"})
def main():
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    features = torch.randn(SAMPLES, 16)
    targets = (features * torch.linspace(1, 2, 16)).sum(dim=1, keepdim=True)
    loader = DataLoader(
        TensorDataset(features, targets), batch_size=BATCH, shuffle=True
    )

    model.train()
    started = False
    for _ in range(EPOCHS):
        for x, y in loader:
            if not started:
                # The resume happens as the first batch is fetched, so this is
                # the first point where `ravex.step()` says where we are.
                started = True
                print(f"starting at step {ravex.step()}", flush=True)

            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if ravex.step() % 50 == 0:
                # From the step, not from the loop's own counter, which a
                # resumed run starts again at 0.
                epoch = (ravex.step() - 1) // len(loader)
                print(f"epoch {epoch:2d}  step {ravex.step():5d}  loss {loss.item():.5f}", flush=True)

    print(f"done at step {ravex.step()}")


if __name__ == "__main__":
    main()
