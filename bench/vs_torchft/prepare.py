"""The benchmark's data: FineWeb-Edu in GPT-2's tokens, as ``train.bin`` and ``val.bin``.

    pip install datasets tiktoken
    python prepare.py --tokens 50000000 --out data

Run once, on the operator's machine, and copied to every node: the same bytes
on every machine is what makes the arms' data comparable, and ``setup.sh``
prints the files' hashes so the operator can see it. The documents of
FineWeb-Edu's ``sample-10BT`` are read in order, each tokenized with GPT-2's
encoding and followed by its end-of-text token, until ``--tokens`` are
written; the last ``--val-share`` of them, whole documents, are the held-out
set. Tokens are little-endian uint32, as ``common.py`` reads them.

``meta.json`` says what the files hold, so a result can name the data it was
trained on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, required=True, help="tokens in total, train and val")
    parser.add_argument("--val-share", type=float, default=0.01)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    args = parser.parse_args()

    import tiktoken
    from datasets import load_dataset

    encoding = tiktoken.get_encoding("gpt2")
    documents = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)
    want_val = int(args.tokens * args.val_share)
    want_train = args.tokens - want_val
    os.makedirs(args.out, exist_ok=True)
    counts = {}
    rows = iter(documents)
    for name, want in (("train.bin", want_train), ("val.bin", want_val)):
        written = 0
        with open(os.path.join(args.out, name), "wb") as handle:
            while written < want:
                tokens = encoding.encode_ordinary(next(rows)["text"]) + [encoding.eot_token]
                tokens = tokens[: want - written]
                np.asarray(tokens, dtype="<u4").tofile(handle)
                written += len(tokens)
        counts[name] = written
        print("%s: %d tokens" % (name, written), flush=True)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump({"dataset": args.dataset, "subset": args.subset, "encoding": "gpt2",
                   "vocab_size": encoding.n_vocab, "dtype": "<u4", "tokens": counts}, handle, indent=2)


if __name__ == "__main__":
    main()
    # Every file is written and closed. A streamed dataset leaves reader
    # threads behind that abort the interpreter as it finalizes (seen with
    # datasets 4 on 2026-11-25: "Aborted (core dumped)" after both files were
    # complete), which would read as a failed preparation; leave without it.
    sys.stdout.flush()
    os._exit(0)
