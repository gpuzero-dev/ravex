"""Data contamination: how much of the benchmarks the training sees (GPU-437).

A model that trained on a benchmark's test items scores well on it without
having learnt what the benchmark measures. The usual check (GPT-3's, and the
ones after it) is n-gram overlap: an item is contaminated when one of its
13-token windows appears in the training data. Ravex makes it during the run,
on a sample of the batches the model is called with, so it shows in Metrics
next to the loss:

* ``data/ngram13_overlap``: the share of the sampled 13-grams that are in a
  benchmark, over the last ``metrics_every`` steps;
* ``data/benchmark_overlap``: the share of benchmark items with at least one
  13-gram seen so far in the run, and ``data/benchmark_overlap/<name>`` for
  each benchmark.

**On the model's own token ids**, read where Ravex already counts tokens (the
forward of a watched model), so nothing is decoded. The benchmarks are
therefore tokenized with the same tokenizer, once, by
``ravex contamination build --tokenizer <name>``, which needs ``datasets``,
``transformers`` and the internet; a node needs neither, only the index file
it writes - a local path, or in the run's bucket (``s3://`` through
Moonclip). An index made with another tokenizer finds nothing, silently: its
name is in the file and in the run's log.

**Exact, not a Bloom filter.** MMLU, HellaSwag, GSM8K and ARC-Challenge come
to a few million 13-grams, a sorted tensor of their 64-bit hashes of some tens
of megabytes, looked up with ``searchsorted`` on the device the ids are on:
no false positive to explain, and nothing that waits for the GPU.

**Sampled.** Every ``contamination_every`` steps, the first
``contamination_rows`` rows of one training forward's ids. The figures are
an estimate of the stream, and cost a few kernels per check.

**Per process.** Rank 0 logs, as for every metric, from the batches its own
forward sees; with data parallelism the other ranks' share of the data is
not in the sample. What has been seen restarts with each execution: a
resumed run counts again from its resume.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

NGRAM = 13
FORMAT = 1
NGRAM_OVERLAP = "data/ngram13_overlap"
BENCHMARK_OVERLAP = "data/benchmark_overlap"

#: What ``build`` reads, by the name a person gives: the Hugging Face dataset,
#: its configuration, its split, and the item's text. HellaSwag's test split
#: has no labels and is not what anybody evaluates on: its validation is.
BENCHMARKS: Dict[str, Tuple[str, Optional[str], str]] = {
    "mmlu": ("cais/mmlu", "all", "test"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
}


def item_text(benchmark: str, row: Dict[str, Any]) -> str:
    """An item as a model would meet it in text: the question and what it
    is answered with."""
    if benchmark == "mmlu":
        return "\n".join([row["question"], *row["choices"]])
    if benchmark == "hellaswag":
        return "\n".join([row["ctx"], *row["endings"]])
    if benchmark == "gsm8k":
        return row["question"] + "\n" + row["answer"]
    if benchmark == "arc_challenge":
        return "\n".join([row["question"], *row["choices"]["text"]])
    raise ValueError("no benchmark %r; there is %s" % (benchmark, ", ".join(sorted(BENCHMARKS))))


def _splitmix64(state: int) -> Tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def _coefficients(n: int) -> List[int]:
    """A fixed odd 64-bit multiplier per position, as signed int64. Made by
    splitmix64 from a constant, not by a torch generator, so an index built
    with one torch version is read by every other."""
    out, state = [], 0x6770753A34333721
    for _ in range(n):
        state, value = _splitmix64(state)
        value |= 1
        out.append(value - (1 << 64) if value >= (1 << 63) else value)
    return out


def ngram_hashes(ids: Any, n: int = NGRAM) -> Any:
    """The 64-bit hash of every window of ``n`` ids along the last dimension:
    ``[..., L]`` to ``[..., L - n + 1]``; empty when ``L < n``. A sum of each
    id times its position's multiplier, wrapping at 64 bits, on the device
    the ids are on."""
    import torch

    ids = ids.to(torch.int64)
    if ids.shape[-1] < n:
        return ids.new_empty(ids.shape[:-1] + (0,))
    coefficients = torch.tensor(_coefficients(n), dtype=torch.int64, device=ids.device)
    return (ids.unfold(-1, n, 1) * coefficients).sum(-1)


class Index:
    """The benchmarks' n-grams: sorted hashes, the item each comes from, and
    which items are whose."""

    def __init__(self, hashes: Any, items: Any, benchmarks: List[Dict[str, Any]], tokenizer: str, n: int) -> None:
        self.hashes = hashes
        self.items = items
        self.benchmarks = benchmarks
        self.tokenizer = tokenizer
        self.n = n
        self.count = sum(int(b["items"]) for b in benchmarks)

    @classmethod
    def build(cls, benchmarks: Dict[str, Sequence[Sequence[int]]], tokenizer: str, n: int = NGRAM) -> "Index":
        """From each benchmark's items as token ids. An item shorter than
        ``n`` tokens has no window and cannot be found; it still counts as an
        item of its benchmark."""
        import torch

        hashes, items, described, first = [], [], [], 0
        for name, rows in benchmarks.items():
            for offset, ids in enumerate(rows):
                found = ngram_hashes(torch.tensor(list(ids), dtype=torch.int64), n).unique()
                hashes.append(found)
                items.append(torch.full_like(found, first + offset, dtype=torch.int32))
            described.append({"name": name, "first": first, "items": len(rows)})
            first += len(rows)
        flat = torch.cat(hashes) if hashes else torch.empty(0, dtype=torch.int64)
        owner = torch.cat(items) if items else torch.empty(0, dtype=torch.int32)
        order = torch.argsort(flat)
        return cls(flat[order], owner[order], described, tokenizer, n)

    def to_bytes(self) -> bytes:
        import torch

        buffer = io.BytesIO()
        torch.save({"format": FORMAT, "n": self.n, "tokenizer": self.tokenizer, "benchmarks": self.benchmarks,
                    "hashes": self.hashes, "items": self.items}, buffer)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Index":
        import torch

        saved = torch.load(io.BytesIO(data), weights_only=True)
        if saved.get("format") != FORMAT:
            raise ValueError("a contamination index of format %r; this Ravex reads %d" % (saved.get("format"), FORMAT))
        return cls(saved["hashes"], saved["items"], list(saved["benchmarks"]), str(saved["tokenizer"]), int(saved["n"]))


def read_index(where: str, storage: Any = None) -> Index:
    """The index at a local path, or at ``s3://bucket/key`` through Moonclip
    with the run's bucket settings (endpoint, region) and the keys from the
    environment, as the run's store is reached."""
    if where.startswith("s3://"):
        from moonclip import Bucket

        bucket, _, key = where[len("s3://"):].partition("/")
        data = Bucket(
            bucket,
            endpoint=getattr(storage, "endpoint", None),
            region=getattr(storage, "region", None),
            path_style=getattr(storage, "path_style", None),
        ).get(key)
        if data is None:
            raise FileNotFoundError("no contamination index at %s" % where)
        return Index.from_bytes(data)
    with open(where, "rb") as handle:
        return Index.from_bytes(handle.read())


def write_index(index: Index, where: str) -> None:
    data = index.to_bytes()
    if where.startswith("s3://"):
        from moonclip import Bucket

        bucket, _, key = where[len("s3://"):].partition("/")
        Bucket(bucket).put(key, data)
        return
    with open(where, "wb") as handle:
        handle.write(data)


class Check:
    """A run's running count against an index, kept on the device of the ids
    it is given: nothing in :meth:`observe` waits for the GPU."""

    def __init__(self, index: Index, rows: int) -> None:
        self.index = index
        self.rows = rows
        self._device: Any = None
        self._seen: Any = None
        self._hits: Any = None
        self._total: Any = None

    def _on(self, device: Any) -> None:
        import torch

        if self._device == device:
            return
        self._device = device
        self.index.hashes = self.index.hashes.to(device)
        self.index.items = self.index.items.to(device)
        self._seen = torch.zeros(self.index.count, dtype=torch.bool, device=device)
        self._hits = torch.zeros((), dtype=torch.int64, device=device)
        self._total = torch.zeros((), dtype=torch.int64, device=device)

    def observe(self, ids: Any) -> None:
        """Compare the first ``rows`` rows of a forward's token ids."""
        import torch

        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        sample = ids[: self.rows]
        self._on(sample.device)
        hashes = ngram_hashes(sample, self.index.n).reshape(-1)
        if hashes.numel() == 0 or self.index.hashes.numel() == 0:
            return
        place = torch.searchsorted(self.index.hashes, hashes).clamp_(max=self.index.hashes.numel() - 1)
        found = self.index.hashes[place] == hashes
        self._hits += found.sum()
        self._total += hashes.numel()
        # A hash can stand for several items (one window in two questions):
        # the first is marked, which counts the item a window first came from.
        self._seen[self.index.items[place[found]].long()] = True

    def figures(self) -> Dict[str, Any]:
        """The metrics since the last call, as device tensors; empty before
        anything was observed. The n-gram share is the interval's, the items
        seen the run's so far."""
        if self._total is None:
            return {}
        # No `.item()` here: it would wait for the GPU. An interval with no
        # window reads 0 over 1.
        out: Dict[str, Any] = {NGRAM_OVERLAP: self._hits.double() / self._total.clamp(min=1).double()}
        out[BENCHMARK_OVERLAP] = self._seen.double().mean()
        for benchmark in self.index.benchmarks:
            first, count = int(benchmark["first"]), int(benchmark["items"])
            if count:
                out["%s/%s" % (BENCHMARK_OVERLAP, benchmark["name"])] = self._seen[first:first + count].double().mean()
        self._hits.zero_()
        self._total.zero_()
        return out


def build_from_hub(names: Iterable[str], tokenizer: str) -> Index:
    """Read the benchmarks from the Hugging Face Hub and tokenize them: the
    one step that needs the internet, ``datasets`` and ``transformers``."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("building an index needs datasets and transformers: pip install datasets transformers") from exc
    tokens = AutoTokenizer.from_pretrained(tokenizer)
    tokenized: Dict[str, List[List[int]]] = {}
    for name in names:
        if name not in BENCHMARKS:
            raise SystemExit("no benchmark %r; there is %s" % (name, ", ".join(sorted(BENCHMARKS))))
        dataset, config, split = BENCHMARKS[name]
        rows = load_dataset(dataset, config, split=split)
        tokenized[name] = [tokens(item_text(name, row), add_special_tokens=False)["input_ids"] for row in rows]
        logger.info("%s: %d items", name, len(tokenized[name]))
    return Index.build(tokenized, tokenizer)
