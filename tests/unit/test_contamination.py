"""The check of the training data against the benchmarks' n-grams (GPU-437):
the hash, the index, the count on clean and on planted data, and a run that
shows it in its metrics."""

from __future__ import annotations

import torch
import torch.nn as nn

import ravex
import ravex.metrics
from ravex._contamination import (
    BENCHMARK_OVERLAP,
    NGRAM_OVERLAP,
    Check,
    Index,
    item_text,
    ngram_hashes,
    read_index,
    write_index,
)

VOCAB = 50_000


def _benchmarks(seed=0):
    """Two benchmarks of token ids, from a vocabulary the clean data never
    draws from, so a clean batch shares no window with them by chance."""
    g = torch.Generator().manual_seed(seed)
    return {
        "quiz": [torch.randint(40_000, VOCAB, (30,), generator=g).tolist() for _ in range(20)],
        "maths": [torch.randint(40_000, VOCAB, (50,), generator=g).tolist() for _ in range(10)]
        + [[40_001, 40_002]],  # shorter than a window: an item nobody can find
    }


def test_the_same_window_hashes_the_same_wherever_it_is():
    ids = torch.arange(40).unsqueeze(0)
    hashes = ngram_hashes(ids)
    assert hashes.shape == (1, 28)
    shifted = ngram_hashes(torch.arange(5, 45).unsqueeze(0))
    assert torch.equal(hashes[0, 5:], shifted[0, :23])
    assert hashes.unique().numel() == 28
    assert ngram_hashes(torch.arange(12).unsqueeze(0)).shape == (1, 0)


def test_the_hash_is_fixed_across_versions():
    # An index is built once and read by every Ravex after: these two values
    # moving means every index built before finds nothing (GPU-437).
    assert int(ngram_hashes(torch.arange(13))[0]) == -708249293553874786
    assert int(ngram_hashes(torch.full((13,), 50_000))[0]) == -9123196985930428752


def test_an_index_goes_to_a_file_and_back(tmp_path):
    index = Index.build(_benchmarks(), "toy")
    write_index(index, str(tmp_path / "index.pt"))
    back = read_index(str(tmp_path / "index.pt"))
    assert back.tokenizer == "toy" and back.n == 13 and back.count == 31
    assert torch.equal(back.hashes, index.hashes) and torch.equal(back.items, index.items)
    assert [b["name"] for b in back.benchmarks] == ["quiz", "maths"]


def test_clean_data_finds_nothing_and_planted_items_are_found():
    benchmarks = _benchmarks()
    check = Check(Index.build(benchmarks, "toy"), rows=4)
    clean = torch.randint(0, 40_000, (4, 64), generator=torch.Generator().manual_seed(1))
    check.observe(clean)
    figures = check.figures()
    assert float(figures[NGRAM_OVERLAP]) == 0.0 and float(figures[BENCHMARK_OVERLAP]) == 0.0

    # Three quiz items and one maths item, inside otherwise clean rows.
    planted = clean.clone()
    planted[0, 10:40] = torch.tensor(benchmarks["quiz"][0])
    planted[1, 0:30] = torch.tensor(benchmarks["quiz"][1])
    planted[2, 20:50] = torch.tensor(benchmarks["quiz"][2])
    planted[3, 5:55] = torch.tensor(benchmarks["maths"][3])
    check.observe(planted)
    figures = check.figures()
    windows = 4 * (64 - 12)
    assert float(figures[NGRAM_OVERLAP]) == (3 * (30 - 12) + (50 - 12)) / windows
    assert float(figures[BENCHMARK_OVERLAP]) == 4 / 31
    assert float(figures[BENCHMARK_OVERLAP + "/quiz"]) == 3 / 20
    assert float(figures[BENCHMARK_OVERLAP + "/maths"]) == 1 / 11
    # The n-gram share is the interval's; the items seen are the run's.
    check.observe(clean)
    figures = check.figures()
    assert float(figures[NGRAM_OVERLAP]) == 0.0 and float(figures[BENCHMARK_OVERLAP]) == 4 / 31


def test_only_the_sampled_rows_are_read():
    benchmarks = _benchmarks()
    check = Check(Index.build(benchmarks, "toy"), rows=1)
    batch = torch.randint(0, 40_000, (2, 40))
    batch[1, :30] = torch.tensor(benchmarks["quiz"][0])
    check.observe(batch)
    assert float(check.figures()[BENCHMARK_OVERLAP]) == 0.0


def test_an_items_text_is_its_question_and_its_answers():
    assert item_text("gsm8k", {"question": "2+2?", "answer": "4"}) == "2+2?\n4"
    assert item_text("arc_challenge", {"question": "Q", "choices": {"text": ["a", "b"]}}) == "Q\na\nb"


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 8)
        self.head = nn.Linear(8, 1)

    def forward(self, input_ids):
        return self.head(self.embed(input_ids)).mean()


def test_a_run_shows_it_and_planted_data_makes_it_climb(storage, tmp_path, monkeypatch):
    benchmarks = _benchmarks()
    write_index(Index.build(benchmarks, "toy"), str(tmp_path / "index.pt"))
    monkeypatch.setenv("RAVEX_CONTAMINATION_INDEX", str(tmp_path / "index.pt"))
    monkeypatch.setenv("RAVEX_CONTAMINATION_EVERY", "1")

    @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, metrics_every=2, histograms_every=0)
    def train():
        torch.manual_seed(0)
        model = _LM()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        g = torch.Generator().manual_seed(2)
        for step in range(1, 7):
            ids = torch.randint(0, 40_000, (4, 64), generator=g)
            if step > 2:
                # From step 3 on, the data carries a quiz item a step.
                ids[0, :30] = torch.tensor(benchmarks["quiz"][step])
            model(input_ids=ids).backward()
            optimizer.step()
            optimizer.zero_grad()

    train()
    scalars = ravex.metrics.read(str(storage))["scalars"]
    seen = scalars[BENCHMARK_OVERLAP]["value"]
    assert seen[0] == 0.0 and seen[-1] > seen[0]
    assert scalars[BENCHMARK_OVERLAP + "/quiz"]["value"][-1] == 4 / 20
    assert max(scalars[NGRAM_OVERLAP]["value"]) > 0
