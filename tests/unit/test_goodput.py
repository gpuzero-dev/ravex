"""The goodput accounting on its own (GPU-522): what counts, what follows the
model through a resume, and what a crash's lost tail adds."""

import pytest

from ravex._goodput import COST_PER_MILLION_TOKENS, GOODPUT, GOODPUT_TOKENS_PER_SECOND, TOKENS_TOTAL, Goodput


def test_time_spent_on_no_step_is_not_goodput():
    goodput = Goodput()
    goodput.add(10.0, lost=10.0, tokens=0, price_per_hour=None)  # the start
    goodput.add(30.0, lost=5.0, tokens=3000, price_per_hour=None)  # steps and one save
    figures = goodput.metrics(gpus=1)
    assert figures[GOODPUT] == pytest.approx(25.0 / 40.0)
    assert figures[GOODPUT_TOKENS_PER_SECOND] == pytest.approx(3000 / 40.0)
    assert figures[TOKENS_TOTAL] == 3000.0
    assert COST_PER_MILLION_TOKENS not in figures


def test_the_cost_is_the_node_s_over_the_tokens_of_its_gpus():
    goodput = Goodput()
    # Two hours at $4 an hour, a million tokens on each of the two GPUs.
    goodput.add(7200.0, lost=0.0, tokens=1_000_000, price_per_hour=4.0)
    assert goodput.metrics(gpus=2)[COST_PER_MILLION_TOKENS] == pytest.approx(4.0)


def test_a_resume_gets_the_sums_of_its_checkpoint_and_the_lost_tail(tmp_path):
    before = Goodput()
    before.add(100.0, lost=0.0, tokens=1000, price_per_hour=1.0)
    at_checkpoint = before.state()  # step 500
    before.add(60.0, lost=0.0, tokens=600, price_per_hour=1.0)
    before.write(str(tmp_path), step=700)  # the crash comes after this

    after = Goodput()
    after.restore(at_checkpoint)
    assert after.count_lost_tail(str(tmp_path), resumed_at=500) == pytest.approx(60.0)
    # The tail's time and cost stay, its tokens do not: they are not in the model.
    assert after.wall == pytest.approx(160.0) and after.productive == pytest.approx(100.0)
    assert after.tokens == 1000 and after.cost == pytest.approx(160.0 / 3600)


def test_no_file_or_a_file_behind_the_resume_adds_nothing(tmp_path):
    goodput = Goodput()
    assert goodput.count_lost_tail(str(tmp_path), resumed_at=500) == 0.0
    goodput.write(str(tmp_path), step=400)
    assert goodput.count_lost_tail(str(tmp_path), resumed_at=500) == 0.0
