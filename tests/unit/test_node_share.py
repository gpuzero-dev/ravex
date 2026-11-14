"""GPU-417: in a run on several nodes each node reads data of its own, and the
script that builds the DataLoader is not told which node it is on.

The split is made under the DataLoader Ravex already tracks, so these build
real loaders through `attach_tracker` with the settings a node of the
platform gets (`outer_member`, `outer_min_nodes`) and read the indices out.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from ravex._patches import attach_tracker  # noqa: E402
from ravex._sampler import NodeShare, node_place  # noqa: E402

SAMPLES = 100
BATCH = 4


def settings(member=None, nodes=2, outer=True, job="run-1"):
    return SimpleNamespace(
        outer_loop=outer, outer_member=member, outer_min_nodes=nodes, outer_job=job
    )


def loader(config, generator=None, shuffle=True):
    built = DataLoader(range(SAMPLES), batch_size=BATCH, shuffle=shuffle, generator=generator)
    attach_tracker(built, config)
    return built


def read(built, batches=None):
    seen = []
    for count, batch in enumerate(built):
        if batches is not None and count >= batches:
            break
        seen.extend(int(index) for index in batch)
        built._ravex_sampler.note_consumed()
    return seen


def test_two_starting_nodes_read_disjoint_halves_of_the_epoch():
    first = read(loader(settings(member=0)))
    second = read(loader(settings(member=1)))
    assert not set(first) & set(second)
    assert sorted(first + second) == list(range(SAMPLES))


def test_a_generator_the_script_seeded_is_kept_and_still_splits():
    first = read(loader(settings(member=0), torch.Generator().manual_seed(7)))
    second = read(loader(settings(member=1), torch.Generator().manual_seed(7)))
    assert not set(first) & set(second)
    assert sorted(first + second) == list(range(SAMPLES))


def test_a_node_alone_reads_what_it_always_read():
    untouched = DataLoader(
        range(SAMPLES), batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(3)
    )
    plain = [int(index) for batch in untouched for index in batch]
    for config in (None, settings(outer=False), settings(member=0, nodes=1)):
        assert read(loader(config, torch.Generator().manual_seed(3))) == plain


def test_a_node_without_a_place_reads_all_the_data_in_its_own_order():
    joiner = read(loader(settings(member=None), torch.Generator().manual_seed(7)))
    starters = read(loader(None, torch.Generator().manual_seed(7)))
    assert sorted(joiner) == list(range(SAMPLES))
    assert joiner != starters


def test_a_resumed_node_carries_on_in_its_own_part():
    whole = read(loader(settings(member=1)))
    first = loader(settings(member=1))
    before = read(first, batches=3)
    state = first._ravex_sampler.state()
    again = loader(settings(member=1))
    again._ravex_sampler.restore(state)
    assert before + read(again) == whole


def test_the_parts_come_from_the_settings_a_node_is_given():
    assert node_place(settings(member=1, nodes=3)) == (1, 3)
    assert node_place(settings(member="2", nodes=3)) == (2, 3)
    assert node_place(settings(member=5, nodes=3)) == (None, 3)
    assert node_place(settings(member=None, nodes=3)) == (None, 3)
    assert node_place(settings(member=0, nodes=1)) is None
    assert node_place(settings(member=0, outer=False)) is None


def test_a_distributed_sampler_is_the_scripts_own_split_and_left_alone():
    from torch.utils.data.distributed import DistributedSampler

    sampler = DistributedSampler(range(SAMPLES), num_replicas=2, rank=0)
    built = DataLoader(range(SAMPLES), batch_size=BATCH, sampler=sampler)
    attach_tracker(built, settings(member=1))
    assert not isinstance(built.batch_sampler.sampler, NodeShare)


def test_a_part_has_the_length_of_the_positions_it_keeps():
    sampler = list(range(10))
    assert [len(NodeShare(sampler, part, 3)) for part in range(3)] == [4, 3, 3]
    assert [list(NodeShare(sampler, part, 3)) for part in range(3)] == [
        [0, 3, 6, 9], [1, 4, 7], [2, 5, 8]
    ]
