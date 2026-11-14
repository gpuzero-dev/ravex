"""GPU-417: two nodes of one run, each reading a shuffled DataLoader the script
built as it would on one node, read no index in common.

The script below knows nothing of nodes: it builds a DataLoader with
``shuffle=True`` and trains. The node's place comes from the settings the
platform gives a starting node (``outer_member``), as on a real run, and the
outer loop is the real one, over a rendezvous of Ravex's own.
"""

import multiprocessing as mp
import os
import sys

import pytest

torch = pytest.importorskip("torch")

# Fixtures, asked for by parameter: a linter sees them unused, pytest does not.
from test_outer_rendezvous import server  # noqa: E402, F401, F811

SAMPLES = 400
BATCH = 4
STEPS = 40


def a_reader(address, job, member, root, queue):
    try:
        for variable in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(variable, None)
        import ravex

        workdir = os.path.join(root, "node%d" % member)
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

        @ravex.train_loop(
            outer_loop=True,
            # One place per starting node, as the platform lists them: a
            # node's place is its number, and only a place in the list is
            # one (GPU-237). Both on the one server, which never dies here.
            outer_rendezvous="%s,%s" % (address, address),
            outer_job=job,
            outer_min_nodes=2,
            outer_member=member,
            outer_inner_steps=10,
            outer_root=os.path.join(workdir, "rounds"),
            outer_deadline=60,
            enabled=True,
            resume=False,
            delta=False,
            checkpoint_every=10**9,
            checkpoint_on_exit=False,
        )
        def train():
            # What a script does on one node, and nothing else.
            data = torch.utils.data.TensorDataset(torch.arange(SAMPLES).float())
            loader = torch.utils.data.DataLoader(data, batch_size=BATCH, shuffle=True)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            seen = []
            for step, (batch,) in enumerate(loader):
                if step >= STEPS:
                    break
                seen.extend(int(value) for value in batch)
                optimizer.zero_grad()
                model(batch.unsqueeze(1)).pow(2).mean().backward()
                optimizer.step()
            return seen

        queue.put((member, train(), None))
    except BaseException:
        import traceback

        queue.put((member, None, traceback.format_exc()))


@pytest.mark.skipif(sys.platform == "darwin", reason="spawn is slow on macOS")
def test_two_nodes_of_a_run_read_no_index_in_common(server, tmp_path):
    context = mp.get_context("spawn")
    queue = context.Queue()
    nodes = [
        context.Process(target=a_reader, args=(server, "split", member, str(tmp_path), queue))
        for member in (0, 1)
    ]
    for node in nodes:
        node.start()
    try:
        results = {}
        for _ in nodes:
            member, seen, failure = queue.get(timeout=240)
            assert failure is None, "node %d failed:\n%s" % (member, failure)
            results[member] = seen
    finally:
        for node in nodes:
            node.join(60)
            if node.is_alive():
                node.kill()
                node.join(10)

    assert len(results[0]) == len(results[1]) == STEPS * BATCH
    assert not set(results[0]) & set(results[1]), "the two nodes read the same samples"
