"""GPU-121: a node that joins a run already in progress.

``test_dist_round.py`` proves the other half of GPU-113's point 4 — a node dies
and the round closes over whoever is left. This is the arrival, and the two are
not symmetric. A departure is safe to notice late: the node that went away is
missing from *everybody's* list of reports at once, because there is nothing to
miss. An arrival noticed by one node and not another is two nodes applying
different averages to the same parameters, for ever, with the loss falling on
both and nothing raising — which is why the interesting assertions here are not
about the newcomer training well, they are about all three holding **one
model**, checked with ``torch.equal`` and not ``allclose``.

Threads, one per node, for the same reason ``test_dist_round.py`` gives: a node
blocks inside ``close_round`` until its peers publish, so driving them in
sequence would be testing a shape the real thing never has.
"""

import threading
import time

import pytest

torch = pytest.importorskip("torch")


from ravex._dist import membership as _membership  # noqa: E402

from test_dist_exchange import FakeStore  # noqa: E402

BASE_WORLD = 2


def test_a_run_every_member_has_left_is_not_waited_for():
    """A node rejoining a run whose nodes have all finished gives up at once.

    On rented machines it waited out its whole deadline - fifteen minutes,
    billed - for members that had ended and would never take it in (GPU-186).
    """
    store = FakeStore()
    for member in range(BASE_WORLD):
        _membership.leave(store, member, 7)
    began = time.monotonic()
    assert not _membership.join(store, None, None, BASE_WORLD, BASE_WORLD, time.monotonic() + 60)
    assert time.monotonic() - began < 5


def test_one_member_still_there_is_waited_for():
    store = FakeStore()
    for member in range(1, BASE_WORLD):
        _membership.leave(store, member, 7)
    began = time.monotonic()
    # Nobody serves the run's state, so it waits to its (short) deadline.
    assert not _membership.join(store, None, None, BASE_WORLD, BASE_WORLD, time.monotonic() + 2)
    assert time.monotonic() - began >= 2


class Clock:
    """A member's clock, moved by hand: the window is in seconds now (GPU-263)."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def a_member(store, clock):
    return _membership.Membership(store, 0, BASE_WORLD, clock=clock)


def test_a_candidate_still_beating_is_served_however_many_rounds_go_by():
    """The flake in GPU-263, as arithmetic.

    The window used to be four rounds from the first sight of the candidate.
    In ``test_dist_join_nodes`` a round is 16-20 ms and the candidate's first
    look at the store comes after a 50 ms sleep, so the four were sometimes
    over before it had read a round: it announced one nobody wrote state for.
    A candidate that keeps beating is served for as many rounds as it takes.
    """
    store, clock = FakeStore(), Clock()
    member = a_member(store, clock)
    _membership.want_in(store, BASE_WORLD)
    for round_number in range(100):
        _membership.beat(store, BASE_WORLD, round_number // 5)
        assert member.wants_state(round_number), round_number
        clock.now += 1.0


def test_a_candidate_that_went_quiet_stops_costing_snapshots_and_can_come_back():
    store, clock = FakeStore(), Clock()
    member = a_member(store, clock)
    _membership.want_in(store, BASE_WORLD)
    _membership.beat(store, BASE_WORLD, 0)

    served = []
    for round_number in range(60):
        served.append(member.wants_state(round_number))
        clock.now += 1.0
    # Served through the grace and not a round past it.
    assert served == [True] * (int(_membership.JOIN_GRACE) + 1) + [False] * (
        60 - int(_membership.JOIN_GRACE) - 1
    )

    _membership.beat(store, BASE_WORLD, 1)
    assert member.wants_state(60)


def test_the_window_stays_open_through_the_round_a_candidate_announced():
    """A candidate slow between its announcement and its round still finds
    the state written for that round, rather than costing every member a
    gather deadline for a round it cannot take part in."""
    store, clock = FakeStore(), Clock()
    member = a_member(store, clock)
    _membership.want_in(store, BASE_WORLD)
    assert member.wants_state(10)

    _membership.announce(store, BASE_WORLD, 14)
    member.refresh(11)
    clock.now += 10 * _membership.JOIN_GRACE
    assert [member.wants_state(r) for r in range(11, 17)] == [
        True, True, True, True, False, False
    ]


def test_a_candidate_that_is_in_is_not_served():
    store, clock = FakeStore(), Clock()
    member = a_member(store, clock)
    _membership.want_in(store, BASE_WORLD)
    assert member.wants_state(3)
    _membership.gave_up(store, BASE_WORLD)
    assert not member.wants_state(4)


def test_a_run_that_ended_while_this_node_was_out_finishes_the_training(tmp_path, monkeypatch):
    """`RunOver` ends the training as finished, not as a failure (GPU-186)."""
    import ravex

    monkeypatch.chdir(tmp_path)
    steps = []

    @ravex.train_loop(storage={"type": "local", "path": str(tmp_path / "store")}, checkpoint_every=10 ** 9)
    def train():
        steps.append(1)
        raise _membership.RunOver("the run ended while node 2 was out of it; every member has finished")

    assert train() is None
    assert steps == [1]


def test_a_script_catching_exceptions_does_not_swallow_run_over():
    """A BaseException, like KeyboardInterrupt: a script's own `except
    Exception` around its step must not catch it and train on alone."""
    assert not issubclass(_membership.RunOver, Exception)


class Loop:
    """What `serve_joins` reads off an outer loop: its round, and a state."""

    def __init__(self, round_number):
        self.round_number = round_number
        self.outer = {"w": torch.zeros(2)}
        self.optimizer = None
        self.model_steps = 0


class Exchange:
    """A member's exchange as far as `serve_joins` uses it: the state goes nowhere."""

    def publish_state(self, payload, round_number):
        pass


def test_a_member_that_takes_a_joiner_in_has_acknowledged_it_before_it_trains_the_round():
    """GPU-279. A round boundary reads the announcements twice: in
    `serve_joins` and again in `peers_at`. One written between the two was
    accepted by the second read, and acknowledged only at the next boundary's
    `serve_joins` - after this member had trained the round and published
    its report, which is all the other member needs to reach the announced
    round and say so on the store. The joiner then found this member's
    acknowledgement missing, published nothing, and asked again for a later
    round, every time the interleaving came round, until its deadline."""
    store = FakeStore()
    member = _membership.Membership(store, 1, BASE_WORLD)
    _membership.serve_joins(store, Exchange(), Loop(2), member)
    # The joiner announces between the boundary's two reads.
    _membership.announce(store, BASE_WORLD, 3)
    member.peers_at(2)
    assert _membership.acknowledged(store, BASE_WORLD, [1]) == []


def test_a_member_acknowledges_before_it_says_the_announced_round_has_come():
    """GPU-279. `serve_joins` wrote the round it was at for the joiner to
    read - which is how a joiner learns its round has come - before the
    acknowledgements, so a joiner could see its round and check them in
    between. The acknowledgement is on the store before the round is."""
    store = FakeStore()
    member = _membership.Membership(store, 1, BASE_WORLD)
    _membership.want_in(store, BASE_WORLD)
    _membership.announce(store, BASE_WORLD, 3)
    seen = []
    real_set = store.set

    def watching(key, value):
        if key == _membership.STATE_AT_KEY:
            seen.append(_membership.acknowledged(store, BASE_WORLD, [1]))
        real_set(key, value)

    store.set = watching
    _membership.serve_joins(store, Exchange(), Loop(2), member)
    assert seen == [[]], "the round was on the store before the acknowledgement: %s" % seen


# ─── a join over slow rounds (GPU-234) ───────────────────────────────────


class Published:
    """Where the members' exchange writes the outer parameters for a round,
    and where the joiner's reads them: one dict, for threads in one process."""

    def __init__(self):
        self.by_round = {}

    def publish_state(self, payload, round_number):
        self.by_round[round_number] = payload

    def fetch_state(self, members, round_number, expectation, deadline):
        from types import SimpleNamespace

        while time.monotonic() < deadline:
            if round_number in self.by_round:
                return SimpleNamespace(delta=self.by_round[round_number])
            time.sleep(0.01)
        return None


class Joiner(Loop):
    """A joining node's loop as far as `apply_state` uses it."""

    def __init__(self):
        from types import SimpleNamespace

        super().__init__(0)
        self.optimizer = SimpleNamespace(buffers={})
        self.written_back = False

    def write_back(self):
        self.written_back = True


def _members_at_a_pace(store, published, seconds, stop, ranks=range(BASE_WORLD)):
    """Members closing a round every ``seconds`` - by default every base
    rank: what the joiner sees of a run is its round on the store, which
    moves only at a boundary."""
    loops = [Loop(5) for _ in ranks]
    members = [_membership.Membership(store, rank, BASE_WORLD) for rank in ranks]
    while not stop.is_set():
        for loop, member in zip(loops, members):
            _membership.serve_joins(store, published, loop, member)
        stop.wait(seconds)
        for loop in loops:
            loop.round_number += 1


def test_a_join_over_rounds_longer_than_its_deadline_still_gets_in():
    """Rounds of 0.4 s and a deadline of 0.6 s: the join waits for the round
    to move, then two more, and its round's state comes at the end of that
    one - about three rounds, twice the deadline. It used to give up every
    time; it now waits the rounds the run is measured to take. The ratio,
    not the speed of the machine, is what the test holds."""
    store = FakeStore()
    published = Published()
    stop = threading.Event()
    members = threading.Thread(target=_members_at_a_pace, args=(store, published, 0.4, stop), daemon=True)
    members.start()
    try:
        joiner = Joiner()
        began = time.monotonic()
        assert _membership.join(store, published, joiner, BASE_WORLD, BASE_WORLD, began + 0.6)
        assert time.monotonic() - began > 0.6, "it got in within the deadline: the test proves nothing"
        assert joiner.round_number >= 7 and joiner.written_back
    finally:
        stop.set()
        members.join(5)


def test_a_run_that_stops_moving_is_not_waited_on_past_its_deadline():
    """The rounds measured while the run moved are no reason to wait once it
    has stopped: a run whose round never reaches the one announced ends the
    join at the deadline, plus at most the rounds one join takes."""
    store = FakeStore()
    published = Published()
    stop = threading.Event()
    members = threading.Thread(target=_members_at_a_pace, args=(store, published, 0.2, stop), daemon=True)
    members.start()
    threading.Timer(0.5, stop.set).start()
    began = time.monotonic()
    assert not _membership.join(store, published, Joiner(), BASE_WORLD, BASE_WORLD, began + 1.0)
    waited = time.monotonic() - began
    assert waited < 1.0 + (_membership.JOIN_MARGIN + 2) * 0.2 + 1.0
    members.join(5)


# ─── a run that lost a starting node (GPU-406) ───────────────────────────


def test_still_serving_is_every_member_that_marked_a_round_since_the_announcement():
    """A joiner that announced when the run was at round 8 waits for the
    members that marked round 8 or later, freshest first - one that marked
    only round 9 may have died right after, and is dialled last. One that
    stopped at round 5 has stopped passing boundaries."""
    store = FakeStore()
    _membership.served(store, 0, 5)
    _membership.served(store, 1, 9)
    _membership.served(store, 2, 10)
    _membership.served(store, 3, 8)
    assert _membership.still_serving(store, [0, 1, 2, 3], 8) == [2, 1, 3]


def test_a_member_with_no_mark_is_not_waited_for():
    """A live member marks before it writes the round a joiner reads, so one
    with no mark at all by then is not passing boundaries: it is left out
    like one whose mark is old (GPU-408)."""
    assert _membership.still_serving(FakeStore(), [0, 1], 10) == []


def test_a_member_that_serves_marks_the_round_before_writing_it():
    """A joiner reads the round, then who served it: the mark is on the store
    first, or the member that wrote the round could read as dead."""
    store = FakeStore()
    member = _membership.Membership(store, 1, BASE_WORLD)
    _membership.want_in(store, BASE_WORLD)
    seen = []
    real_set = store.set

    def watching(key, value):
        if key == _membership.STATE_AT_KEY:
            seen.append(_membership.still_serving(store, [0, 1], 7))
        real_set(key, value)

    store.set = watching
    _membership.serve_joins(store, Exchange(), Loop(7), member)
    assert seen == [[1]], seen


def test_a_join_gets_in_when_a_starting_node_has_died():
    """Rank 0 is gone without a word - no `leave` - and rank 1 carries on.
    The joiner waits for rank 1's acknowledgement alone, and gets in; it used
    to wait for rank 0's until its deadline, every time (GPU-406)."""
    store = FakeStore()
    published = Published()
    stop = threading.Event()
    members = threading.Thread(
        target=_members_at_a_pace, args=(store, published, 0.1, stop),
        kwargs=dict(ranks=[1]), daemon=True,
    )
    members.start()
    try:
        joiner = Joiner()
        assert _membership.join(
            store, published, joiner, BASE_WORLD, BASE_WORLD, time.monotonic() + 10
        )
        assert joiner.written_back
        assert not store.check([_membership.ACK_KEY % (0, BASE_WORLD)])
    finally:
        stop.set()
        members.join(5)


class Blinking:
    """One member's view of the store, which fails every call while ``down``
    is set: a network blip, not a death."""

    def __init__(self, store):
        self.store = store
        self.down = False

    def __getattr__(self, name):
        method = getattr(self.store, name)

        def call(*args, **kwargs):
            if self.down:
                raise ConnectionError("store unreachable")
            return method(*args, **kwargs)

        return call


def test_one_failed_boundary_is_a_retry_not_a_death():
    """A live member whose store calls all fail at the boundary after the
    announcement writes neither its acknowledgement nor its mark there.
    Counted from the boundary before the admit round, that read as death and
    the joiner published without it. Counted from the round announced from,
    the member's earlier mark keeps it in, and the joiner asks again instead
    (GPU-406, review round 1)."""
    store = FakeStore()
    published = Published()
    stop = threading.Event()
    blinking = Blinking(store)
    stores = [store, blinking]
    loops = [Loop(5) for _ in range(BASE_WORLD)]
    members = [
        _membership.Membership(stores[rank], rank, BASE_WORLD) for rank in range(BASE_WORLD)
    ]
    stopped = set()
    blinked = []

    def run():
        while not stop.is_set():
            # Rank 1 loses the store for exactly one boundary: the first one
            # at which the joiner's announcement is there to be read.
            announced = store.check([_membership.JOIN_KEY % BASE_WORLD])
            blinking.down = announced and not blinked
            if blinking.down:
                blinked.append(loops[1].round_number)
            for rank, (loop, member) in enumerate(zip(loops, members)):
                if rank in stopped:
                    continue
                try:
                    _membership.serve_joins(stores[rank], published, loop, member)
                except _membership.MembershipError:
                    # Meeting the announcement late is this member's own stop;
                    # what is under test is the joiner's side.
                    stopped.add(rank)
                except Exception:
                    pass
            blinking.down = False
            stop.wait(0.3)
            for loop in loops:
                loop.round_number += 1

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        joiner = Joiner()
        assert _membership.join(
            store, published, joiner, BASE_WORLD, BASE_WORLD, time.monotonic() + 15
        )
    finally:
        stop.set()
        thread.join(5)
    assert blinked, "the blip never happened: the test proves nothing"
    first = int(store.get(_membership.ACK_KEY % (0, BASE_WORLD)).decode("utf-8"))
    assert joiner.round_number != first, (
        "the joiner entered round %d, the round rank 1 never acknowledged "
        "because of one failed boundary" % first
    )


# ─── a run whose starting nodes have all been replaced (GPU-408) ─────────


def test_the_members_are_the_starting_nodes_and_every_number_that_got_in():
    """Rank 2 got in, so it wrote its mark; rank 3 announced and never did -
    a candidate that gave up, which will never acknowledge or say it left."""
    store = FakeStore()
    _membership.announce(store, 2, 5)
    _membership.served(store, 2, 5)
    _membership.announce(store, 3, 7)
    assert _membership.run_members(store, BASE_WORLD, 4) == [0, 1, 2]
    assert _membership.run_members(store, BASE_WORLD, 0) == [1, 2]


def test_a_node_that_joined_acknowledges_the_next_joiner():
    """It used to write no acknowledgement, so a joiner could only ever be
    taken in by starting nodes."""
    store = FakeStore()
    joined = _membership.Membership(store, BASE_WORLD, BASE_WORLD)
    _membership.announce(store, BASE_WORLD + 1, 6)
    _membership.serve_joins(store, Exchange(), Loop(4), joined)
    assert _membership.acknowledged(store, BASE_WORLD + 1, [BASE_WORLD]) == []


def test_a_join_marks_the_round_it_entered_at():
    """The mark is what makes the node a member for the next joiner."""
    store = FakeStore()
    published = Published()
    stop = threading.Event()
    members = threading.Thread(
        target=_members_at_a_pace, args=(store, published, 0.1, stop), daemon=True,
    )
    members.start()
    try:
        joiner = Joiner()
        assert _membership.join(
            store, published, joiner, BASE_WORLD, BASE_WORLD, time.monotonic() + 10
        )
        mark = store.get(_membership.SERVED_KEY % BASE_WORLD).decode("utf-8")
        assert int(mark) == joiner.round_number
    finally:
        stop.set()
        members.join(5)


def test_a_join_gets_in_when_every_starting_node_has_been_replaced():
    """Rank 0 left the run and said so, rank 1 died without a word, and rank
    2, which joined in their place, is the run. A new node gets in from rank
    2 alone. It used to give up at once, reading a run whose starting nodes
    had all left as a run that was over, or else wait for state only
    starting nodes served until its deadline (GPU-408)."""
    store = FakeStore()
    _membership.leave(store, 0, 4)
    _membership.announce(store, BASE_WORLD, 3)
    _membership.served(store, BASE_WORLD, 3)
    published = Published()
    stop = threading.Event()
    members = threading.Thread(
        target=_members_at_a_pace, args=(store, published, 0.1, stop),
        kwargs=dict(ranks=[BASE_WORLD]), daemon=True,
    )
    members.start()
    newcomer = BASE_WORLD + 1
    try:
        joiner = Joiner()
        assert _membership.join(
            store, published, joiner, newcomer, BASE_WORLD, time.monotonic() + 10
        )
        assert joiner.written_back
        assert store.check([_membership.ACK_KEY % (BASE_WORLD, newcomer)])
    finally:
        stop.set()
        members.join(5)
