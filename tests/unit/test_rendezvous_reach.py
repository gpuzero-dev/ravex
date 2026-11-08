"""GPU-242: a node that cannot reach the rendezvous says so in minutes.

On 2026-10-18 node 1 of two RTX 4090 pods never reached node 0's
rendezvous, twice; node 0 waited the whole ``outer_deadline`` (15 minutes)
and the run failed with a generic error, two GPUs paid for nothing. What a
node now tells apart, before it becomes a client of the server: a dial that
is refused, or a name that does not resolve, is a server or a machine not up
yet, and is waited for; dials that go unanswered for ``silence`` seconds are
a network that does not carry this pair, raised as ``RegionUnreachable`` -
the name a launcher matches to send the run back to the queue (GPU-233).

The dials are faked: these tests pin the classification, not a network.
"""

import socket
import time

import pytest

from ravex._dist import rendezvous
from ravex._dist.exchange import RegionUnreachable


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(rendezvous, "DIAL_RETRY", 0.01)


def dials(monkeypatch, *outcomes):
    """Each dial takes the next outcome - an exception to raise, or ``"ok"``
    - and the last one repeats. Returns the list of dials made."""
    made = []

    class Sock:
        def close(self):
            pass

        def shutdown(self, how):
            pass

    def create_connection(address, timeout=None):
        outcome = outcomes[min(len(made), len(outcomes) - 1)]
        made.append(address)
        if outcome == "ok":
            return Sock()
        raise outcome

    monkeypatch.setattr(rendezvous.socket, "create_connection", create_connection)
    return made


def test_a_rendezvous_that_never_answers_is_the_machines_being_apart(monkeypatch):
    dials(monkeypatch, socket.timeout("timed out"))
    with pytest.raises(RegionUnreachable) as raised:
        rendezvous.reach("10.0.0.1:29400", time.monotonic() + 30, silence=0.2)
    # The name is what a launcher matches, on the traceback's last line.
    assert type(raised.value).__name__ == "RegionUnreachable"
    assert "10.0.0.1:29400" in str(raised.value) and "other machines" in str(raised.value)


def test_a_refusal_is_a_server_not_up_yet_and_is_waited_for(monkeypatch):
    made = dials(monkeypatch, ConnectionRefusedError("refused"))
    started = time.monotonic()
    rendezvous.reach("10.0.0.1:29400", started + 0.3, silence=0.05)
    assert time.monotonic() - started >= 0.3 and len(made) > 1


def test_a_name_that_does_not_resolve_yet_is_a_machine_not_made_yet(monkeypatch):
    dials(monkeypatch, socket.gaierror("Name or service not known"))
    rendezvous.reach("node-0.runpod.internal:29400", time.monotonic() + 0.2, silence=0.05)


def test_an_answer_ends_the_silence(monkeypatch):
    """Silence that a refusal breaks starts again from zero: a node 0 that
    is booting answers in between, and the pair is not apart."""
    pattern = [socket.timeout("t")] * 3 + [ConnectionRefusedError("r")]
    dials(monkeypatch, *(pattern * 50))
    rendezvous.reach("10.0.0.1:29400", time.monotonic() + 0.4, silence=0.1)


def test_a_server_that_answers_is_reached_at_once(monkeypatch):
    made = dials(monkeypatch, ConnectionRefusedError("r"), "ok")
    rendezvous.reach("10.0.0.1:29400", time.monotonic() + 30)
    assert len(made) == 2


def test_past_the_deadline_the_connect_says_what_it_always_said(monkeypatch):
    """A server never up before ``outer_deadline`` is not this: the connect
    that follows fails as it did, so nothing is called apart that was late."""
    dials(monkeypatch, ConnectionRefusedError("r"))
    rendezvous.reach("10.0.0.1:29400", time.monotonic() - 1)
