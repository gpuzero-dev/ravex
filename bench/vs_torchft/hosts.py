"""The machines a benchmark arm runs on, and the four things the controller does to them.

The harness rents nothing. It is given a list of hosts that are already up and
reachable - over SSH, or this machine itself for the CPU tests - in a JSON
file::

    {"hosts": [
      {"name": "eu-0", "ssh": "root@203.0.113.7 -p 40022 -i ~/.ssh/bench",
       "address": "10.65.0.2", "region": "eu", "workdir": "/workspace/bench"},
      {"name": "local", "local": true, "address": "127.0.0.1", "workdir": "/tmp/bench"}
    ]}

``address`` is where the *other hosts* reach this one - on RunPod, the pod's
Global Networking address, not the SSH endpoint - and is what each framework
is told to advertise. ``ssh`` is how the operator's machine reaches it, as the
arguments ``ssh`` takes before the command.

What the controller needs from a host is to start a process in the
background, to kill that process the way a preemption would, to read a file
from an offset, and to copy a file home. Nothing else, so a provider's own
API never has to be part of the harness: a preemption by the provider is
the operator's hook, run beside the kill.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Process:
    """A process the controller started, enough to kill it later."""

    host: "Host"
    pid: int
    log: str
    popen: Optional[subprocess.Popen] = None

    def alive(self) -> bool:
        return self.host.alive(self)


@dataclass
class Host:
    name: str
    address: str
    workdir: str
    region: str = ""
    ssh: str = ""
    local: bool = False
    #: Prepended to every command on this host, after the environment: the
    #: interpreter of a virtual environment, say ("/workspace/venv/bin/").
    python: str = "python"
    extra: Dict[str, str] = field(default_factory=dict)

    # The four operations, local or over SSH.

    def start(self, command: List[str], env: Dict[str, str], log: str) -> Process:
        """Start ``command`` in the background, in a session of its own, output to ``log``.

        A session of its own, so a kill takes the whole process group - a
        trainer and anything it started - as a preemption takes the machine.
        """
        log = self.path(log)
        if self.local:
            os.makedirs(os.path.dirname(log), exist_ok=True)
            out = open(log, "ab")
            popen = subprocess.Popen(
                command, cwd=self.workdir, env={**os.environ, **env}, stdout=out, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            out.close()
            return Process(self, popen.pid, log, popen)
        assignments = " ".join("%s=%s" % (k, shlex.quote(v)) for k, v in env.items())
        script = "mkdir -p %s && cd %s && %s setsid nohup %s >> %s 2>&1 < /dev/null & echo $!" % (
            shlex.quote(os.path.dirname(log)), shlex.quote(self.workdir), ("env " + assignments) if assignments else "",
            shlex.join(command), shlex.quote(log),
        )
        pid = int(self._ssh(script).strip().splitlines()[-1])
        return Process(self, pid, log)

    def kill(self, process: Process) -> None:
        """SIGKILL the process's group: no SIGTERM, no final checkpoint, as when a spot machine is taken."""
        if self.local:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.popen is not None:
                process.popen.wait()
            return
        self._ssh("kill -9 -- -%d 2>/dev/null || kill -9 %d 2>/dev/null || true" % (process.pid, process.pid))

    def alive(self, process: Process) -> bool:
        if self.local:
            return process.popen is not None and process.popen.poll() is None
        return self._ssh("kill -0 %d 2>/dev/null && echo yes || echo no" % process.pid).strip() == "yes"

    def read(self, path: str, offset: int = 0) -> bytes:
        """The bytes of ``path`` from ``offset`` on; empty when it does not exist yet."""
        path = self.path(path)
        if self.local:
            try:
                with open(path, "rb") as handle:
                    handle.seek(offset)
                    return handle.read()
            except FileNotFoundError:
                return b""
        return self._ssh_bytes("tail -c +%d %s 2>/dev/null || true" % (offset + 1, shlex.quote(path)))

    def fetch(self, path: str, destination: str) -> bool:
        """Copy ``path`` home to ``destination``. False when it is not there."""
        data = self.read(path)
        if not data:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        with open(destination, "wb") as handle:
            handle.write(data)
        return True

    def path(self, relative: str) -> str:
        return relative if os.path.isabs(relative) or relative.startswith("/") else os.path.join(self.workdir, relative).replace("\\", "/")

    def _ssh_bytes(self, script: str) -> bytes:
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"] + shlex.split(self.ssh) + [script]
        for attempt in range(3):
            done = subprocess.run(command, capture_output=True, timeout=120)
            # 255 is ssh's own failure, a dropped connection; anything else
            # is the remote command's answer and is returned as it is.
            if done.returncode != 255:
                return done.stdout
            time.sleep(2 * (attempt + 1))
        raise RuntimeError("ssh to %s failed: %s" % (self.name, done.stderr.decode(errors="replace").strip()))

    def _ssh(self, script: str) -> str:
        return self._ssh_bytes(script).decode("utf-8", errors="replace")


def load(path: str) -> Dict[str, Host]:
    """The hosts of a ``hosts.json``, by name."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    hosts: Dict[str, Host] = {}
    for entry in raw["hosts"]:
        known = {k: entry[k] for k in ("name", "address", "workdir", "region", "ssh", "local", "python") if k in entry}
        extra = {k: str(v) for k, v in entry.items() if k not in known}
        host = Host(**known, extra=extra)
        if not host.local and not host.ssh:
            raise ValueError("host %s has neither \"ssh\" nor \"local\": true" % host.name)
        hosts[host.name] = host
    return hosts
