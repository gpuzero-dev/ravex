"""The harness's tests: the controller and the collector anywhere, the runs on Linux only.

The runs start a rendezvous or a lighthouse and several training processes
talking over sockets, and their verdict comes from the Linux container
(``Dockerfile.cpu``), as every multi-process test of Ravex's does: on Windows
the children of a failed test outlive it and break the next one.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
