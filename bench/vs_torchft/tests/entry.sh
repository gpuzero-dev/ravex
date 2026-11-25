#!/bin/sh
# Copy the mounted checkout in without its virtual environment and results,
# install Ravex from the copy, and run the harness's tests (arguments go to
# pytest). OMP_NUM_THREADS=1 because several nodes share the container's cores.
set -e
mkdir -p /work/ravex
tar -C /mnt/ravex --exclude=./.venv --exclude=./.git --exclude='*/__pycache__' --exclude=./bench/vs_torchft/results -cf - . | tar -C /work/ravex -xf -
pip install --no-deps --no-build-isolation -q /work/ravex
cd /work/ravex/bench/vs_torchft
export OMP_NUM_THREADS=1
exec python -m pytest tests -q "$@"
