#!/bin/sh
# Copy the mounted checkouts in without their virtual environments and data,
# install Ravex from the copy, and run the harness's tests (arguments go to
# pytest). OMP_NUM_THREADS=1 because several nodes share the container's cores.
set -e
mkdir -p /work/ravex /work/deepseek-1b
tar -C /mnt/ravex --exclude=./.venv --exclude=./.git --exclude='*/__pycache__' --exclude=./bench/vs_torchft/results -cf - . | tar -C /work/ravex -xf -
tar -C /mnt/deepseek-1b --exclude=./.venv --exclude=./.git --exclude=./data --exclude=./runs -cf - . | tar -C /work/deepseek-1b -xf -
pip install --no-deps --no-build-isolation -q /work/ravex
cd /work/ravex/bench/vs_torchft
export DEEPSEEK_1B=/work/deepseek-1b OMP_NUM_THREADS=1
exec python -m pytest tests -q "$@"
