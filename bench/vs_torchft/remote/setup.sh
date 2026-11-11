#!/bin/sh
# Prepare one rented GPU machine for the benchmark, from three archives the
# operator copied into /workspace:
#
#   ravex.tar        git -C ravex archive -o ravex.tar <commit>
#   deepseek-1b.tar  git -C deepseek-1b archive -o deepseek-1b.tar <commit>
#   data.tar         train.bin, val.bin and meta.json from deepseek-1b's prepare.py
#
# Archives rather than a clone, so a machine runs exactly the commits the
# results name, whether or not they have been pushed, and needs no access to
# anything but the operator's copy. The virtual environment sees the image's
# torch (built for its CUDA) instead of downloading another.
#
# It ends by printing what the go/no-go needs from a machine before anything
# is started: the private-network address, the GPU and its memory, the
# versions, and the data's hashes, which must be the same on every machine.
set -eu
cd /workspace
for name in ravex deepseek-1b data; do
    mkdir -p "$name"
    tar -xf "$name.tar" -C "$name"
done
python3 -m venv --system-site-packages venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q ./ravex -r ravex/bench/vs_torchft/requirements.txt
echo "== private address (a 10.x on podnet1 is Global Networking)"
hostname -I
echo "== gpu"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== versions"
venv/bin/python -c "import torch, torchft, ravex, moonclip; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchft', torchft.__file__); print('ravex', ravex.__version__)"
ls -l venv/bin/torchft_lighthouse venv/bin/ravex
echo "== data"
sha256sum data/train.bin data/val.bin
