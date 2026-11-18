"""The default pre-training metrics: perplexity, throughput, MFU, and what to chart per layer (GPU-436).

A script with nothing but the decorator should see what a pre-training run is
judged by, without computing any of it itself. Everything here is about the
model and the machine and nothing about the script's loop, so it lives apart
from the runtime that calls it.

**Perplexity** is ``exp`` of the loss the script logs as ``train/loss`` or
``val/loss``: Ravex cannot see the loss on its own (that would mean wrapping
``backward()``), but when the script logs it under the names charts look
for, the perplexity comes with it. Queued on the device like the loss
itself, never read back on the training thread.

**Throughput** counts tokens at the model's door: the forward pre-hook that
already marks the top of each iteration reads the shape of the first integer
tensor the model is called with, the token ids, at every forward in training.
A shape is metadata, so this never waits for the GPU, and it counts every
micro-batch of an accumulation. A model fed no integer tensor (an image model,
embeddings computed outside) gets no throughput, rather than a guess.

**MFU** is the FLOPs the model needs per token, times the tokens per second,
over what the GPU can do. The FLOPs are the usual estimate: ``6 x`` the
parameters that multiply (every trainable one but an embedding's lookup
table, which is an index, not a product), plus the attention's ``12 x layers
x width x sequence`` when the model says its layers and width the way Hugging
Face configurations do. The GPU's peak comes from :data:`PEAKS`, per GPU model
and per compute dtype, and each parameter is charged at the peak of the dtype
it is multiplied in: its own, or the autocast dtype when the forward runs
under autocast. A model whose components run in different dtypes - fp8
experts beside bf16 attention - gets the peak of that mix, not of one of them.
That is the compute dtype, which is not the dtype a component is *saved* at
(``ravex.save_dtype``): an fp8 checkpoint of weights trained in bf16 was
computed at bf16's peak. A GPU missing from the table, or a dtype it has no
peak for, gets no MFU: a number made up is worse than none.

**Per layer** names the parameters worth charting in a transformer: the
embedding; attention and MLP of the first, middle and last block; the final
norm and the output head. Found by structure, not by a list of model names:
the blocks are the longest ``ModuleList``, attention and MLP are their
submodules named the way every common implementation names them. A model
without such a list falls back to its largest parameters.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("ravex")

#: The names whose perplexity Ravex logs beside them, and under what name.
PERPLEXITY = {"train/loss": "train/ppl", "val/loss": "val/ppl"}

TOKENS_PER_SECOND = "perf/tokens_per_s_per_gpu"
MFU = "perf/mfu"

#: Dense peak TFLOPS per GPU model and compute dtype, from each vendor's
#: datasheet: without sparsity, and for GeForce cards with the fp32
#: accumulation torch's matmuls use. ``match`` is tested in order against
#: ``torch.cuda.get_device_name()``, lowercased, so a more specific name comes
#: before the one it contains (``h100 nvl`` is not an SXM H100). A GPU whose
#: datasheet does not give a number for a dtype has no entry for it.
PEAKS: Sequence[Tuple[str, Mapping[str, float]]] = (
    # NVL and the GH200's are left out on purpose: their published figures
    # vary with the configuration, and a wrong peak is a wrong MFU.
    ("h100 nvl", {}),
    ("h100 pcie", {"bf16": 756.5, "fp16": 756.5, "fp8": 1513.0, "tf32": 378.0, "fp32": 51.2}),
    ("h100", {"bf16": 989.4, "fp16": 989.4, "fp8": 1978.9, "tf32": 494.7, "fp32": 66.9}),
    ("h200", {"bf16": 989.4, "fp16": 989.4, "fp8": 1978.9, "tf32": 494.7, "fp32": 66.9}),
    ("a100", {"bf16": 312.0, "fp16": 312.0, "tf32": 156.0, "fp32": 19.5}),
    ("l40s", {"bf16": 362.1, "fp16": 362.1, "fp8": 733.0, "tf32": 183.0, "fp32": 91.6}),
    ("a40", {"bf16": 149.7, "fp16": 149.7, "tf32": 74.8, "fp32": 37.4}),
    ("rtx 4090", {"bf16": 165.2, "fp16": 165.2, "tf32": 82.6, "fp32": 82.6}),
    ("rtx 3090", {"bf16": 71.0, "fp16": 71.0, "tf32": 35.6, "fp32": 35.6}),
    ("mi300x", {"bf16": 1307.4, "fp16": 1307.4, "fp8": 2614.9, "tf32": 653.7, "fp32": 163.4}),
)

#: Exact names, for the GPUs whose name is a prefix of another's: an ``A10``
#: is not an ``A100``, and an ``L4`` is not an ``L40S``.
EXACT_PEAKS: Mapping[str, Mapping[str, float]] = {
    "nvidia a10": {"bf16": 125.0, "fp16": 125.0, "tf32": 62.5, "fp32": 31.2},
    "nvidia l4": {"bf16": 121.0, "fp16": 121.0, "fp8": 242.5, "tf32": 60.0, "fp32": 30.3},
}


def peaks_for(device_name: Optional[str]) -> Optional[Mapping[str, float]]:
    """The peak TFLOPS per dtype of the GPU called ``device_name``, or None when unknown."""
    if not device_name:
        return None
    name = " ".join(device_name.lower().split())
    if name in EXACT_PEAKS:
        return EXACT_PEAKS[name]
    for match, peaks in PEAKS:
        if match in name:
            return peaks or None
    return None


def dtype_key(dtype: Any, tf32: bool = False) -> Optional[str]:
    """The :data:`PEAKS` key a torch dtype is computed at, or None for one with no tensor-core peak."""
    text = str(dtype).replace("torch.", "")
    if text == "bfloat16":
        return "bf16"
    if text == "float16":
        return "fp16"
    if text.startswith("float8"):
        return "fp8"
    if text == "float32":
        return "tf32" if tf32 else "fp32"
    return None


# ─── perplexity ──────────────────────────────────────────────────────


def with_perplexity(values: Dict[str, Any]) -> Dict[str, Any]:
    """``values`` with ``train/ppl`` and ``val/ppl`` added beside the losses it holds.

    Prepared values in, prepared values out: a :class:`ravex._metrics.Scalar`
    whose pending value may be a device tensor, so the ``exp`` is queued work
    too. A perplexity the script logged itself is kept as it is.
    """
    from ravex._metrics import Scalar

    out = values
    for loss, ppl in PERPLEXITY.items():
        value = values.get(loss)
        if not isinstance(value, Scalar) or ppl in values:
            continue
        derived = _exp(value.pending)
        if derived is None:
            continue
        if out is values:
            out = dict(values)
        out[ppl] = Scalar(derived)
    return out


def _exp(pending: Any) -> Any:
    if hasattr(pending, "exp"):
        # A tensor: exp on the device, in float32 at least, so a bf16 loss
        # of 10 is not a perplexity rounded to the nearest 64.
        try:
            return pending.float().exp()
        except Exception:
            return None
    try:
        return math.exp(float(pending))
    except OverflowError:
        return math.inf
    except (TypeError, ValueError):
        return None


def mean_perplexity(means: Dict[str, float]) -> Dict[str, float]:
    """Perplexity out of a mean loss, for the model's series of a multi-node run.

    The mean of the nodes' perplexities is not the perplexity of the mean
    loss, and the second is the one that says what the model does.
    """
    for loss, ppl in PERPLEXITY.items():
        if loss in means:
            try:
                means[ppl] = math.exp(means[loss])
            except OverflowError:
                means[ppl] = math.inf
    return means


# ─── tokens ──────────────────────────────────────────────────────────


def token_ids(args: Sequence[Any], kwargs: Optional[Mapping[str, Any]]) -> Any:
    """The token ids a forward was called with: the first integer tensor, or None.

    ``input_ids`` by name first, as Hugging Face models are called, then the
    positional arguments in order. Integer means the ids of a vocabulary;
    floats are already embeddings or pixels, and have no tokens to count.
    """
    if kwargs:
        candidate = kwargs.get("input_ids")
        if _is_ids(candidate):
            return candidate
    for value in args:
        if _is_ids(value):
            return value
    if kwargs:
        for value in kwargs.values():
            if _is_ids(value):
                return value
    return None


def _is_ids(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None or not hasattr(value, "numel"):
        return False
    if getattr(dtype, "is_floating_point", True) or getattr(dtype, "is_complex", False):
        return False
    return str(dtype) != "torch.bool"


# ─── FLOPs ───────────────────────────────────────────────────────────


def _config_value(config: Any, names: Iterable[str]) -> Optional[int]:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def attention_shape(model: Any) -> Optional[Tuple[int, int]]:
    """``(layers, width)`` from the model's configuration, as Hugging Face names them, or None."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    layers = _config_value(config, ("num_hidden_layers", "n_layer", "num_layers", "n_layers"))
    width = _config_value(config, ("hidden_size", "n_embd", "d_model", "dim"))
    if layers is None or width is None:
        return None
    return layers, width


def multiplied_parameters(model: Any) -> List[Any]:
    """The trainable parameters a token is multiplied by: all but an embedding's lookup table.

    An embedding's weight that the output head also uses (tied) is a product
    there, and stays.
    """
    import torch.nn as nn

    lookups = set()
    products = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            lookups.add(id(module.weight))
        elif isinstance(module, nn.Linear):
            products.add(id(module.weight))
    seen = set()
    out = []
    for param in model.parameters():
        if id(param) in seen or not param.requires_grad:
            continue
        seen.add(id(param))
        if id(param) in lookups and id(param) not in products:
            continue
        out.append(param)
    return out


class Flops:
    """What one token costs the model, and what the GPU could do at that cost's dtype mix.

    Built once per run, at the first throughput point: the parameters do not
    change dtype mid-run, and walking a model's modules every few steps would
    be the cost this module is meant to avoid.
    """

    def __init__(self, per_token: float, attention_per_position: float, peak_tflops: float) -> None:
        #: ``6 x`` the multiplied parameters.
        self.per_token = per_token
        #: ``12 x layers x width``, to multiply by the sequence length; 0 when unknown.
        self.attention_per_position = attention_per_position
        #: The GPU's peak for this model's dtype mix, in TFLOPS.
        self.peak_tflops = peak_tflops

    def mfu(self, tokens_per_second: float, sequence: Optional[int]) -> float:
        per_token = self.per_token + self.attention_per_position * (sequence or 0)
        return tokens_per_second * per_token / (self.peak_tflops * 1e12)


def flops_for(model: Any, peaks: Mapping[str, float], autocast: Optional[Any], tf32: bool) -> Optional[Flops]:
    """The model's :class:`Flops` on a GPU with ``peaks``, or None when a dtype it uses has no peak.

    Each parameter is charged at its compute dtype: ``autocast`` when the
    forward runs under it (only for floating parameters it would cast, as
    autocast does), else its own. The effective peak is the one at which the
    whole token's work would take as long as its parts at their own peaks:
    ``sum(flops) / sum(flops_i / peak_i)``.
    """
    params = multiplied_parameters(model)
    if not params:
        return None
    total = 0.0
    seconds = 0.0
    for param in params:
        dtype = autocast if autocast is not None and param.dtype.is_floating_point else param.dtype
        key = dtype_key(dtype, tf32)
        peak = peaks.get(key) if key else None
        if not peak:
            logger.info("No MFU: the GPU has no known peak for %s, which part of the model computes in.", dtype)
            return None
        flops = 6.0 * param.numel()
        total += flops
        seconds += flops / peak
    shape = attention_shape(model)
    attention = 12.0 * shape[0] * shape[1] if shape else 0.0
    return Flops(total, attention, total / seconds)


# ─── what to chart per layer ────────────────────────────────────────

_ATTENTION = re.compile(r"(^|[._])(self_)?attn|attention([._]|$)", re.IGNORECASE)
_MLP = re.compile(r"(^|[._])(mlp|ffn|ff|feed_forward|feedforward|experts?|moe)([._]|$)", re.IGNORECASE)

#: How many parameters a model with no blocks to find gets charted: its largest.
LARGEST_PARAMETERS = 4

#: The most parameters charted per layer in all, however a block is built: a
#: block of a hundred experts charts the first few, not the hundred.
MAX_PER_PART = 4


def _blocks(model: Any) -> Tuple[str, List[Any]]:
    """The model's blocks: its longest ``ModuleList`` of one class, with its name."""
    import torch.nn as nn

    best: Tuple[str, List[Any]] = ("", [])
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        children = list(module)
        if len({type(child) for child in children}) != 1:
            continue
        if len(children) > len(best[1]):
            best = (name, children)
    return best


def _part(block: Any, pattern: "re.Pattern[str]", prefix: str) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for name, param in block.named_parameters():
        if param.dim() < 2 or not pattern.search(name):
            continue
        out.append((prefix + name, param))
        if len(out) >= MAX_PER_PART:
            break
    return out


def layer_parameters(model: Any) -> List[Tuple[str, Any]]:
    """``(name, parameter)`` worth charting per layer, in the model's own names.

    The embedding; attention and MLP of the first, middle and last block; the
    final norm and the output head, which are what is left after the blocks.
    Its largest parameters for a model with no blocks to find.
    """
    import torch.nn as nn

    blocks_name, blocks = _blocks(model)
    named = list(model.named_parameters())
    if not blocks:
        largest = sorted(named, key=lambda item: item[1].numel(), reverse=True)
        return [(name, param) for name, param in largest[:LARGEST_PARAMETERS] if param.numel() >= 2]

    chosen: List[Tuple[str, Any]] = []
    seen = set()

    def add(name: str, param: Any) -> None:
        if id(param) not in seen and param.numel() >= 2:
            seen.add(id(param))
            chosen.append((name, param))

    owners = {id(param): name for name, param in named}
    for module in model.modules():
        if isinstance(module, nn.Embedding) and id(module.weight) in owners:
            add(owners[id(module.weight)], module.weight)
    for index in sorted({0, len(blocks) // 2, len(blocks) - 1}):
        prefix = "%s.%d." % (blocks_name, index) if blocks_name else "%d." % index
        for name, param in _part(blocks[index], _ATTENTION, prefix) + _part(blocks[index], _MLP, prefix):
            add(name, param)
    # After the blocks: whatever is not inside them nor the embedding - the
    # final norm and the head, in the order the model declares them.
    inside = blocks_name + "."
    for name, param in named:
        if not name.startswith(inside):
            add(name, param)
    return chosen
