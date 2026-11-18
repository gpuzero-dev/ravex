"""The default pre-training metrics (GPU-436): perplexity, throughput, MFU, and the charts per layer.

What a script with nothing but the decorator should see about a pre-training
run. The names are stable: a chart looks for them and computes none of them.
"""

import math
import types

import pytest
import torch
import torch.nn as nn

import ravex
import ravex.metrics
from ravex import _metrics, _pretraining
from ravex._pretraining import (
    Flops,
    attention_shape,
    flops_for,
    layer_parameters,
    mean_perplexity,
    peaks_for,
    token_ids,
    with_perplexity,
)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = nn.ModuleDict({"q": nn.Linear(width, width), "o": nn.Linear(width, width)})
        self.mlp = nn.ModuleDict({"up": nn.Linear(width, 2 * width), "down": nn.Linear(2 * width, width)})
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        x = x + self.attn["o"](torch.tanh(self.attn["q"](self.norm(x))))
        return x + self.mlp["down"](torch.relu(self.mlp["up"](x)))


class TinyLM(nn.Module):
    """A transformer's shape, small: embedding, a list of blocks, a final norm, a head."""

    def __init__(self, vocab=32, width=8, layers=5):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(Block(width) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = types.SimpleNamespace(num_hidden_layers=layers, hidden_size=width)

    def forward(self, input_ids=None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


class TestPeaks:
    @pytest.mark.parametrize("name, bf16", [
        ("NVIDIA H100 80GB HBM3", 989.4),
        ("NVIDIA H100 PCIe", 756.5),
        ("NVIDIA A100-SXM4-80GB", 312.0),
        ("NVIDIA A100 80GB PCIe", 312.0),
        ("NVIDIA A10", 125.0),
        ("NVIDIA L4", 121.0),
        ("NVIDIA L40S", 362.1),
        ("NVIDIA GeForce RTX 4090", 165.2),
        ("NVIDIA RTX A6000", 154.8),
        ("AMD Instinct MI300X", 1307.4),
    ])
    def test_a_known_gpu(self, name, bf16):
        assert peaks_for(name)["bf16"] == bf16

    @pytest.mark.parametrize("name", ["NVIDIA H100 NVL", "Tesla T4", "", None, "NVIDIA A10G", "NVIDIA RTX 6000 Ada Generation"])
    def test_an_unknown_gpu_has_no_peak_rather_than_a_guess(self, name):
        assert peaks_for(name) is None


class TestFlops:
    PEAKS = {"bf16": 100.0, "fp32": 10.0, "tf32": 50.0}

    def test_six_times_the_parameters_that_multiply(self):
        model = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 3))
        flops = flops_for(model, self.PEAKS, None, tf32=False)
        # The embedding's table is a lookup, not a product: only the Linear.
        assert flops.per_token == 6.0 * (4 * 3 + 3)
        assert flops.peak_tflops == 10.0

    def test_a_tied_embedding_is_a_product_in_the_head(self):
        embed, head = nn.Embedding(10, 4), nn.Linear(4, 10, bias=False)
        head.weight = embed.weight
        flops = flops_for(nn.ModuleList([embed, head]), self.PEAKS, None, tf32=False)
        assert flops.per_token == 6.0 * 40

    def test_autocast_charges_every_float_parameter_at_its_dtype(self):
        flops = flops_for(nn.Linear(4, 4), self.PEAKS, torch.bfloat16, tf32=False)
        assert flops.peak_tflops == 100.0

    def test_a_mix_of_dtypes_gets_the_peak_of_the_mix(self):
        """A component computed at another precision than the rest, as with
        experts in a lower one: each part at its own peak."""
        model = nn.Sequential(nn.Linear(10, 10, bias=False), nn.Linear(10, 10, bias=False).to(torch.bfloat16))
        flops = flops_for(model, self.PEAKS, None, tf32=False)
        assert flops.peak_tflops == pytest.approx(2 / (1 / 10.0 + 1 / 100.0))

    def test_tf32_is_its_own_peak(self):
        assert flops_for(nn.Linear(4, 4), self.PEAKS, None, tf32=True).peak_tflops == 50.0

    def test_a_dtype_the_gpu_has_no_peak_for_is_no_mfu(self):
        assert flops_for(nn.Linear(4, 4), {"bf16": 100.0}, None, tf32=False) is None

    def test_attention_from_the_configuration(self):
        model = TinyLM(layers=5, width=8)
        assert attention_shape(model) == (5, 8)
        flops = flops_for(model, self.PEAKS, None, tf32=False)
        assert flops.attention_per_position == 12.0 * 5 * 8
        assert attention_shape(nn.Linear(2, 2)) is None

    def test_mfu(self):
        flops = Flops(per_token=6e9, attention_per_position=0.0, peak_tflops=100.0)
        # 10k tokens/s of 6 GFLOPs each is 60 TFLOPS: 60% of 100.
        assert flops.mfu(10_000, sequence=None) == pytest.approx(0.6)


class TestTokens:
    def test_input_ids_by_name_first(self):
        ids = torch.zeros(2, 7, dtype=torch.long)
        mask = torch.ones(2, 7, dtype=torch.long)
        assert token_ids((), {"attention_mask": mask, "input_ids": ids}) is ids

    def test_the_first_integer_tensor_in_order(self):
        ids = torch.zeros(3, 5, dtype=torch.int32)
        assert token_ids((torch.randn(2), ids), {}) is ids

    def test_floats_and_masks_are_no_tokens(self):
        assert token_ids((torch.randn(2, 4), torch.ones(2, dtype=torch.bool)), {}) is None


class TestPerplexity:
    def test_beside_the_losses(self):
        values = _metrics.prepare_all({"train/loss": 2.0, "val/loss": torch.tensor(1.5), "other": 3.0})
        out = with_perplexity(values)
        assert float(out["train/ppl"].pending) == pytest.approx(math.exp(2.0))
        assert float(out["val/ppl"].pending) == pytest.approx(math.exp(1.5))
        assert "other/ppl" not in out and "train/ppl" not in values

    def test_one_the_script_logged_itself_is_kept(self):
        values = _metrics.prepare_all({"train/loss": 2.0, "train/ppl": 99.0})
        assert with_perplexity(values)["train/ppl"].pending == 99.0

    def test_a_huge_loss_is_an_infinite_perplexity_not_an_error(self):
        assert with_perplexity(_metrics.prepare_all({"train/loss": 1e6}))["train/ppl"].pending == math.inf

    def test_several_nodes_get_the_perplexity_of_the_mean_loss(self):
        means = _metrics.model_means([(1, {"train/loss": [1.0, 1]}), (1, {"train/loss": [3.0, 1]})])
        assert means["train/ppl"] == pytest.approx(math.exp(2.0))
        assert mean_perplexity({}) == {}


class Expert(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w1 = nn.Linear(width, 2 * width, bias=False)
        self.w2 = nn.Linear(2 * width, width, bias=False)


class MoEBlock(nn.Module):
    """A mixture's block: its list of experts is longer than the model's list of blocks."""

    def __init__(self, width, experts):
        super().__init__()
        self.attn = nn.Linear(width, width, bias=False)
        self.ffn = nn.ModuleDict({"experts": nn.ModuleList(Expert(width) for _ in range(experts))})


class TinyMoE(nn.Module):
    def __init__(self, width=8, layers=3, experts=8, active=2):
        super().__init__()
        self.embed = nn.Embedding(16, width)
        self.layers = nn.ModuleList(MoEBlock(width, experts) for _ in range(layers))
        self.head = nn.Linear(width, 16, bias=False)
        self.args = types.SimpleNamespace(n_layers=layers, dim=width, n_routed_experts=experts,
                                          n_activated_experts=active)


class TestMixtureOfExperts:
    def test_the_blocks_are_the_list_that_holds_the_experts_not_one_of_them(self):
        names = [name for name, _ in layer_parameters(TinyMoE(layers=3, experts=8))]
        # The first, middle and last of the model's three blocks, not of one
        # block's eight experts.
        for index in (0, 1, 2):
            assert "layers.%d.attn.weight" % index in names

    def test_only_the_active_share_of_the_experts_counts(self):
        model = TinyMoE(width=8, layers=1, experts=8, active=2)
        flops = flops_for(model, {"fp32": 10.0}, None, tf32=False)
        experts = 8 * (8 * 16 + 16 * 8)
        dense = 8 * 8 + 8 * 16
        assert flops.per_token == pytest.approx(6.0 * (dense + experts * 2 / 8))

    def test_a_mixture_that_does_not_say_its_k_has_no_mfu(self):
        model = TinyMoE()
        del model.args.n_activated_experts
        assert flops_for(model, {"fp32": 10.0}, None, tf32=False) is None

    def test_the_shape_from_args_too(self):
        assert attention_shape(TinyMoE(width=8, layers=3)) == (3, 8)


class TestLayers:
    def test_embedding_three_blocks_norm_and_head(self):
        names = [name for name, _ in layer_parameters(TinyLM(layers=5))]
        assert names[0] == "embed.weight"
        for index in (0, 2, 4):
            for part in ("attn.q.weight", "attn.o.weight", "mlp.up.weight", "mlp.down.weight"):
                assert "blocks.%d.%s" % (index, part) in names
        assert not [name for name in names if name.startswith(("blocks.1.", "blocks.3."))]
        assert names[-3:] == ["norm.weight", "norm.bias", "lm_head.weight"]

    def test_a_model_without_blocks_charts_its_largest(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 2))
        assert [name for name, _ in layer_parameters(model)] == ["0.weight", "1.weight", "0.bias", "1.bias"]


def train_tiny(steps, **config):
    """A tiny language model trained with nothing but the decorator and a logged loss."""

    @ravex.train_loop(backend="torch_save", checkpoint_every=10_000, **config)
    def train():
        torch.manual_seed(0)
        model = TinyLM()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(steps):
            ids = torch.randint(0, 32, (4, 6))
            logits = model(input_ids=ids)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 32), ids.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ravex.log_metrics({"train/loss": loss.detach()})
            with torch.no_grad():
                model(input_ids=ids)

    train()


class TestWithTheDecorator:
    def test_throughput_and_perplexity(self, storage):
        train_tiny(6, metrics_every=2, histograms_every=0)
        scalars = ravex.metrics.read(str(storage))["scalars"]
        tokens = scalars["perf/tokens_per_s_per_gpu"]
        assert tokens["step"] == [2, 4, 6] and all(value > 0 for value in tokens["value"])
        loss, ppl = scalars["train/loss"], scalars["train/ppl"]
        assert ppl["step"] == loss["step"]
        assert ppl["value"] == pytest.approx([math.exp(value) for value in loss["value"]], rel=1e-5)
        # On a CPU: no peak to measure against, so no MFU rather than a guess.
        assert "perf/mfu" not in scalars

    def test_the_histograms_per_layer(self, storage):
        train_tiny(4, histograms_every=2)
        history = ravex.metrics.read(str(storage))
        histograms = history["histograms"]
        for name in ("embed.weight", "blocks.0.attn.q.weight", "blocks.4.mlp.down.weight", "lm_head.weight"):
            assert [entry["step"] for entry in histograms["train/weights/" + name]] == [2, 4]
            assert "train/grads/" + name in histograms
            ratios = history["scalars"]["train/updates/" + name]
            assert ratios["step"] == [2, 4] and all(0 < value < 1 for value in ratios["value"])
        # The outputs of the first, middle and last block, once per histogram step.
        for index in (0, 2, 4):
            assert [entry["step"] for entry in histograms["train/activations/blocks.%d" % index]] == [2, 4]
        assert "train/activations/blocks.1" not in histograms

    def test_training_metrics_false_charts_nothing_per_layer(self, storage):
        train_tiny(4, histograms_every=2, training_metrics=False)
        history = ravex.metrics.read(str(storage))
        assert history["histograms"] == {}
        assert not [name for name in history["scalars"] if name.startswith("train/updates/")]


def test_the_work_of_a_histogram_step_does_not_grow_with_the_model():
    """The cost of a histogram step is bounded by construction: the same
    parameters charted however deep the model, and each tensor sampled down
    to a fixed size however wide. Timing it here, on a CPU, would measure the
    CPU; the 1B measurement on a GPU is GPU-436's acceptance."""
    shallow = len(layer_parameters(TinyLM(layers=4)))
    deep = len(layer_parameters(TinyLM(layers=48)))
    assert shallow == deep
    wide = torch.zeros(_metrics.HISTOGRAM_MAX_ELEMENTS * 3 + 5)
    assert _metrics._histogram(wide).stride == 4
    assert _metrics.update_sample(wide).numel() <= _metrics.UPDATE_SAMPLE_ELEMENTS
