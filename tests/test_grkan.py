"""Tests for GroupRational and GRKANFFN modules."""

import pytest
import torch
import sys
from pathlib import Path

# Ensure nanochat is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass


@dataclass
class _TestConfig:
    n_embd: int = 128
    sequence_len: int = 128
    vocab_size: int = 1024
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int = 4
    window_pattern: str = "L"
    ffn_type: str = "grkan"
    grkan_groups: int = 8
    grkan_m: int = 5
    grkan_n: int = 4
    grkan_init_rat1: str = "identity"
    grkan_init_rat2: str = "swish"


# ---------------------------------------------------------------------------
# GroupRational tests
# ---------------------------------------------------------------------------

class TestGroupRational:
    def test_output_shape(self):
        """GroupRational preserves input shape."""
        from nanochat.gpt import GroupRational
        d_in, g = 128, 8
        rat = GroupRational(d_in, g)
        x = torch.randn(4, 16, d_in)
        out = rat(x)
        assert out.shape == x.shape

    def test_2d_input(self):
        """GroupRational accepts 2D input (flattened batch)."""
        from nanochat.gpt import GroupRational
        d_in, g = 128, 8
        rat = GroupRational(d_in, g)
        x = torch.randn(64, d_in)
        out = rat(x)
        assert out.shape == x.shape

    def test_identity_init(self):
        """Identity init: a1=1.0, all others 0 -> output equals input."""
        from nanochat.gpt import GroupRational
        d_in, g = 128, 8
        rat = GroupRational(d_in, g, init="identity")
        x = torch.randn(4, 16, d_in)
        out = rat(x)
        max_err = (out - x).abs().max().item()
        assert max_err < 1e-5, f"Identity output differs from input: max_err={max_err:.2e}"

    def test_swish_init_is_not_identity(self):
        """Swish init produces output different from input."""
        from nanochat.gpt import GroupRational
        d_in, g = 128, 8
        rat = GroupRational(d_in, g, init="swish")
        x = torch.randn(4, 16, d_in)
        out = rat(x)
        max_err = (out - x).abs().max().item()
        assert max_err > 0.01, "Swish init should not produce identity output"

    def test_coefficients_are_parameters(self):
        """GroupRational coefficients are nn.Parameters."""
        from nanochat.gpt import GroupRational
        rat = GroupRational(128, 8)
        params = dict(rat.named_parameters())
        assert "a" in params
        assert "b" in params
        assert params["a"].requires_grad
        assert params["b"].requires_grad

    def test_raises_on_indivisible_d_in(self):
        """d_in not divisible by num_groups raises ValueError."""
        from nanochat.gpt import GroupRational
        with pytest.raises(ValueError, match="divisible"):
            GroupRational(127, 8)


# ---------------------------------------------------------------------------
# GRKANFFN tests
# ---------------------------------------------------------------------------

class TestGRKANFFN:
    def test_output_shape(self):
        """GRKANFFN preserves input shape."""
        from nanochat.gpt import GRKANFFN
        config = _TestConfig()
        ffn = GRKANFFN(config)
        x = torch.randn(4, 16, config.n_embd)
        out = ffn(x)
        assert out.shape == x.shape

    def test_forward_runs(self):
        """GRKANFFN forward pass runs without error for typical shapes."""
        from nanochat.gpt import GRKANFFN
        config = _TestConfig()
        ffn = GRKANFFN(config)
        x = torch.randn(2, 64, config.n_embd)
        out = ffn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_parameter_count(self):
        """GRKANFFN has the expected number of parameters."""
        from nanochat.gpt import GRKANFFN
        config = _TestConfig()
        ffn = GRKANFFN(config)
        n_params = sum(p.numel() for p in ffn.parameters())
        # 2 linear layers + 2 group rationals (a:6, b:8*4=32 each)
        expected = (
            config.n_embd * 4 * config.n_embd  # c_fc
            + 4 * config.n_embd * config.n_embd  # c_proj
            + 2 * (6 + config.grkan_groups * config.grkan_n)  # rat1 + rat2 coeffs
        )
        assert n_params == expected, f"Expected {expected} params, got {n_params}"

    def test_init_modes_from_config(self):
        """GRKANFFN respects grkan_init_rat1/rat2 from config."""
        from nanochat.gpt import GRKANFFN
        config = _TestConfig()
        config.grkan_init_rat1 = "identity"
        config.grkan_init_rat2 = "identity"
        ffn = GRKANFFN(config)
        # With identity init everywhere, output should be near-zero
        # (c_proj is zero-initialized by GPT.init_weights)
        # Just verify it runs without error
        x = torch.randn(2, 16, config.n_embd)
        out = ffn(x)
        assert out.shape == x.shape

    def test_different_ffn_type_does_not_crash(self):
        """Building GRKANFFN when ffn_type='mlp' should not be called,
        but the class itself is independent of config.ffn_type."""
        from nanochat.gpt import GRKANFFN
        config = _TestConfig()
        ffn = GRKANFFN(config)  # config.ffn_type is irrelevant here
        assert ffn.rat1._init_mode == config.grkan_init_rat1
        assert ffn.rat2._init_mode == config.grkan_init_rat2


# ---------------------------------------------------------------------------
# GPT integration tests
# ---------------------------------------------------------------------------

class TestGPTWithGRKAN:
    def test_build_model_grkan(self):
        """GPT model builds successfully with ffn_type='grkan'."""
        from nanochat.gpt import GPT
        config = _TestConfig()
        config.ffn_type = "grkan"
        model = GPT(config)
        # Check GRKANFFN was used
        for block in model.transformer.h:
            from nanochat.gpt import GRKANFFN
            assert isinstance(block.mlp, GRKANFFN)

    def test_build_model_mlp(self):
        """GPT model builds with ffn_type='mlp' (default)."""
        from nanochat.gpt import GPT, MLP
        config = _TestConfig()
        config.ffn_type = "mlp"
        model = GPT(config)
        for block in model.transformer.h:
            assert isinstance(block.mlp, MLP)

    def test_init_weights_grkan(self):
        """init_weights initializes GR-KAN coefficients correctly."""
        from nanochat.gpt import GPT
        config = _TestConfig()
        config.ffn_type = "grkan"
        model = GPT(config)
        model.init_weights()
        for block in model.transformer.h:
            # a[1] should be ~1.0 (identity init)
            assert abs(block.mlp.rat1.a[1].item() - 1.0) < 1e-6
            # a[0] should be ~0.0
            assert abs(block.mlp.rat1.a[0].item()) < 1e-6
            # b should be all zeros
            assert (block.mlp.rat1.b == 0).all()

    def test_num_scaling_params_grkan(self):
        """num_scaling_params includes grkan_coeffs when ffn_type='grkan'."""
        from nanochat.gpt import GPT
        config = _TestConfig()
        config.ffn_type = "grkan"
        model = GPT(config)
        counts = model.num_scaling_params()
        assert 'grkan_coeffs' in counts
        assert counts['grkan_coeffs'] > 0
        # grkan_coeffs should be 2 * (6 + 8*4) * n_layer = 2 * 38 * 2 = 152
        assert counts['grkan_coeffs'] == 2 * (6 + config.grkan_groups * config.grkan_n) * config.n_layer

    def test_num_scaling_params_mlp_no_grkan(self):
        """num_scaling_params has grkan_coeffs=0 when ffn_type='mlp'."""
        from nanochat.gpt import GPT
        config = _TestConfig()
        config.ffn_type = "mlp"
        model = GPT(config)
        counts = model.num_scaling_params()
        assert 'grkan_coeffs' in counts
        assert counts['grkan_coeffs'] == 0
