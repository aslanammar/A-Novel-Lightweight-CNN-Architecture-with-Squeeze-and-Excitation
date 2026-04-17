"""
Tests for model architecture components.

Run with::

    pytest tests/test_model.py -v
"""

import pytest
import torch

from model.se_block import SEBlock
from model.lightweight_cnn import (
    ConvBNReLU,
    DepthSepBlock,
    SEDepthSepBlock,
    MaizeSENet,
    build_model,
)


# ---------------------------------------------------------------------------
# SEBlock tests
# ---------------------------------------------------------------------------


class TestSEBlock:
    def test_output_shape_matches_input(self):
        """SE block must not change spatial dimensions or channel count."""
        x = torch.randn(2, 64, 28, 28)
        se = SEBlock(64)
        out = se(x)
        assert out.shape == x.shape

    def test_output_shape_small_channels(self):
        """SE block with very few channels (< reduction) should still work."""
        x = torch.randn(1, 8, 4, 4)
        se = SEBlock(8, reduction=16)  # mid = max(8//16, 4) = 4
        out = se(x)
        assert out.shape == x.shape

    def test_scale_values_bounded(self):
        """Excitation output must lie in [0, 1] (Sigmoid), so scaled features
        must be <= original absolute values."""
        torch.manual_seed(0)
        x = torch.rand(4, 32, 8, 8) + 0.1  # strictly positive
        se = SEBlock(32)
        out = se(x)
        assert (out.abs() <= x.abs() + 1e-6).all(), \
            "SE-scaled values should not exceed original magnitudes"

    def test_different_batch_sizes(self):
        se = SEBlock(16)
        for bs in [1, 4, 16]:
            x = torch.randn(bs, 16, 7, 7)
            assert se(x).shape == (bs, 16, 7, 7)

    def test_gradient_flows(self):
        """Gradients must flow back through the SE block."""
        x = torch.randn(2, 32, 8, 8, requires_grad=True)
        se = SEBlock(32)
        loss = se(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# ConvBNReLU tests
# ---------------------------------------------------------------------------


class TestConvBNReLU:
    def test_output_shape(self):
        layer = ConvBNReLU(3, 32, kernel_size=3, stride=2)
        x = torch.randn(1, 3, 224, 224)
        assert layer(x).shape == (1, 32, 112, 112)

    def test_no_negative_values_after_relu(self):
        layer = ConvBNReLU(8, 16)
        x = torch.randn(2, 8, 14, 14)
        assert (layer(x) >= 0).all()


# ---------------------------------------------------------------------------
# DepthSepBlock tests
# ---------------------------------------------------------------------------


class TestDepthSepBlock:
    def test_output_shape_stride1(self):
        block = DepthSepBlock(64, 128, stride=1)
        x = torch.randn(2, 64, 28, 28)
        assert block(x).shape == (2, 128, 28, 28)

    def test_output_shape_stride2(self):
        block = DepthSepBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 28, 28)
        assert block(x).shape == (2, 128, 14, 14)

    def test_fewer_params_than_standard_conv(self):
        """DW-Sep should have strictly fewer params than a standard 3×3 Conv."""
        import torch.nn as nn
        dw_block = DepthSepBlock(64, 128)
        std_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        dw_params = sum(p.numel() for p in dw_block.parameters())
        std_params = sum(p.numel() for p in std_conv.parameters())
        assert dw_params < std_params, (
            f"DepthSepBlock ({dw_params}) should have fewer params than "
            f"standard Conv ({std_params})"
        )


# ---------------------------------------------------------------------------
# SEDepthSepBlock tests
# ---------------------------------------------------------------------------


class TestSEDepthSepBlock:
    def test_output_shape(self):
        block = SEDepthSepBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 56, 56)
        assert block(x).shape == (2, 64, 28, 28)


# ---------------------------------------------------------------------------
# MaizeSENet full model tests
# ---------------------------------------------------------------------------


class TestMaizeSENet:
    @pytest.fixture
    def model(self):
        return MaizeSENet(num_classes=4)

    def test_output_shape_default(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 4)

    def test_output_shape_custom_classes(self):
        model = MaizeSENet(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        assert model(x).shape == (1, 10)

    def test_parameter_count_reasonable(self, model):
        """MaizeSENet should have < 3 M parameters (lightweight requirement)."""
        n_params = model.count_parameters()
        assert n_params < 3_000_000, (
            f"Model has {n_params:,} parameters – expected < 3M for 'lightweight'"
        )

    def test_parameter_count_not_trivial(self, model):
        """Model must have at least 100 K parameters to be non-trivial."""
        n_params = model.count_parameters()
        assert n_params > 100_000, f"Model only has {n_params:,} parameters"

    def test_eval_mode_deterministic(self, model):
        """In eval mode, two forward passes on the same input must agree."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_feature_maps_keys(self, model):
        x = torch.randn(1, 3, 224, 224)
        maps = model.feature_maps(x)
        expected_keys = {"stem", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"}
        assert set(maps.keys()) == expected_keys

    def test_feature_maps_shapes_decrease(self, model):
        """Spatial resolution should monotonically decrease through stages."""
        x = torch.randn(1, 3, 224, 224)
        maps = model.feature_maps(x)
        stages = ["stem", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]
        prev_h = 224
        for stage in stages:
            _, _, h, w = maps[stage].shape
            assert h <= prev_h, f"{stage}: h={h} should be <= prev {prev_h}"
            assert h == w, f"{stage}: expected square feature map, got {h}×{w}"
            prev_h = h

    def test_gradient_flows_full_model(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_single_sample_batch(self, model):
        """Batch size of 1 must work correctly (BatchNorm edge case)."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# build_model factory tests
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_default_construction(self):
        model = build_model()
        assert isinstance(model, MaizeSENet)

    def test_custom_params(self):
        model = build_model(num_classes=7, dropout=0.2, se_reduction=8)
        x = torch.randn(1, 3, 224, 224)
        assert model(x).shape == (1, 7)

    def test_load_from_checkpoint(self, tmp_path):
        """Verify that build_model correctly loads saved weights."""
        model_orig = build_model(num_classes=4)
        ckpt_path = tmp_path / "model.pth"
        torch.save(model_orig.state_dict(), ckpt_path)

        model_loaded = build_model(num_classes=4, pretrained_path=str(ckpt_path))
        for (n1, p1), (n2, p2) in zip(
            model_orig.named_parameters(), model_loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter {n1} differs after loading"

    def test_load_from_checkpoint_dict(self, tmp_path):
        """Verify loading from a full training checkpoint dict."""
        model_orig = build_model(num_classes=4)
        ckpt_path = tmp_path / "checkpoint.pth"
        torch.save({"model_state_dict": model_orig.state_dict()}, ckpt_path)

        model_loaded = build_model(num_classes=4, pretrained_path=str(ckpt_path))
        for (_, p1), (_, p2) in zip(
            model_orig.named_parameters(), model_loaded.named_parameters()
        ):
            assert torch.allclose(p1, p2)
