"""
 * [INPUT]: 依赖 pytest, torch, src.models.autoencoder 的 Autoencoder3D
 * [OUTPUT]: 提供 Autoencoder3D 的单元测试套件
 * [POS]: tests/unit/ 的核心模型验证器, 确保自编码器结构与前向行为正确
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import pytest
import torch

from src.models.autoencoder import Autoencoder3D


class TestAutoencoder3D:
    """Autoencoder3D 的 TDD 单元测试."""

    def test_forward_output_shape_matches_input(self):
        """前向传播输出 shape 应与输入相同."""
        model = Autoencoder3D(
            encoder_name="swin_unetr",
            freeze_encoder=True,
            pretrained=False,
        )
        x = torch.randn(2, 1, 64, 64, 64)
        out = model(x)
        assert out.shape == x.shape

    def test_encoder_frozen_when_freeze_encoder_true(self):
        """freeze_encoder=True 时编码器参数应被冻结."""
        model = Autoencoder3D(
            encoder_name="swin_unetr",
            freeze_encoder=True,
            pretrained=False,
        )
        encoder_params = list(model.encoder.parameters())
        assert len(encoder_params) > 0
        for p in encoder_params:
            assert not p.requires_grad

    def test_encoder_unfrozen_when_freeze_encoder_false(self):
        """freeze_encoder=False 时编码器参数应可训练."""
        model = Autoencoder3D(
            encoder_name="swin_unetr",
            freeze_encoder=False,
            pretrained=False,
        )
        for p in model.encoder.parameters():
            assert p.requires_grad

    def test_reconstruction_head_output_channels(self):
        """重建头输出通道数应为 1."""
        model = Autoencoder3D(
            encoder_name="swin_unetr",
            freeze_encoder=True,
            pretrained=False,
        )
        x = torch.randn(1, 1, 64, 64, 64)
        out = model(x)
        assert out.shape[1] == 1

    def test_batch_size_two_runs(self):
        """batch size=2 时应正常运行."""
        model = Autoencoder3D(
            encoder_name="swin_unetr",
            freeze_encoder=True,
            pretrained=False,
        )
        x = torch.randn(2, 1, 64, 64, 64)
        out = model(x)
        assert out.shape == torch.Size([2, 1, 64, 64, 64])

    def test_decoder_parameters_require_grad(self):
        """解码器参数应始终可训练."""
        model = Autoencoder3D(
            encoder_name="swin_unetr",
            freeze_encoder=True,
            pretrained=False,
        )
        for p in model.decoder.parameters():
            assert p.requires_grad
        for p in model.reconstruction_head.parameters():
            assert p.requires_grad

    def test_gradient_flow_through_decoder(self):
        """梯度应能流经解码器到可训练参数."""
        model = Autoencoder3D(
            encoder_name="swin_unetr",
            freeze_encoder=True,
            pretrained=False,
        )
        x = torch.randn(1, 1, 64, 64, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # decoder params have grad
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
        assert has_grad
