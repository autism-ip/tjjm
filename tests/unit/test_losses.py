"""
 * [INPUT]: 依赖 pytest, torch, src.training.losses 的 WeightedMSELoss
 * [OUTPUT]: 提供 WeightedMSELoss 的单元测试套件
 * [POS]: tests/unit/ 的核心损失函数验证器, 确保加权 MSE 行为正确
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import pytest
import torch

from src.training.losses import WeightedMSELoss


class TestWeightedMSELoss:
    """WeightedMSELoss 的 TDD 单元测试."""

    def test_loss_value_is_nonnegative(self):
        """损失值必须非负."""
        loss_fn = WeightedMSELoss(k=5.0)
        pred = torch.randn(2, 1, 8, 8, 8)
        target = torch.randn(2, 1, 8, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0.0

    def test_loss_on_zeros_is_zero(self):
        """全零输入与全零目标，损失应为 0."""
        loss_fn = WeightedMSELoss(k=5.0)
        pred = torch.zeros(2, 1, 8, 8, 8)
        target = torch.zeros(2, 1, 8, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_weight_computation_negative_values(self):
        """HU <= -300 (x_norm <= 0) 时权重应为 1."""
        loss_fn = WeightedMSELoss(k=5.0)
        # 构造全 -1000 HU 的 target，归一化后 <= 0
        target = torch.full((2, 1, 8, 8, 8), -1000.0)
        weights = loss_fn.compute_weights(target)
        assert torch.allclose(weights, torch.ones_like(weights))

    def test_weight_computation_positive_values(self):
        """HU > -300 (x_norm > 0) 时权重应按公式增长."""
        loss_fn = WeightedMSELoss(k=5.0)
        # 构造全 0 HU 的 target，归一化后 x_norm = 0.3
        target = torch.zeros(2, 1, 8, 8, 8)
        weights = loss_fn.compute_weights(target)
        expected = (1.0 + 5.0 * 0.3) ** 2  # = 6.25
        assert torch.allclose(weights, torch.full_like(weights, expected))

    def test_weight_computation_mixed(self):
        """混合正负值时权重分别计算."""
        loss_fn = WeightedMSELoss(k=5.0)
        target = torch.tensor([[-1000.0, 0.0, 100.0]])
        # expand to 5D
        target = target.view(1, 1, 1, 1, 3)
        weights = loss_fn.compute_weights(target)
        # -1000 -> x_norm = (-1000+300)/1000 = -0.7 <= 0 -> w=1
        # 0 -> x_norm = 0.3 -> w=(1+5*0.3)^2=6.25
        # 100 -> x_norm = 0.4 -> w=(1+5*0.4)^2=9.0
        assert weights[0, 0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert weights[0, 0, 0, 0, 1].item() == pytest.approx(6.25, abs=1e-5)
        assert weights[0, 0, 0, 0, 2].item() == pytest.approx(9.0, abs=1e-5)

    def test_different_k_values(self):
        """不同 k 值产生不同权重."""
        target = torch.zeros(1, 1, 4, 4, 4)
        loss_fn_k2 = WeightedMSELoss(k=2.0)
        loss_fn_k5 = WeightedMSELoss(k=5.0)
        w2 = loss_fn_k2.compute_weights(target)
        w5 = loss_fn_k5.compute_weights(target)
        assert not torch.allclose(w2, w5)
        assert w2[0, 0, 0, 0, 0].item() == pytest.approx((1.0 + 2.0 * 0.3) ** 2, abs=1e-5)
        assert w5[0, 0, 0, 0, 0].item() == pytest.approx((1.0 + 5.0 * 0.3) ** 2, abs=1e-5)

    def test_gradient_flow(self):
        """损失应能反向传播到 pred."""
        loss_fn = WeightedMSELoss(k=5.0)
        pred = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
        target = torch.randn(2, 1, 8, 8, 8)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)
