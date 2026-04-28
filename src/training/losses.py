"""
 * [INPUT]: 依赖 torch.nn, torch.nn.functional
 * [OUTPUT]: 对外提供 WeightedMSELoss
 * [POS]: src/training/ 的损失函数集合, 被 trainer 与测试消费
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Weighted MSE Loss
# ============================================================

class WeightedMSELoss(nn.Module):
    """
    加权 MSE 损失.
    对 HU > -300 的区域给予更高权重, 强化肺部软组织重建精度.
    权重公式: w = (1 + k * relu(x_norm))^2
    其中 x_norm = (HU + 1000) / 1300 - 1, 将 HU 映射到 [-1, 1].
    """

    def __init__(self, k: float = 5.0):
        super().__init__()
        self.k = k

    def compute_weights(self, target: torch.Tensor) -> torch.Tensor:
        """计算像素级权重."""
        x_norm = (target + 300.0) / 1000.0
        w = (1.0 + self.k * F.relu(x_norm)) ** 2
        return w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.compute_weights(target)
        diff = pred - target
        loss = torch.mean(weights * (diff ** 2))
        return loss
