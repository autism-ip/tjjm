"""
 * [INPUT]: 依赖 torch, math
 * [OUTPUT]: 对外提供 DDPMScheduler, DDIMSampler
 * [POS]: src/models/ 的扩散工具骨架, 当前项目主路径为自编码器, 此模块预留扩展
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import math
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# DDPM Scheduler
# ============================================================

class DDPMScheduler:
    """
    DDPM beta 调度与前向加噪.
    预留接口, 当前项目未启用完整扩散训练.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.betas = self._build_betas(beta_start, beta_end, beta_schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _build_betas(
        self, beta_start: float, beta_end: float, schedule: str
    ) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, self.num_train_timesteps)
        if schedule == "scaled_linear":
            return torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps) ** 2
        raise ValueError(f"Unknown beta schedule: {schedule}")

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """前向扩散: q(x_t | x_0)."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(
            -1, 1, 1, 1, 1
        )
        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise


# ============================================================
# DDIM Sampler
# ============================================================

class DDIMSampler:
    """
    DDIM 确定性采样, 用于推理加速.
    预留接口, 当前项目未启用完整扩散推理.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """单步 DDIM 去噪."""
        # Simplified deterministic step (eta=0)
        alpha_prod_t = self._alpha_cumprod(timestep)
        alpha_prod_t_prev = self._alpha_cumprod(max(timestep - 1, 0))

        pred_original_sample = (
            sample - math.sqrt(1 - alpha_prod_t) * model_output
        ) / math.sqrt(alpha_prod_t)

        pred_sample_direction = math.sqrt(1 - alpha_prod_t_prev) * model_output
        prev_sample = math.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        return prev_sample

    def _alpha_cumprod(self, timestep: int) -> float:
        # Placeholder linear approximation
        return 1.0 - (timestep / self.num_train_timesteps)
