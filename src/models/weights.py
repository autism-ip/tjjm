"""
 * [INPUT]: 依赖 torch, monai.networks.nets.SwinUNETR, os, warnings
 * [OUTPUT]: 对外提供 load_swin_unetr_pretrained, freeze_encoder, unfreeze_layers
 * [POS]: src/models/ 的权重管理工具, 被 autoencoder 初始化流程消费
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import os
import warnings
from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.networks.utils import copy_model_state
from monai.networks.nets.swin_unetr import filter_swinunetr


# ============================================================
# Pretrained Weight Loading
# ============================================================

# MONAI 官方 SSL 预训练权重 (encoder only, ~50k 3D volumes)
SSL_PRETRAINED_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/"
    "releases/download/0.8.1/ssl_pretrained_weights.pth"
)


def load_swin_unetr_pretrained(
    model: SwinUNETR,
    checkpoint_path: str | None = None,
) -> None:
    """
    加载 SwinUNETR 预训练权重 (encoder only).

    使用 MONAI 官方 SSL 预训练权重, 通过 filter_swinunetr
    将权重键名映射到当前模型结构, 仅加载 encoder (swinViT) 部分.

    PyTorch 2.6+ 默认 weights_only=True, 但旧版权重包含 numpy scalar,
    故使用 weights_only=False (权重来自可信的 MONAI 官方仓库).
    """
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        _load_weights_file(model, checkpoint_path)
        return

    # 尝试下载并加载官方 SSL 预训练权重
    try:
        from monai.apps import download_url

        cache_dir = os.path.expanduser("~/.cache/monai/pretrained")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, "ssl_pretrained_weights.pth")

        if not os.path.exists(local_path):
            warnings.warn(
                f"Downloading SwinUNETR SSL pretrained weights from {SSL_PRETRAINED_URL} ..."
            )
            download_url(SSL_PRETRAINED_URL, local_path)

        _load_weights_file(model, local_path)

    except Exception as exc:
        warnings.warn(
            f"Failed to load pretrained weights: {exc}. "
            f"Using random initialization. "
            f"If behind a proxy, manually download {SSL_PRETRAINED_URL} "
            f"and pass checkpoint_path=<local_path>."
        )


def _load_weights_file(model: SwinUNETR, path: str) -> None:
    """
    从本地文件加载 SSL 预训练权重.

    [NOTE] weights_only=False 原因:
    - PyTorch 2.6+ 默认 weights_only=True
    - MONAI 旧版权重文件包含 numpy.core.multiarray.scalar
    - 该类型不在 PyTorch 默认安全白名单中
    - 权重文件来自可信的 MONAI 官方 GitHub Release, 风险可控

    [NOTE] filter_swinunetr 作用:
    - SSL 权重键名以 'encoder.' 开头
    - SwinUNETR 内部 encoder 名为 'swinViT'
    - filter_swinunetr 做键名映射并过滤无关层 (mask_token, out.conv 等)
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # SSL 权重结构: checkpoint["model"] 包含 encoder.* 键
    if "model" in checkpoint:
        ssl_weights = checkpoint["model"]
    elif "state_dict" in checkpoint:
        ssl_weights = checkpoint["state_dict"]
    else:
        ssl_weights = checkpoint

    # 使用 MONAI 官方的 filter_swinunetr 进行键名映射
    dst_dict, loaded, not_loaded = copy_model_state(
        model, ssl_weights, filter_func=filter_swinunetr
    )

    if loaded:
        warnings.warn(
            f"Loaded {len(loaded)} pretrained layers into SwinUNETR encoder."
        )
    if not_loaded:
        warnings.warn(
            f"Could not load {len(not_loaded)} layers (expected for decoder/head): "
            f"{not_loaded[:5]}..."
        )


# ============================================================
# Freeze / Unfreeze Utilities
# ============================================================

def freeze_encoder(model: nn.Module) -> None:
    """冻结模型所有参数."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_layers(model: nn.Module, layer_names: Sequence[str]) -> None:
    """按名称解冻指定层或其子模块."""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                break
