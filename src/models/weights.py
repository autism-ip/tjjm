"""
 * [INPUT]: 依赖 torch, monai.networks.nets.SwinUNETR, os/pathlib, warnings
 * [OUTPUT]: 对外提供 load_swin_unetr_pretrained, freeze_encoder, unfreeze_layers
 * [POS]: src/models/ 的权重管理工具, 被 autoencoder 初始化流程消费，并将 MONAI 预训练权重缓存收口到项目内可写目录
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import os
import warnings
from pathlib import Path
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


def _resolve_pretrained_cache_path() -> Path:
    """
    将预训练权重缓存收口到项目内可写目录。
    """
    project_root = Path(__file__).resolve().parents[2]
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", project_root / ".cache")
    ).expanduser()
    cache_dir = cache_root / "monai" / "pretrained"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "ssl_pretrained_weights.pth"


def _resolve_pretrained_weights_source(
    checkpoint_path: str | None,
) -> Path | None:
    """
    优先使用显式传入的本地权重文件。
    """
    if checkpoint_path is None:
        return None

    path = Path(checkpoint_path).expanduser()
    return path if path.is_file() else None


def _download_pretrained_weights(local_path: Path) -> Path:
    """
    下载并缓存官方 SSL 预训练权重。
    """
    from monai.apps import download_url

    if not local_path.exists():
        warnings.warn(
            f"Downloading SwinUNETR SSL pretrained weights from {SSL_PRETRAINED_URL} ..."
        )
        download_url(SSL_PRETRAINED_URL, str(local_path))

    return local_path


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
    local_checkpoint = _resolve_pretrained_weights_source(checkpoint_path)
    if local_checkpoint is not None:
        _load_weights_file(model, str(local_checkpoint))
        return

    # 尝试下载并加载官方 SSL 预训练权重
    try:
        local_path = _download_pretrained_weights(_resolve_pretrained_cache_path())
        _load_weights_file(model, str(local_path))

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
    ssl_weights = _extract_state_dict(checkpoint)

    _copy_filtered_weights(model, ssl_weights)


def _extract_state_dict(checkpoint: dict) -> dict:
    """
    兼容 MONAI 常见 checkpoint 结构。
    """
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _copy_filtered_weights(model: SwinUNETR, ssl_weights: dict) -> None:
    """
    使用 MONAI 官方过滤器完成键名映射与权重拷贝。
    """
    # 使用 MONAI 官方的 filter_swinunetr 进行键名映射
    _, loaded, not_loaded = copy_model_state(
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
