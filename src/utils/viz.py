#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 os/pathlib 的缓存目录初始化，依赖 matplotlib 的绘图，依赖 numpy 的张量操作
 * [OUTPUT]: 对外提供 ensure_matplotlib_config_dir()、visualize_slice()、visualize_anomaly_map()、plot_roc_curve()
 * [POS]: src/utils/ 的可视化引擎与 Matplotlib 缓存初始化入口，被 detection/、evaluation/、notebooks/ 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
from pathlib import Path
from typing import Optional, Tuple


# ============================================================
# Matplotlib/fontconfig 缓存目录初始化
# ============================================================
def ensure_matplotlib_config_dir() -> Path:
    """
    为 Matplotlib 与 fontconfig 提供稳定、可写的项目内缓存目录。
    """
    project_root = Path(__file__).resolve().parents[2]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", project_root / ".cache")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    mpl_config_dir = Path(
        os.environ.get("MPLCONFIGDIR", cache_root / "matplotlib")
    ).expanduser()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    return mpl_config_dir


ensure_matplotlib_config_dir()

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# 单张切片可视化
# ============================================================
def visualize_slice(
    ct: np.ndarray,
    slice_idx: int,
    title: Optional[str] = None,
    cmap: str = "gray",
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    显示 CT 单张轴向切片。

    Args:
        ct: CT 体积，形状 (D, H, W) 或 (C, D, H, W)。
        slice_idx: 要显示的切片索引（沿 D 轴）。
        title: 图像标题。
        cmap: 颜色映射。
        figsize: 图像尺寸。
        save_path: 若提供，保存到该路径。

    Returns:
        matplotlib Figure 对象。
    """
    if ct.ndim == 4:
        ct = ct[0]

    if slice_idx < 0 or slice_idx >= ct.shape[0]:
        raise ValueError(
            f"slice_idx {slice_idx} out of bounds for depth {ct.shape[0]}"
        )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(ct[slice_idx], cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================
# 异常热图叠加可视化
# ============================================================
def visualize_anomaly_map(
    ct: np.ndarray,
    anomaly_map: np.ndarray,
    slice_idx: int,
    alpha: float = 0.5,
    cmap_ct: str = "gray",
    cmap_heat: str = "hot",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    并排显示 CT 切片、异常热图、叠加图。

    Args:
        ct: CT 体积，形状 (D, H, W) 或 (C, D, H, W)。
        anomaly_map: 异常分数体积，形状与 ct 一致。
        slice_idx: 切片索引。
        alpha: 热图叠加透明度。
        cmap_ct: CT 颜色映射。
        cmap_heat: 热图颜色映射。
        figsize: 图像尺寸。
        save_path: 若提供，保存到该路径。

    Returns:
        matplotlib Figure 对象。
    """
    if ct.ndim == 4:
        ct = ct[0]
    if anomaly_map.ndim == 4:
        anomaly_map = anomaly_map[0]

    if ct.shape != anomaly_map.shape:
        raise ValueError(
            f"Shape mismatch: ct {ct.shape} vs anomaly_map {anomaly_map.shape}"
        )

    ct_slice = ct[slice_idx]
    heat_slice = anomaly_map[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # CT 原图
    axes[0].imshow(ct_slice, cmap=cmap_ct)
    axes[0].set_title("CT Slice")
    axes[0].axis("off")

    # 异常热图
    im = axes[1].imshow(heat_slice, cmap=cmap_heat)
    axes[1].set_title("Anomaly Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 叠加
    axes[2].imshow(ct_slice, cmap=cmap_ct)
    axes[2].imshow(heat_slice, cmap=cmap_heat, alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================
# ROC 曲线绘制
# ============================================================
def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制 ROC 曲线。

    Args:
        fpr: 假阳性率数组。
        tpr: 真阳性率数组。
        auc: AUC 值。
        figsize: 图像尺寸。
        save_path: 若提供，保存到该路径。

    Returns:
        matplotlib Figure 对象。
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
