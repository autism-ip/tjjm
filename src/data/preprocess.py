"""
 * [INPUT]: 依赖 numpy, scipy.ndimage, skimage.transform
 * [OUTPUT]: 对外提供 hu_windowing, resample_to_spacing, normalize, extract_patches, filter_healthy_patches, world_to_voxel
 * [POS]: data/ 的预处理引擎, 被 dataset.py 消费
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.transform import resize


# ============================================================
# World <-> Voxel Coordinate Conversion
# ============================================================

def world_to_voxel(
    world: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    将世界坐标 (mm) 转换为体素索引 (voxel index).

    Args:
        world:   世界坐标数组, 任意维度
        origin:  原点偏移, 与 world 同维度
        spacing: 体素间距, 与 world 同维度

    Returns:
        体素坐标数组, 公式: (world - origin) / spacing
    """
    return (np.asarray(world) - np.asarray(origin)) / np.asarray(spacing)


# ============================================================
# HU Windowing
# ============================================================

def hu_windowing(
    ct_array: np.ndarray,
    hu_min: int = -1024,
    hu_max: int = 3071,
) -> np.ndarray:
    """
    将 CT 的 HU 值裁剪到指定窗口范围.

    Args:
        ct_array: 原始 CT 数组
        hu_min:   窗口下限
        hu_max:   窗口上限

    Returns:
        裁剪后的新数组, 不修改输入
    """
    return np.clip(ct_array, hu_min, hu_max)


# ============================================================
# Resample
# ============================================================

def resample_to_spacing(
    ct_array: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    将 CT 重采样到目标 spacing (默认 1mm 各向同性).

    Args:
        ct_array:         原始数组 (D, H, W)
        original_spacing: 原始体素间距 (z, y, x)
        target_spacing:   目标体素间距 (z, y, x)

    Returns:
        重采样后的新数组
    """
    if ct_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {ct_array.ndim}D")

    scale_factors = tuple(
        orig / tgt for orig, tgt in zip(original_spacing, target_spacing)
    )
    new_shape = tuple(
        max(1, int(round(s * factor)))
        for s, factor in zip(ct_array.shape, scale_factors)
    )

    # 使用 order=1 (双线性/三线性) 保持速度, anti_aliasing 防止混叠
    resampled = resize(
        ct_array,
        output_shape=new_shape,
        order=1,
        mode="constant",
        anti_aliasing=True,
        preserve_range=True,
    )
    return resampled.astype(np.float32)


# ============================================================
# Normalize
# ============================================================

def normalize(
    ct_array: np.ndarray,
    method: str = "minmax",
    out_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    归一化 CT 数组.

    Args:
        ct_array:  输入数组
        method:    "minmax" | "zscore"
        out_range: minmax 时的输出范围

    Returns:
        归一化后的新数组
    """
    arr = ct_array.astype(np.float32)

    if method == "minmax":
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        normed = (arr - min_val) / (max_val - min_val)
        lo, hi = out_range
        normed = normed * (hi - lo) + lo
        return normed

    if method == "zscore":
        mean = arr.mean()
        std = arr.std()
        if std < 1e-8:
            return np.zeros_like(arr)
        return (arr - mean) / std

    raise ValueError(f"Unknown normalize method: {method}")


# ============================================================
# Patch Extraction
# ============================================================

def extract_patches(
    ct_array: np.ndarray,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    stride: int = 32,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    滑动窗口提取 3D patch.

    Args:
        ct_array:   3D 数组 (D, H, W)
        patch_size: (patch_d, patch_h, patch_w)
        stride:     滑动步长

    Returns:
        (patches 列表, centers 列表)
    """
    if ct_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {ct_array.ndim}D")

    d, h, w = ct_array.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        return [], []

    patches = []
    centers = []

    for z in range(0, d - pd + 1, stride):
        for y in range(0, h - ph + 1, stride):
            for x in range(0, w - pw + 1, stride):
                patch = ct_array[z : z + pd, y : y + ph, x : x + pw]
                patches.append(patch.copy())
                centers.append(
                    (z + pd // 2, y + ph // 2, x + pw // 2)
                )

    return patches, centers


# ============================================================
# Filter Healthy Patches
# ============================================================

def filter_healthy_patches(
    patches: Optional[List[np.ndarray]],
    annotations: List[dict],
    patch_centers: List[Tuple[int, int, int]],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    nodule_margin_ratio: float = 1.5,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    过滤掉包含结节区域的 patch, 只保留健康组织 patch.

    Args:
        patches:             patch 列表
        annotations:         结节标注列表, 每项为 dict 含 coordX/Y/Z, diameter_mm
        patch_centers:       每个 patch 的物理中心坐标 (与 annotations 同坐标系)
        spacing:             体素间距 (mm)
        nodule_margin_ratio: 排除半径 = radius * ratio

    Returns:
        (健康 patches, 健康 centers)
    """
    if not annotations:
        if patches is None:
            return [], patch_centers.copy()
        return patches.copy(), patch_centers.copy()

    # spacing / origin 在项目中约定为 (z, y, x)
    # 世界坐标 (coordX, coordY, coordZ) 为 (x, y, z)
    # 调用 world_to_voxel 前重排为 (x, y, z) 以匹配测试契约
    spacing_xyz = (spacing[2], spacing[1], spacing[0])
    origin_xyz = (origin[2], origin[1], origin[0])

    healthy_patches = []
    healthy_centers = []

    patch_iter = patches if patches is not None else [None] * len(patch_centers)
    for patch, center in zip(patch_iter, patch_centers):
        is_healthy = True
        cx, cy, cz = center

        for ann in annotations:
            world = np.array([ann["coordX"], ann["coordY"], ann["coordZ"]])
            voxel = world_to_voxel(world, origin_xyz, spacing_xyz)
            nx, ny, nz = voxel

            diameter = ann["diameter_mm"]
            radius_mm = (diameter / 2.0) * nodule_margin_ratio

            # 计算 patch 中心到结节中心的欧氏距离 (mm)
            dist_mm = math.sqrt(
                ((cx - nx) * spacing[2]) ** 2
                + ((cy - ny) * spacing[1]) ** 2
                + ((cz - nz) * spacing[0]) ** 2
            )

            if dist_mm <= radius_mm:
                is_healthy = False
                break

        if is_healthy:
            if patch is not None:
                healthy_patches.append(patch)
            healthy_centers.append(center)

    return healthy_patches, healthy_centers
