"""
 * [INPUT]: 依赖 math, numpy
 * [OUTPUT]: 对外提供 world_to_voxel, extract_patches, filter_healthy_patches
 * [POS]: data/ 的 patch 与坐标子模块，被 preprocess.py 门面与 dataset.py 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import math
from typing import List, Optional, Tuple

import numpy as np


def world_to_voxel(
    world: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    将世界坐标 (mm) 转换为体素索引。
    """
    return (np.asarray(world) - np.asarray(origin)) / np.asarray(spacing)


def extract_patches(
    ct_array: np.ndarray,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    stride: int = 32,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    滑动窗口提取 3D patch。
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
                centers.append((z + pd // 2, y + ph // 2, x + pw // 2))

    return patches, centers


def filter_healthy_patches(
    patches: Optional[List[np.ndarray]],
    annotations: List[dict],
    patch_centers: List[Tuple[int, int, int]],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    nodule_margin_ratio: float = 1.5,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    过滤掉包含结节区域的 patch, 只保留健康组织 patch。
    """
    if not annotations:
        if patches is None:
            return [], patch_centers.copy()
        return patches.copy(), patch_centers.copy()

    spacing_xyz = (spacing[2], spacing[1], spacing[0])
    origin_xyz = (origin[2], origin[1], origin[0])

    healthy_patches = []
    healthy_centers = []
    patch_iter = patches if patches is not None else [None] * len(patch_centers)

    for patch, center in zip(patch_iter, patch_centers):
        if _is_healthy_patch(
            center=center,
            annotations=annotations,
            spacing=spacing,
            spacing_xyz=spacing_xyz,
            origin_xyz=origin_xyz,
            nodule_margin_ratio=nodule_margin_ratio,
        ):
            if patch is not None:
                healthy_patches.append(patch)
            healthy_centers.append(center)

    return healthy_patches, healthy_centers


def _is_healthy_patch(
    center: Tuple[int, int, int],
    annotations: List[dict],
    spacing: Tuple[float, float, float],
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    nodule_margin_ratio: float,
) -> bool:
    cx, cy, cz = center

    for ann in annotations:
        voxel = _annotation_world_to_voxel(ann, origin_xyz, spacing_xyz)
        radius_mm = (ann["diameter_mm"] / 2.0) * nodule_margin_ratio
        dist_mm = _patch_distance_mm(center=(cx, cy, cz), voxel=voxel, spacing=spacing)
        if dist_mm <= radius_mm:
            return False

    return True


def _annotation_world_to_voxel(
    annotation: dict,
    origin_xyz: Tuple[float, float, float],
    spacing_xyz: Tuple[float, float, float],
) -> np.ndarray:
    world = np.array(
        [annotation["coordX"], annotation["coordY"], annotation["coordZ"]]
    )
    return world_to_voxel(world, origin_xyz, spacing_xyz)


def _patch_distance_mm(
    center: Tuple[int, int, int],
    voxel: np.ndarray,
    spacing: Tuple[float, float, float],
) -> float:
    cx, cy, cz = center
    nx, ny, nz = voxel
    return math.sqrt(
        ((cx - nx) * spacing[2]) ** 2
        + ((cy - ny) * spacing[1]) ** 2
        + ((cz - nz) * spacing[0]) ** 2
    )
