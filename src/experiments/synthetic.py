"""
 * [INPUT]: 依赖 numpy, src.experiments.analysis 的统计接口
 * [OUTPUT]: 对外提供 spherical_mask, inject_spherical_anomaly, evaluate_synthetic_sensitivity
 * [POS]: src/experiments/ 的合成异常实验层, 负责敏感性测试与干预分析
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def spherical_mask(
    shape: Sequence[int],
    center: Sequence[int] | None = None,
    radius: int = 5,
) -> np.ndarray:
    """
    生成 3D 球形掩码。
    """
    if len(shape) != 3:
        raise ValueError("shape must be 3D")

    center = tuple(int(value) for value in (center or tuple(dim // 2 for dim in shape)))
    grids = np.ogrid[tuple(slice(0, dim) for dim in shape)]
    distance_sq = np.zeros(shape, dtype=np.float32)
    for axis, grid in enumerate(grids):
        distance_sq += (grid - center[axis]) ** 2
    return distance_sq <= float(radius) ** 2


def inject_spherical_anomaly(
    volume: np.ndarray,
    center: Sequence[int] | None = None,
    radius: int = 5,
    intensity: float = 1.0,
    mode: str = "add",
) -> tuple[np.ndarray, np.ndarray]:
    """
    向体积中注入球形合成异常。

    mode:
        add - 在病灶区域上做加法扰动
        set - 直接覆盖为指定强度
    """
    arr = np.asarray(volume, dtype=np.float32)
    mask = spherical_mask(arr.shape, center=center, radius=radius)
    modified = arr.copy()

    if mode == "add":
        modified[mask] = modified[mask] + intensity
    elif mode == "set":
        modified[mask] = intensity
    else:
        raise ValueError(f"Unknown synthetic mode: {mode}")

    return modified, mask


def evaluate_synthetic_sensitivity(
    volume: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    radii: Sequence[int] = (4, 6, 8),
    intensities: Sequence[float] = (0.5, 1.0),
    center: Sequence[int] | None = None,
    mode: str = "add",
) -> list[dict[str, float | int]]:
    """
    对不同合成异常参数组合做敏感性评估。
    score_fn 应返回与输入体积同形状的 anomaly score map。
    """
    base = np.asarray(volume, dtype=np.float32)
    results: list[dict[str, float | int]] = []

    for radius in radii:
        for intensity in intensities:
            modified, mask = inject_spherical_anomaly(
                base,
                center=center,
                radius=radius,
                intensity=float(intensity),
                mode=mode,
            )
            scores = np.asarray(score_fn(modified), dtype=np.float32)
            if scores.shape != base.shape:
                raise ValueError("score_fn must return a map with the same shape as volume")

            lesion_scores = scores[mask]
            background_scores = scores[~mask]
            lesion_mean = float(np.mean(lesion_scores)) if lesion_scores.size else 0.0
            background_mean = float(np.mean(background_scores)) if background_scores.size else 0.0

            results.append(
                {
                    "radius": int(radius),
                    "intensity": float(intensity),
                    "lesion_voxels": int(mask.sum()),
                    "lesion_mean": lesion_mean,
                    "background_mean": background_mean,
                    "contrast": float(lesion_mean - background_mean),
                    "ratio": float(lesion_mean / (background_mean + 1e-8)),
                }
            )
    return results


def make_difference_score_fn(reference: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    构造一个基于参考体积的差异分数函数，供 CLI 默认使用。
    """
    base = np.asarray(reference, dtype=np.float32)

    def _score_fn(candidate: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(candidate, dtype=np.float32) - base)

    return _score_fn
