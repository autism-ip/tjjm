"""
 * [INPUT]: 依赖 numpy, skimage.transform
 * [OUTPUT]: 对外提供 hu_windowing, resample_to_spacing, normalize
 * [POS]: data/ 的强度与重采样子模块，被 preprocess.py 门面与 dataset.py 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from typing import Tuple

import numpy as np
from skimage.transform import resize


def hu_windowing(
    ct_array: np.ndarray,
    hu_min: int = -1024,
    hu_max: int = 3071,
) -> np.ndarray:
    """
    将 CT 的 HU 值裁剪到指定窗口范围。
    """
    return np.clip(ct_array, hu_min, hu_max)


def resample_to_spacing(
    ct_array: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    将 CT 重采样到目标 spacing。
    """
    if ct_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {ct_array.ndim}D")

    scale_factors = tuple(
        orig / tgt for orig, tgt in zip(original_spacing, target_spacing)
    )
    new_shape = tuple(
        max(1, int(round(size * factor)))
        for size, factor in zip(ct_array.shape, scale_factors)
    )

    resampled = resize(
        ct_array,
        output_shape=new_shape,
        order=1,
        mode="constant",
        anti_aliasing=True,
        preserve_range=True,
    )
    return resampled.astype(np.float32)


def normalize(
    ct_array: np.ndarray,
    method: str = "minmax",
    out_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    归一化 CT 数组。
    """
    arr = ct_array.astype(np.float32)

    if method == "minmax":
        return _normalize_minmax(arr, out_range)
    if method == "zscore":
        return _normalize_zscore(arr)

    raise ValueError(f"Unknown normalize method: {method}")


def _normalize_minmax(
    arr: np.ndarray,
    out_range: Tuple[float, float],
) -> np.ndarray:
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)

    lo, hi = out_range
    normed = (arr - min_val) / (max_val - min_val)
    return normed * (hi - lo) + lo


def _normalize_zscore(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - mean) / std
