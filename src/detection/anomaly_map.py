"""
 * [INPUT]: 依赖 numpy, skimage.filters.threshold_otsu, skimage.measure.label
 * [OUTPUT]: 对外提供 compute_anomaly_map, threshold_anomaly_map, postprocess_connected_components
 * [POS]: src/detection 的基础后处理原语，负责把重建误差变成异常图和稳定二值掩码
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
 */
"""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label


# ============================================================
# Anomaly Map Computation
# ============================================================

def compute_anomaly_map(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """计算原图和重建图之间的逐体素绝对误差。"""
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    return np.abs(original - reconstructed)


# ============================================================
# Thresholding
# ============================================================

def threshold_anomaly_map(
    anomaly_map: np.ndarray,
    method: str = "otsu",
    threshold: float | None = None,
) -> np.ndarray:
    """把异常图阈值化成 0/1 掩码。"""
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    if method == "otsu":
        thresh = float(threshold_otsu(anomaly_map))
    elif method == "fixed":
        if threshold is None:
            raise ValueError("fixed method requires threshold")
        thresh = float(threshold)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return (anomaly_map > thresh).astype(np.uint8)


# ============================================================
# Connected Component Postprocess
# ============================================================

def postprocess_connected_components(
    binary_map: np.ndarray,
    *,
    min_size_voxels: int = 0,
    keep_largest_component: bool = False,
    return_labeled: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, list[int]]:
    """删除小连通域，必要时只保留最大连通域。"""
    binary = np.asarray(binary_map, dtype=np.uint8) > 0
    if binary.ndim != 3:
        raise ValueError(f"Expected 3D binary map, got {binary.ndim}D")
    if not np.any(binary):
        empty = binary.astype(np.uint8)
        if return_labeled:
            return empty, np.zeros_like(empty, dtype=np.int32), []
        return empty

    labeled = label(binary.astype(np.uint8), connectivity=1)
    component_ids = np.unique(labeled)
    component_ids = component_ids[component_ids != 0]
    if component_ids.size == 0:
        empty = np.zeros_like(binary, dtype=np.uint8)
        if return_labeled:
            return empty, np.zeros_like(empty, dtype=np.int32), []
        return empty

    component_sizes = np.bincount(labeled.reshape(-1))
    component_sizes[0] = 0
    kept_mask = component_sizes >= int(min_size_voxels)
    kept_mask[0] = False

    if not np.any(kept_mask):
        empty = np.zeros_like(binary, dtype=np.uint8)
        if return_labeled:
            return empty, np.zeros_like(empty, dtype=np.int32), []
        return empty

    if keep_largest_component:
        largest_id = int(np.argmax(component_sizes))
        kept_mask = np.zeros_like(kept_mask, dtype=bool)
        kept_mask[largest_id] = True

    kept_ids = np.flatnonzero(kept_mask).tolist()
    cleaned = kept_mask[labeled]
    cleaned_uint8 = cleaned.astype(np.uint8)
    if return_labeled:
        cleaned_labels = (labeled * cleaned).astype(np.int32)
        return cleaned_uint8, cleaned_labels.astype(np.int32), kept_ids
    return cleaned_uint8
