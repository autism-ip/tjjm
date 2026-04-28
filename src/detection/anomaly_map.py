"""
 * [INPUT]: 依赖 numpy, skimage.filters
 * [OUTPUT]: 对外提供 compute_anomaly_map, threshold_anomaly_map
 * [POS]: src/detection/ 的异常热图生成器，被 sliding_window 与评估流程消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
from skimage.filters import threshold_otsu


# ============================================================
# Anomaly Map Computation
# ============================================================

def compute_anomaly_map(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """
    计算逐体素绝对差异图
    INPUT:  original      (D, H, W) 原始 CT
            reconstructed (D, H, W) 重建 CT
    OUTPUT: anomaly_map   (D, H, W) 非负差异图
    """
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
    """
    将连续异常图二值化
    method: "otsu" | "fixed"
    threshold: 固定阈值，method="fixed" 时必填
    OUTPUT: (D, H, W) uint8 二值图，0 或 1
    """
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    if method == "otsu":
        thresh = threshold_otsu(anomaly_map)
    elif method == "fixed":
        if threshold is None:
            raise ValueError("fixed method requires threshold")
        thresh = threshold
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return (anomaly_map > thresh).astype(np.uint8)
