"""
 * [INPUT]: 依赖 detection.anomaly_map, detection.sliding_window, detection.fusion
 * [OUTPUT]: 对外提供 compute_anomaly_map, threshold_anomaly_map, sliding_window_reconstruct
 * [POS]: src/detection/ 的入口，聚合检测层全部公共接口
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from src.detection.anomaly_map import compute_anomaly_map, threshold_anomaly_map
from src.detection.inference import SlidingWindowDetector
from src.detection.sliding_window import sliding_window_reconstruct

__all__ = [
    "compute_anomaly_map",
    "SlidingWindowDetector",
    "threshold_anomaly_map",
    "sliding_window_reconstruct",
]
