"""
 * [INPUT]: 依赖 detection.anomaly_map, detection.inference, detection.sliding_window
 * [OUTPUT]: 对外提供 compute_anomaly_map, threshold_anomaly_map, postprocess_connected_components, SlidingWindowDetector, sliding_window_reconstruct
 * [POS]: src/detection 的统一导出层，向训练外流程暴露检测与后处理最小接口
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
 */
"""

from src.detection.anomaly_map import (
    compute_anomaly_map,
    postprocess_connected_components,
    threshold_anomaly_map,
)
from src.detection.inference import SlidingWindowDetector
from src.detection.sliding_window import sliding_window_reconstruct

__all__ = [
    "compute_anomaly_map",
    "postprocess_connected_components",
    "SlidingWindowDetector",
    "threshold_anomaly_map",
    "sliding_window_reconstruct",
]
