"""
 * [INPUT]: 依赖 src.experiments.analysis, src.experiments.synthetic, src.experiments.io
 * [OUTPUT]: 对外提供实验层公共接口
 * [POS]: src/experiments/ 的入口，聚合健康统计、合成异常、汇总与输入适配函数
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from src.experiments.analysis import (
    save_summary,
    save_summary_text,
    summarize_anomaly_map_files,
    summarize_anomaly_maps,
    summarize_metric_records,
    summarize_metric_reports,
)
from src.experiments.io import iter_input_paths, load_array, load_report
from src.experiments.synthetic import (
    evaluate_synthetic_sensitivity,
    inject_spherical_anomaly,
    make_difference_score_fn,
    spherical_mask,
)

__all__ = [
    "save_summary",
    "save_summary_text",
    "summarize_anomaly_map_files",
    "summarize_anomaly_maps",
    "summarize_metric_records",
    "summarize_metric_reports",
    "iter_input_paths",
    "load_array",
    "load_report",
    "evaluate_synthetic_sensitivity",
    "inject_spherical_anomaly",
    "make_difference_score_fn",
    "spherical_mask",
]
