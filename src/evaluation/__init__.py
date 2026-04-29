"""
 * [INPUT]: 依赖 evaluation.metrics, evaluation.reporter, evaluation.luna16
 * [OUTPUT]: 对外提供 dice_score, compute_auc, compute_metrics, EvaluationReporter, load_luna16_annotations, evaluate_luna16_case, evaluate_luna16_detection_dir, evaluate_luna16_threshold_sweep, build_luna16_froc_curve, select_luna16_operating_point
 * [POS]: src/evaluation 的统一导出层，向脚本与实验协议暴露可复用评估接口
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
 */
"""

from src.evaluation.luna16 import (
    build_luna16_froc_curve,
    evaluate_luna16_case,
    evaluate_luna16_detection_dir,
    evaluate_luna16_threshold_sweep,
    load_luna16_annotations,
    select_luna16_operating_point,
)
from src.evaluation.metrics import compute_auc, compute_metrics, dice_score
from src.evaluation.reporter import EvaluationReporter

__all__ = [
    "dice_score",
    "compute_auc",
    "compute_metrics",
    "EvaluationReporter",
    "load_luna16_annotations",
    "evaluate_luna16_case",
    "evaluate_luna16_detection_dir",
    "evaluate_luna16_threshold_sweep",
    "build_luna16_froc_curve",
    "select_luna16_operating_point",
]
