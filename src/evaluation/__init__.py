"""
 * [INPUT]: 依赖 evaluation.metrics, evaluation.reporter, evaluation.luna16
 * [OUTPUT]: 对外提供 dice_score, compute_auc, compute_metrics, EvaluationReporter, evaluate_luna16_case, evaluate_luna16_detection_dir, evaluate_luna16_threshold_sweep, load_luna16_annotations
 * [POS]: src/evaluation/ 的入口，聚合评估层全部公共接口
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from src.evaluation.metrics import dice_score, compute_auc, compute_metrics
from src.evaluation.luna16 import (
    evaluate_luna16_case,
    evaluate_luna16_detection_dir,
    evaluate_luna16_threshold_sweep,
    load_luna16_annotations,
)
from src.evaluation.reporter import EvaluationReporter

__all__ = [
    "dice_score",
    "compute_auc",
    "compute_metrics",
    "EvaluationReporter",
    "evaluate_luna16_case",
    "evaluate_luna16_detection_dir",
    "evaluate_luna16_threshold_sweep",
    "load_luna16_annotations",
]
