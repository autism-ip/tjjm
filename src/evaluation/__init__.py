"""
 * [INPUT]: 依赖 evaluation.metrics, evaluation.reporter
 * [OUTPUT]: 对外提供 dice_score, compute_auc, compute_metrics, EvaluationReporter
 * [POS]: src/evaluation/ 的入口，聚合评估层全部公共接口
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from src.evaluation.metrics import dice_score, compute_auc, compute_metrics
from src.evaluation.reporter import EvaluationReporter

__all__ = [
    "dice_score",
    "compute_auc",
    "compute_metrics",
    "EvaluationReporter",
]
