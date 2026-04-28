"""
 * [INPUT]: 依赖 numpy, skimage.filters, sklearn.metrics
 * [OUTPUT]: 对外提供 dice_score, compute_auc, compute_metrics
 * [POS]: src/evaluation/ 的核心指标计算器，被 reporter 与测试直接消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_auc_score, roc_curve


MASK_METRICS = ("dice", "recall", "precision", "specificity", "f1")
VALID_METRICS = set(MASK_METRICS) | {"auc"}


# ============================================================
# Dice Score
# ============================================================

def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算 Dice 相似系数
    Dice = 2 * |pred ∩ target| / (|pred| + |target|)
    """
    pred = pred.astype(np.float32).flatten()
    target = target.astype(np.float32).flatten()
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)


# ============================================================
# AUC
# ============================================================

def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 ROC-AUC
    scores: 异常分数，越高越异常
    labels: 0 = 正常，1 = 异常
    """
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


# ============================================================
# Thresholding
# ============================================================

def _threshold_scores(
    scores: np.ndarray,
    method: str = "otsu",
    threshold: float | None = None,
) -> np.ndarray:
    """
    将连续异常分数二值化。
    method: "otsu" | "fixed"
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return np.zeros_like(scores, dtype=np.uint8)
    if method == "otsu":
        thresh = threshold_otsu(scores)
    elif method == "fixed":
        if threshold is None:
            raise ValueError("fixed method requires threshold")
        thresh = threshold
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return (scores > thresh).astype(np.uint8)


def _requested_metrics(metrics_list) -> list[str]:
    if metrics_list is None:
        return list(MASK_METRICS)
    if isinstance(metrics_list, str):
        metrics = [metrics_list]
    else:
        metrics = list(metrics_list)
    requested = [str(metric).lower() for metric in metrics]
    unknown = [metric for metric in requested if metric not in VALID_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")
    return requested


def _flat_binary(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values).reshape(-1) > 0).astype(np.uint8)


def _roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    if len(np.unique(labels)) < 2:
        return [0.0, 1.0], [0.0, 1.0]
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr.tolist(), tpr.tolist()


def _metric_inputs(
    pred_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    *,
    preds: np.ndarray | None = None,
    gts: np.ndarray | None = None,
    threshold_method: str = "otsu",
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_input = preds if preds is not None else pred_mask
    gt_input = gts if gts is not None else gt_mask
    if pred_input is None or gt_input is None:
        raise ValueError("compute_metrics requires predictions and ground truth")

    scores = np.asarray(pred_input, dtype=np.float32).reshape(-1)
    gt = _flat_binary(gt_input)
    if scores.shape != gt.shape:
        raise ValueError("predictions and ground truth must have the same size")

    if preds is None:
        pred = _flat_binary(scores)
    else:
        pred = _threshold_scores(scores, threshold_method, threshold)
    return scores, pred, gt


def _mask_metric_values(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = float(np.sum(pred * gt))
    fp = float(np.sum(pred * (1 - gt)))
    fn = float(np.sum((1 - pred) * gt))
    tn = float(np.sum((1 - pred) * (1 - gt)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "dice": dice_score(pred, gt),
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
    }


# ============================================================
# Full Metrics Suite
# ============================================================

def compute_metrics(
    pred_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    *,
    preds: np.ndarray | None = None,
    gts: np.ndarray | None = None,
    threshold_method: str = "otsu",
    threshold: float | None = None,
    metrics_list=None,
) -> dict:
    """
    计算评估指标。
    pred_mask/gt_mask: 已二值化 mask。
    preds/gts: 连续异常分数与标签，先阈值化再算 mask 指标。
    """
    requested = _requested_metrics(metrics_list)
    scores, pred, gt = _metric_inputs(
        pred_mask,
        gt_mask,
        preds=preds,
        gts=gts,
        threshold_method=threshold_method,
        threshold=threshold,
    )
    mask_values = _mask_metric_values(pred, gt)
    metrics = {name: mask_values[name] for name in requested if name in mask_values}

    if "auc" in requested:
        metrics["auc"] = compute_auc(scores, gt)
        metrics["fpr"], metrics["tpr"] = _roc_points(scores, gt)

    return metrics
