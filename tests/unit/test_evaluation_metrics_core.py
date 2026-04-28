"""
 * [INPUT]: 依赖 numpy, pytest, src.evaluation.metrics
 * [OUTPUT]: 对外提供 evaluation.metrics 的核心单元测试
 * [POS]: tests/unit/ 的纯指标验证器，覆盖 Dice / AUC / F1 与连续分数阈值化契约
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
import pytest

from src.evaluation.metrics import compute_auc, compute_metrics, dice_score


def test_dice_perfect_match():
    pred = np.ones((10, 10, 10), dtype=np.float32)
    target = np.ones((10, 10, 10), dtype=np.float32)
    assert dice_score(pred, target) == pytest.approx(1.0, rel=1e-6)


def test_dice_no_overlap():
    pred = np.ones((10, 10, 10), dtype=np.float32)
    target = np.zeros((10, 10, 10), dtype=np.float32)
    assert dice_score(pred, target) == pytest.approx(0.0, rel=1e-6)


def test_dice_half_overlap():
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    target = np.zeros((10, 10, 10), dtype=np.float32)
    pred[:5, :, :] = 1.0
    target[:5, :, :] = 1.0
    pred[5:7, :, :] = 1.0
    assert dice_score(pred, target) == pytest.approx(5.0 / 6.0, rel=1e-6)


def test_dice_empty_both():
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    target = np.zeros((10, 10, 10), dtype=np.float32)
    assert dice_score(pred, target) == pytest.approx(1.0, rel=1e-6)


def test_auc_perfect_classification():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(1.0, rel=1e-6)


def test_auc_worst_classification():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(0.0, rel=1e-6)


def test_auc_random():
    rng = np.random.RandomState(42)
    scores = rng.rand(200)
    labels = rng.randint(0, 2, size=200)
    auc = compute_auc(scores, labels)
    assert 0.3 < auc < 0.7


def test_compute_metrics_perfect():
    pred = np.ones((10, 10, 10), dtype=np.float32)
    gt = np.ones((10, 10, 10), dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    assert metrics["dice"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["recall"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["precision"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["specificity"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["f1"] == pytest.approx(1.0, rel=1e-6)


def test_compute_metrics_zero():
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    gt = np.zeros((10, 10, 10), dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    assert metrics["dice"] == pytest.approx(1.0, rel=1e-6)


def test_f1_equals_harmonic_mean():
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    gt = np.zeros((10, 10, 10), dtype=np.float32)
    pred[:3, :, :] = 1.0
    gt[:5, :, :] = 1.0
    metrics = compute_metrics(pred, gt)
    p = metrics["precision"]
    r = metrics["recall"]
    expected_f1 = 2.0 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    assert metrics["f1"] == pytest.approx(expected_f1, rel=1e-6)


def test_compute_metrics_empty_pred():
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    gt = np.ones((10, 10, 10), dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    assert metrics["dice"] == pytest.approx(0.0, rel=1e-6)
    assert metrics["recall"] == pytest.approx(0.0, rel=1e-6)


def test_compute_metrics_thresholds_scores_with_requested_auc():
    scores = np.array([0.05, 0.90, 0.10, 0.80], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)

    metrics = compute_metrics(
        preds=scores,
        gts=labels,
        threshold_method="fixed",
        threshold=0.5,
        metrics_list=["dice", "recall", "precision", "specificity", "f1", "auc"],
    )

    assert metrics["dice"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["recall"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["precision"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["specificity"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["f1"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["auc"] == pytest.approx(1.0, rel=1e-6)
    assert "fpr" in metrics
    assert "tpr" in metrics


def test_compute_metrics_omits_auc_when_not_requested():
    scores = np.array([0.05, 0.90, 0.10, 0.80], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)

    metrics = compute_metrics(
        preds=scores,
        gts=labels,
        threshold_method="fixed",
        threshold=0.5,
        metrics_list=["dice"],
    )

    assert metrics == {"dice": pytest.approx(1.0, rel=1e-6)}
