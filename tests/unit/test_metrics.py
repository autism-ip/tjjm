"""
 * [INPUT]: 依赖 importlib/sys/pathlib, numpy, pytest, omegaconf, evaluation.metrics，按需导入 scripts.detect 与可视化模块
 * [OUTPUT]: 对外提供 metrics 模块、检测入口评估/checkpoint 回退与 Matplotlib 初始化约束的单元测试
 * [POS]: tests/unit/ 的核心验证器，覆盖 Dice / AUC / F1、入口评估数组规整、旧 checkpoint 兼容与绘图库缓存目录收敛
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import importlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.evaluation.metrics import dice_score, compute_auc, compute_metrics


# ============================================================
# Dice Score Tests
# ============================================================

def test_dice_perfect_match():
    """完美匹配时 Dice 应为 1.0"""
    pred = np.ones((10, 10, 10), dtype=np.float32)
    target = np.ones((10, 10, 10), dtype=np.float32)
    assert dice_score(pred, target) == pytest.approx(1.0, rel=1e-6)


def test_dice_no_overlap():
    """完全不重叠时 Dice 应为 0.0"""
    pred = np.ones((10, 10, 10), dtype=np.float32)
    target = np.zeros((10, 10, 10), dtype=np.float32)
    assert dice_score(pred, target) == pytest.approx(0.0, rel=1e-6)


def test_dice_half_overlap():
    """一半重叠时 Dice 应为 0.5"""
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    target = np.zeros((10, 10, 10), dtype=np.float32)
    pred[:5, :, :] = 1.0
    target[:5, :, :] = 1.0
    pred[5:7, :, :] = 1.0
    # intersection = 5*10*10 = 500
    # union = 7*10*10 + 5*10*10 = 1200
    # dice = 2*500 / (700+500) = 1000/1200 = 5/6
    assert dice_score(pred, target) == pytest.approx(5.0 / 6.0, rel=1e-6)


def test_dice_empty_both():
    """两者都为空时 Dice 应为 1.0（约定）"""
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    target = np.zeros((10, 10, 10), dtype=np.float32)
    assert dice_score(pred, target) == pytest.approx(1.0, rel=1e-6)


# ============================================================
# AUC Tests
# ============================================================

def test_auc_perfect_classification():
    """完美分类时 AUC 应为 1.0"""
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(1.0, rel=1e-6)


def test_auc_worst_classification():
    """完全反向分类时 AUC 应为 0.0"""
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(0.0, rel=1e-6)


def test_auc_random():
    """随机分数 AUC 约 0.5"""
    rng = np.random.RandomState(42)
    scores = rng.rand(200)
    labels = rng.randint(0, 2, size=200)
    auc = compute_auc(scores, labels)
    assert 0.3 < auc < 0.7


# ============================================================
# compute_metrics Tests
# ============================================================

def test_compute_metrics_perfect():
    """完美预测时所有指标应为 1.0"""
    pred = np.ones((10, 10, 10), dtype=np.float32)
    gt = np.ones((10, 10, 10), dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    assert metrics["dice"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["recall"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["precision"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["specificity"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["f1"] == pytest.approx(1.0, rel=1e-6)


def test_compute_metrics_zero():
    """全零预测且全零真值时约定 Dice=1.0，其余按定义"""
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    gt = np.zeros((10, 10, 10), dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    assert metrics["dice"] == pytest.approx(1.0, rel=1e-6)


def test_f1_equals_harmonic_mean():
    """F1 = 2 * (P * R) / (P + R)"""
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
    """预测全零，真值非零：recall=0, precision=nan/0, dice=0"""
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    gt = np.ones((10, 10, 10), dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    assert metrics["dice"] == pytest.approx(0.0, rel=1e-6)
    assert metrics["recall"] == pytest.approx(0.0, rel=1e-6)


def test_compute_metrics_thresholds_scores_with_requested_auc():
    """连续异常分数应先阈值化，再按需返回 AUC/ROC。"""
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
    """AUC 是可选指标，未请求时不污染报告字典。"""
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


def test_detect_evaluation_arrays_accept_numpy_outputs():
    """检测入口的评估收集器应处理 detector 返回的 numpy 数组。"""
    from scripts.detect import _collect_evaluation_arrays

    results = {
        "case_a": {
            "anomaly_map": np.array([[0.1, 0.9]], dtype=np.float32),
            "ground_truth": np.array([[0, 1]], dtype=np.uint8),
        },
        "case_b": {
            "anomaly_map": np.array([[0.2, 0.8]], dtype=np.float32),
            "ground_truth": np.array([[0, 1]], dtype=np.uint8),
        },
    }

    preds, gts = _collect_evaluation_arrays(results)

    np.testing.assert_allclose(preds, np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32))
    np.testing.assert_array_equal(gts, np.array([0, 1, 0, 1], dtype=np.uint8))


def test_detect_checkpoint_without_encoder_name_uses_supported_default(monkeypatch, tmp_path):
    """旧 checkpoint 缺少 encoder_name 时，检测入口应回退到唯一受支持的编码器。"""
    from scripts import detect

    captured = {}

    class DummyModel:
        def __init__(self, encoder_name, pretrained):
            captured["encoder_name"] = encoder_name
            captured["pretrained"] = pretrained

        def load_state_dict(self, state_dict):
            captured["state_dict"] = state_dict

        def to(self, device):
            captured["device"] = str(device)
            return self

        def eval(self):
            captured["eval_called"] = True
            return self

    class DummyDetector:
        def __init__(self, model, patch_size, stride, batch_size, device):
            captured["detector_args"] = {
                "patch_size": patch_size,
                "stride": stride,
                "batch_size": batch_size,
                "device": str(device),
            }

        def run_directory(self, test_ct_dir):
            captured["test_ct_dir"] = test_ct_dir
            return {}

    monkeypatch.setattr(detect, "Autoencoder3D", DummyModel)
    monkeypatch.setattr(detect, "SlidingWindowDetector", DummyDetector)
    monkeypatch.setattr(detect, "setup_logging", lambda: None)
    monkeypatch.setattr(detect.torch, "load", lambda path, map_location: {"state_dict": {"weight": 1}})

    cfg = OmegaConf.create(
        {
            "data": {
                "output_dir": str(tmp_path / "outputs"),
                "test_ct_dir": str(tmp_path / "ct"),
            },
            "model": {
                "checkpoint_path": str(tmp_path / "model.ckpt"),
            },
            "detection": {
                "patch_size": [64, 64, 64],
                "stride": 32,
                "batch_size": 2,
            },
        }
    )

    detect.main.__wrapped__(cfg)

    assert captured["encoder_name"] == "swin_unetr"
    assert captured["pretrained"] is False
    assert captured["state_dict"] == {"weight": 1}
    assert captured["eval_called"] is True


def test_viz_import_initializes_project_matplotlib_cache(monkeypatch):
    """viz 模块导入前应绑定项目内稳定可写的 Matplotlib/fontconfig 缓存目录。"""
    project_root = Path(__file__).resolve().parents[2]
    cache_root = project_root / ".cache"
    mpl_config_dir = cache_root / "matplotlib"

    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delitem(sys.modules, "src.utils.viz", raising=False)

    importlib.import_module("src.utils.viz")

    assert Path(os.environ["MPLCONFIGDIR"]) == mpl_config_dir
    assert Path(os.environ["XDG_CACHE_HOME"]) == cache_root
    assert mpl_config_dir.is_dir()
    assert cache_root.is_dir()


def test_reporter_import_reuses_existing_matplotlib_cache(monkeypatch, tmp_path):
    """reporter 导入不应覆盖已有缓存配置。"""
    custom_cache = tmp_path / "custom-cache"
    custom_mpl = custom_cache / "matplotlib"
    custom_mpl.mkdir(parents=True)

    monkeypatch.setenv("XDG_CACHE_HOME", str(custom_cache))
    monkeypatch.setenv("MPLCONFIGDIR", str(custom_mpl))
    monkeypatch.delitem(sys.modules, "src.evaluation.reporter", raising=False)
    monkeypatch.delitem(sys.modules, "src.utils.viz", raising=False)

    importlib.import_module("src.evaluation.reporter")

    assert Path(os.environ["XDG_CACHE_HOME"]) == custom_cache
    assert Path(os.environ["MPLCONFIGDIR"]) == custom_mpl


def test_detect_import_initializes_matplotlib_cache_before_model_imports(monkeypatch):
    """detect 入口应在导入模型链之前固定项目内 Matplotlib 缓存目录。"""
    project_root = Path(__file__).resolve().parents[2]
    cache_root = project_root / ".cache"
    mpl_config_dir = cache_root / "matplotlib"

    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delitem(sys.modules, "scripts.detect", raising=False)

    importlib.import_module("scripts.detect")

    assert Path(os.environ["XDG_CACHE_HOME"]) == cache_root
    assert Path(os.environ["MPLCONFIGDIR"]) == mpl_config_dir
    assert cache_root.is_dir()
    assert mpl_config_dir.is_dir()
