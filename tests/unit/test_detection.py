"""
 * [INPUT]: 依赖 pathlib, numpy, pytest, torch, SimpleITK, detection.anomaly_map, detection.inference, detection.sliding_window
 * [OUTPUT]: 对外提供 detection 模块的单元测试
 * [POS]: tests/unit/ 的检测层验证器，覆盖异常图 / 阈值 / 重建尺寸 / 输出空间元数据
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from src.detection.anomaly_map import compute_anomaly_map, threshold_anomaly_map
from src.detection.inference import SlidingWindowDetector
from src.detection.sliding_window import sliding_window_reconstruct


# ============================================================
# Anomaly Map Tests
# ============================================================

def test_anomaly_map_non_negative():
    """异常图计算结果必须非负"""
    original = np.random.randn(64, 64, 64).astype(np.float32)
    reconstructed = np.random.randn(64, 64, 64).astype(np.float32)
    anomaly_map = compute_anomaly_map(original, reconstructed)
    assert np.all(anomaly_map >= 0)


def test_anomaly_map_zero_when_identical():
    """原始与重建完全一致时异常图应为全零"""
    original = np.random.randn(64, 64, 64).astype(np.float32)
    anomaly_map = compute_anomaly_map(original, original)
    assert np.allclose(anomaly_map, 0.0, atol=1e-6)


def test_threshold_binary_output():
    """阈值处理后值只能为 0 或 1"""
    anomaly_map = np.random.rand(32, 32, 32).astype(np.float32)
    binary = threshold_anomaly_map(anomaly_map, method="otsu")
    assert set(np.unique(binary).tolist()).issubset({0, 1})


def test_threshold_fixed_value():
    """固定阈值按预期分割"""
    anomaly_map = np.array([0.1, 0.5, 0.9], dtype=np.float32).reshape(3, 1, 1)
    binary = threshold_anomaly_map(anomaly_map, method="fixed", threshold=0.5)
    expected = np.array([0, 0, 1], dtype=np.uint8).reshape(3, 1, 1)
    assert np.array_equal(binary, expected)


# ============================================================
# Sliding Window Reconstruction Tests
# ============================================================

def test_reconstruction_same_shape():
    """重建结果与输入 CT 同尺寸"""

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()
    ct_scan = torch.randn(1, 128, 128, 128)
    reconstructed = sliding_window_reconstruct(
        model, ct_scan, patch_size=(64, 64, 64), stride=32, batch_size=4
    )
    assert reconstructed.shape == ct_scan.shape


def test_reconstruction_values_in_range():
    """重建值应在合理范围（输入归一化后约 [-1, 1]）"""

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x * 0.5

    model = DummyModel()
    ct_scan = torch.randn(1, 64, 64, 64)
    reconstructed = sliding_window_reconstruct(
        model, ct_scan, patch_size=(64, 64, 64), stride=64, batch_size=1
    )
    assert reconstructed.shape == ct_scan.shape
    assert torch.isfinite(reconstructed).all()


def test_save_nifti_preserves_reference_metadata(tmp_path: Path):
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    reference = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    reference.SetSpacing((0.7, 0.8, 2.5))
    reference.SetOrigin((-10.0, 20.0, 30.0))

    output_path = tmp_path / "anomaly.nii.gz"
    SlidingWindowDetector._save_nifti(arr, output_path, reference_image=reference)

    restored = sitk.ReadImage(str(output_path))
    assert restored.GetSpacing() == pytest.approx(reference.GetSpacing(), rel=1e-6)
    assert restored.GetOrigin() == pytest.approx(reference.GetOrigin(), rel=1e-6)
