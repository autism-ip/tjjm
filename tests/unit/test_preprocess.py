"""
 * [INPUT]: 依赖 numpy, pytest, scipy.ndimage, skimage.transform
 * [OUTPUT]: 对外提供 preprocess 模块的单元测试
 * [POS]: tests/unit/ 的数据预处理验证器, 覆盖 HU 窗口/重采样/归一化/patch 提取/健康筛选
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import numpy as np
import pytest

from src.data.preprocess import (
    hu_windowing,
    resample_to_spacing,
    normalize,
    extract_patches,
    filter_healthy_patches,
)


# ============================================================
# HU Windowing Tests
# ============================================================

def test_hu_windowing_clips_min():
    """低于 hu_min 的值应被裁剪到 hu_min"""
    arr = np.array([-2000, -1024, -500], dtype=np.float32)
    out = hu_windowing(arr, hu_min=-1024, hu_max=3071)
    assert out[0] == pytest.approx(-1024, rel=1e-6)
    assert out[1] == pytest.approx(-1024, rel=1e-6)
    assert out[2] == pytest.approx(-500, rel=1e-6)


def test_hu_windowing_clips_max():
    """高于 hu_max 的值应被裁剪到 hu_max"""
    arr = np.array([3000, 3071, 4000], dtype=np.float32)
    out = hu_windowing(arr, hu_min=-1024, hu_max=3071)
    assert out[0] == pytest.approx(3000, rel=1e-6)
    assert out[1] == pytest.approx(3071, rel=1e-6)
    assert out[2] == pytest.approx(3071, rel=1e-6)


def test_hu_windowing_range():
    """裁剪后数值范围应在 [hu_min, hu_max] 内"""
    rng = np.random.RandomState(42)
    arr = rng.randint(-3000, 5000, size=(100, 100, 100)).astype(np.float32)
    out = hu_windowing(arr, hu_min=-1024, hu_max=3071)
    assert out.min() >= -1024
    assert out.max() <= 3071


def test_hu_windowing_does_not_mutate():
    """不应修改输入数组（不可变性）"""
    arr = np.array([-2000, 0, 4000], dtype=np.float32)
    original = arr.copy()
    hu_windowing(arr, hu_min=-1024, hu_max=3071)
    np.testing.assert_array_equal(arr, original)


# ============================================================
# Resample Tests
# ============================================================

def test_resample_to_spacing_shape():
    """重采样到 1mm spacing 后 shape 应正确缩放"""
    arr = np.ones((100, 100, 100), dtype=np.float32)
    original_spacing = (2.0, 2.0, 2.5)
    target_spacing = (1.0, 1.0, 1.0)
    out = resample_to_spacing(arr, original_spacing, target_spacing)
    # 期望 shape: (200, 200, 250)
    assert out.shape == (200, 200, 250)


def test_resample_to_spacing_identity():
    """spacing 已为目标值时 shape 不变"""
    arr = np.ones((64, 64, 64), dtype=np.float32)
    out = resample_to_spacing(arr, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert out.shape == (64, 64, 64)


def test_resample_to_spacing_values_preserved():
    """均匀数组重采样后数值应保持一致"""
    arr = np.full((50, 50, 50), 3.14, dtype=np.float32)
    out = resample_to_spacing(arr, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    assert np.allclose(out, 3.14, atol=0.1)


def test_resample_to_spacing_does_not_mutate():
    """不应修改输入数组"""
    arr = np.ones((32, 32, 32), dtype=np.float32)
    original = arr.copy()
    resample_to_spacing(arr, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    np.testing.assert_array_equal(arr, original)


# ============================================================
# Normalize Tests
# ============================================================

def test_normalize_minmax_range():
    """minmax 归一化后范围应在 [-1, 1]"""
    arr = np.array([0, 50, 100], dtype=np.float32)
    out = normalize(arr, method="minmax", out_range=(-1, 1))
    assert out.min() == pytest.approx(-1.0, rel=1e-6)
    assert out.max() == pytest.approx(1.0, rel=1e-6)


def test_normalize_minmax_zero():
    """全零数组归一化后应为全零"""
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    out = normalize(arr, method="minmax", out_range=(-1, 1))
    np.testing.assert_array_equal(out, np.zeros_like(arr))


def test_normalize_zscore_mean_std():
    """zscore 归一化后均值约 0，标准差约 1"""
    rng = np.random.RandomState(42)
    arr = rng.randn(100, 100, 100).astype(np.float32) * 10 + 5
    out = normalize(arr, method="zscore")
    assert abs(out.mean()) < 0.01
    assert abs(out.std() - 1.0) < 0.01


def test_normalize_does_not_mutate():
    """不应修改输入数组"""
    arr = np.array([10, 20, 30], dtype=np.float32)
    original = arr.copy()
    normalize(arr, method="minmax", out_range=(-1, 1))
    np.testing.assert_array_equal(arr, original)


# ============================================================
# Patch Extraction Tests
# ============================================================

def test_extract_patches_count():
    """patch 提取数量应正确"""
    arr = np.ones((128, 128, 128), dtype=np.float32)
    patch_size = (64, 64, 64)
    stride = 32
    patches, centers = extract_patches(arr, patch_size, stride)
    # 每维: (128 - 64) // 32 + 1 = 3
    expected = 3 * 3 * 3
    assert len(patches) == expected
    assert len(centers) == expected


def test_extract_patches_shape():
    """每个 patch 的 shape 应正确"""
    arr = np.ones((100, 100, 100), dtype=np.float32)
    patches, centers = extract_patches(arr, (64, 64, 64), 32)
    for p in patches:
        assert p.shape == (64, 64, 64)


def test_extract_patches_centers():
    """patch 中心坐标应正确"""
    arr = np.zeros((64, 64, 64), dtype=np.float32)
    patches, centers = extract_patches(arr, (64, 64, 64), 64)
    assert len(patches) == 1
    assert centers[0] == (32, 32, 32)


def test_extract_patches_does_not_mutate():
    """不应修改输入数组"""
    arr = np.ones((128, 128, 128), dtype=np.float32)
    original = arr.copy()
    extract_patches(arr, (64, 64, 64), 32)
    np.testing.assert_array_equal(arr, original)


def test_extract_patches_empty_when_too_small():
    """CT 小于 patch_size 时应返回空列表"""
    arr = np.ones((32, 32, 32), dtype=np.float32)
    patches, centers = extract_patches(arr, (64, 64, 64), 32)
    assert len(patches) == 0
    assert len(centers) == 0


# ============================================================
# Filter Healthy Patches Tests
# ============================================================

def test_filter_healthy_patches_removes_nodule():
    """包含结节的 patch 应被过滤掉"""
    patches = [np.ones((64, 64, 64)) for _ in range(3)]
    # patch 中心: (32,32,32), (96,96,96), (160,160,160)
    centers = [(32, 32, 32), (96, 96, 96), (160, 160, 160)]
    annotations = [
        {"seriesuid": "1", "coordX": 32.0, "coordY": 32.0, "coordZ": 32.0, "diameter_mm": 10.0}
    ]
    healthy, healthy_centers = filter_healthy_patches(
        patches, annotations, centers, spacing=(1.0, 1.0, 1.0), nodule_margin_ratio=1.5
    )
    assert len(healthy) == 2
    assert len(healthy_centers) == 2
    assert (32, 32, 32) not in healthy_centers


def test_filter_healthy_patches_keeps_far_away():
    """远离结节的 patch 应保留"""
    patches = [np.ones((64, 64, 64)) for _ in range(2)]
    centers = [(1000, 1000, 1000), (2000, 2000, 2000)]
    annotations = [
        {"seriesuid": "1", "coordX": 0.0, "coordY": 0.0, "coordZ": 0.0, "diameter_mm": 10.0}
    ]
    healthy, healthy_centers = filter_healthy_patches(
        patches, annotations, centers, spacing=(1.0, 1.0, 1.0), nodule_margin_ratio=1.5
    )
    assert len(healthy) == 2


def test_filter_healthy_patches_empty_annotations():
    """空标注时应保留所有 patch"""
    patches = [np.ones((64, 64, 64)) for _ in range(3)]
    centers = [(32, 32, 32), (96, 96, 96), (160, 160, 160)]
    healthy, healthy_centers = filter_healthy_patches(
        patches, [], centers, spacing=(1.0, 1.0, 1.0), nodule_margin_ratio=1.5
    )
    assert len(healthy) == 3


def test_filter_healthy_patches_margin_ratio():
    """margin_ratio 应扩大排除范围"""
    patches = [np.ones((64, 64, 64)) for _ in range(1)]
    centers = [(50, 50, 50)]
    annotations = [
        {"seriesuid": "1", "coordX": 0.0, "coordY": 0.0, "coordZ": 0.0, "diameter_mm": 20.0}
    ]
    # 直径 20, 半径 10, margin 1.5 -> 排除距离 15
    # 中心距离 sqrt(50^2*3) ~ 86.6 > 15, 应保留
    healthy, _ = filter_healthy_patches(
        patches, annotations, centers, spacing=(1.0, 1.0, 1.0), nodule_margin_ratio=1.5
    )
    assert len(healthy) == 1

    # margin 扩大到 10 -> 排除距离 100, 应排除
    healthy2, _ = filter_healthy_patches(
        patches, annotations, centers, spacing=(1.0, 1.0, 1.0), nodule_margin_ratio=10.0
    )
    assert len(healthy2) == 0
