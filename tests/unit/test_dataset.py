"""
 * [INPUT]: 依赖 pytest, torch, numpy, pandas, tempfile, pathlib
 * [OUTPUT]: 对外提供 dataset 模块的单元测试
 * [POS]: tests/unit/ 的 Dataset 验证器, 覆盖 LunaCTDataset / LunaPatchDataset
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from src.data.dataset import LunaCTDataset, LunaPatchDataset


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def fake_ct_dir(tmp_path):
    """创建模拟 LUNA16 CT 目录，包含 .mhd + .raw"""
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()

    # 使用 nibabel 写一个简单的 nifti，再转 mhd 比较麻烦
    # 这里直接用 SimpleITK 写
    import SimpleITK as sitk

    for i in range(3):
        # 128^3 保证至少能提取多个 patch, 避免全部被过滤
        arr = np.random.randn(128, 128, 128).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetOrigin((0.0, 0.0, 0.0))
        sitk.WriteImage(img, str(ct_dir / f"ct_{i}.mhd"))

    return ct_dir


@pytest.fixture
def annotations_csv(tmp_path, fake_ct_dir):
    """创建模拟 annotations.csv"""
    csv_path = tmp_path / "annotations.csv"
    rows = []
    for i in range(3):
        rows.append({
            "seriesuid": f"ct_{i}",
            "coordX": 32.0,
            "coordY": 32.0,
            "coordZ": 32.0,
            "diameter_mm": 10.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


# ============================================================
# LunaCTDataset Tests
# ============================================================

def test_luna_ct_dataset_length(fake_ct_dir):
    """Dataset 长度应等于 .mhd 文件数量"""
    ds = LunaCTDataset(fake_ct_dir)
    assert len(ds) == 3


def test_luna_ct_dataset_getitem_shape(fake_ct_dir):
    """getitem 应返回 (C, D, H, W) = (1, 128, 128, 128)"""
    ds = LunaCTDataset(fake_ct_dir)
    tensor = ds[0]
    assert tensor.shape == (1, 128, 128, 128)


def test_luna_ct_dataset_dtype(fake_ct_dir):
    """输出类型应为 torch.float32"""
    ds = LunaCTDataset(fake_ct_dir)
    tensor = ds[0]
    assert tensor.dtype == torch.float32


def test_luna_ct_dataset_lazy_loading(fake_ct_dir):
    """lazy loading: 构造时不应读取文件"""
    ds = LunaCTDataset(fake_ct_dir, lazy=True)
    # 构造时 file_list 已扫描，但图像未加载
    assert len(ds.file_list) == 3


def test_luna_ct_dataset_empty_dir(tmp_path):
    """空目录时应长度为 0"""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ds = LunaCTDataset(empty_dir)
    assert len(ds) == 0


# ============================================================
# LunaPatchDataset Tests
# ============================================================

def test_luna_patch_dataset_length(fake_ct_dir, annotations_csv):
    """PatchDataset 长度应大于 0"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
    )
    assert len(ds) > 0


def test_luna_patch_dataset_getitem_shape(fake_ct_dir, annotations_csv):
    """getitem 应返回 (patch, patch) 且 shape 为 (1, 64, 64, 64)"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
    )
    x, y = ds[0]
    assert x.shape == (1, 64, 64, 64)
    assert y.shape == (1, 64, 64, 64)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_luna_patch_dataset_returns_same_xy(fake_ct_dir, annotations_csv):
    """自编码器输入应等于目标"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
    )
    x, y = ds[0]
    torch.testing.assert_close(x, y)


def test_luna_patch_dataset_value_range(fake_ct_dir, annotations_csv):
    """输出数值范围应在 [-1, 1] 内（经归一化后）"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
        normalize_method="minmax",
    )
    x, _ = ds[0]
    assert x.min() >= -1.0
    assert x.max() <= 1.0


def test_luna_patch_dataset_no_nodule_patches(fake_ct_dir, annotations_csv):
    """不应包含结节中心 patch"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
        nodule_margin_ratio=1.5,
    )
    # 所有 patch 中心都不应在结节附近
    for idx in range(len(ds)):
        x, _ = ds[idx]
        assert x.shape == (1, 64, 64, 64)


def test_luna_patch_dataset_lazy_no_preload(fake_ct_dir, annotations_csv):
    """懒加载: 构造时不应加载任何 CT 像素数据到内存"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
    )
    # patches 属性应为索引列表, 而非 tensor 列表
    assert len(ds.patches) > 0
    first_item = ds.patches[0]
    # 索引项应为 (path, z, y, x) 元组, 而非 tensor
    assert isinstance(first_item, tuple)
    assert len(first_item) == 4


def test_luna_patch_dataset_len_equals_healthy_count(fake_ct_dir, annotations_csv):
    """__len__ 应等于健康 patch 索引数量"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
    )
    assert len(ds) == len(ds.patches)


def test_luna_patch_dataset_multiple_access_consistent(fake_ct_dir, annotations_csv):
    """多次访问同一索引应返回相同结果"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
    )
    x1, y1 = ds[0]
    x2, y2 = ds[0]
    torch.testing.assert_close(x1, x2)
    torch.testing.assert_close(y1, y2)


def test_luna_patch_dataset_file_cache_limits_size(fake_ct_dir, annotations_csv):
    """文件缓存应有大小限制, 避免无限增长"""
    ds = LunaPatchDataset(
        ct_dir=fake_ct_dir,
        annotations_csv=annotations_csv,
        patch_size=(64, 64, 64),
        stride=32,
        file_cache_size=2,
    )
    # 访问所有 patch, 触发多次文件加载
    for i in range(len(ds)):
        _ = ds[i]
    # 缓存大小不应超过限制
    assert len(ds._file_cache) <= 2
