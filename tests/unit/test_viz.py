#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 src.utils.viz 的绘图函数，依赖 pytest 的 fixture 与 tmp_path
 * [OUTPUT]: 对外提供 viz 模块的单元测试集合
 * [POS]: tests/unit/ 的可视化测试模块，覆盖缓存初始化、切片显示、异常热图、ROC 曲线
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ============================================================
# ensure_matplotlib_config_dir
# ============================================================
class TestEnsureMatplotlibConfigDir:
    """测试 Matplotlib 缓存目录初始化函数。"""

    def test_returns_path(self):
        """应返回 Path 对象。"""
        from src.utils.viz import ensure_matplotlib_config_dir

        result = ensure_matplotlib_config_dir()
        assert isinstance(result, Path)

    def test_directory_exists(self):
        """返回的目录应在磁盘上真实存在。"""
        from src.utils.viz import ensure_matplotlib_config_dir

        result = ensure_matplotlib_config_dir()
        assert result.exists()
        assert result.is_dir()

    def test_sets_environment_variables(self):
        """应设置 MPLCONFIGDIR 和 XDG_CACHE_HOME 环境变量。"""
        from src.utils.viz import ensure_matplotlib_config_dir

        ensure_matplotlib_config_dir()
        assert "MPLCONFIGDIR" in os.environ
        assert "XDG_CACHE_HOME" in os.environ


# ============================================================
# visualize_slice
# ============================================================
class TestVisualizeSlice:
    """测试 CT 单张切片可视化函数。"""

    @pytest.fixture(autouse=True)
    def mock_plt(self):
        """模块级 patch matplotlib.pyplot，避免真实绘图与 GUI 依赖。"""
        with patch("src.utils.viz.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            yield mock_plt

    def test_3d_array_normal(self, mock_plt):
        """3D 数组应正常显示。"""
        from src.utils.viz import visualize_slice

        ct = np.random.rand(10, 64, 64)
        fig = visualize_slice(ct, slice_idx=5)

        mock_plt.subplots.assert_called_once_with(1, 1, figsize=(6, 6))
        assert fig is mock_plt.subplots.return_value[0]

    def test_4d_array_takes_first_channel(self, mock_plt):
        """4D 数组应取第一通道后显示。"""
        from src.utils.viz import visualize_slice

        ct = np.random.rand(1, 10, 64, 64)
        fig = visualize_slice(ct, slice_idx=5)

        mock_plt.subplots.assert_called_once()
        assert fig is mock_plt.subplots.return_value[0]

    def test_slice_idx_out_of_bounds_raises(self, mock_plt):
        """slice_idx 越界时应抛出 ValueError。"""
        from src.utils.viz import visualize_slice

        ct = np.random.rand(10, 64, 64)

        with pytest.raises(ValueError, match="out of bounds"):
            visualize_slice(ct, slice_idx=-1)

        with pytest.raises(ValueError, match="out of bounds"):
            visualize_slice(ct, slice_idx=10)

    def test_save_path_saves_file(self, mock_plt, tmp_path):
        """提供 save_path 时应保存文件。"""
        from src.utils.viz import visualize_slice

        ct = np.random.rand(10, 64, 64)
        save_path = str(tmp_path / "output" / "slice.png")
        fig = visualize_slice(ct, slice_idx=5, save_path=save_path)

        assert Path(save_path).parent.exists()
        fig.savefig.assert_called_once_with(save_path, dpi=150, bbox_inches="tight")

    def test_title_is_set(self, mock_plt):
        """title 参数应正确设置到 ax。"""
        from src.utils.viz import visualize_slice

        ct = np.random.rand(10, 64, 64)
        mock_ax = mock_plt.subplots.return_value[1]
        visualize_slice(ct, slice_idx=5, title="Test Title")

        mock_ax.set_title.assert_called_once_with("Test Title")


# ============================================================
# visualize_anomaly_map
# ============================================================
class TestVisualizeAnomalyMap:
    """测试异常热图叠加可视化函数。"""

    @pytest.fixture(autouse=True)
    def mock_plt(self):
        """模块级 patch matplotlib.pyplot。"""
        with patch("src.utils.viz.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = [MagicMock(), MagicMock(), MagicMock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            yield mock_plt

    def test_normal_overlay(self, mock_plt):
        """正常 3D 数组应生成三张子图。"""
        from src.utils.viz import visualize_anomaly_map

        ct = np.random.rand(10, 64, 64)
        anomaly = np.random.rand(10, 64, 64)
        fig = visualize_anomaly_map(ct, anomaly, slice_idx=5)

        mock_plt.subplots.assert_called_once_with(1, 3, figsize=(12, 5))
        assert fig is mock_plt.subplots.return_value[0]

    def test_4d_array_handling(self, mock_plt):
        """4D 数组应正确处理第一通道。"""
        from src.utils.viz import visualize_anomaly_map

        ct = np.random.rand(1, 10, 64, 64)
        anomaly = np.random.rand(1, 10, 64, 64)
        fig = visualize_anomaly_map(ct, anomaly, slice_idx=5)

        assert fig is mock_plt.subplots.return_value[0]

    def test_shape_mismatch_raises(self, mock_plt):
        """ct 与 anomaly_map shape 不一致时应抛出 ValueError。"""
        from src.utils.viz import visualize_anomaly_map

        ct = np.random.rand(10, 64, 64)
        anomaly = np.random.rand(10, 32, 32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            visualize_anomaly_map(ct, anomaly, slice_idx=5)

    def test_save_path_saves_file(self, mock_plt, tmp_path):
        """提供 save_path 时应保存文件。"""
        from src.utils.viz import visualize_anomaly_map

        ct = np.random.rand(10, 64, 64)
        anomaly = np.random.rand(10, 64, 64)
        save_path = str(tmp_path / "output" / "anomaly.png")
        fig = visualize_anomaly_map(ct, anomaly, slice_idx=5, save_path=save_path)

        assert Path(save_path).parent.exists()
        fig.savefig.assert_called_once_with(save_path, dpi=150, bbox_inches="tight")


# ============================================================
# plot_roc_curve
# ============================================================
class TestPlotRocCurve:
    """测试 ROC 曲线绘制函数。"""

    @pytest.fixture(autouse=True)
    def mock_plt(self):
        """模块级 patch matplotlib.pyplot。"""
        with patch("src.utils.viz.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            yield mock_plt

    def test_normal_plot(self, mock_plt):
        """正常输入应绘制 ROC 曲线。"""
        from src.utils.viz import plot_roc_curve

        fpr = np.array([0.0, 0.1, 0.5, 1.0])
        tpr = np.array([0.0, 0.3, 0.8, 1.0])
        auc = 0.85
        fig = plot_roc_curve(fpr, tpr, auc)

        mock_plt.subplots.assert_called_once_with(1, 1, figsize=(6, 6))
        assert fig is mock_plt.subplots.return_value[0]

    def test_save_path_saves_file(self, mock_plt, tmp_path):
        """提供 save_path 时应保存文件。"""
        from src.utils.viz import plot_roc_curve

        fpr = np.array([0.0, 0.1, 0.5, 1.0])
        tpr = np.array([0.0, 0.3, 0.8, 1.0])
        auc = 0.85
        save_path = str(tmp_path / "output" / "roc.png")
        fig = plot_roc_curve(fpr, tpr, auc, save_path=save_path)

        assert Path(save_path).parent.exists()
        fig.savefig.assert_called_once_with(save_path, dpi=150, bbox_inches="tight")
