"""
 * [INPUT]: 依赖 pytest, numpy, src.data.preprocess
 * [OUTPUT]: 世界坐标与体素坐标转换的单元测试
 * [POS]: tests/unit/ 的坐标系统验证器
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import math
import numpy as np
import pytest


class TestWorldToVoxel:
    """世界坐标 -> 体素坐标转换测试."""

    def test_origin_zero_identity(self):
        """origin=(0,0,0) 时 world=voxel."""
        from src.data.preprocess import world_to_voxel
        world = np.array([10.0, 20.0, 30.0])
        origin = np.array([0.0, 0.0, 0.0])
        spacing = np.array([1.0, 1.0, 1.0])
        voxel = world_to_voxel(world, origin, spacing)
        np.testing.assert_allclose(voxel, [10.0, 20.0, 30.0])

    def test_negative_origin(self):
        """origin=(-100,-100,-100), spacing=(1,1,2)."""
        from src.data.preprocess import world_to_voxel
        world = np.array([10.0, 20.0, 30.0])
        origin = np.array([-100.0, -100.0, -100.0])
        spacing = np.array([1.0, 1.0, 2.0])
        voxel = world_to_voxel(world, origin, spacing)
        # (10 - (-100)) / 1 = 110
        # (20 - (-100)) / 1 = 120
        # (30 - (-100)) / 2 = 65
        np.testing.assert_allclose(voxel, [110.0, 120.0, 65.0])

    def test_non_uniform_spacing(self):
        """非各向同性 spacing."""
        from src.data.preprocess import world_to_voxel
        world = np.array([5.0, 10.0, 15.0])
        origin = np.array([0.0, 0.0, 0.0])
        spacing = np.array([0.5, 1.0, 2.5])
        voxel = world_to_voxel(world, origin, spacing)
        np.testing.assert_allclose(voxel, [10.0, 10.0, 6.0])


class TestFilterHealthyPatchesWithCoordinateConversion:
    """filter_healthy_patches 坐标转换集成测试."""

    def test_excludes_nodule_with_nonzero_origin(self):
        """
        关键测试：当 origin != 0 时，不转换坐标会导致错误。
        
        场景：
        - CT origin = (-100, -100, -100) mm
        - spacing = (1, 1, 1) mm
        - 结节世界坐标 = (10, 20, 30) mm
        - patch 中心体素坐标 = (110, 120, 65) (对应世界坐标 (10,20,30))
        
        如果不做转换，代码会认为 patch 中心 (110,120,65) 和结节 (10,20,30)
        距离很远，错误地将包含结节的 patch 判定为健康。
        """
        from src.data.preprocess import filter_healthy_patches

        # 单个 patch，中心在体素坐标 (110, 120, 130)
        # world=(10,20,30), origin=(-100,-100,-100), spacing=(1,1,1)
        # voxel = (world - origin) / spacing = (110, 120, 130)
        patch = np.zeros((8, 8, 8), dtype=np.float32)
        patches = [patch]
        centers = [(110, 120, 130)]

        # 结节世界坐标 (10, 20, 30) mm，直径 10 mm
        annotations = [{
            "coordX": 10.0,
            "coordY": 20.0,
            "coordZ": 30.0,
            "diameter_mm": 10.0,
        }]

        # 正确的 origin 和 spacing
        origin = (-100.0, -100.0, -100.0)
        spacing = (1.0, 1.0, 1.0)

        healthy_p, healthy_c = filter_healthy_patches(
            patches, annotations, centers,
            spacing=spacing, origin=origin, nodule_margin_ratio=1.0
        )

        # patch 中心经坐标转换后对应世界坐标 (10,20,30)，正好是结节中心
        # 距离 = 0 mm <= radius 5 mm，应该被排除
        assert len(healthy_p) == 0
        assert len(healthy_c) == 0

    def test_keeps_far_away_patch_with_nonzero_origin(self):
        """远离结节的 patch 应被保留."""
        from src.data.preprocess import filter_healthy_patches

        patch = np.zeros((8, 8, 8), dtype=np.float32)
        # 体素坐标 (0, 0, 0) 对应世界坐标 (-100, -100, -100)
        patches = [patch]
        centers = [(0, 0, 0)]

        annotations = [{
            "coordX": 10.0,
            "coordY": 20.0,
            "coordZ": 30.0,
            "diameter_mm": 10.0,
        }]

        origin = (-100.0, -100.0, -100.0)
        spacing = (1.0, 1.0, 1.0)

        healthy_p, healthy_c = filter_healthy_patches(
            patches, annotations, centers,
            spacing=spacing, origin=origin, nodule_margin_ratio=1.0
        )

        # 世界坐标距离 = sqrt((-100-10)^2 + (-100-20)^2 + (-100-30)^2) ≈ 216 mm
        # >> 5 mm radius，应该保留
        assert len(healthy_p) == 1
        assert len(healthy_c) == 1

    def test_origin_zero_backward_compatible(self):
        """origin=(0,0,0) 时行为应与旧代码一致."""
        from src.data.preprocess import filter_healthy_patches

        patch = np.zeros((8, 8, 8), dtype=np.float32)
        # 体素坐标 = 世界坐标 (因为 origin=0, spacing=1)
        patches = [patch]
        centers = [(10, 20, 30)]

        annotations = [{
            "coordX": 10.0,
            "coordY": 20.0,
            "coordZ": 30.0,
            "diameter_mm": 10.0,
        }]

        healthy_p, healthy_c = filter_healthy_patches(
            patches, annotations, centers,
            spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), nodule_margin_ratio=1.0
        )

        assert len(healthy_p) == 0
        assert len(healthy_c) == 0
