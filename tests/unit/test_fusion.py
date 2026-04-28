#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 pytest, torch, src.detection.fusion
 * [OUTPUT]: 对外提供 overlap_average_fusion 与 finalize_fusion 的完整单元测试
 * [POS]: tests/unit/ 的融合模块测试，覆盖单patch、多patch重叠、边界、除零防护
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import pytest
import torch

from src.detection.fusion import overlap_average_fusion, finalize_fusion


# ============================================================
# overlap_average_fusion
# ============================================================


def test_overlap_average_fusion_single_patch():
    """正常单 patch 累加：canvas 与 count_map 正确更新。"""
    canvas = torch.zeros(1, 8, 8, 8)
    count_map = torch.zeros(8, 8, 8)
    patch = torch.ones(1, 4, 4, 4)

    new_canvas, new_count = overlap_average_fusion(canvas, count_map, patch, (0, 0, 0))

    # 累加区域应为 1
    assert torch.all(new_canvas[0, :4, :4, :4] == 1.0)
    # 未累加区域应为 0
    assert torch.all(new_canvas[0, 4:, :, :] == 0.0)
    assert torch.all(new_count[:4, :4, :4] == 1.0)
    assert torch.all(new_count[4:, :, :] == 0.0)


def test_overlap_average_fusion_multiple_overlapping_patches():
    """多个 patch 重叠区域正确累加。"""
    canvas = torch.zeros(1, 8, 8, 8)
    count_map = torch.zeros(8, 8, 8)
    patch1 = torch.ones(1, 4, 4, 4)
    patch2 = torch.ones(1, 4, 4, 4) * 2

    canvas, count_map = overlap_average_fusion(canvas, count_map, patch1, (0, 0, 0))
    canvas, count_map = overlap_average_fusion(canvas, count_map, patch2, (2, 2, 2))

    # 重叠区域 (2:4, 2:4, 2:4) 应为 1 + 2 = 3
    overlap = canvas[0, 2:4, 2:4, 2:4]
    assert torch.all(overlap == 3.0)

    # 非重叠区域各自保留
    assert torch.all(canvas[0, 0:2, 0:2, 0:2] == 1.0)
    assert torch.all(canvas[0, 4:6, 4:6, 4:6] == 2.0)

    # count_map 重叠区域应为 2
    assert torch.all(count_map[2:4, 2:4, 2:4] == 2.0)
    assert torch.all(count_map[0:2, 0:2, 0:2] == 1.0)
    assert torch.all(count_map[4:6, 4:6, 4:6] == 1.0)


def test_overlap_average_fusion_different_coords():
    """不同坐标位置放置 patch，边界互不重叠。"""
    canvas = torch.zeros(1, 16, 16, 16)
    count_map = torch.zeros(16, 16, 16)
    patch = torch.ones(1, 4, 4, 4) * 5

    coords_list = [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)]
    for coords in coords_list:
        canvas, count_map = overlap_average_fusion(canvas, count_map, patch, coords)

    for z, y, x in coords_list:
        assert torch.all(canvas[0, z : z + 4, y : y + 4, x : x + 4] == 5.0)
        assert torch.all(count_map[z : z + 4, y : y + 4, x : x + 4] == 1.0)

    # 中间未覆盖区域保持 0
    assert torch.all(canvas[0, 8:, 8:, 8:] == 0.0)
    assert torch.all(count_map[8:, 8:, 8:] == 0.0)


def test_overlap_average_fusion_edge_exact_fit():
    """patch 边界恰好贴边，不越界。"""
    canvas = torch.zeros(1, 4, 4, 4)
    count_map = torch.zeros(4, 4, 4)
    patch = torch.ones(1, 4, 4, 4) * 7

    # 从 (0,0,0) 开始，patch 尺寸恰好等于 canvas 尺寸
    new_canvas, new_count = overlap_average_fusion(canvas, count_map, patch, (0, 0, 0))

    assert torch.all(new_canvas == 7.0)
    assert torch.all(new_count == 1.0)


# ============================================================
# finalize_fusion
# ============================================================


def test_finalize_fusion_normal_division():
    """正常除以 count_map，无重叠区域。"""
    canvas = torch.ones(1, 4, 4, 4) * 6.0
    count_map = torch.ones(4, 4, 4) * 2.0

    result = finalize_fusion(canvas, count_map)

    assert torch.all(result == 3.0)


def test_finalize_fusion_overlap_average():
    """重叠区域平均正确：count_map > 1 的区域被正确平均。"""
    canvas = torch.zeros(1, 4, 4, 4)
    count_map = torch.zeros(4, 4, 4)

    # 构造一个 count_map 不均匀的场景
    canvas[0, :2, :2, :2] = 4.0   # 被累加 2 次，每次 2
    count_map[:2, :2, :2] = 2.0

    canvas[0, 2:, 2:, 2:] = 9.0   # 被累加 3 次，每次 3
    count_map[2:, 2:, 2:] = 3.0

    result = finalize_fusion(canvas, count_map)

    assert torch.all(result[0, :2, :2, :2] == 2.0)
    assert torch.all(result[0, 2:, 2:, 2:] == 3.0)


def test_finalize_fusion_zero_count_clamp():
    """count_map 为 0 的区域被 clamp 为 1，避免除零。"""
    canvas = torch.ones(1, 4, 4, 4) * 5.0
    count_map = torch.zeros(4, 4, 4)

    result = finalize_fusion(canvas, count_map)

    # 零区域被 clamp 为 1，所以结果保持 5.0
    assert torch.all(result == 5.0)


def test_finalize_fusion_4d_count_map():
    """count_map 已经是 4 维时，unsqueeze 不触发，直接除。"""
    canvas = torch.ones(1, 4, 4, 4) * 8.0
    count_map = torch.ones(1, 4, 4, 4) * 4.0

    result = finalize_fusion(canvas, count_map)

    assert result.shape == canvas.shape
    assert torch.all(result == 2.0)
