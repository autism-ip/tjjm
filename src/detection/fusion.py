"""
 * [INPUT]: 依赖 torch
 * [OUTPUT]: 对外提供 overlap_average_fusion（备用融合函数）
 * [POS]: src/detection/ 的融合辅助模块，MONAI 内置融合为主，此模块为兜底
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch


# ============================================================
# Overlap Average Fusion
# ============================================================

def overlap_average_fusion(
    canvas: torch.Tensor,
    count_map: torch.Tensor,
    patch: torch.Tensor,
    coords: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将单个 patch 结果累加到全局 canvas，并更新计数图
    INPUT:  canvas    (C, D, H, W) 全局累加器
            count_map (D, H, W)    计数累加器
            patch     (C, d, h, w) 当前 patch 重建结果
            coords    (z, y, x)    起始坐标
    OUTPUT: (updated_canvas, updated_count_map)
    """
    z, y, x = coords
    _, pd, ph, pw = patch.shape

    canvas[:, z : z + pd, y : y + ph, x : x + pw] += patch
    count_map[z : z + pd, y : y + ph, x : x + pw] += 1

    return canvas, count_map


def finalize_fusion(canvas: torch.Tensor, count_map: torch.Tensor) -> torch.Tensor:
    """
    用计数图对 canvas 做平均，消除重叠区域重复累加
    """
    count_map = count_map.unsqueeze(0) if count_map.dim() == 3 else count_map
    count_map = count_map.clamp(min=1)
    return canvas / count_map
