"""
 * [INPUT]: 依赖 data.intensity, data.patches
 * [OUTPUT]: 对外提供 hu_windowing, resample_to_spacing, normalize, extract_patches, filter_healthy_patches, world_to_voxel
 * [POS]: data/ 的兼容门面，维持历史导入路径，同时把强度变换与 patch/坐标逻辑分发到专用子模块
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

from src.data.intensity import hu_windowing, normalize, resample_to_spacing
from src.data.patches import extract_patches, filter_healthy_patches, world_to_voxel

__all__ = [
    "world_to_voxel",
    "hu_windowing",
    "resample_to_spacing",
    "normalize",
    "extract_patches",
    "filter_healthy_patches",
]
