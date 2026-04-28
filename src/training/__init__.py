"""
 * [INPUT]: 依赖 training.trainer, training.callbacks
 * [OUTPUT]: 对外提供 AutoencoderLightningModule, ReconstructionVisualizationCallback
 * [POS]: src/training/ 的入口，聚合训练层全部公共接口
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from src.training.trainer import AutoencoderLightningModule
from src.training.callbacks import ReconstructionVisualizationCallback

__all__ = [
    "AutoencoderLightningModule",
    "ReconstructionVisualizationCallback",
]
