"""
 * [INPUT]: 依赖 torch, pytorch_lightning.callbacks, matplotlib
 * [OUTPUT]: 对外提供 ReconstructionVisualizationCallback
 * [POS]: src/training/ 的可视化与检查点回调集合
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


# ============================================================
# Reconstruction Visualization Callback
# ============================================================

class ReconstructionVisualizationCallback(pl.Callback):
    """
    每 N 个 epoch 可视化原始 patch vs 重建 patch
    """

    def __init__(self, every_n_epochs: int = 5, num_samples: int = 4, save_dir: str = "./viz"):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0 or epoch == 0:
            return

        dataloader = trainer.datamodule.val_dataloader()
        if dataloader is None:
            return

        batch = next(iter(dataloader))
        x, _ = batch
        x = x[: self.num_samples].to(pl_module.device)

        with torch.no_grad():
            recon = pl_module(x)

        self._plot_and_save(x, recon, epoch)

    def _plot_and_save(self, original: torch.Tensor, reconstructed: torch.Tensor, epoch: int):
        n = original.size(0)
        fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i in range(n):
            orig_slice = original[i, 0, original.size(2) // 2, :, :].cpu().numpy()
            recon_slice = reconstructed[i, 0, reconstructed.size(2) // 2, :, :].cpu().numpy()

            axes[i, 0].imshow(orig_slice, cmap="gray")
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon_slice, cmap="gray")
            axes[i, 1].set_title("Reconstructed")
            axes[i, 1].axis("off")

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"recon_epoch_{epoch:04d}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ============================================================
# ModelCheckpoint re-export with project defaults
# ============================================================

__all__ = [
    "ReconstructionVisualizationCallback",
    "ModelCheckpoint",
]
