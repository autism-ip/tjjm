"""
 * [INPUT]: 依赖 torch, pytorch_lightning
 * [OUTPUT]: 对外提供 AutoencoderLightningModule
 * [POS]: src/training/ 的核心训练器，封装自编码器训练循环
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import pytorch_lightning as pl
import torch


# ============================================================
# Autoencoder Lightning Module
# ============================================================

class AutoencoderLightningModule(pl.LightningModule):
    """
    基于 PyTorch Lightning 的自编码器训练模块
    INPUT batch: (patch, patch) — 自编码器输入=目标
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_cfg: dict,
        scheduler_cfg: dict | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        reconstructed = self(x)
        loss = self.loss_fn(reconstructed, target)

        if batch_idx % 10 == 0:
            self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        reconstructed = self(x)
        loss = self.loss_fn(reconstructed, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            **self.optimizer_cfg,
        )

        if self.scheduler_cfg is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **self.scheduler_cfg,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
