#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 hydra-core 的 @hydra.main，依赖 omegaconf 的 DictConfig
 * [OUTPUT]: 对外提供自编码器训练入口 main() 函数
 * [POS]: scripts/ 的训练入口，被 CLI 直接调用
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from src.models.autoencoder import Autoencoder3D
from src.data.dataset import LunaPatchDataset
from src.training.trainer import AutoencoderLightningModule
from src.training.losses import WeightedMSELoss
from src.utils.logging import setup_logging


# ============================================================
# LungPatchDataModule
# ============================================================

class LungPatchDataModule(LightningDataModule):
    """
    LunaPatchDataset 的 Lightning DataModule 封装.
    负责训练/验证 DataLoader 的创建.
    """

    def __init__(
        self,
        dataset_dir: str,
        luna16_raw_dir: str,
        hu_min: int,
        hu_max: int,
        target_spacing: list[float],
        patch_size: list[int],
        batch_size: int,
        num_workers: int,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.luna16_raw_dir = Path(luna16_raw_dir)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_spacing = tuple(target_spacing)
        self.patch_size = tuple(patch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio

    def setup(self, stage: str | None = None):
        annotations_csv = self.luna16_raw_dir / "annotations.csv"
        full_dataset = LunaPatchDataset(
            ct_dir=self.dataset_dir,
            annotations_csv=annotations_csv,
            patch_size=self.patch_size,
            stride=32,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
            target_spacing=self.target_spacing,
        )
        val_len = int(len(full_dataset) * self.val_ratio)
        train_len = len(full_dataset) - val_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ============================================================
# Main
# ============================================================

@hydra.main(
    config_path="../configs",
    config_name="train_autoencoder",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    自编码器训练主入口。

    流程:
        1. 解析配置并打印
        2. 初始化数据模块
        3. 初始化 Lightning 模型
        4. 组装 Trainer + callbacks
        5. 执行训练
        6. 保存最终模型
    """
    # --------------------------------------------------
    # 1. 配置解析与日志
    # --------------------------------------------------
    OmegaConf.resolve(cfg)
    setup_logging()

    print("=" * 60)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # --------------------------------------------------
    # 2. 数据模块
    # --------------------------------------------------
    datamodule = LungPatchDataModule(
        dataset_dir=cfg.data.dataset_dir,
        luna16_raw_dir=cfg.data.luna16_raw_dir,
        hu_min=cfg.data.hu_min,
        hu_max=cfg.data.hu_max,
        target_spacing=cfg.data.target_spacing,
        patch_size=cfg.data.patch_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # --------------------------------------------------
    # 3. 模型 + 损失 + Lightning 模块
    # --------------------------------------------------
    model = Autoencoder3D(
        encoder_name=cfg.model.encoder_name,
        freeze_encoder=cfg.model.freeze_encoder,
        feature_size=48,
        use_checkpoint=cfg.model.use_checkpoint,
        pretrained=cfg.model.encoder_pretrained,
    )

    loss_fn = WeightedMSELoss(k=cfg.loss.weight_k)

    optimizer_cfg = {
        "lr": cfg.training.optimizer.lr,
        "weight_decay": cfg.training.optimizer.weight_decay,
    }
    scheduler_cfg = {
        "T_max": cfg.training.scheduler.T_max,
        "eta_min": cfg.training.scheduler.eta_min,
    }

    pl_module = AutoencoderLightningModule(
        model=model,
        loss_fn=loss_fn,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
    )

    # --------------------------------------------------
    # 4. Callbacks + Logger + Trainer
    # --------------------------------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="autoencoder-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.training.checkpoint_dir, "tb_logs"),
        name="autoencoder",
    )

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        enable_progress_bar=True,
    )

    # --------------------------------------------------
    # 5. 训练
    # --------------------------------------------------
    trainer.fit(pl_module, datamodule=datamodule)

    # --------------------------------------------------
    # 6. 保存最终模型
    # --------------------------------------------------
    final_path = os.path.join(cfg.training.checkpoint_dir, "final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
