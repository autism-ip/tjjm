#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 pytest, re, omegaconf, src.models.autoencoder, src.data.dataset, src.training.trainer
 * [OUTPUT]: 提供训练流程集成测试——Hydra runtime 插值、配置加载、模型/数据集/训练器实例化
 * [POS]: tests/integration/ 的训练管道验证器
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import re
import sys
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder import Autoencoder3D
from src.data.dataset import LunaPatchDataset
from src.training.trainer import AutoencoderLightningModule
from src.training.losses import WeightedMSELoss


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def config_path() -> Path:
    return PROJECT_ROOT / "configs" / "train_autoencoder.yaml"


@pytest.fixture
def detect_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "detect.yaml"


def assert_hydra_run_dir_resolves(cfg, prefix: str) -> None:
    raw = OmegaConf.to_container(cfg, resolve=False)["hydra"]["run"]["dir"]
    assert raw == f"{prefix}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}"
    OmegaConf.resolve(cfg)
    assert re.fullmatch(
        rf"{re.escape(prefix)}/\d{{4}}-\d{{2}}-\d{{2}}/\d{{2}}-\d{{2}}-\d{{2}}",
        cfg.hydra.run.dir,
    )


# ============================================================
# Config Loading Tests
# ============================================================

class TestConfigLoading:
    """验证 YAML 配置可被正确加载与解析."""

    def test_train_config_exists_and_loads(self, config_path):
        assert config_path.exists(), f"Config not found: {config_path}"
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert "data" in raw
        assert "model" in raw
        assert "training" in raw

    def test_train_config_omegaconf_resolve(self, config_path):
        cfg = OmegaConf.load(config_path)
        assert_hydra_run_dir_resolves(cfg, "./outputs")
        assert cfg.data.patch_size == [64, 64, 64]
        assert cfg.model.encoder_name == "swin_unetr"

    def test_detect_config_exists_and_loads(self, detect_config_path):
        assert detect_config_path.exists()
        cfg = OmegaConf.load(detect_config_path)
        assert_hydra_run_dir_resolves(cfg, "./outputs/detection")
        assert "data" in cfg
        assert "model" in cfg
        assert "detection" in cfg


# ============================================================
# Model Instantiation Tests
# ============================================================

class TestModelInstantiation:
    """验证 Autoencoder3D 可按配置参数实例化."""

    def test_model_from_config_params(self, config_path):
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = Autoencoder3D(
            encoder_name=cfg.model.encoder_name,
            freeze_encoder=cfg.model.freeze_encoder,
            feature_size=48,
            use_checkpoint=cfg.model.use_checkpoint,
            pretrained=False,
        )
        assert isinstance(model, torch.nn.Module)

    def test_model_forward_with_config_patch_size(self, config_path):
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        patch_size = cfg.data.patch_size
        model = Autoencoder3D(
            encoder_name=cfg.model.encoder_name,
            freeze_encoder=True,
            feature_size=48,
            use_checkpoint=False,
            pretrained=False,
        )
        x = torch.randn(1, 1, *patch_size)
        out = model(x)
        assert out.shape == x.shape


# ============================================================
# Dataset Instantiation Tests
# ============================================================

class TestDatasetInstantiation:
    """验证 LunaPatchDataset 可按配置参数实例化."""

    def test_dataset_class_exists(self):
        assert LunaPatchDataset is not None

    def test_dataset_init_signature(self):
        import inspect
        sig = inspect.signature(LunaPatchDataset.__init__)
        params = list(sig.parameters.keys())
        assert "ct_dir" in params
        assert "annotations_csv" in params
        assert "patch_size" in params


# ============================================================
# Lightning Module Instantiation Tests
# ============================================================

class TestLightningModuleInstantiation:
    """验证 AutoencoderLightningModule 可按配置实例化."""

    def test_lightning_module_from_components(self, config_path):
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = Autoencoder3D(
            encoder_name=cfg.model.encoder_name,
            freeze_encoder=cfg.model.freeze_encoder,
            feature_size=48,
            use_checkpoint=False,
            pretrained=False,
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
        assert pl_module is not None
        assert hasattr(pl_module, "model")
        assert hasattr(pl_module, "loss_fn")


# ============================================================
# Script Import Tests
# ============================================================

class TestScriptImports:
    """验证训练与推理脚本可被导入而不报错."""

    def test_train_script_importable(self):
        from scripts.train_autoencoder import main
        assert callable(main)

    def test_detect_script_importable(self):
        from scripts.detect import main
        assert callable(main)
