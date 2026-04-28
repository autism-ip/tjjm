#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 hydra-core 的初始化与 Compose API，依赖 omegaconf 的 DictConfig
 * [OUTPUT]: 对外提供 load_config()、merge_configs() 两个工具函数
 * [POS]: src/utils/ 的配置中心，被 scripts/ 和 notebooks/ 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from hydra import initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Union[str, Path],
    config_name: str = "train_autoencoder",
    overrides: Optional[list] = None,
) -> DictConfig:
    """
    从指定目录加载 Hydra 配置。

    Args:
        config_path: 配置文件所在目录的绝对路径。
        config_name: 主配置文件名（不含 .yaml 后缀）。
        overrides: Hydra 命令行风格的覆盖列表，例如 ["data.batch_size=8"]。

    Returns:
        解析并 resolve 后的 DictConfig。

    Raises:
        FileNotFoundError: 配置目录不存在时抛出。
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config directory not found: {config_path}")

    overrides = overrides or []

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    OmegaConf.resolve(cfg)
    return cfg


def merge_configs(base: DictConfig, override: Union[DictConfig, Dict[str, Any]]) -> DictConfig:
    """
    合并两个配置，override 中的键优先。

    Args:
        base: 基础配置。
        override: 覆盖配置，可以是 DictConfig 或普通 dict。

    Returns:
        合并后的新 DictConfig，不修改原始对象。
    """
    if isinstance(override, dict):
        override = OmegaConf.create(override)

    # OmegaConf.merge 返回新的 DictConfig，保证不可变性
    merged = OmegaConf.merge(base, override)
    OmegaConf.resolve(merged)
    return merged
