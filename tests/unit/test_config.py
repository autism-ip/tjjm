#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 pytest 的 fixture 与 mock，依赖 omegaconf 的 DictConfig/OmegaConf，依赖 hydra 的初始化 API
 * [OUTPUT]: 对外提供 load_config 与 merge_configs 的全覆盖单元测试
 * [POS]: tests/unit/ 的配置加载测试模块，验证 src/utils/config.py 的契约行为
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from src.utils.config import load_config, merge_configs


# ============================================================
# load_config 测试
# ============================================================

class TestLoadConfig:
    """验证 load_config 对路径检查、Hydra 初始化、overrides 传递、默认值的处理。"""

    def test_loads_existing_config_dir(self, tmp_path: Path) -> None:
        """
        正常加载现有配置目录，返回 resolve 后的 DictConfig。
        """
        fake_cfg = OmegaConf.create({"model": {"encoder_name": "swin_unetr"}})

        with (
            patch("src.utils.config.initialize_config_dir") as mock_init,
            patch("src.utils.config.compose", return_value=fake_cfg) as mock_compose,
        ):
            result = load_config(config_path=str(tmp_path), config_name="train_autoencoder")

        assert isinstance(result, DictConfig)
        assert result.model.encoder_name == "swin_unetr"
        mock_init.assert_called_once_with(config_dir=str(tmp_path.resolve()), version_base=None)
        mock_compose.assert_called_once_with(config_name="train_autoencoder", overrides=[])

    def test_raises_file_not_found_for_missing_dir(self) -> None:
        """
        传入不存在的目录时，立即抛出 FileNotFoundError，不进入 Hydra 初始化。
        """
        nonexistent = "/nonexistent/path/to/configs"

        with pytest.raises(FileNotFoundError, match="Config directory not found"):
            load_config(config_path=nonexistent)

    def test_overrides_passed_to_compose(self, tmp_path: Path) -> None:
        """
        overrides 参数正确透传给 compose。
        """
        fake_cfg = OmegaConf.create({"data": {"batch_size": 8}})
        overrides = ["data.batch_size=8", "training.max_epochs=50"]

        with (
            patch("src.utils.config.initialize_config_dir") as mock_init,
            patch("src.utils.config.compose", return_value=fake_cfg) as mock_compose,
        ):
            load_config(config_path=str(tmp_path), config_name="train_autoencoder", overrides=overrides)

        mock_compose.assert_called_once_with(config_name="train_autoencoder", overrides=overrides)

    def test_default_config_name(self, tmp_path: Path) -> None:
        """
        不传入 config_name 时，默认使用 "train_autoencoder"。
        """
        fake_cfg = OmegaConf.create({})

        with (
            patch("src.utils.config.initialize_config_dir"),
            patch("src.utils.config.compose", return_value=fake_cfg) as mock_compose,
        ):
            load_config(config_path=str(tmp_path))

        mock_compose.assert_called_once_with(config_name="train_autoencoder", overrides=[])

    def test_returns_resolved_config(self, tmp_path: Path) -> None:
        """
        返回的配置经过 OmegaConf.resolve 处理，插值已展开。
        """
        fake_cfg = OmegaConf.create({"a": 1, "b": "${a}"})

        with (
            patch("src.utils.config.initialize_config_dir"),
            patch("src.utils.config.compose", return_value=fake_cfg),
        ):
            result = load_config(config_path=str(tmp_path))

        assert result.b == 1


# ============================================================
# merge_configs 测试
# ============================================================

class TestMergeConfigs:
    """验证 merge_configs 对 DictConfig/dict 合并、深层嵌套、不可变性的处理。"""

    def test_dictconfig_override_takes_precedence(self) -> None:
        """
        两个 DictConfig 合并时，override 中的同名键覆盖 base 值。
        """
        base = OmegaConf.create({"lr": 1e-4, "epochs": 100})
        override = OmegaConf.create({"lr": 5e-5})

        merged = merge_configs(base, override)

        assert merged.lr == 5e-5
        assert merged.epochs == 100

    def test_plain_dict_override_converted_and_merged(self) -> None:
        """
        dict 类型的 override 被正确转换为 DictConfig 后合并。
        """
        base = OmegaConf.create({"batch_size": 4, "num_workers": 4})
        override = {"batch_size": 8}

        merged = merge_configs(base, override)

        assert merged.batch_size == 8
        assert merged.num_workers == 4
        assert isinstance(merged, DictConfig)

    def test_deep_nested_merge(self) -> None:
        """
        深层嵌套配置合并时，子层级正确覆盖与保留。
        """
        base = OmegaConf.create({
            "model": {"encoder": "swin", "pretrained": True},
            "training": {"optimizer": {"name": "adam", "lr": 1e-4}},
        })
        override = OmegaConf.create({
            "training": {"optimizer": {"lr": 5e-5}},
        })

        merged = merge_configs(base, override)

        assert merged.model.encoder == "swin"
        assert merged.model.pretrained is True
        assert merged.training.optimizer.name == "adam"
        assert merged.training.optimizer.lr == 5e-5

    def test_does_not_mutate_original_base(self) -> None:
        """
        合并后不修改原始 base 配置，体现不可变性。
        """
        base = OmegaConf.create({"a": 1, "b": {"c": 2}})
        override = OmegaConf.create({"a": 99, "b": {"c": 999}})
        base_snapshot = OmegaConf.to_yaml(base)

        merged = merge_configs(base, override)

        assert merged.a == 99
        assert merged.b.c == 999
        assert OmegaConf.to_yaml(base) == base_snapshot
        assert base.a == 1
        assert base.b.c == 2

    def test_returns_resolved_config(self) -> None:
        """
        合并结果经过 OmegaConf.resolve，插值表达式已展开。
        """
        base = OmegaConf.create({"x": 10, "y": "${x}"})
        override = OmegaConf.create({"x": 20})

        merged = merge_configs(base, override)

        assert merged.y == 20
