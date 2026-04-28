#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 pytest 的 fixture 机制，依赖 torch 的张量构造，依赖 omegaconf 的 resolver 注册，依赖 pathlib/shutil/uuid 的临时目录管理
 * [OUTPUT]: 对外提供 Hydra runtime resolver、session/device/fixture、自定义 tmp_path 等全局测试资源
 * [POS]: tests/ 的根配置，被所有测试模块共享，并替代 pytest 内建 tmpdir 插件以规避当前沙箱文件系统权限冲突
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from datetime import datetime
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
import torch
from omegaconf import OmegaConf


def _hydra_now(pattern: str) -> str:
    return datetime.now().strftime(pattern)


def _test_cache_root() -> Path:
    """
    返回测试基础设施使用的仓库内缓存根目录。
    """
    root = Path.cwd() / ".cache" / "pytest-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_tmp_path() -> Path:
    """
    创建当前测试用例独占的临时目录。
    """
    path = _test_cache_root() / f"case-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


@pytest.fixture(scope="session", autouse=True)
def hydra_runtime_resolvers() -> None:
    """
    注册 Hydra 在 @hydra.main 启动时提供的 now resolver。
    """
    OmegaConf.register_new_resolver(
        "now",
        _hydra_now,
        use_cache=True,
        replace=True,
    )


@pytest.fixture
def tmp_path() -> Path:
    """
    提供仓库内可控的临时目录，绕开 pytest 内建 tmpdir 插件在当前沙箱下的权限问题。
    """
    path = _make_tmp_path()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """
    返回可用的计算设备。

    优先 CUDA，其次 MPS，最后 CPU。
    作用域为 session，整个测试周期只创建一次。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def sample_3d_patch() -> torch.Tensor:
    """
    单个 3D patch，形状 (C, D, H, W) = (1, 64, 64, 64)。

    用于单样本前向传播测试。
    """
    return torch.randn(1, 64, 64, 64)


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """
    小批量 3D patch，形状 (B, C, D, H, W) = (2, 1, 64, 64, 64)。

    用于批量前向传播与损失计算测试。
    """
    return torch.randn(2, 1, 64, 64, 64)


@pytest.fixture
def sample_ct_volume() -> torch.Tensor:
    """
    模拟完整 CT 体积，形状 (1, 128, 128, 128)。

    用于滑动窗口推理与异常检测测试。
    """
    return torch.randn(1, 128, 128, 128)


@pytest.fixture
def sample_mask() -> torch.Tensor:
    """
    模拟二值标注 mask，形状 (128, 128, 128)。

    用于评估指标计算测试。
    """
    mask = torch.zeros(128, 128, 128)
    mask[50:70, 50:70, 50:70] = 1.0
    return mask
