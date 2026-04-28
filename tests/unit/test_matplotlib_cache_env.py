"""
 * [INPUT]: 依赖 importlib, os, pathlib, sys, src.utils.viz, src.evaluation.reporter, scripts.detect
 * [OUTPUT]: 对外提供 Matplotlib 缓存环境测试
 * [POS]: tests/unit/ 的环境守卫验证器，覆盖 viz/reporter/detect 对项目内缓存目录的收敛行为
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import importlib
import os
from pathlib import Path
import sys


def test_viz_import_initializes_project_matplotlib_cache(monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    cache_root = project_root / ".cache"
    mpl_config_dir = cache_root / "matplotlib"

    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delitem(sys.modules, "src.utils.viz", raising=False)

    importlib.import_module("src.utils.viz")

    assert Path(os.environ["MPLCONFIGDIR"]) == mpl_config_dir
    assert Path(os.environ["XDG_CACHE_HOME"]) == cache_root
    assert mpl_config_dir.is_dir()
    assert cache_root.is_dir()


def test_reporter_import_reuses_existing_matplotlib_cache(monkeypatch, tmp_path):
    custom_cache = tmp_path / "custom-cache"
    custom_mpl = custom_cache / "matplotlib"
    custom_mpl.mkdir(parents=True)

    monkeypatch.setenv("XDG_CACHE_HOME", str(custom_cache))
    monkeypatch.setenv("MPLCONFIGDIR", str(custom_mpl))
    monkeypatch.delitem(sys.modules, "src.evaluation.reporter", raising=False)
    monkeypatch.delitem(sys.modules, "src.utils.viz", raising=False)

    importlib.import_module("src.evaluation.reporter")

    assert Path(os.environ["XDG_CACHE_HOME"]) == custom_cache
    assert Path(os.environ["MPLCONFIGDIR"]) == custom_mpl


def test_detect_import_initializes_matplotlib_cache_before_model_imports(monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    cache_root = project_root / ".cache"
    mpl_config_dir = cache_root / "matplotlib"

    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delitem(sys.modules, "scripts.detect", raising=False)

    importlib.import_module("scripts.detect")

    assert Path(os.environ["XDG_CACHE_HOME"]) == cache_root
    assert Path(os.environ["MPLCONFIGDIR"]) == mpl_config_dir
    assert cache_root.is_dir()
    assert mpl_config_dir.is_dir()
