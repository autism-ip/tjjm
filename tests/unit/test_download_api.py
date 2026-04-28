"""
 * [INPUT]: 依赖 importlib, pytest, pandas, src.data.download 的公共下载 API
 * [OUTPUT]: 对外提供下载入口契约单元测试
 * [POS]: tests/unit/ 的 CLI 下载契约守卫, 防止 scripts/download_data.py 与 data/download.py 再次漂移或默认撒谎
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import importlib
import sys

import pandas as pd
import pytest

from src.data import download as download_module


def test_download_data_script_imports_public_api_without_network():
    """CLI 模块导入应只绑定公共 API, 不触发下载。"""
    module = importlib.import_module("scripts.download_data")

    assert module.download_luna16 is download_module.download_luna16
    assert module.download_lidc_idri is download_module.download_lidc_idri


def test_data_package_exports_download_api():
    """src.data 包出口应包含 CLI 依赖的下载函数。"""
    data_package = importlib.import_module("src.data")

    assert data_package.download_luna16 is download_module.download_luna16
    assert data_package.download_lidc_idri is download_module.download_lidc_idri
    assert "download_luna16" in data_package.__all__
    assert "download_lidc_idri" in data_package.__all__


def test_download_data_cli_defaults_to_supported_dataset(monkeypatch, tmp_path):
    """CLI 默认行为只应承诺当前真实支持的 LUNA16。"""
    module = importlib.import_module("scripts.download_data")
    calls = []

    def fake_luna16(**kwargs):
        calls.append(("luna16", kwargs))

    def fake_lidc_idri(**kwargs):
        calls.append(("lidc-idri", kwargs))
        raise AssertionError("default CLI path should not hit lidc-idri")

    monkeypatch.setattr(module, "download_luna16", fake_luna16)
    monkeypatch.setattr(module, "download_lidc_idri", fake_lidc_idri)
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--output-dir", str(tmp_path)])

    args = module.parse_args()
    assert args.dataset == "luna16"

    module.main()

    assert calls == [
        (
            "luna16",
            {
                "output_dir": str(tmp_path / "LUNA16"),
                "subset": None,
                "extract": True,
            },
        )
    ]


def test_download_data_cli_help_marks_lidc_as_not_implemented(monkeypatch, capsys):
    """help 文案必须诚实暴露 LIDC-IDRI 仍是占位入口。"""
    module = importlib.import_module("scripts.download_data")
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--help"])

    with pytest.raises(SystemExit, match="0"):
        module.parse_args()

    help_text = capsys.readouterr().out
    assert "default: luna16" in help_text
    assert "lidc-idri" in help_text
    lowered = help_text.lower()
    assert "not" in lowered
    assert "implemented" in lowered


def test_download_data_cli_lidc_option_fails_honestly(monkeypatch, tmp_path):
    """显式选 LIDC-IDRI 时, CLI 应直达未实现入口而不是伪装完成。"""
    module = importlib.import_module("scripts.download_data")
    calls = []

    def fake_luna16(**kwargs):
        calls.append(("luna16", kwargs))

    def fake_lidc_idri(**kwargs):
        calls.append(("lidc-idri", kwargs))
        raise NotImplementedError("LIDC-IDRI download is not implemented yet")

    monkeypatch.setattr(module, "download_luna16", fake_luna16)
    monkeypatch.setattr(module, "download_lidc_idri", fake_lidc_idri)
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_data.py", "--dataset", "lidc-idri", "--output-dir", str(tmp_path)],
    )

    with pytest.raises(NotImplementedError, match="not implemented"):
        module.main()

    assert calls == [
        (
            "lidc-idri",
            {
                "output_dir": str(tmp_path / "LIDC-IDRI"),
                "subset": None,
                "extract": True,
            },
        )
    ]


def test_download_luna16_delegates_to_real_downloader(monkeypatch, tmp_path):
    """函数 API 应收敛到 Luna16Downloader, 不再停留在想象函数。"""
    calls = []

    def fake_download(self, use_kaggle=True, extract=True):
        calls.append((self.raw_dir, self.processed_dir, use_kaggle, extract))

    def fake_manifest(self):
        return pd.DataFrame({"seriesuid": ["scan-a", "scan-b"]})

    monkeypatch.setattr(download_module.Luna16Downloader, "download", fake_download)
    monkeypatch.setattr(download_module.Luna16Downloader, "get_manifest", fake_manifest)

    output_dir = tmp_path / "raw" / "LUNA16"
    manifest = download_module.download_luna16(
        output_dir=output_dir,
        subset=1,
        extract=False,
        use_kaggle=False,
    )

    assert calls == [
        (output_dir, tmp_path / "processed" / "LUNA16", False, False)
    ]
    assert manifest["seriesuid"].tolist() == ["scan-a"]


def test_luna16_downloader_respects_extract_flag(monkeypatch, tmp_path):
    """extract=False 时 Kaggle 命令不应携带 --unzip。"""
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))

    downloader = download_module.Luna16Downloader(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    monkeypatch.setattr(download_module.subprocess, "run", fake_run)
    monkeypatch.setattr(downloader, "_validate_file_pairs", lambda: [])

    downloader.download(use_kaggle=True, extract=False)

    assert calls
    assert "--unzip" not in calls[0][0]


def test_download_lidc_idri_fails_explicitly_without_fake_download(tmp_path):
    """LIDC-IDRI 尚无真实下载器时, 入口必须诚实失败。"""
    with pytest.raises(NotImplementedError, match="LIDC-IDRI"):
        download_module.download_lidc_idri(tmp_path / "LIDC-IDRI")
