#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 importlib, sys, pandas, pytest, src.data.download
 * [OUTPUT]: 对外提供下载器契约测试
 * [POS]: tests/unit/ 的下载入口验证器，覆盖脚本导入、CLI 转发与少量真实 CT 文件选择逻辑
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import importlib
import sys

import pandas as pd
import pytest

from src.data import download as download_module


class _FakeFile:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakePage:
    def __init__(self, names: list[str], next_token: str | None = None) -> None:
        self.dataset_files = [_FakeFile(name) for name in names]
        self.nextPageToken = next_token


class _FakeKaggleApi:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages
        self.downloads: list[tuple[str, str, str, bool, bool]] = []
        self.authenticated = False

    def authenticate(self) -> None:
        self.authenticated = True

    def dataset_list_files(self, dataset, page_token=None, page_size=200):
        if page_token is None:
            return self.pages[0]
        if page_token == "page-2" and len(self.pages) > 1:
            return self.pages[1]
        return _FakePage([], None)

    def dataset_download_file(self, dataset, file_name, path=None, force=False, quiet=True):
        self.downloads.append((dataset, file_name, path, force, quiet))


def test_download_data_script_imports_public_api_without_network():
    """CLI 入口必须只暴露真实下载 API，不额外引入网络副作用。"""
    module = importlib.import_module("scripts.download_data")

    assert module.download_luna16 is download_module.download_luna16
    assert module.download_lidc_idri is download_module.download_lidc_idri


def test_data_package_exports_download_api():
    """src.data 必须把下载 API 透出，保证 CLI 和包导入同构。"""
    data_package = importlib.import_module("src.data")

    assert data_package.download_luna16 is download_module.download_luna16
    assert data_package.download_lidc_idri is download_module.download_lidc_idri
    assert "download_luna16" in data_package.__all__
    assert "download_lidc_idri" in data_package.__all__


def test_download_data_cli_defaults_to_supported_dataset(monkeypatch, tmp_path):
    """CLI 默认只走可用的 LUNA16 路径。"""
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
    """显式选择 LIDC-IDRI 时，CLI 必须直达未实现入口而不是伪装完成。"""
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
    """公共 API 必须把 subset 和目录约束传给真实下载器。"""
    calls = []

    def fake_download(self, use_kaggle=True, extract=True, subset=None):
        calls.append((self.raw_dir, self.processed_dir, use_kaggle, extract, subset))

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
        (output_dir, tmp_path / "processed" / "LUNA16", False, False, 1)
    ]
    assert manifest["seriesuid"].tolist() == ["scan-a"]


def test_luna16_downloader_downloads_one_real_pair_by_default(monkeypatch, tmp_path):
    """默认只下载一份真实 CT 配对，避免把整包 LUNA16 拉进来。"""
    pages = [
        _FakePage(
            [
                "annotations.csv",
                "subset0/subset0/111111111111111111111111111111.mhd",
                "subset0/subset0/111111111111111111111111111111.raw",
                "subset0/subset0/222222222222222222222222222222.mhd",
                "subset0/subset0/222222222222222222222222222222.raw",
            ],
            next_token=None,
        )
    ]
    api = _FakeKaggleApi(pages)
    downloader = download_module.Luna16Downloader(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    monkeypatch.setattr(download_module, "KaggleApi", lambda: api)
    monkeypatch.setattr(downloader, "_extract_if_needed", lambda: None)
    monkeypatch.setattr(downloader, "_validate_file_pairs", lambda: [])

    downloader.download(use_kaggle=True, extract=False)

    assert api.authenticated is True
    assert [item[1] for item in api.downloads] == [
        "annotations.csv",
        "subset0/subset0/111111111111111111111111111111.mhd",
        "subset0/subset0/111111111111111111111111111111.raw",
    ]


def test_download_lidc_idri_fails_explicitly_without_fake_download(tmp_path):
    """LIDC-IDRI 入口仍然必须明确失败，不能假装实现。"""
    with pytest.raises(NotImplementedError, match="LIDC-IDRI"):
        download_module.download_lidc_idri(tmp_path / "LIDC-IDRI")
