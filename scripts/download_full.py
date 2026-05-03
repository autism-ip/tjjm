#!/usr/bin/env python3
"""Standalone LUNA16 full dataset downloader (no project import chain)."""

import os
import sys
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


KAGGLE_DATASET = "namnguyenhoang1/luna16-full-dataset-until-23-feb-2026"
ANNOTATIONS_FILE = "annotations.csv"


def download_full(output_dir: str = "/root/autodl-tmp/data/raw/LUNA16") -> None:
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    print("Kaggle API 认证成功")

    # 收集所有文件
    all_files = []
    page_token = None
    while True:
        page = api.dataset_list_files(
            KAGGLE_DATASET,
            page_token=page_token,
            page_size=200,
        )
        for f in page.dataset_files:
            all_files.append(f.name)
        print(f"  已发现 {len(all_files)} 个文件...")
        if not page.next_page_token:
            break
        page_token = page.next_page_token

    # 只保留 annotations.csv + .mhd + .raw
    target_files = [ANNOTATIONS_FILE] + sorted(set(
        f for f in all_files if f.endswith((".mhd", ".raw"))
    ))
    print(f"需要下载 {len(target_files)} 个文件")

    # 检查已下载的文件，跳过已存在且非空的
    to_download = []
    for f in target_files:
        dest = raw_dir / f
        if dest.exists() and dest.stat().st_size > 0:
            continue
        to_download.append(f)
    print(f"跳过已下载 {len(target_files) - len(to_download)} 个，剩余 {len(to_download)} 个")

    # 逐个下载
    for i, file_name in enumerate(to_download, 1):
        print(f"[{i}/{len(to_download)}] 下载 {file_name} ...", end=" ", flush=True)
        try:
            api.dataset_download_file(
                KAGGLE_DATASET,
                file_name,
                path=str(raw_dir),
                force=True,
                quiet=True,
            )
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    # 解压 zip
    for zip_path in raw_dir.glob("*.zip"):
        print(f"解压 {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    # 统计结果
    mhd_count = len(list(raw_dir.rglob("*.mhd")))
    raw_count = len(list(raw_dir.rglob("*.raw")))
    print(f"\n下载完成！mhd: {mhd_count}, raw: {raw_count}")


if __name__ == "__main__":
    download_full()
