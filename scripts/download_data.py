#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 argparse 的命令行解析，依赖 src/data/download.py 的数据下载逻辑
 * [OUTPUT]: 对外提供数据下载 CLI 入口 main() 函数
 * [POS]: scripts/ 的数据准备入口，被 CI / 用户手动调用，默认只承诺可用的 LUNA16
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.download import download_luna16, download_lidc_idri


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Download and prepare lung CT datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["luna16", "lidc-idri"],
        default="luna16",
        help=(
            "Which dataset to download (default: luna16). "
            "'lidc-idri' remains listed, but that download path is not implemented yet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Directory to store downloaded data (default: ./data/raw)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Download only a subset of N CT volume pairs; None defaults to 1 real pair",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction, only download archives",
    )
    return parser.parse_args()


def main() -> None:
    """数据下载主入口。"""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "luna16":
        print("=" * 60)
        print("Downloading LUNA16...")
        download_luna16(
            output_dir=os.path.join(args.output_dir, "LUNA16"),
            subset=args.subset,
            extract=not args.no_extract,
        )

    if args.dataset == "lidc-idri":
        print("=" * 60)
        print("Downloading LIDC-IDRI (not implemented yet)...")
        download_lidc_idri(
            output_dir=os.path.join(args.output_dir, "LIDC-IDRI"),
            subset=args.subset,
            extract=not args.no_extract,
        )

    print("=" * 60)
    print("Download complete.")


if __name__ == "__main__":
    main()
