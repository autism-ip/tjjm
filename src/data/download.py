"""
 * [INPUT]: 依赖 pathlib, subprocess, zipfile, pandas
 * [OUTPUT]: 对外提供 Luna16Downloader 类、download_luna16/download_lidc_idri CLI 函数
 * [POS]: data/ 的数据获取器, 负责 LUNA16 下载/解压/校验与 CLI 下载契约
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import subprocess
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd


__all__ = [
    "Luna16Downloader",
    "download_luna16",
    "download_lidc_idri",
]


# ============================================================
# Luna16Downloader
# ============================================================

class Luna16Downloader:
    """
    LUNA16 数据集下载器.
    支持 Kaggle API 自动下载或手动放置后校验.
    """

    KAGGLE_DATASET = "eliasmarcon/luna-16"
    EXPECTED_FILES = ["annotations.csv", "candidates.csv"]

    def __init__(self, raw_dir: Path, processed_dir: Path):
        """
        Args:
            raw_dir:       原始数据存放目录
            processed_dir: 处理后数据存放目录
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def download(self, use_kaggle: bool = True, extract: bool = True) -> None:
        """
        下载 LUNA16 数据集.

        Args:
            use_kaggle: True 时使用 Kaggle API, False 时仅校验已存在文件
            extract:    True 时解压 raw_dir 中的 zip 文件
        """
        if use_kaggle:
            self._download_kaggle(extract=extract)

        if extract:
            self._extract_if_needed()
        self._validate_file_pairs()

    def get_manifest(self) -> pd.DataFrame:
        """
        生成文件清单, 列出所有完整的 .mhd + .raw 对.

        Returns:
            DataFrame 含列: seriesuid, mhd_path, raw_path, valid
        """
        records = []
        for mhd_path in sorted(self.raw_dir.rglob("*.mhd")):
            raw_path = mhd_path.with_suffix(".raw")
            valid = raw_path.exists() and raw_path.stat().st_size > 0
            records.append({
                "seriesuid": mhd_path.stem,
                "mhd_path": str(mhd_path),
                "raw_path": str(raw_path) if raw_path.exists() else None,
                "valid": valid,
            })
        return pd.DataFrame(records)

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    def _download_kaggle(self, extract: bool = True) -> None:
        """通过 Kaggle API 下载数据集."""
        cmd = [
            "kaggle", "datasets", "download",
            "-d", self.KAGGLE_DATASET,
            "-p", str(self.raw_dir),
        ]
        if extract:
            cmd.append("--unzip")
        subprocess.run(cmd, check=True)

    def _extract_if_needed(self) -> None:
        """解压 raw_dir 中的 zip 文件."""
        for zip_path in self.raw_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.raw_dir)

    def _validate_file_pairs(self) -> List[Tuple[Path, Path]]:
        """
        验证 .mhd + .raw 文件对是否完整.

        Returns:
            完整文件对列表
        """
        valid_pairs = []
        for mhd_path in self.raw_dir.rglob("*.mhd"):
            raw_path = mhd_path.with_suffix(".raw")
            if raw_path.exists() and raw_path.stat().st_size > 0:
                valid_pairs.append((mhd_path, raw_path))
        return valid_pairs


# ============================================================
# CLI 函数 API
# ============================================================

def _default_processed_dir(raw_dir: Path) -> Path:
    """从 raw/LUNA16 推导 processed/LUNA16, 其他路径落在自身 processed 子目录。"""
    if raw_dir.parent.name == "raw":
        return raw_dir.parent.parent / "processed" / raw_dir.name
    return raw_dir / "processed"


def _limit_manifest(manifest: pd.DataFrame, subset: int | None) -> pd.DataFrame:
    """按 CLI subset 返回清单切片, 不改变真实下载行为。"""
    if subset is None:
        return manifest
    if subset < 0:
        raise ValueError("subset must be non-negative")
    return manifest.head(subset).reset_index(drop=True)


def download_luna16(
    output_dir: str | Path,
    subset: int | None = None,
    extract: bool = True,
    use_kaggle: bool = True,
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    CLI 友好的 LUNA16 下载入口.

    Args:
        output_dir:     原始数据目录
        subset:         返回清单的前 N 条, 不伪装成部分下载
        extract:        是否解压 zip 文件
        use_kaggle:     是否调用 Kaggle API
        processed_dir:  处理后目录, 默认从 output_dir 推导

    Returns:
        LUNA16 文件清单 DataFrame
    """
    raw_dir = Path(output_dir)
    resolved_processed_dir = (
        Path(processed_dir) if processed_dir is not None else _default_processed_dir(raw_dir)
    )

    downloader = Luna16Downloader(
        raw_dir=raw_dir,
        processed_dir=resolved_processed_dir,
    )
    downloader.download(use_kaggle=use_kaggle, extract=extract)
    return _limit_manifest(downloader.get_manifest(), subset)


def download_lidc_idri(
    output_dir: str | Path,
    subset: int | None = None,
    extract: bool = True,
) -> pd.DataFrame:
    """
    LIDC-IDRI 下载入口占位.

    当前代码库没有真实 LIDC-IDRI 下载器; 入口必须存在以保证 CLI 可导入,
    但不能假装已经实现下载能力.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "LIDC-IDRI download is not implemented yet; add a real downloader before using this CLI path."
    )
