"""
 * [INPUT]: 依赖 pathlib, pandas, kaggle.api.kaggle_api_extended.KaggleApi
 * [OUTPUT]: 对外提供 Luna16Downloader 类、download_luna16/download_lidc_idri CLI 函数
 * [POS]: data/ 的数据获取器, 负责 LUNA16 文件级下载/校验与 CLI 下载契约
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

# 延迟导入 Kaggle API，避免模块加载时触发认证
KaggleApi = None


def _get_kaggle_api():
    """懒加载 Kaggle API"""
    global KaggleApi
    if KaggleApi is None:
        from kaggle.api.kaggle_api_extended import KaggleApi as _KaggleApi
        KaggleApi = _KaggleApi
    return KaggleApi()


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

    KAGGLE_DATASET = "namnguyenhoang1/luna16-full-dataset-until-23-feb-2026"
    ANNOTATIONS_FILE = "annotations.csv"

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

    def download(
        self,
        use_kaggle: bool = True,
        extract: bool = True,
        subset: int | None = None,
    ) -> None:
        """
        下载 LUNA16 真实 CT 文件.

        Args:
            use_kaggle: True 时使用 Kaggle API, False 时仅校验已存在文件
            extract:    保留兼容参数; 真实文件下载不需要解压
            subset:     需要下载的 CT 体积对数量, None 时默认只取 1 份
        """
        if use_kaggle:
            self._download_kaggle(subset=subset)

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

    def _download_kaggle(self, subset: int | None = None) -> None:
        """通过 Kaggle API 下载少量真实 CT 体积对."""
        api = _get_kaggle_api()
        api.authenticate()

        for file_name in self._select_download_files(api, subset=subset):
            api.dataset_download_file(
                self.KAGGLE_DATASET,
                file_name,
                path=str(self.raw_dir),
                force=True,
                quiet=True,
            )

    def _select_download_files(
        self,
        api,
        subset: int | None = None,
    ) -> list[str]:
        """挑选 annotations.csv 与 N 份 .mhd/.raw 配对文件。subset=None 时下载全部。"""
        selected = [self.ANNOTATIONS_FILE]
        page_token: str | None = None

        if subset is not None:
            # 下载前 N 份 CT
            if subset < 1:
                raise ValueError("subset must be at least 1")
            series_found = 0
            while series_found < subset:
                page = api.dataset_list_files(
                    self.KAGGLE_DATASET,
                    page_token=page_token,
                    page_size=200,
                )
                selected.extend(self._collect_scan_pairs(page.dataset_files, subset - series_found))
                series_found = (len(selected) - 1) // 2
                if series_found >= subset or not page.next_page_token:
                    break
                page_token = page.next_page_token
            if series_found < subset:
                raise RuntimeError(
                    f"Unable to find {subset} LUNA16 CT pairs in dataset {self.KAGGLE_DATASET}"
                )
        else:
            # 下载全部 CT 文件
            while True:
                page = api.dataset_list_files(
                    self.KAGGLE_DATASET,
                    page_token=page_token,
                    page_size=200,
                )
                for file_info in page.dataset_files:
                    name = getattr(file_info, "name", "")
                    if name.endswith((".mhd", ".raw")):
                        selected.append(name)
                if not page.next_page_token:
                    break
                page_token = page.next_page_token
            # 去重（annotations.csv 已在首位）
            selected = list(dict.fromkeys(selected))

        return selected

    @staticmethod
    def _collect_scan_pairs(files: Iterable, remaining_pairs: int) -> list[str]:
        """从 Kaggle 文件清单里收集 mhd/raw 配对."""
        selected: list[str] = []
        for file_info in files:
            name = getattr(file_info, "name", "")
            if not name.endswith(".mhd"):
                continue
            selected.append(name)
            selected.append(name[:-4] + ".raw")
            if len(selected) // 2 >= remaining_pairs:
                break
        return selected

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
    downloader.download(use_kaggle=use_kaggle, extract=extract, subset=subset)
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
