"""
 * [INPUT]: 依赖 torch, numpy, pandas, pathlib, SimpleITK, data.intensity, data.patches
 * [OUTPUT]: 对外提供 LunaCTDataset, LunaPatchDataset
 * [POS]: data/ 的 PyTorch Dataset 实现, 被 training/ 和 detection/ 消费
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from src.data.intensity import hu_windowing, normalize, resample_to_spacing
from src.data.patches import (
    extract_patches,
    filter_healthy_patches,
)


# ============================================================
# LunaCTDataset
# ============================================================

class LunaCTDataset(Dataset):
    """
    加载 LUNA16 .mhd/.raw 文件, 应用完整预处理流水线.
    """

    def __init__(
        self,
        ct_dir: Path,
        hu_min: int = -1024,
        hu_max: int = 3071,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalize_method: str = "minmax",
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        lazy: bool = True,
    ):
        """
        Args:
            ct_dir:           CT 文件目录
            hu_min/max:       HU 窗口范围
            target_spacing:   重采样目标间距
            normalize_method: "minmax" | "zscore"
            normalize_range:  minmax 输出范围
            lazy:             True 时构造不加载图像
        """
        self.ct_dir = Path(ct_dir)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_spacing = target_spacing
        self.normalize_method = normalize_method
        self.normalize_range = normalize_range

        self.file_list = sorted(self.ct_dir.glob("*.mhd"))
        self._cache = {} if not lazy else None

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        path = self.file_list[idx]
        tensor = self._load_and_preprocess(path)

        if self._cache is not None:
            self._cache[idx] = tensor

        return tensor

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """读取 .mhd 并应用完整预处理."""
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = img.GetSpacing()[::-1]  # SimpleITK 顺序 (x,y,z) -> (z,y,x)

        arr = hu_windowing(arr, self.hu_min, self.hu_max)
        arr = resample_to_spacing(arr, spacing, self.target_spacing)
        arr = normalize(arr, self.normalize_method, self.normalize_range)

        # 添加 channel 维度: (1, D, H, W)
        arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr)


# ============================================================
# LunaPatchDataset
# ============================================================

class LunaPatchDataset(Dataset):
    """
    从预处理后的 CT 中提取健康 patch, 用于自编码器训练.
    输出 (patch, patch) 因为输入=目标.
    懒加载: 构造时只建立索引, __getitem__ 时才读取文件并提取单个 patch.
    """

    def __init__(
        self,
        ct_dir: Path,
        annotations_csv: Path,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        stride: int = 32,
        hu_min: int = -1024,
        hu_max: int = 3071,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalize_method: str = "minmax",
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        nodule_margin_ratio: float = 1.5,
        file_cache_size: int = 4,
    ):
        """
        Args:
            ct_dir:              CT 文件目录
            annotations_csv:     结节标注 CSV
            patch_size:          提取 patch 尺寸
            stride:              滑动窗口步长
            nodule_margin_ratio: 排除结节区域的 margin 倍数
            file_cache_size:     预处理后的 CT 文件缓存数量上限
        """
        self.ct_dir = Path(ct_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_spacing = target_spacing
        self.normalize_method = normalize_method
        self.normalize_range = normalize_range
        self.nodule_margin_ratio = nodule_margin_ratio
        self.file_cache_size = file_cache_size

        self.file_list = sorted(self.ct_dir.glob("*.mhd"))
        self.annotations_df = pd.read_csv(annotations_csv)

        # 索引: (path, z, y, x) 元组列表, 不存储像素数据
        self.patches: list[Tuple[Path, int, int, int]] = []
        self._file_cache: dict[Path, np.ndarray] = {}
        self._build_patch_index()

    def _build_patch_index(self) -> None:
        """遍历所有 CT, 只记录健康 patch 的位置索引."""
        for path in self.file_list:
            seriesuid = path.stem

            img = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            spacing = img.GetSpacing()[::-1]
            origin = img.GetOrigin()

            arr = hu_windowing(arr, self.hu_min, self.hu_max)
            arr = resample_to_spacing(arr, spacing, self.target_spacing)
            arr = normalize(arr, self.normalize_method, self.normalize_range)

            _, centers = extract_patches(arr, self.patch_size, self.stride)

            # 过滤包含结节的 patch
            ann_rows = self.annotations_df[
                self.annotations_df["seriesuid"] == seriesuid
            ]
            annotations = ann_rows.to_dict("records")

            _, healthy_centers = filter_healthy_patches(
                None,
                annotations,
                centers,
                spacing=self.target_spacing,
                origin=origin,
                nodule_margin_ratio=self.nodule_margin_ratio,
            )

            for cz, cy, cx in healthy_centers:
                self.patches.append((path, cz, cy, cx))

    def _load_file(self, path: Path) -> np.ndarray:
        """读取并预处理单个 CT 文件, 带 LRU 风格缓存."""
        if path in self._file_cache:
            return self._file_cache[path]

        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = img.GetSpacing()[::-1]

        arr = hu_windowing(arr, self.hu_min, self.hu_max)
        arr = resample_to_spacing(arr, spacing, self.target_spacing)
        arr = normalize(arr, self.normalize_method, self.normalize_range)

        # 缓存淘汰: 超过上限时清空 (简单策略, 可替换为 OrderedDict LRU)
        if len(self._file_cache) >= self.file_cache_size:
            self._file_cache.clear()
        self._file_cache[path] = arr

        return arr

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, cz, cy, cx = self.patches[idx]
        arr = self._load_file(path)

        pd, ph, pw = self.patch_size
        # 计算左上角坐标
        z = cz - pd // 2
        y = cy - ph // 2
        x = cx - pw // 2

        patch = arr[z : z + pd, y : y + ph, x : x + pw]
        tensor = torch.from_numpy(np.expand_dims(patch, axis=0))
        return tensor, tensor
