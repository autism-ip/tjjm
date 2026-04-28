"""
 * [INPUT]: 依赖 pathlib, json, numpy, SimpleITK
 * [OUTPUT]: 对外提供 load_array, load_report, iter_input_paths
 * [POS]: src/experiments/ 的输入适配层, 负责读取实验输入与报告文件
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def load_array(path: str | Path) -> np.ndarray:
    """
    读取 numpy / NIfTI / MetaImage 体积或数组文件。
    """
    file_path = Path(path)
    suffixes = "".join(file_path.suffixes).lower()

    if suffixes.endswith(".npy"):
        return np.asarray(np.load(file_path))

    if suffixes.endswith(".npz"):
        archive = np.load(file_path)
        if "arr_0" in archive:
            return np.asarray(archive["arr_0"])
        first_key = next(iter(archive.files))
        return np.asarray(archive[first_key])

    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz") or suffixes.endswith(".mhd"):
        import SimpleITK as sitk

        image = sitk.ReadImage(str(file_path))
        return sitk.GetArrayFromImage(image).astype(np.float32)

    raise ValueError(f"Unsupported array format: {file_path}")


def load_report(path: str | Path) -> dict:
    """读取 JSON 指标报告。"""
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_input_paths(
    inputs: Sequence[str | Path] | None = None,
    input_dir: str | Path | None = None,
    patterns: Iterable[str] = ("*.npy", "*.npz", "*.nii", "*.nii.gz", "*.mhd"),
) -> list[Path]:
    """
    收集实验输入路径。
    """
    paths: list[Path] = []
    if inputs:
        paths.extend(Path(item) for item in inputs)
    if input_dir is not None:
        root = Path(input_dir)
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))
    return paths
