"""
 * [INPUT]: 依赖 torch, numpy, SimpleITK, detection.sliding_window, detection.anomaly_map
 * [OUTPUT]: 对外提供 SlidingWindowDetector
 * [POS]: src/detection/ 的高阶推理封装，被 scripts/detect.py 直接消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from src.detection.anomaly_map import compute_anomaly_map
from src.detection.sliding_window import sliding_window_reconstruct


# ============================================================
# SlidingWindowDetector
# ============================================================

class SlidingWindowDetector:
    """
    对整目录 CT 做滑动窗口重建与异常检测。

    遍历 .mhd 文件 -> 加载 -> 重建 -> 计算异常热图 -> 返回结构化结果。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: tuple[int, int, int] = (64, 64, 64),
        stride: int = 32,
        batch_size: int = 4,
        device: torch.device | None = None,
    ):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def run_directory(self, test_ct_dir: str | Path) -> dict[str, dict]:
        """
        遍历目录中所有 .mhd 文件，返回每个 case 的重建结果。

        Returns:
            dict[case_id, {
                "anomaly_map": np.ndarray (D, H, W),
                "ct_volume":   np.ndarray (D, H, W),
                "save_fn":     callable(arr, path),
            }]
        """
        results: dict[str, dict] = {}
        for path in sorted(Path(test_ct_dir).glob("*.mhd")):
            case_id = path.stem
            result = self._process_single(path)
            results[case_id] = result
        return results

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------

    def _process_single(self, path: Path) -> dict:
        """处理单个 CT 文件：加载 -> 重建 -> 异常图。"""
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)

        # (1, D, H, W) tensor
        ct_tensor = torch.from_numpy(np.expand_dims(arr, axis=0))

        reconstructed = sliding_window_reconstruct(
            self.model,
            ct_tensor,
            self.patch_size,
            self.stride,
            self.batch_size,
        )

        # 转回 numpy，去掉 channel 维度 -> (D, H, W)
        recon_np = reconstructed.squeeze(0).cpu().numpy()
        anomaly_map = compute_anomaly_map(arr, recon_np)

        return {
            "anomaly_map": anomaly_map,
            "ct_volume": arr,
            "save_fn": lambda candidate, output_path, reference=img: self._save_nifti(
                candidate,
                output_path,
                reference_image=reference,
            ),
        }

    @staticmethod
    def _save_nifti(
        arr: np.ndarray,
        path: str | Path,
        reference_image: sitk.Image | None = None,
    ) -> None:
        """将 numpy array 保存为 NIfTI 格式，并尽量继承原始空间信息。"""
        img = sitk.GetImageFromArray(arr)
        if reference_image is not None and tuple(img.GetSize()) == tuple(reference_image.GetSize()):
            img.CopyInformation(reference_image)
        sitk.WriteImage(img, str(path))
