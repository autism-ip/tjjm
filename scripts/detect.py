#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 hydra-core 的 @hydra.main，依赖 omegaconf 的 DictConfig，依赖 numpy 的数组规整，依赖 os/pathlib 的缓存目录初始化
 * [OUTPUT]: 对外提供异常检测推理入口 main()、_collect_evaluation_arrays()、_resolve_encoder_name() 与 _ensure_matplotlib_cache_env() 函数
 * [POS]: scripts/ 的推理入口，被 CLI 直接调用，并在模型导入前收敛 Matplotlib 缓存目录
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _ensure_matplotlib_cache_env() -> Path:
    """在导入依赖链前固定项目内 Matplotlib/fontconfig 缓存目录。"""
    project_root = Path(PROJECT_ROOT)
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", project_root / ".cache")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    mpl_config_dir = Path(
        os.environ.get("MPLCONFIGDIR", cache_root / "matplotlib")
    ).expanduser()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    return mpl_config_dir


_ensure_matplotlib_cache_env()

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from src.models.autoencoder import Autoencoder3D
from src.detection.inference import SlidingWindowDetector
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporter import save_report
from src.utils.viz import visualize_anomaly_map, plot_roc_curve
from src.utils.logging import setup_logging

SUPPORTED_ENCODER_NAME = "swin_unetr"


def _as_numpy_flat(values: Any) -> np.ndarray:
    """把 numpy/torch 输出统一压平成一维 numpy 数组。"""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1)


def _collect_evaluation_arrays(results: dict) -> tuple[np.ndarray, np.ndarray]:
    """收集多病例异常分数与标注，供 metrics API 统一阈值化。"""
    preds = []
    gts = []
    for result in results.values():
        preds.append(_as_numpy_flat(result["anomaly_map"]))
        gts.append(_as_numpy_flat(result["ground_truth"]))

    if not preds:
        return np.array([], dtype=np.float32), np.array([], dtype=np.uint8)

    return np.concatenate(preds), np.concatenate(gts)


def _resolve_encoder_name(checkpoint: dict[str, Any]) -> str:
    """对旧 checkpoint 缺失字段做兼容，收敛到项目唯一支持的编码器。"""
    return checkpoint.get("encoder_name") or SUPPORTED_ENCODER_NAME


@hydra.main(
    config_path="../configs",
    config_name="detect",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    异常检测推理主入口。

    流程:
        1. 解析配置
        2. 加载训练好的模型
        3. 对测试 CT 做滑动窗口重建
        4. 生成异常热图
        5. 如有标注则计算评估指标
        6. 保存结果
    """
    # --------------------------------------------------
    # 1. 配置解析
    # --------------------------------------------------
    OmegaConf.resolve(cfg)
    setup_logging()

    print("=" * 60)
    print("Detection Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.data.output_dir, exist_ok=True)

    # --------------------------------------------------
    # 2. 加载模型
    # --------------------------------------------------
    print(f"Loading checkpoint from: {cfg.model.checkpoint_path}")
    checkpoint = torch.load(cfg.model.checkpoint_path, map_location=device)

    # 从 checkpoint 中恢复模型结构参数
    model = Autoencoder3D(
        encoder_name=_resolve_encoder_name(checkpoint),
        pretrained=False,
    )
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # 3. 滑动窗口推理
    # --------------------------------------------------
    detector = SlidingWindowDetector(
        model=model,
        patch_size=cfg.detection.patch_size,
        stride=cfg.detection.stride,
        batch_size=cfg.detection.batch_size,
        device=device,
    )

    results = detector.run_directory(cfg.data.test_ct_dir)

    # --------------------------------------------------
    # 4. 生成并保存异常热图
    # --------------------------------------------------
    for case_id, result in results.items():
        anomaly_map = result["anomaly_map"]
        ct_volume = result["ct_volume"]

        # 保存热图 NIfTI
        out_nii = os.path.join(cfg.data.output_dir, f"{case_id}_anomaly.nii.gz")
        result["save_fn"](anomaly_map, out_nii)

        # 可视化中间切片
        mid_z = anomaly_map.shape[2] // 2
        viz_path = os.path.join(cfg.data.output_dir, f"{case_id}_viz.png")
        visualize_anomaly_map(ct_volume, anomaly_map, mid_z, save_path=viz_path)

    # --------------------------------------------------
    # 5. 评估（若存在标注）
    # --------------------------------------------------
    if results and "ground_truth" in next(iter(results.values())):
        preds, gts = _collect_evaluation_arrays(results)

        metrics = compute_metrics(
            preds=preds,
            gts=gts,
            threshold_method=cfg.evaluation.threshold_method,
            metrics_list=cfg.evaluation.metrics,
        )

        # 保存评估报告
        report_path = os.path.join(cfg.data.output_dir, "report.json")
        save_report(metrics, report_path)

        # 绘制 ROC
        if "fpr" in metrics and "tpr" in metrics:
            roc_path = os.path.join(cfg.data.output_dir, "roc_curve.png")
            plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics.get("auc", 0.0), save_path=roc_path)

        print("Evaluation metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    # --------------------------------------------------
    # 6. 完成
    # --------------------------------------------------
    print(f"Detection results saved to: {cfg.data.output_dir}")


if __name__ == "__main__":
    main()
