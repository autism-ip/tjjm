"""
 * [INPUT]: 依赖 json, csv, os/pathlib 的缓存目录初始化, numpy, sklearn.metrics, matplotlib
 * [OUTPUT]: 对外提供 save_report, EvaluationReporter
 * [POS]: src/evaluation/ 的报告生成器，聚合指标计算与可视化输出
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from src.evaluation.metrics import compute_auc, compute_metrics


def _ensure_matplotlib_config_dir() -> Path:
    """在导入 pyplot 前固定项目内缓存目录。"""
    project_root = Path(__file__).resolve().parents[2]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", project_root / ".cache")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    mpl_config_dir = Path(
        os.environ.get("MPLCONFIGDIR", cache_root / "matplotlib")
    ).expanduser()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    return mpl_config_dir


_ensure_matplotlib_config_dir()

import matplotlib.pyplot as plt


# ============================================================
# Evaluation Reporter
# ============================================================

def save_report(metrics: dict, report_path: str | Path) -> None:
    """将指标字典保存为 JSON 报告。"""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


class EvaluationReporter:
    """
    生成 JSON/CSV 评估报告，保存异常热图与 ROC 曲线
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        anomaly_scores: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        case_id: str = "unknown",
    ) -> dict:
        """
        生成单病例评估报告并持久化
        """
        metrics = compute_metrics(pred_mask, gt_mask)

        if anomaly_scores is not None and labels is not None:
            metrics["auc"] = compute_auc(anomaly_scores, labels)

        self._save_json(metrics, case_id)
        self._save_csv(metrics, case_id)

        if anomaly_scores is not None and labels is not None:
            self._plot_roc(anomaly_scores, labels, case_id)

        self._save_anomaly_map(pred_mask, case_id)

        return metrics

    def _save_json(self, metrics: dict, case_id: str):
        path = self.output_dir / f"{case_id}_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def _save_csv(self, metrics: dict, case_id: str):
        path = self.output_dir / f"{case_id}_metrics.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "metric", "value"])
            for key, value in metrics.items():
                writer.writerow([case_id, key, value])

    def _plot_roc(self, scores: np.ndarray, labels: np.ndarray, case_id: str):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_val = compute_auc(scores, labels)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {case_id}")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        path = self.output_dir / f"{case_id}_roc.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _save_anomaly_map(self, pred_mask: np.ndarray, case_id: str):
        slice_idx = pred_mask.shape[0] // 2
        slice_img = pred_mask[slice_idx, :, :]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(slice_img, cmap="hot")
        ax.set_title(f"Anomaly Map - {case_id}")
        ax.axis("off")

        path = self.output_dir / f"{case_id}_anomaly.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
