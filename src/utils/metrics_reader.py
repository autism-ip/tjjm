"""
 * [INPUT]: 依赖 json, pathlib
 * [OUTPUT]: 对外提供 MetricsReader
 * [POS]: src/utils/ 的指标读取器，支持可视化而无需重跑实验
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
from pathlib import Path


__all__ = [
    "MetricsReader",
]


# ============================================================
# Metrics Reader
# ============================================================

class MetricsReader:
    """
    读取持久化的指标数据，支持可视化而无需重跑实验。
    
    使用方式:
        reader = MetricsReader("./outputs/metrics")
        
        # 列出所有训练运行
        runs = reader.list_runs("training")
        
        # 加载训练数据
        data = reader.load_training_run("run_20260503_120000")
        
        # 获取绘图数据
        summary = reader.get_training_summary("run_20260503_120000")
        froc_sens, froc_fp = reader.get_froc_data("run_20260503_150000")
    """

    def __init__(self, metrics_dir: str):
        """
        Args:
            metrics_dir: 指标根目录（通常是 outputs/metrics）
        """
        self.metrics_dir = Path(metrics_dir)

    def list_runs(self, run_type: str) -> list[str]:
        """
        列出指定类型的所有运行。
        
        Args:
            run_type: 运行类型（training/detection/evaluation/experiments）
        
        Returns:
            运行 ID 列表（按时间排序）
        """
        run_dir = self.metrics_dir / run_type
        if not run_dir.exists():
            return []
        return sorted([d.name for d in run_dir.iterdir() if d.is_dir()])

    def load_training_run(self, run_id: str) -> dict:
        """
        加载训练运行数据。
        
        Args:
            run_id: 运行 ID
        
        Returns:
            包含 config、records、summary 的字典
        """
        run_dir = self.metrics_dir / "training" / run_id
        return {
            "config": self._load_json(run_dir / "config.json"),
            "records": self._load_json(run_dir / "records.json"),
            "summary": self._load_json(run_dir / "summary.json"),
        }

    def load_detection_run(self, run_id: str) -> dict:
        """
        加载检测运行数据。
        
        Args:
            run_id: 运行 ID
        
        Returns:
            包含 config、cases、summary 的字典
        """
        run_dir = self.metrics_dir / "detection" / run_id
        return {
            "config": self._load_json(run_dir / "config.json"),
            "cases": self._load_json(run_dir / "detection_cases.json"),
            "summary": self._load_json(run_dir / "detection_summary.json"),
        }

    def load_evaluation_run(self, run_id: str) -> dict:
        """
        加载评估运行数据。
        
        Args:
            run_id: 运行 ID
        
        Returns:
            包含 config、case_metrics、global_metrics、froc_curve 等的字典
        """
        run_dir = self.metrics_dir / "evaluation" / run_id
        return {
            "config": self._load_json(run_dir / "config.json"),
            "case_metrics": self._load_json(run_dir / "case_metrics.json"),
            "global_metrics": self._load_json(run_dir / "global_metrics.json"),
            "froc_curve": self._load_json(run_dir / "froc_curve.json"),
            "threshold_sweep": self._load_json(run_dir / "threshold_sweep.json"),
            "recommended_op": self._load_json(run_dir / "recommended_op.json"),
        }

    def load_experiment_run(self, run_id: str) -> dict:
        """
        加载实验运行数据。
        
        Args:
            run_id: 运行 ID
        
        Returns:
            包含 protocol、synthetic_results、ablation_results 等的字典
        """
        run_dir = self.metrics_dir / "experiments" / run_id
        return {
            "protocol": self._load_json(run_dir / "protocol.json"),
            "synthetic_results": self._load_json(run_dir / "synthetic_results.json"),
            "ablation_results": self._load_json(run_dir / "ablation_results.json"),
            "compare_results": self._load_json(run_dir / "compare_results.json"),
        }

    def get_training_summary(self, run_id: str) -> dict:
        """
        获取训练汇总（用于快速绘图）。
        
        Args:
            run_id: 运行 ID
        
        Returns:
            包含 epochs、train_loss、val_loss 等列表的字典
        """
        data = self.load_training_run(run_id)
        summary = data.get("summary", {})
        return {
            "epochs": summary.get("epoch_history", []),
            "train_loss": summary.get("train_loss_history", []),
            "val_loss": summary.get("val_loss_history", []),
            "best_val_loss": summary.get("best_val_loss"),
            "best_epoch": summary.get("best_epoch"),
            "total_epochs": summary.get("total_epochs"),
            "final_train_loss": summary.get("final_train_loss"),
            "final_val_loss": summary.get("final_val_loss"),
            "epoch_time_stats": summary.get("epoch_time_stats"),
            "gpu_memory_stats": summary.get("gpu_memory_stats"),
            "gradient_norm_stats": summary.get("gradient_norm_stats"),
        }

    def get_froc_data(self, run_id: str) -> tuple[list[float], list[float]]:
        """
        获取 FROC 曲线数据（用于绘图）。
        
        Args:
            run_id: 运行 ID
        
        Returns:
            (sensitivities, fp_per_case) 元组
        """
        data = self.load_evaluation_run(run_id)
        froc = data.get("froc_curve", [])
        sensitivities = [p.get("sensitivity", 0) for p in froc]
        fp_per_case = [p.get("fp_per_case", 0) for p in froc]
        return sensitivities, fp_per_case

    def get_detection_summary(self, run_id: str) -> dict:
        """
        获取检测汇总（用于快速绘图）。
        
        Args:
            run_id: 运行 ID
        
        Returns:
            包含检测统计的字典
        """
        data = self.load_detection_run(run_id)
        return data.get("summary", {})

    def get_evaluation_summary(self, run_id: str) -> dict:
        """
        获取评估汇总（用于快速绘图）。
        
        Args:
            run_id: 运行 ID
        
        Returns:
            包含评估统计的字典
        """
        run_dir = self.metrics_dir / "evaluation" / run_id
        return self._load_json(run_dir / "evaluation_summary.json")

    def compare_runs(self, run_type: str, run_ids: list[str]) -> dict:
        """
        比较多个运行。
        
        Args:
            run_type: 运行类型
            run_ids: 运行 ID 列表
        
        Returns:
            包含各运行数据的字典
        """
        results = {}
        for run_id in run_ids:
            if run_type == "training":
                results[run_id] = self.load_training_run(run_id)
            elif run_type == "detection":
                results[run_id] = self.load_detection_run(run_id)
            elif run_type == "evaluation":
                results[run_id] = self.load_evaluation_run(run_id)
            elif run_type == "experiments":
                results[run_id] = self.load_experiment_run(run_id)
        return results

    def _load_json(self, path: Path) -> dict | list:
        """加载 JSON 文件"""
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
