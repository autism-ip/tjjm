"""
 * [INPUT]: 依赖 json, time, pathlib, numpy
 * [OUTPUT]: 对外提供 BaseCollector, TrainingCollector, DetectionCollector, EvaluationCollector
 * [POS]: src/utils/ 的数据收集器，负责训练/检测/评估全流程数据持久化
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np


__all__ = [
    "BaseCollector",
    "TrainingCollector",
    "DetectionCollector",
    "EvaluationCollector",
]


# ============================================================
# Base Collector
# ============================================================

class BaseCollector:
    """
    基础收集器，提供通用持久化功能。
    
    所有收集器继承此类，实现特定维度的数据收集。
    数据先缓存在内存中，调用 flush() 时写入磁盘。
    """

    def __init__(self, output_dir: str, run_id: str | None = None):
        """
        Args:
            output_dir: 输出根目录
            run_id: 运行ID，默认使用时间戳
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._metadata = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "collector_type": self.__class__.__name__,
        }

    def append(self, record: dict) -> None:
        """追加记录到缓冲区"""
        record["timestamp"] = time.time()
        self._buffer.append(record)

    def flush(self) -> None:
        """刷新缓冲区到磁盘"""
        if self._buffer:
            self._save_records()
            self._save_summary()
            self._buffer.clear()

    def save_config(self, config: dict) -> None:
        """保存配置快照"""
        config_with_meta = {
            **self._metadata,
            "config": config,
        }
        self._save_json(config_with_meta, "config.json")

    def _save_records(self) -> None:
        """保存原始记录"""
        self._save_json(self._buffer, "records.json")

    def _save_summary(self) -> None:
        """保存汇总统计（子类重写）"""
        pass

    def _save_json(self, data: dict | list, filename: str) -> None:
        """保存JSON文件"""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _compute_array_stats(self, arr: np.ndarray) -> dict:
        """计算数组统计信息"""
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


# ============================================================
# Training Collector
# ============================================================

class TrainingCollector(BaseCollector):
    """
    训练数据收集器。
    
    收集训练过程中的 loss、学习率、梯度范数、epoch 耗时、GPU 显存等数据。
    每个 epoch 自动刷新到磁盘。
    """

    def __init__(self, output_dir: str, run_id: str | None = None):
        super().__init__(output_dir, run_id)
        self._step_records: list[dict] = []
        self._epoch_records: list[dict] = []

    def log_step(
        self,
        step: int,
        train_loss: float,
        lr: float | None = None,
        grad_norm: float | None = None,
    ) -> None:
        """
        记录训练步数据。
        
        Args:
            step: 全局步数
            train_loss: 训练 loss
            lr: 当前学习率
            grad_norm: 梯度范数
        """
        record = {
            "type": "step",
            "step": step,
            "train_loss": float(train_loss),
        }
        if lr is not None:
            record["learning_rate"] = float(lr)
        if grad_norm is not None:
            record["gradient_norm"] = float(grad_norm)
        
        self._step_records.append(record)
        self.append(record)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time: float | None = None,
        gpu_memory_mb: int | None = None,
    ) -> None:
        """
        记录 epoch 数据。
        
        Args:
            epoch: 当前 epoch
            train_loss: epoch 平均训练 loss
            val_loss: epoch 平均验证 loss
            epoch_time: epoch 耗时（秒）
            gpu_memory_mb: GPU 显存使用（MB）
        """
        record = {
            "type": "epoch",
            "epoch": epoch,
            "train_loss_epoch": float(train_loss),
            "val_loss_epoch": float(val_loss),
        }
        if epoch_time is not None:
            record["epoch_time"] = float(epoch_time)
        if gpu_memory_mb is not None:
            record["gpu_memory_mb"] = int(gpu_memory_mb)
        
        self._epoch_records.append(record)
        self.append(record)
        self.flush()  # 每 epoch 刷新

    def _save_summary(self) -> None:
        """保存训练汇总"""
        if not self._epoch_records:
            return
        
        epochs = [r["epoch"] for r in self._epoch_records]
        train_losses = [r["train_loss_epoch"] for r in self._epoch_records]
        val_losses = [r["val_loss_epoch"] for r in self._epoch_records]
        
        summary = {
            **self._metadata,
            "total_epochs": len(epochs),
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": min(val_losses),
            "best_epoch": epochs[val_losses.index(min(val_losses))],
            "train_loss_history": train_losses,
            "val_loss_history": val_losses,
            "epoch_history": epochs,
        }
        
        # 添加可选统计
        epoch_times = [r.get("epoch_time") for r in self._epoch_records 
                       if r.get("epoch_time") is not None]
        if epoch_times:
            summary["epoch_time_stats"] = {
                "mean": float(np.mean(epoch_times)),
                "std": float(np.std(epoch_times)),
                "total": float(sum(epoch_times)),
            }
        
        gpu_memories = [r.get("gpu_memory_mb") for r in self._epoch_records 
                        if r.get("gpu_memory_mb") is not None]
        if gpu_memories:
            summary["gpu_memory_stats"] = {
                "mean": int(np.mean(gpu_memories)),
                "max": int(np.max(gpu_memories)),
            }
        
        grad_norms = [r.get("gradient_norm") for r in self._step_records 
                      if r.get("gradient_norm") is not None]
        if grad_norms:
            summary["gradient_norm_stats"] = {
                "mean": float(np.mean(grad_norms)),
                "std": float(np.std(grad_norms)),
                "max": float(np.max(grad_norms)),
            }
        
        self._save_json(summary, "summary.json")


# ============================================================
# Detection Collector
# ============================================================

class DetectionCollector(BaseCollector):
    """
    检测数据收集器。
    
    收集每个 case 的异常热图统计、重建误差、处理时间等数据。
    """

    def __init__(self, output_dir: str, run_id: str | None = None):
        super().__init__(output_dir, run_id)
        self._case_records: list[dict] = []

    def log_case(
        self,
        case_id: str,
        anomaly_map: np.ndarray,
        ct_volume: np.ndarray | None = None,
        processing_time: float | None = None,
        anomaly_map_path: str | None = None,
    ) -> None:
        """
        记录检测 case 数据。
        
        Args:
            case_id: 病例 ID
            anomaly_map: 异常热图数组
            ct_volume: 原始 CT 数组（可选）
            processing_time: 处理时间（秒）
            anomaly_map_path: 异常热图保存路径
        """
        record = {
            "type": "case",
            "case_id": case_id,
            "anomaly_map_stats": self._compute_array_stats(anomaly_map),
            "anomaly_map_shape": list(anomaly_map.shape),
        }
        
        if ct_volume is not None:
            record["ct_stats"] = self._compute_array_stats(ct_volume)
            record["ct_shape"] = list(ct_volume.shape)
        
        if processing_time is not None:
            record["processing_time"] = float(processing_time)
        
        if anomaly_map_path is not None:
            record["anomaly_map_path"] = anomaly_map_path
        
        self._case_records.append(record)
        self.append(record)

    def flush(self) -> None:
        """刷新到磁盘"""
        if self._buffer:
            self._save_json(self._case_records, "detection_cases.json")
            self._save_summary()
            self._buffer.clear()

    def _save_summary(self) -> None:
        """保存检测汇总"""
        if not self._case_records:
            return
        
        times = [r.get("processing_time", 0) for r in self._case_records]
        means = [r["anomaly_map_stats"]["mean"] for r in self._case_records]
        
        summary = {
            **self._metadata,
            "total_cases": len(self._case_records),
            "total_time": float(sum(times)),
            "avg_time_per_case": float(np.mean(times)),
            "std_time_per_case": float(np.std(times)),
            "anomaly_score_stats": {
                "mean": float(np.mean(means)),
                "std": float(np.std(means)),
                "min": float(np.min(means)),
                "max": float(np.max(means)),
            },
            "case_ids": [r["case_id"] for r in self._case_records],
        }
        self._save_json(summary, "detection_summary.json")


# ============================================================
# Evaluation Collector
# ============================================================

class EvaluationCollector(BaseCollector):
    """
    评估数据收集器。
    
    收集病例级指标、全局指标、FROC 曲线、阈值扫描结果等数据。
    """

    def __init__(self, output_dir: str, run_id: str | None = None):
        super().__init__(output_dir, run_id)
        self._case_records: list[dict] = []

    def log_case_metrics(self, case_id: str, metrics: dict) -> None:
        """
        记录病例级指标。
        
        Args:
            case_id: 病例 ID
            metrics: 指标字典
        """
        record = {
            "type": "case_metrics",
            "case_id": case_id,
            **metrics,
        }
        self._case_records.append(record)
        self.append(record)

    def log_global_metrics(self, metrics: dict) -> None:
        """记录全局指标"""
        self._save_json(metrics, "global_metrics.json")

    def log_froc_curve(self, froc_points: list) -> None:
        """记录 FROC 曲线"""
        self._save_json(froc_points, "froc_curve.json")

    def log_threshold_sweep(self, sweep_results: list) -> None:
        """记录阈值扫描结果"""
        self._save_json(sweep_results, "threshold_sweep.json")

    def log_recommended_operating_point(self, op_point: dict) -> None:
        """记录推荐工作点"""
        self._save_json(op_point, "recommended_op.json")

    def flush(self) -> None:
        """刷新到磁盘"""
        if self._buffer:
            self._save_json(self._case_records, "case_metrics.json")
            self._save_summary()
            self._buffer.clear()

    def _save_summary(self) -> None:
        """保存评估汇总"""
        if not self._case_records:
            return
        
        # 提取数值字段
        numeric_fields = ["nodule_recall", "fp_components", "case_auc", "case_ap"]
        summary = {
            **self._metadata,
            "total_cases": len(self._case_records),
        }
        
        for field in numeric_fields:
            values = [r.get(field) for r in self._case_records 
                      if r.get(field) is not None]
            if values:
                summary[f"{field}_stats"] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        
        self._save_json(summary, "evaluation_summary.json")
