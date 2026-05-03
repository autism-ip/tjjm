"""
Tests for data collection and metrics reading.
"""

import json
import numpy as np
import pytest

from src.utils.collector import (
    BaseCollector,
    TrainingCollector,
    DetectionCollector,
    EvaluationCollector,
)
from src.utils.metrics_reader import MetricsReader


# ============================================================
# BaseCollector Tests
# ============================================================

class TestBaseCollector:
    """BaseCollector 基础功能测试"""

    def test_init_creates_directory(self, tmp_path):
        """初始化时创建输出目录"""
        collector = BaseCollector(str(tmp_path), "test_run")
        assert collector.output_dir.exists()
        assert collector.output_dir == tmp_path / "test_run"

    def test_init_default_run_id(self, tmp_path):
        """默认 run_id 使用时间戳"""
        collector = BaseCollector(str(tmp_path))
        assert collector.run_id is not None
        assert len(collector.run_id) > 0

    def test_append_adds_to_buffer(self, tmp_path):
        """append 添加记录到缓冲区"""
        collector = BaseCollector(str(tmp_path), "test_run")
        collector.append({"key": "value"})
        assert len(collector._buffer) == 1
        assert collector._buffer[0]["key"] == "value"

    def test_append_adds_timestamp(self, tmp_path):
        """append 自动添加时间戳"""
        collector = BaseCollector(str(tmp_path), "test_run")
        collector.append({"key": "value"})
        assert "timestamp" in collector._buffer[0]

    def test_save_config(self, tmp_path):
        """save_config 保存配置快照"""
        collector = BaseCollector(str(tmp_path), "test_run")
        config = {"learning_rate": 0.001, "batch_size": 4}
        collector.save_config(config)
        
        config_path = collector.output_dir / "config.json"
        assert config_path.exists()
        
        with open(config_path, "r") as f:
            saved = json.load(f)
        assert saved["config"] == config
        assert "run_id" in saved
        assert "created_at" in saved

    def test_compute_array_stats(self, tmp_path):
        """_compute_array_stats 计算数组统计"""
        collector = BaseCollector(str(tmp_path), "test_run")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = collector._compute_array_stats(arr)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p50" in stats
        assert "p90" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0


# ============================================================
# TrainingCollector Tests
# ============================================================

class TestTrainingCollector:
    """TrainingCollector 训练数据收集测试"""

    def test_log_step(self, tmp_path):
        """记录训练步数据"""
        collector = TrainingCollector(str(tmp_path), "test_run")
        collector.log_step(step=100, train_loss=0.5, lr=0.001, grad_norm=1.5)
        
        assert len(collector._step_records) == 1
        record = collector._step_records[0]
        assert record["step"] == 100
        assert record["train_loss"] == 0.5
        assert record["learning_rate"] == 0.001
        assert record["gradient_norm"] == 1.5

    def test_log_step_optional_fields(self, tmp_path):
        """记录训练步数据（可选字段）"""
        collector = TrainingCollector(str(tmp_path), "test_run")
        collector.log_step(step=100, train_loss=0.5)
        
        record = collector._step_records[0]
        assert "learning_rate" not in record
        assert "gradient_norm" not in record

    def test_log_epoch(self, tmp_path):
        """记录 epoch 数据"""
        collector = TrainingCollector(str(tmp_path), "test_run")
        collector.log_epoch(
            epoch=1, train_loss=0.5, val_loss=0.4,
            epoch_time=10.5, gpu_memory_mb=1024
        )
        
        assert len(collector._epoch_records) == 1
        record = collector._epoch_records[0]
        assert record["epoch"] == 1
        assert record["train_loss_epoch"] == 0.5
        assert record["val_loss_epoch"] == 0.4
        assert record["epoch_time"] == 10.5
        assert record["gpu_memory_mb"] == 1024

    def test_log_epoch_flushes_to_disk(self, tmp_path):
        """log_epoch 自动刷新到磁盘"""
        collector = TrainingCollector(str(tmp_path), "test_run")
        collector.log_epoch(epoch=1, train_loss=0.5, val_loss=0.4)
        
        # 检查文件是否生成
        records_path = collector.output_dir / "records.json"
        summary_path = collector.output_dir / "summary.json"
        assert records_path.exists()
        assert summary_path.exists()

    def test_summary_content(self, tmp_path):
        """汇总内容正确"""
        collector = TrainingCollector(str(tmp_path), "test_run")
        
        # 记录多个 epoch
        for epoch in range(3):
            collector.log_epoch(
                epoch=epoch,
                train_loss=0.5 - epoch * 0.1,
                val_loss=0.4 - epoch * 0.1,
                epoch_time=10.0 + epoch,
            )
        
        # 读取汇总
        with open(collector.output_dir / "summary.json", "r") as f:
            summary = json.load(f)
        
        assert summary["total_epochs"] == 3
        assert summary["best_val_loss"] == pytest.approx(0.2)
        assert summary["best_epoch"] == 2
        assert len(summary["train_loss_history"]) == 3
        assert len(summary["val_loss_history"]) == 3


# ============================================================
# DetectionCollector Tests
# ============================================================

class TestDetectionCollector:
    """DetectionCollector 检测数据收集测试"""

    def test_log_case(self, tmp_path):
        """记录检测 case 数据"""
        collector = DetectionCollector(str(tmp_path), "test_run")
        anomaly_map = np.random.rand(64, 64, 64)
        
        collector.log_case(
            case_id="case_001",
            anomaly_map=anomaly_map,
            processing_time=5.0,
            anomaly_map_path="/path/to/anomaly.nii.gz"
        )
        
        assert len(collector._case_records) == 1
        record = collector._case_records[0]
        assert record["case_id"] == "case_001"
        assert record["processing_time"] == 5.0
        assert record["anomaly_map_path"] == "/path/to/anomaly.nii.gz"
        assert "anomaly_map_stats" in record
        assert "anomaly_map_shape" in record

    def test_log_case_with_ct_volume(self, tmp_path):
        """记录检测 case 数据（包含 CT volume）"""
        collector = DetectionCollector(str(tmp_path), "test_run")
        anomaly_map = np.random.rand(64, 64, 64)
        ct_volume = np.random.rand(64, 64, 64)
        
        collector.log_case(
            case_id="case_001",
            anomaly_map=anomaly_map,
            ct_volume=ct_volume,
        )
        
        record = collector._case_records[0]
        assert "ct_stats" in record
        assert "ct_shape" in record

    def test_flush_creates_summary(self, tmp_path):
        """flush 创建汇总文件"""
        collector = DetectionCollector(str(tmp_path), "test_run")
        
        for i in range(3):
            anomaly_map = np.random.rand(64, 64, 64)
            collector.log_case(
                case_id=f"case_{i:03d}",
                anomaly_map=anomaly_map,
                processing_time=float(i + 1),
            )
        
        collector.flush()
        
        summary_path = collector.output_dir / "detection_summary.json"
        assert summary_path.exists()
        
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        assert summary["total_cases"] == 3
        assert summary["total_time"] == 6.0
        assert len(summary["case_ids"]) == 3


# ============================================================
# EvaluationCollector Tests
# ============================================================

class TestEvaluationCollector:
    """EvaluationCollector 评估数据收集测试"""

    def test_log_case_metrics(self, tmp_path):
        """记录病例级指标"""
        collector = EvaluationCollector(str(tmp_path), "test_run")
        
        collector.log_case_metrics(
            case_id="case_001",
            metrics={"nodule_recall": 1.0, "fp_components": 2, "case_auc": 0.85}
        )
        
        assert len(collector._case_records) == 1
        record = collector._case_records[0]
        assert record["case_id"] == "case_001"
        assert record["nodule_recall"] == 1.0
        assert record["fp_components"] == 2
        assert record["case_auc"] == 0.85

    def test_log_global_metrics(self, tmp_path):
        """记录全局指标"""
        collector = EvaluationCollector(str(tmp_path), "test_run")
        
        metrics = {
            "lesion_recall": 0.8,
            "fp_per_case": 1.5,
            "case_auc": 0.82,
        }
        collector.log_global_metrics(metrics)
        
        path = collector.output_dir / "global_metrics.json"
        assert path.exists()
        
        with open(path, "r") as f:
            saved = json.load(f)
        assert saved == metrics

    def test_log_froc_curve(self, tmp_path):
        """记录 FROC 曲线"""
        collector = EvaluationCollector(str(tmp_path), "test_run")
        
        froc_points = [
            {"sensitivity": 0.9, "fp_per_case": 0.5},
            {"sensitivity": 0.8, "fp_per_case": 1.0},
            {"sensitivity": 0.7, "fp_per_case": 2.0},
        ]
        collector.log_froc_curve(froc_points)
        
        path = collector.output_dir / "froc_curve.json"
        assert path.exists()
        
        with open(path, "r") as f:
            saved = json.load(f)
        assert len(saved) == 3

    def test_log_threshold_sweep(self, tmp_path):
        """记录阈值扫描结果"""
        collector = EvaluationCollector(str(tmp_path), "test_run")
        
        sweep_results = [
            {"percentile": 99.0, "threshold": 0.5, "lesion_recall": 0.9},
            {"percentile": 99.5, "threshold": 0.6, "lesion_recall": 0.8},
        ]
        collector.log_threshold_sweep(sweep_results)
        
        path = collector.output_dir / "threshold_sweep.json"
        assert path.exists()

    def test_log_recommended_operating_point(self, tmp_path):
        """记录推荐工作点"""
        collector = EvaluationCollector(str(tmp_path), "test_run")
        
        op_point = {
            "percentile": 99.5,
            "threshold": 0.6,
            "sensitivity": 0.8,
            "fp_per_case": 1.0,
        }
        collector.log_recommended_operating_point(op_point)
        
        path = collector.output_dir / "recommended_op.json"
        assert path.exists()

    def test_flush_creates_summary(self, tmp_path):
        """flush 创建汇总文件"""
        collector = EvaluationCollector(str(tmp_path), "test_run")
        
        for i in range(3):
            collector.log_case_metrics(
                case_id=f"case_{i:03d}",
                metrics={
                    "nodule_recall": 0.8 + i * 0.1,
                    "fp_components": i,
                    "case_auc": 0.7 + i * 0.1,
                }
            )
        
        collector.flush()
        
        summary_path = collector.output_dir / "evaluation_summary.json"
        assert summary_path.exists()
        
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        assert summary["total_cases"] == 3
        assert "nodule_recall_stats" in summary
        assert "fp_components_stats" in summary
        assert "case_auc_stats" in summary


# ============================================================
# MetricsReader Tests
# ============================================================

class TestMetricsReader:
    """MetricsReader 指标读取测试"""

    def test_list_runs_empty(self, tmp_path):
        """列出空目录的运行"""
        reader = MetricsReader(str(tmp_path))
        runs = reader.list_runs("training")
        assert runs == []

    def test_list_runs(self, tmp_path):
        """列出训练运行"""
        # 创建测试目录
        (tmp_path / "training" / "run_001").mkdir(parents=True)
        (tmp_path / "training" / "run_002").mkdir(parents=True)
        
        reader = MetricsReader(str(tmp_path))
        runs = reader.list_runs("training")
        assert len(runs) == 2
        assert "run_001" in runs
        assert "run_002" in runs

    def test_load_training_run(self, tmp_path):
        """加载训练运行数据"""
        run_dir = tmp_path / "training" / "run_001"
        run_dir.mkdir(parents=True)
        
        # 创建测试文件
        config = {
            "run_id": "run_001",
            "created_at": "2026-05-03T12:00:00",
            "collector_type": "TrainingCollector",
            "config": {"learning_rate": 0.001, "batch_size": 4}
        }
        summary = {
            "total_epochs": 10,
            "train_loss_history": [0.5, 0.4, 0.3],
            "val_loss_history": [0.45, 0.35, 0.25],
            "best_val_loss": 0.25,
            "best_epoch": 2,
        }
        
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f)
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f)
        
        reader = MetricsReader(str(tmp_path))
        data = reader.load_training_run("run_001")
        
        assert data["config"]["config"]["learning_rate"] == 0.001
        assert data["summary"]["total_epochs"] == 10
        assert len(data["summary"]["train_loss_history"]) == 3

    def test_load_detection_run(self, tmp_path):
        """加载检测运行数据"""
        run_dir = tmp_path / "detection" / "run_001"
        run_dir.mkdir(parents=True)
        
        summary = {"total_cases": 100, "total_time": 500.0}
        with open(run_dir / "detection_summary.json", "w") as f:
            json.dump(summary, f)
        
        reader = MetricsReader(str(tmp_path))
        data = reader.load_detection_run("run_001")
        
        assert data["summary"]["total_cases"] == 100

    def test_load_evaluation_run(self, tmp_path):
        """加载评估运行数据"""
        run_dir = tmp_path / "evaluation" / "run_001"
        run_dir.mkdir(parents=True)
        
        froc = [
            {"sensitivity": 0.9, "fp_per_case": 0.5},
            {"sensitivity": 0.8, "fp_per_case": 1.0},
        ]
        with open(run_dir / "froc_curve.json", "w") as f:
            json.dump(froc, f)
        
        reader = MetricsReader(str(tmp_path))
        data = reader.load_evaluation_run("run_001")
        
        assert len(data["froc_curve"]) == 2

    def test_get_training_summary(self, tmp_path):
        """获取训练汇总"""
        run_dir = tmp_path / "training" / "run_001"
        run_dir.mkdir(parents=True)
        
        summary = {
            "epoch_history": [0, 1, 2],
            "train_loss_history": [0.5, 0.4, 0.3],
            "val_loss_history": [0.45, 0.35, 0.25],
            "best_val_loss": 0.25,
            "best_epoch": 2,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f)
        
        reader = MetricsReader(str(tmp_path))
        result = reader.get_training_summary("run_001")
        
        assert result["epochs"] == [0, 1, 2]
        assert result["train_loss"] == [0.5, 0.4, 0.3]
        assert result["val_loss"] == [0.45, 0.35, 0.25]
        assert result["best_val_loss"] == 0.25

    def test_get_froc_data(self, tmp_path):
        """获取 FROC 曲线数据"""
        run_dir = tmp_path / "evaluation" / "run_001"
        run_dir.mkdir(parents=True)
        
        froc = [
            {"sensitivity": 0.9, "fp_per_case": 0.5},
            {"sensitivity": 0.8, "fp_per_case": 1.0},
            {"sensitivity": 0.7, "fp_per_case": 2.0},
        ]
        with open(run_dir / "froc_curve.json", "w") as f:
            json.dump(froc, f)
        
        reader = MetricsReader(str(tmp_path))
        sensitivities, fp_per_case = reader.get_froc_data("run_001")
        
        assert sensitivities == [0.9, 0.8, 0.7]
        assert fp_per_case == [0.5, 1.0, 2.0]

    def test_compare_runs(self, tmp_path):
        """比较多个运行"""
        for run_id in ["run_001", "run_002"]:
            run_dir = tmp_path / "training" / run_id
            run_dir.mkdir(parents=True)
            summary = {"total_epochs": 10, "best_val_loss": 0.3}
            with open(run_dir / "summary.json", "w") as f:
                json.dump(summary, f)
        
        reader = MetricsReader(str(tmp_path))
        results = reader.compare_runs("training", ["run_001", "run_002"])
        
        assert len(results) == 2
        assert "run_001" in results
        assert "run_002" in results

    def test_missing_file_handling(self, tmp_path):
        """缺失文件处理"""
        reader = MetricsReader(str(tmp_path))
        data = reader.load_training_run("nonexistent")
        
        assert data["config"] == {}
        assert data["records"] == {}
        assert data["summary"] == {}
