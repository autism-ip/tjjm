# 多维度数据收集方案

## 1. 设计目标

**核心原则**: 实验数据必须持久化，支持后续可视化和分析而无需重跑实验。

### 1.1 收集维度

| 维度 | 数据类型 | 收集频率 | 用途 |
|------|----------|----------|------|
| 训练 | loss、lr、grad_norm、epoch_time、gpu_memory | 每步/每epoch | 训练曲线、性能分析 |
| 检测 | anomaly_map_stats、reconstruction_error、processing_time | 每case | 检测统计、性能分析 |
| 评估 | case_metrics、global_metrics、froc_curve、threshold_sweep | 每case/全局 | 性能评估、报告生成 |
| 实验 | synthetic_results、ablation_results、compare_results | 每实验 | 实验对比、消融分析 |

### 1.2 存储格式

- **主格式**: JSON（结构化、可读、易解析）
- **辅助格式**: CSV（表格数据）、NPY（数值数组）
- **命名规范**: `{类型}_{时间戳}.json` 或 `{类型}_{run_id}.json`

## 2. 收集器架构

### 2.1 类层次

```
BaseCollector (基础收集器)
├── TrainingCollector (训练数据)
├── DetectionCollector (检测数据)
├── EvaluationCollector (评估数据)
└── ExperimentCollector (实验数据)
```

### 2.2 基础收集器

```python
class BaseCollector:
    """基础收集器，提供通用持久化功能"""
    
    def __init__(self, output_dir: str, run_id: str):
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = []
        self._metadata = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "collector_type": self.__class__.__name__
        }
    
    def append(self, record: dict):
        """追加记录到缓冲区"""
        record["timestamp"] = time.time()
        self._buffer.append(record)
    
    def flush(self):
        """刷新缓冲区到磁盘"""
        if self._buffer:
            self._save_records()
            self._save_summary()
            self._buffer.clear()
    
    def save_config(self, config: dict):
        """保存配置快照"""
        self._save_json(config, "config.json")
    
    def _save_records(self):
        """保存原始记录"""
        self._save_json(self._buffer, "records.json")
    
    def _save_summary(self):
        """保存汇总统计（子类重写）"""
        pass
    
    def _save_json(self, data: dict | list, filename: str):
        """保存JSON文件"""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
```

## 3. 训练数据收集

### 3.1 收集字段

| 字段 | 类型 | 说明 | 收集时机 |
|------|------|------|----------|
| step | int | 全局步数 | 每步 |
| epoch | int | 当前epoch | 每epoch |
| train_loss | float | 训练loss | 每步 |
| val_loss | float | 验证loss | 每epoch |
| learning_rate | float | 学习率 | 每epoch |
| gradient_norm | float | 梯度范数 | 每步 |
| epoch_time | float | epoch耗时(秒) | 每epoch |
| gpu_memory_mb | int | GPU显存(MB) | 每epoch |
| batch_size | int | 批次大小 | 配置 |
| num_patches | int | patch数量 | 配置 |

### 3.2 实现

```python
class TrainingCollector(BaseCollector):
    """训练数据收集器"""
    
    def __init__(self, output_dir: str, run_id: str):
        super().__init__(output_dir, run_id)
        self._step_records = []
        self._epoch_records = []
    
    def log_step(self, step: int, train_loss: float, 
                 lr: float = None, grad_norm: float = None):
        """记录训练步数据"""
        record = {
            "type": "step",
            "step": step,
            "train_loss": train_loss,
        }
        if lr is not None:
            record["learning_rate"] = lr
        if grad_norm is not None:
            record["gradient_norm"] = grad_norm
        self._step_records.append(record)
        self.append(record)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  epoch_time: float = None, gpu_memory_mb: int = None):
        """记录epoch数据"""
        record = {
            "type": "epoch",
            "epoch": epoch,
            "train_loss_epoch": train_loss,
            "val_loss_epoch": val_loss,
        }
        if epoch_time is not None:
            record["epoch_time"] = epoch_time
        if gpu_memory_mb is not None:
            record["gpu_memory_mb"] = gpu_memory_mb
        self._epoch_records.append(record)
        self.append(record)
        self.flush()  # 每epoch刷新
    
    def _save_summary(self):
        """保存训练汇总"""
        if not self._epoch_records:
            return
        
        epochs = [r["epoch"] for r in self._epoch_records]
        train_losses = [r["train_loss_epoch"] for r in self._epoch_records]
        val_losses = [r["val_loss_epoch"] for r in self._epoch_records]
        
        summary = {
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
        if self._step_records:
            grad_norms = [r.get("gradient_norm") for r in self._step_records 
                         if r.get("gradient_norm") is not None]
            if grad_norms:
                summary["gradient_norm_stats"] = {
                    "mean": float(np.mean(grad_norms)),
                    "std": float(np.std(grad_norms)),
                    "max": float(np.max(grad_norms))
                }
        
        self._save_json(summary, "summary.json")
```

## 4. 检测数据收集

### 4.1 收集字段

| 字段 | 类型 | 说明 | 收集时机 |
|------|------|------|----------|
| case_id | str | 病例ID | 每case |
| anomaly_map_stats | dict | 异常热图统计 | 每case |
| reconstruction_error | dict | 重建误差统计 | 每case |
| processing_time | float | 处理时间(秒) | 每case |
| ct_shape | tuple | CT尺寸 | 每case |
| anomaly_map_path | str | 热图保存路径 | 每case |

### 4.2 实现

```python
class DetectionCollector(BaseCollector):
    """检测数据收集器"""
    
    def __init__(self, output_dir: str, run_id: str):
        super().__init__(output_dir, run_id)
        self._case_records = []
    
    def log_case(self, case_id: str, anomaly_map: np.ndarray,
                 ct_volume: np.ndarray = None, processing_time: float = None,
                 anomaly_map_path: str = None):
        """记录检测case数据"""
        record = {
            "type": "case",
            "case_id": case_id,
            "anomaly_map_stats": self._compute_array_stats(anomaly_map),
            "anomaly_map_shape": anomaly_map.shape,
        }
        
        if ct_volume is not None:
            record["ct_stats"] = self._compute_array_stats(ct_volume)
            record["ct_shape"] = ct_volume.shape
        
        if processing_time is not None:
            record["processing_time"] = processing_time
        
        if anomaly_map_path is not None:
            record["anomaly_map_path"] = anomaly_map_path
        
        self._case_records.append(record)
        self.append(record)
    
    def flush(self):
        """刷新到磁盘"""
        if self._buffer:
            self._save_json(self._case_records, "detection_cases.json")
            self._save_summary()
            self._buffer.clear()
    
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
    
    def _save_summary(self):
        """保存检测汇总"""
        if not self._case_records:
            return
        
        times = [r.get("processing_time", 0) for r in self._case_records]
        means = [r["anomaly_map_stats"]["mean"] for r in self._case_records]
        
        summary = {
            "total_cases": len(self._case_records),
            "total_time": sum(times),
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
```

## 5. 评估数据收集

### 5.1 收集字段

| 字段 | 类型 | 说明 | 收集时机 |
|------|------|------|----------|
| case_id | str | 病例ID | 每case |
| has_nodule | bool | 是否有结节 | 每case |
| nodule_recall | float | 结节召回率 | 每case |
| fp_components | int | 假阳性数 | 每case |
| case_auc | float | 病例AUC | 每case |
| lesion_recall | float | 全局结节召回 | 全局 |
| fp_per_case | float | 每病例假阳性 | 全局 |
| froc_curve | list | FROC曲线点 | 全局 |
| recommended_op | dict | 推荐工作点 | 全局 |

### 5.2 实现

```python
class EvaluationCollector(BaseCollector):
    """评估数据收集器"""
    
    def __init__(self, output_dir: str, run_id: str):
        super().__init__(output_dir, run_id)
        self._case_records = []
    
    def log_case_metrics(self, case_id: str, metrics: dict):
        """记录病例级指标"""
        record = {
            "type": "case_metrics",
            "case_id": case_id,
            **metrics,
        }
        self._case_records.append(record)
        self.append(record)
    
    def log_global_metrics(self, metrics: dict):
        """记录全局指标"""
        self._save_json(metrics, "global_metrics.json")
    
    def log_froc_curve(self, froc_points: list):
        """记录FROC曲线"""
        self._save_json(froc_points, "froc_curve.json")
    
    def log_threshold_sweep(self, sweep_results: list):
        """记录阈值扫描结果"""
        self._save_json(sweep_results, "threshold_sweep.json")
    
    def log_recommended_operating_point(self, op_point: dict):
        """记录推荐工作点"""
        self._save_json(op_point, "recommended_op.json")
    
    def flush(self):
        """刷新到磁盘"""
        if self._buffer:
            self._save_json(self._case_records, "case_metrics.json")
            self._save_summary()
            self._buffer.clear()
    
    def _save_summary(self):
        """保存评估汇总"""
        if not self._case_records:
            return
        
        # 提取数值字段
        numeric_fields = ["nodule_recall", "fp_components", "case_auc", "case_ap"]
        summary = {"total_cases": len(self._case_records)}
        
        for field in numeric_fields:
            values = [r.get(field) for r in self._case_records if r.get(field) is not None]
            if values:
                summary[f"{field}_stats"] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        
        self._save_json(summary, "evaluation_summary.json")
```

## 6. 可视化数据读取

### 6.1 MetricsReader

```python
class MetricsReader:
    """读取持久化的指标数据，支持可视化而无需重跑实验"""
    
    def __init__(self, metrics_dir: str):
        self.metrics_dir = Path(metrics_dir)
    
    def list_runs(self, run_type: str) -> list[str]:
        """列出所有运行"""
        run_dir = self.metrics_dir / run_type
        if not run_dir.exists():
            return []
        return sorted([d.name for d in run_dir.iterdir() if d.is_dir()])
    
    def load_training_run(self, run_id: str) -> dict:
        """加载训练运行数据"""
        run_dir = self.metrics_dir / "training" / run_id
        return {
            "config": self._load_json(run_dir / "config.json"),
            "records": self._load_json(run_dir / "records.json"),
            "summary": self._load_json(run_dir / "summary.json"),
        }
    
    def load_detection_run(self, run_id: str) -> dict:
        """加载检测运行数据"""
        run_dir = self.metrics_dir / "detection" / run_id
        return {
            "config": self._load_json(run_dir / "config.json"),
            "cases": self._load_json(run_dir / "detection_cases.json"),
            "summary": self._load_json(run_dir / "detection_summary.json"),
        }
    
    def load_evaluation_run(self, run_id: str) -> dict:
        """加载评估运行数据"""
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
        """加载实验运行数据"""
        run_dir = self.metrics_dir / "experiments" / run_id
        return {
            "protocol": self._load_json(run_dir / "protocol.json"),
            "synthetic_results": self._load_json(run_dir / "synthetic_results.json"),
            "ablation_results": self._load_json(run_dir / "ablation_results.json"),
            "compare_results": self._load_json(run_dir / "compare_results.json"),
        }
    
    def get_training_summary(self, run_id: str) -> dict:
        """获取训练汇总（用于快速绘图）"""
        data = self.load_training_run(run_id)
        summary = data.get("summary", {})
        return {
            "epochs": summary.get("epoch_history", []),
            "train_loss": summary.get("train_loss_history", []),
            "val_loss": summary.get("val_loss_history", []),
            "best_val_loss": summary.get("best_val_loss"),
            "best_epoch": summary.get("best_epoch"),
        }
    
    def get_froc_data(self, run_id: str) -> tuple[list, list]:
        """获取FROC曲线数据（用于绘图）"""
        data = self.load_evaluation_run(run_id)
        froc = data.get("froc_curve", [])
        sensitivities = [p.get("sensitivity", 0) for p in froc]
        fp_per_case = [p.get("fp_per_case", 0) for p in froc]
        return sensitivities, fp_per_case
    
    def _load_json(self, path: Path) -> dict | list:
        """加载JSON文件"""
        if not path.exists():
            return {} if "summary" in path.name or "config" in path.name else []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {} if "summary" in path.name or "config" in path.name else []
```

## 7. 集成到现有代码

### 7.1 训练脚本集成

```python
# scripts/train_autoencoder.py
from src.utils.collector import TrainingCollector

def main(cfg):
    # 初始化收集器
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    collector = TrainingCollector(
        output_dir=cfg.get("data_collection", {}).get("output_dir", "./outputs/metrics/training"),
        run_id=run_id
    )
    collector.save_config(OmegaConf.to_container(cfg, resolve=True))
    
    # 训练循环中收集数据
    for epoch in range(max_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(...)
        val_loss = validate(...)
        epoch_time = time.time() - start_time
        
        collector.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            epoch_time=epoch_time,
            gpu_memory_mb=get_gpu_memory()
        )
    
    collector.flush()
```

### 7.2 检测脚本集成

```python
# scripts/detect.py
from src.utils.collector import DetectionCollector

def main(cfg):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    collector = DetectionCollector(
        output_dir=cfg.get("data_collection", {}).get("output_dir", "./outputs/metrics/detection"),
        run_id=run_id
    )
    collector.save_config(OmegaConf.to_container(cfg, resolve=True))
    
    # 检测循环中收集数据
    for case_id, ct_path in test_cases:
        start_time = time.time()
        anomaly_map = detect_single(ct_path)
        processing_time = time.time() - start_time
        
        collector.log_case(
            case_id=case_id,
            anomaly_map=anomaly_map,
            processing_time=processing_time,
            anomaly_map_path=str(output_dir / f"{case_id}_anomaly.nii.gz")
        )
    
    collector.flush()
```

## 8. 目录结构

```
outputs/metrics/
├── training/
│   ├── run_20260503_120000/
│   │   ├── config.json
│   │   ├── records.json
│   │   └── summary.json
│   └── ...
├── detection/
│   ├── run_20260503_140000/
│   │   ├── config.json
│   │   ├── detection_cases.json
│   │   └── detection_summary.json
│   └── ...
├── evaluation/
│   ├── run_20260503_150000/
│   │   ├── config.json
│   │   ├── case_metrics.json
│   │   ├── global_metrics.json
│   │   ├── froc_curve.json
│   │   ├── threshold_sweep.json
│   │   ├── recommended_op.json
│   │   └── evaluation_summary.json
│   └── ...
└── experiments/
    ├── exp_20260503_160000/
    │   ├── protocol.json
    │   ├── synthetic_results.json
    │   ├── ablation_results.json
    │   └── compare_results.json
    └── ...
```

## 9. 可视化示例

### 9.1 训练曲线

```python
reader = MetricsReader("./outputs/metrics")
summary = reader.get_training_summary("run_20260503_120000")

plt.figure(figsize=(10, 6))
plt.plot(summary["epochs"], summary["train_loss"], label="Train Loss")
plt.plot(summary["epochs"], summary["val_loss"], label="Val Loss")
plt.axvline(x=summary["best_epoch"], color="r", linestyle="--", label=f"Best Epoch ({summary['best_epoch']})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.legend()
plt.savefig("training_curve.png")
```

### 9.2 FROC曲线

```python
reader = MetricsReader("./outputs/metrics")
sensitivities, fp_per_case = reader.get_froc_data("run_20260503_150000")

plt.figure(figsize=(8, 6))
plt.plot(fp_per_case, sensitivities, "b-o")
plt.xlabel("False Positives per Case")
plt.ylabel("Sensitivity")
plt.title("FROC Curve")
plt.grid(True)
plt.savefig("froc_curve.png")
```

## 10. 扩展性

### 10.1 新增维度

如需新增收集维度，只需：
1. 继承 `BaseCollector`
2. 实现 `log_xxx()` 方法
3. 重写 `_save_summary()`

### 10.2 新增可视化

如需新增可视化，只需：
1. 使用 `MetricsReader` 加载数据
2. 实现绘图函数
3. 无需修改收集器或重跑实验
