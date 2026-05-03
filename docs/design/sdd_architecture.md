# SDD - 系统架构设计

## 1. 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI 入口层 (scripts/)                    │
│  train_autoencoder.py │ detect.py │ run_experiments.py      │
└───────────────────────┼─────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│                   配置层 (configs/)                          │
│  train_autoencoder.yaml │ detect.yaml │ experiments.yaml     │
└───────────────────────┼─────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│                   业务层 (src/)                              │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ data/   │  │ models/  │  │training/ │  │detection/│      │
│  │         │  │          │  │          │  │          │      │
│  │download │  │autoencoder│ │ trainer  │  │sliding   │      │
│  │preprocess│ │weights   │  │losses    │  │anomaly   │      │
│  │dataset  │  │diffusion │  │callbacks │  │fusion    │      │
│  └─────────┘  └──────────┘  └──────────┘  └──────────┘      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │evaluation│  │experiments│ │ utils/   │                   │
│  │          │  │          │  │          │                   │
│  │metrics   │  │synthetic │  │logging   │                   │
│  │luna16    │  │analysis  │  │viz       │                   │
│  │reporter  │  │protocol  │  │config    │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│               数据收集层 (src/utils/collector.py)            │
│  TrainingCollector │ DetectionCollector │ EvaluationCollector │
└───────────────────────┼─────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│               持久化层 (outputs/metrics/)                    │
│  training/ │ detection/ │ evaluation/ │ experiments/         │
└─────────────────────────────────────────────────────────────┘
```

## 2. 模块职责矩阵

| 模块 | 职责 | 输入 | 输出 | 测试覆盖 |
|------|------|------|------|----------|
| `data/download` | Kaggle下载 | 配置 | CT文件 | ✓ |
| `data/preprocess` | HU窗/重采样/归一化 | .mhd/.raw | 预处理volume | ✓ |
| `data/dataset` | PyTorch Dataset | 预处理volume | (patch, patch) | ✓ |
| `models/autoencoder` | SwinUNETR+UNet | CT patch | 重建patch | ✓ |
| `models/weights` | 预训练加载 | checkpoint路径 | 加载结果 | ✓ |
| `training/trainer` | LightningModule | batch | loss | △ |
| `training/losses` | WeightedMSELoss | pred/target | loss | ✓ |
| `training/callbacks` | 可视化回调 | epoch | PNG | ✗ |
| `detection/sliding_window` | MONAI滑窗推理 | volume | 重建volume | ✓ |
| `detection/anomaly_map` | 误差计算 | orig/recon | anomaly_map | ✓ |
| `detection/inference` | 目录推理 | CT目录 | NIfTI热图 | △ |
| `evaluation/metrics` | Dice/AUC/F1 | pred/gt | scores | ✓ |
| `evaluation/luna16` | FROC评估 | 热图+annotation | 报告 | ✓ |
| `evaluation/reporter` | 报告生成 | metrics | JSON/CSV/PNG | ✗ |
| `experiments/synthetic` | 合成异常 | volume | 敏感性报告 | ✓ |
| `experiments/analysis` | 统计汇总 | 多报告 | 汇总报告 | ✓ |
| `experiments/protocol` | 协议生成 | 配置 | Markdown/JSON | ✓ |
| `utils/collector` | 数据收集 | 训练/检测/评估数据 | 持久化JSON | ✓ |

## 3. 数据流设计

### 3.1 训练数据流

```
LUNA16 CT files
    ↓ SimpleITK读取
HU volume [-1000, 400]
    ↓ HU窗/重采样/归一化
normalized volume [0, 1]
    ↓ patch提取 + 结节过滤
healthy patches
    ↓ DataLoader
batch (B, 1, 64, 64, 64)
    ↓ Autoencoder3D
reconstructed patches
    ↓ WeightedMSELoss
loss → backward → optimizer
    ↓
TrainingCollector收集:
  - train_loss (每步)
  - val_loss (每epoch)
  - learning_rate (每epoch)
  - gradient_norm (每步)
  - epoch_time (每epoch)
  - gpu_memory (每epoch)
    ↓
持久化到 outputs/metrics/training/{run_id}/
```

### 3.2 检测数据流

```
测试CT volume
    ↓ SlidingWindowInferer
overlap patches (50%重叠)
    ↓ Autoencoder3D
reconstructed patches
    ↓ Gaussian加权融合
reconstructed volume
    ↓ |original - reconstructed|
anomaly map (D, H, W)
    ↓
DetectionCollector收集:
  - anomaly_map_stats (mean, std, min, max, percentiles)
  - reconstruction_error (mse, mae)
  - processing_time
    ↓
持久化到 outputs/metrics/detection/{run_id}/
```

### 3.3 评估数据流

```
anomaly maps
    ↓ 百分位阈值扫描
多个阈值结果
    ↓ 与annotation球形mask交叉
nodule_hits / fp_components
    ↓ 汇总统计
lesion_recall / fp_per_case
    ↓ FROC曲线
推荐工作点
    ↓
EvaluationCollector收集:
  - case_metrics (auc, ap, recall, fp)
  - global_metrics (lesion_recall, fp_per_case)
  - froc_curve_points
  - threshold_stats
    ↓
持久化到 outputs/metrics/evaluation/{run_id}/
```

## 4. 数据收集架构

### 4.1 收集器层次

```python
class BaseCollector:
    """基础收集器，提供通用持久化功能"""
    def __init__(self, output_dir: str, run_id: str):
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = []
    
    def append(self, record: dict):
        """追加记录到缓冲区"""
        self._buffer.append(record)
    
    def flush(self):
        """刷新缓冲区到磁盘"""
        if self._buffer:
            self._save_json(self._buffer, "records.json")
            self._buffer.clear()
    
    def _save_json(self, data: dict | list, filename: str):
        """保存JSON文件"""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class TrainingCollector(BaseCollector):
    """训练数据收集器"""
    def log_step(self, step: int, train_loss: float, lr: float, grad_norm: float):
        self.append({
            "type": "step",
            "step": step,
            "train_loss": train_loss,
            "learning_rate": lr,
            "gradient_norm": grad_norm,
            "timestamp": time.time()
        })
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  epoch_time: float, gpu_memory: int):
        self.append({
            "type": "epoch",
            "epoch": epoch,
            "train_loss_epoch": train_loss,
            "val_loss_epoch": val_loss,
            "epoch_time": epoch_time,
            "gpu_memory_mb": gpu_memory,
            "timestamp": time.time()
        })
        self.flush()  # 每epoch刷新


class DetectionCollector(BaseCollector):
    """检测数据收集器"""
    def log_case(self, case_id: str, anomaly_map: np.ndarray, 
                 processing_time: float):
        self.append({
            "type": "case",
            "case_id": case_id,
            "anomaly_map_stats": {
                "mean": float(np.mean(anomaly_map)),
                "std": float(np.std(anomaly_map)),
                "min": float(np.min(anomaly_map)),
                "max": float(np.max(anomaly_map)),
                "p50": float(np.percentile(anomaly_map, 50)),
                "p90": float(np.percentile(anomaly_map, 90)),
                "p95": float(np.percentile(anomaly_map, 95)),
                "p99": float(np.percentile(anomaly_map, 99))
            },
            "processing_time": processing_time,
            "timestamp": time.time()
        })
    
    def flush(self):
        if self._buffer:
            self._save_json(self._buffer, "detection_cases.json")
            self._save_summary()
            self._buffer.clear()
    
    def _save_summary(self):
        """保存汇总统计"""
        times = [r["processing_time"] for r in self._buffer]
        summary = {
            "total_cases": len(self._buffer),
            "total_time": sum(times),
            "avg_time": np.mean(times),
            "std_time": np.std(times)
        }
        self._save_json(summary, "detection_summary.json")


class EvaluationCollector(BaseCollector):
    """评估数据收集器"""
    def log_case_metrics(self, case_id: str, metrics: dict):
        self.append({
            "type": "case_metrics",
            "case_id": case_id,
            **metrics,
            "timestamp": time.time()
        })
    
    def log_global_metrics(self, metrics: dict):
        self._save_json(metrics, "global_metrics.json")
    
    def log_froc_curve(self, froc_points: list):
        self._save_json(froc_points, "froc_curve.json")
    
    def log_threshold_sweep(self, sweep_results: list):
        self._save_json(sweep_results, "threshold_sweep.json")
```

### 4.2 目录结构

```
outputs/metrics/
├── training/
│   ├── run_20260503_120000/
│   │   ├── config.json              # 训练配置快照
│   │   ├── records.json             # 所有step/epoch记录
│   │   ├── summary.json             # 汇总统计
│   │   └── checkpoints/             # checkpoint路径记录
│   └── run_20260503_130000/
│       └── ...
├── detection/
│   ├── run_20260503_140000/
│   │   ├── config.json              # 检测配置快照
│   │   ├── detection_cases.json     # 每个case的详细记录
│   │   ├── detection_summary.json   # 汇总统计
│   │   └── anomaly_maps/            # 异常热图路径记录
│   └── ...
├── evaluation/
│   ├── run_20260503_150000/
│   │   ├── config.json              # 评估配置快照
│   │   ├── case_metrics.json        # 每个case的指标
│   │   ├── global_metrics.json      # 全局指标
│   │   ├── froc_curve.json          # FROC曲线点
│   │   ├── threshold_sweep.json     # 阈值扫描结果
│   │   └── recommended_op.json      # 推荐工作点
│   └── ...
└── experiments/
    ├── exp_20260503_160000/
    │   ├── protocol.json            # 实验协议
    │   ├── synthetic_results.json   # 合成异常结果
    │   ├── ablation_results.json    # 消融结果
    │   └── compare_results.json     # 对比结果
    └── ...
```

## 5. 配置扩展

### 5.1 新增配置项

```yaml
# configs/train_autoencoder.yaml 新增
data_collection:
  enabled: true
  output_dir: "./outputs/metrics/training"
  log_every_n_steps: 10
  save_checkpoints: true
  save_reconstructions: true
  save_gradient_stats: true

# configs/detect.yaml 新增
data_collection:
  enabled: true
  output_dir: "./outputs/metrics/detection"
  save_anomaly_maps: true
  save_processing_stats: true

# configs/experiments.yaml 新增
data_collection:
  enabled: true
  output_dir: "./outputs/metrics/experiments"
  save_synthetic_details: true
  save_ablation_details: true
```

## 6. 可视化接口

### 6.1 数据读取器

```python
class MetricsReader:
    """读取持久化的指标数据"""
    def __init__(self, metrics_dir: str):
        self.metrics_dir = Path(metrics_dir)
    
    def list_runs(self, run_type: str) -> list[str]:
        """列出所有运行"""
        run_dir = self.metrics_dir / run_type
        return sorted([d.name for d in run_dir.iterdir() if d.is_dir()])
    
    def load_training_run(self, run_id: str) -> dict:
        """加载训练运行数据"""
        run_dir = self.metrics_dir / "training" / run_id
        return {
            "config": self._load_json(run_dir / "config.json"),
            "records": self._load_json(run_dir / "records.json"),
            "summary": self._load_json(run_dir / "summary.json")
        }
    
    def load_detection_run(self, run_id: str) -> dict:
        """加载检测运行数据"""
        run_dir = self.metrics_dir / "detection" / run_id
        return {
            "config": self._load_json(run_dir / "config.json"),
            "cases": self._load_json(run_dir / "detection_cases.json"),
            "summary": self._load_json(run_dir / "detection_summary.json")
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
            "recommended_op": self._load_json(run_dir / "recommended_op.json")
        }
    
    def _load_json(self, path: Path) -> dict | list:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
```

## 7. 已知问题与修复方案

| 问题 | 严重度 | 修复方案 |
|------|--------|----------|
| losses.py文档与实现不一致 | P0 | 修正docstring |
| train_loss缺少epoch级聚合 | P1 | 设置on_epoch=True |
| callbacks.py datamodule硬依赖 | P1 | 改用trainer.val_dataloaders |
| 文件缓存策略 | P2 | 实现LRU淘汰 |
| 索引构建耗时 | P2 | 持久化pickle缓存 |
| feature_size硬编码 | P3 | 提升为配置项 |
| detect.py评估分支不执行 | P2 | 补充ground_truth传递 |
| fusion.py未使用 | P3 | 清理或激活 |
