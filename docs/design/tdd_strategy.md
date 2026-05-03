# TDD - 测试驱动设计策略

## 1. 测试金字塔

```
                    ┌─────────────────┐
                    │   E2E测试 (3-5) │  端到端流程验证
                    │   test_e2e_*    │
                    └─────────────────┘
                  ┌─────────────────────┐
                  │  集成测试 (15-20)   │  模块间协作
                  │  test_integration  │
                  └─────────────────────┘
              ┌─────────────────────────────┐
              │     单元测试 (150+)         │  纯函数/类
              │     test_unit_*            │
              └─────────────────────────────┘
```

## 2. 测试覆盖目标

| 层级 | 当前 | 目标 | 优先级 |
|------|------|------|--------|
| 单元测试 | 124 | 150+ | 高 |
| 集成测试 | 12 | 20+ | 中 |
| E2E测试 | 0 | 3-5 | 高 |
| 总通过率 | 162 passed | 200+ passed | - |

## 3. 新增测试计划

### 3.1 E2E测试（最高优先级）

```python
# tests/e2e/test_full_pipeline.py
class TestFullPipeline:
    """端到端流程测试"""
    
    def test_train_detect_evaluate_pipeline(self):
        """完整流程：训练1epoch → 检测 → 评估"""
        # Given: 配置 + 2个CT
        # When: 执行train → detect → evaluate
        # Then: 生成报告，指标合理
    
    def test_reproducibility(self):
        """多seed复现性"""
        # Given: seed=0,1,2
        # When: 训练+检测
        # Then: 结果方差 < 阈值
    
    def test_checkpoint_resume(self):
        """断点续训"""
        # Given: 训练50epoch的checkpoint
        # When: 从checkpoint继续训练
        # Then: loss继续下降

    def test_data_collection_persistence(self):
        """数据收集持久化"""
        # Given: 训练配置开启数据收集
        # When: 执行训练
        # Then: outputs/metrics/training/ 下生成JSON文件
        # And: 可通过MetricsReader读取

    def test_visualization_without_rerun(self):
        """无需重跑的可视化"""
        # Given: 已收集的训练数据
        # When: 调用可视化接口
        # Then: 生成loss曲线、FROC曲线等
```

### 3.2 单元测试补充

```python
# tests/unit/test_trainer_behavior.py
class TestAutoencoderLightningModule:
    """训练器行为测试"""
    
    def test_training_step_returns_loss(self):
        """training_step返回标量loss"""
        # Given: batch (x, target)
        # When: 调用training_step
        # Then: 返回requires_grad=True的标量
    
    def test_validation_step_logs_val_loss(self):
        """validation_step记录val_loss"""
        # Given: batch (x, target)
        # When: 调用validation_step
        # Then: log被调用，on_epoch=True
    
    def test_configure_optimizers_returns_adamw(self):
        """优化器配置正确"""
        # When: 调用configure_optimizers
        # Then: 返回AdamW优化器
    
    def test_epoch_level_logging(self):
        """epoch级loss聚合正确"""
        # Given: 多个batch
        # When: 完成一个epoch
        # Then: train_loss_epoch被正确计算

# tests/unit/test_callbacks.py
class TestReconstructionVisualizationCallback:
    """回调测试"""
    
    def test_saves_png_on_epoch_end(self):
        """epoch结束时保存PNG"""
    
    def test_skips_epoch_0(self):
        """跳过epoch 0"""

# tests/unit/test_reporter.py
class TestEvaluationReporter:
    """报告生成器测试"""
    
    def test_generate_report_json(self):
        """生成JSON报告"""
    
    def test_generate_report_csv(self):
        """生成CSV报告"""
    
    def test_plot_roc_curve(self):
        """绘制ROC曲线"""

# tests/unit/test_collector.py
class TestTrainingCollector:
    """数据收集器测试"""
    
    def test_log_step(self):
        """记录step数据"""
    
    def test_log_epoch(self):
        """记录epoch数据"""
    
    def test_flush_to_disk(self):
        """刷新到磁盘"""
    
    def test_load_by_metrics_reader(self):
        """通过MetricsReader加载"""

class TestDetectionCollector:
    """检测数据收集器测试"""
    
    def test_log_case(self):
        """记录case数据"""
    
    def test_anomaly_map_stats(self):
        """异常热图统计"""
    
    def test_processing_time(self):
        """处理时间记录"""

class TestEvaluationCollector:
    """评估数据收集器测试"""
    
    def test_log_case_metrics(self):
        """记录case指标"""
    
    def test_log_global_metrics(self):
        """记录全局指标"""
    
    def test_log_froc_curve(self):
        """记录FROC曲线"""
    
    def test_log_threshold_sweep(self):
        """记录阈值扫描"""

# tests/unit/test_metrics_reader.py
class TestMetricsReader:
    """指标读取器测试"""
    
    def test_list_runs(self):
        """列出所有运行"""
    
    def test_load_training_run(self):
        """加载训练运行"""
    
    def test_load_detection_run(self):
        """加载检测运行"""
    
    def test_load_evaluation_run(self):
        """加载评估运行"""
    
    def test_missing_file_handling(self):
        """缺失文件处理"""
```

### 3.3 集成测试补充

```python
# tests/integration/test_detection_pipeline.py
class TestDetectionPipeline:
    """检测流程集成测试"""
    
    def test_run_directory(self):
        """目录级推理流程"""
    
    def test_nifti_metadata_preservation(self):
        """NIfTI元数据保留"""
    
    def test_detection_collector_integration(self):
        """检测收集器集成"""
        # Given: SlidingWindowDetector + DetectionCollector
        # When: 执行检测
        # Then: 收集器正确记录数据

# tests/integration/test_luna16_evaluation.py
class TestLuna16Evaluation:
    """LUNA16评估集成测试"""
    
    def test_threshold_sweep(self):
        """阈值扫描流程"""
    
    def test_froc_curve_generation(self):
        """FROC曲线生成"""
    
    def test_evaluation_collector_integration(self):
        """评估收集器集成"""

# tests/integration/test_data_collection.py
class TestDataCollection:
    """数据收集集成测试"""
    
    def test_training_with_collection(self):
        """训练+数据收集"""
    
    def test_detection_with_collection(self):
        """检测+数据收集"""
    
    def test_evaluation_with_collection(self):
        """评估+数据收集"""
    
    def test_metrics_reader_integration(self):
        """MetricsReader集成"""
```

## 4. 测试实现策略

### 4.1 Mock策略

```python
# 网络依赖全部mock
@pytest.fixture
def mock_kaggle_api(monkeypatch):
    """Mock Kaggle API"""
    ...

# 文件I/O使用tmp_path
@pytest.fixture
def sample_ct_files(tmp_path):
    """创建测试用CT文件"""
    ...

# GPU测试自动跳过
@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("GPU not available")

# 数据收集器使用tmp目录
@pytest.fixture
def collector(tmp_path):
    """创建测试用收集器"""
    return TrainingCollector(str(tmp_path), "test_run")
```

### 4.2 参数化测试

```python
@pytest.mark.parametrize("method", ["otsu", "fixed"])
def test_threshold_methods(method):
    """测试不同阈值方法"""

@pytest.mark.parametrize("radius,intensity", [(4, 0.5), (6, 1.0), (8, 0.5)])
def test_synthetic_anomaly(radius, intensity):
    """测试不同合成异常参数"""
```

## 5. 测试执行计划

### Phase 1: 修复已知问题（1-2天）
- 修复losses.py文档
- 修复train_loss logging
- 修复callbacks datamodule依赖
- 编写trainer行为测试
- 编写callbacks测试

### Phase 2: 数据收集系统（2-3天）
- 实现TrainingCollector
- 实现DetectionCollector
- 实现EvaluationCollector
- 实现MetricsReader
- 编写收集器测试

### Phase 3: 完整训练验证（3-5天）
- 启动完整训练（100 epochs）
- 验证数据收集
- 编写E2E测试
- 验证可视化接口

### Phase 4: 检测评估验证（2-3天）
- 执行全量检测
- 执行FROC评估
- 验证检测数据收集
- 验证评估数据收集

## 6. 成功标准

| 阶段 | 指标 | 目标值 |
|------|------|--------|
| 测试覆盖率 | 单元测试 | 150+ |
| 测试通过率 | pytest | 200+ passed |
| E2E覆盖 | 端到端测试 | 3-5 |
| 数据收集 | 持久化完整性 | 100% |
| 可视化 | 无需重跑 | 支持 |
