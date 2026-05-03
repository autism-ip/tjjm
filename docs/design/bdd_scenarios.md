# BDD - 行为驱动设计场景

## 核心用户故事

作为肺部CT异常检测研究员，我希望建立一个完整的3D肺部CT异常检测系统，以便能够：
1. 在健康CT数据上训练自编码器
2. 利用重建误差检测异常区域
3. 通过LUNA16弱标注评估检测性能
4. 生成可复现的实验报告

---

## 场景1: 数据准备

```gherkin
Feature: LUNA16数据预处理
  为了训练健康自编码器
  作为研究员
  我需要准备健康的CT patch数据

  Scenario: 从原始CT提取健康patch
    Given LUNA16数据集已下载到 /root/autodl-tmp/data/raw/LUNA16
    And   annotations.csv包含结节标注
    When  执行数据预处理
    Then  应该生成健康patch用于训练
    And   应该过滤掉包含结节的patch
    And   应该保持病例级划分（train/val/test不泄漏）
    And   预处理结果应持久化到 data/processed/

  Scenario: 数据质量验证
    Given 预处理完成
    When  检查patch统计
    Then  patch值范围应在[0, 1]之间
    And   patch尺寸应为64x64x64
    And   不应包含结节区域
```

---

## 场景2: 模型训练

```gherkin
Feature: 3D自编码器训练
  为了学习健康CT的正常结构
  作为研究员
  我需要训练一个自编码器

  Scenario: 正常训练流程
    Given 健康patch数据集已准备
    And   配置文件train_autoencoder.yaml已设置
    When  训练3D自编码器（SwinUNETR encoder + U-Net decoder）
    Then  训练loss应该持续下降
    And   验证loss应该同步下降（无过拟合）
    And   重建可视化应该显示清晰的CT结构
    And   模型checkpoint应该保存top-3 + last

  Scenario: 训练数据持久化
    Given 训练正在运行
    When  检查输出目录
    Then  应该生成TensorBoard日志
    And   应该保存每epoch的指标JSON
    And   应该保存重建可视化PNG
    And   应该保存checkpoint文件

  Scenario: 训练中断恢复
    Given 训练在epoch 50中断
    And   存在checkpoint文件
    When  从checkpoint继续训练
    Then  loss应该从上次中断点继续下降
    And   epoch计数应该正确延续
```

---

## 场景3: 异常检测

```gherkin
Feature: 滑动窗口异常检测
  为了检测CT中的异常区域
  作为研究员
  我需要对测试CT执行推理

  Scenario: 单CT检测
    Given 训练好的自编码器checkpoint
    And   测试CT volume
    When  对测试CT执行滑动窗口推理
    Then  应该生成逐体素的异常热图
    And   热图应该保存为NIfTI格式（保留spacing/origin）
    And   应该生成可视化PNG（原图|热图|叠加）

  Scenario: 批量检测
    Given 测试CT目录
    When  执行目录级推理
    Then  每个CT应该生成对应的anomaly map
    And   应该保存检测统计摘要
    And   应该记录每个case的处理时间
```

---

## 场景4: 评估验证

```gherkin
Feature: LUNA16弱标注评估
  为了量化检测性能
  作为研究员
  我需要执行FROC风格评估

  Scenario: 阈值扫描
    Given 异常热图和LUNA16 annotations.csv
    When  执行FROC风格阈值扫描
    Then  应该计算结节召回率
    And   应该计算每病例假阳性数
    And   应该生成FROC曲线
    And   应该选择推荐工作点

  Scenario: 评估报告生成
    Given 评估完成
    When  检查输出目录
    Then  应该生成JSON格式的评估报告
    And   应该生成CSV格式的指标表
    And   应该生成ROC曲线PNG
    And   应该生成异常热图可视化
```

---

## 场景5: 实验复现

```gherkin
Feature: 实验协议生成
  为了确保实验可复现
  作为研究员
  我需要生成结构化的实验报告

  Scenario: 完整实验协议
    Given 实验配置文件
    When  执行实验协议
    Then  应该生成完整的实验报告（JSON/Markdown）
    And   多seed实验应该显示结果稳定性
    And   消融实验应该量化各组件贡献

  Scenario: 合成异常敏感性
    Given 健康CT volume
    When  注入合成球形异常
    Then  应该计算病灶区vs背景区对比度
    And   应该生成敏感性曲线
    And   应该记录不同半径/强度的检测率
```

---

## 验收标准

| 阶段 | 指标 | 目标值 |
|------|------|--------|
| 训练 | val_loss | < 0.01 |
| 检测 | 结节召回率 | > 0.7 |
| 评估 | fp_per_case | < 3 |
| 复现 | seed方差 | < 0.05 |

---

## 完整实验矩阵

### 实验类型总览

| 类型 | 目的 | 实验数量 |
|------|------|----------|
| 复现性实验 | 验证结果稳定性 | 3 seeds × 3 repeats = 9 runs |
| 对比实验 | 与基线方法对比 | 6 种方法 |
| 消融实验 | 量化各组件贡献 | 7 维度 × 2-3 选项 |
| 敏感性实验 | 合成异常/阈值/后处理 | 3 类实验 |
| 鲁棒性实验 | 不同数据条件 | 4 维度 × 3-4 选项 |
| 效率实验 | 计算资源分析 | 1 综合测试 |

### 一、对比实验（Baselines）

| 方法 | 描述 | 目的 |
|------|------|------|
| Random | 随机异常图 | 下限参考 |
| Intensity Threshold | 固定 HU 阈值 | 传统方法基线 |
| AE (无预训练) | 随机初始化编码器 | 预训练收益对比 |
| AE + Pretrained | SwinUNETR SSL 预训练 | 当前主方法 |
| AE + Sliding Window | 主方法 + 滑窗推理 | 滑窗收益验证 |
| AE + Postprocess | 主方法 + 连通域后处理 | 后处理收益验证 |

### 二、消融实验（Ablation）

| 维度 | 选项 | 默认值 |
|------|------|--------|
| 编码器架构 | swin_unetr, resnet, vit | swin_unetr |
| 损失函数 | weighted_mse, mse, ssim | weighted_mse |
| Patch尺寸 | 32, 64, 128 | 64 |
| 滑窗重叠率 | 25%, 50%, 75% | 50% |
| 预训练策略 | random, imagenet, swin_ssl | swin_ssl |
| 冻结策略 | true, false | true |
| 解码器深度 | 2, 3, 4 | 4 |

### 三、敏感性实验（Sensitivity）

| 实验 | 参数 | 范围 |
|------|------|------|
| 合成异常半径 | radius | 4, 6, 8, 12, 16 voxels |
| 合成异常强度 | intensity | 0.3, 0.5, 1.0, 2.0 |
| 合成异常形状 | shape | sphere, cube |
| 阈值百分位 | percentile | 99.0, 99.5, 99.9 |
| 最小连通域 | min_voxels | 0, 10, 50, 100 |

### 四、复现性实验（Reproducibility）

| 配置 | 值 |
|------|-----|
| 种子数 | 3 (seed=0, 1, 2) |
| 每seed重复 | 3次 |
| 总运行数 | 9 |
| 验收标准 | 结果方差 < 0.05 |

### 五、鲁棒性实验（Robustness）

| 实验 | 参数 | 范围 |
|------|------|------|
| 数据量敏感 | train_ratio | 25%, 50%, 75%, 100% |
| 噪声鲁棒 | noise_sigma | 0.01, 0.05, 0.1 |
| 分辨率敏感 | spacing | 0.5, 1.0, 2.0 mm |
| HU窗敏感 | hu_range | [-1000,400], [-1000,200], [-500,400] |

### 六、效率实验（Efficiency）

| 指标 | 测量内容 |
|------|----------|
| 训练时间 | 每epoch耗时 |
| 推理时间 | 每case耗时 |
| GPU显存 | 峰值显存占用 |
| 模型参数 | 参数量统计 |
| Checkpoint大小 | 模型文件大小 |

---

## 实验执行顺序

```
1. Smoke Test (1 epoch)
   ↓
2. 复现性实验 (3 seeds × 3 repeats)
   ↓
3. 对比实验 (6 种基线方法)
   ↓
4. 消融实验 (7 维度消融)
   ↓
5. 敏感性实验 (合成异常/阈值/后处理)
   ↓
6. 鲁棒性实验 (数据量/噪声/分辨率/HU窗)
   ↓
7. 效率分析 (时间/显存/参数)
   ↓
8. 最终报告生成
```

---

## 数据收集

所有实验数据自动持久化到 `outputs/metrics/`，支持后续可视化而无需重跑实验：

```
outputs/metrics/
├── training/          # 训练数据
├── detection/         # 检测数据
├── evaluation/        # 评估数据
└── experiments/       # 实验数据
    ├── comparison/    # 对比实验
    ├── ablation/      # 消融实验
    ├── sensitivity/   # 敏感性实验
    ├── robustness/    # 鲁棒性实验
    └── efficiency/    # 效率实验
```
