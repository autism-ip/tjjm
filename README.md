# Lung-Diffusion-Anomaly

基于 3D 自编码重建的无监督肺部异常检测系统。

核心思路：用仅学习过健康肺组织分布的自编码器重建测试 CT，重建误差大的区域即被标记为异常。无需病灶标注即可训练，适用于早期筛查与罕见病变发现。

---

## 技术栈

PyTorch + MONAI + PyTorch Lightning + Hydra/OmegaConf

## 核心特性

- **无监督训练** — 仅需健康 CT 数据即可训练，摆脱对精细病灶标注的依赖
- **SwinUNETR 编码器** — 利用 MONAI 预训练权重，医学图像特征提取能力更强
- **3D 滑动窗口推理** — 对完整 CT 体积进行 patch 级重建与异常热图生成
- **Hydra 配置驱动** — 实验参数 YAML 化，训练/推理/评估全链路可复现
- **Lightning 训练框架** — 自动混合精度、梯度裁剪、Checkpoint、EarlyStopping 开箱即用

## 项目架构

```
src/
├── data/        # 数据层 — 下载、预处理、Dataset、健康 patch 筛选
├── models/      # 模型层 — SwinUNETR 自编码器、权重迁移、DDPM/DDIM 工具
├── training/    # 训练层 — Lightning Module、损失函数、回调
├── detection/   # 检测层 — 目录推理、滑动窗口重建、异常热图、融合
├── evaluation/  # 评估层 — Dice/AUC 等指标、报告生成
└── utils/       # 工具层 — 配置、日志、可视化

scripts/
├── train_autoencoder.py   # 自编码器训练入口
├── detect.py              # 异常检测推理入口
├── download_data.py       # 数据集下载入口
└── run_experiments.py     # 实验调度入口

configs/
├── train_autoencoder.yaml # 训练配置
├── detect.yaml            # 推理配置
└── experiments.yaml       # 实验配置
```

## 快速开始

### 1. 环境安装

```bash
# Python >= 3.10
pip install -r requirements.txt
pip install -e .
```

### 2. 数据准备

```bash
# 下载 LUNA16 数据集到 ./data/raw/LUNA16
python scripts/download_data.py
```

### 3. 训练自编码器

```bash
python scripts/train_autoencoder.py
```

训练配置位于 `configs/train_autoencoder.yaml`，默认参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| patch_size | [64, 64, 64] | 3D patch 尺寸 |
| batch_size | 4 | 训练批次 |
| max_epochs | 100 | 最大训练轮数 |
| lr | 1e-4 | AdamW 学习率 |
| precision | 16-mixed | 自动混合精度 |

Checkpoint 自动保存至 `./checkpoints/autoencoder/`。

### 4. 异常检测推理

```bash
python scripts/detect.py
```

推理配置位于 `configs/detect.yaml`，需指定：

```yaml
model:
  checkpoint_path: "./checkpoints/autoencoder/best.ckpt"

data:
  test_ct_dir: "./data/test"
  output_dir: "./outputs/detection"
```

输出包括异常热图 NIfTI、可视化切片、评估指标报告。

## 实验运行

```bash
# 健康数据统计
python scripts/run_experiments.py health

# 合成敏感性实验
python scripts/run_experiments.py synthetic

# 消融对比实验
python scripts/run_experiments.py ablation
```

## 测试

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 引用

```bibtex
@software{lung_diffusion_anomaly,
  title = {Lung-Diffusion-Anomaly: Unsupervised Lung Anomaly Detection via 3D Autoencoder Reconstruction},
  year = {2026},
}
```
