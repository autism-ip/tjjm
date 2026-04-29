# Lung-Diffusion-Anomaly

基于 3D 自编码器的肺部异常检测系统。核心目标很直接：用健康 CT 学到正常结构，再用重建误差定位异常区域。

这份 README 只做一件事：把仓库架构、每条流程、当前已验证状态讲清楚，方便你自己复现，而不是每次都靠我口头带跑。

## 项目地图

```text
.
├── CLAUDE.md              # 项目宪法，记录全局架构、约定和变更协议
├── configs/               # Hydra 配置，训练 / 检测 / 实验参数都在这里
├── scripts/               # 主流程入口：下载、训练、检测、实验
├── src/                   # 业务源码
├── tests/                 # 单测与集成测试，保证入口和核心模块可复现
├── data/                  # 本地数据与验证产物，默认不进版本控制
├── outputs/               # 运行产物输出目录
├── docs/                  # 补充文档
└── requirements.txt       # 运行依赖
```

### `src/` 结构

```text
src/
├── data/        # LUNA16 下载、CT 预处理、Dataset、健康 patch 选择
├── models/      # SwinUNETR 自编码器、权重迁移、冻结/解冻工具
├── training/    # LightningModule、损失函数、训练回调
├── detection/   # 滑窗重建、异常图、目录级推理
├── evaluation/  # Dice / AUC / Precision / Recall / F1，以及 LUNA16 弱标注病例/结节级指标
├── experiments/ # 健康统计、合成敏感度、ablation 汇总
└── utils/       # 配置、日志、可视化
```

### `scripts/` 入口

```text
scripts/
├── download_data.py     # 下载少量真实 LUNA16 CT 配对
├── train_autoencoder.py  # 训练入口
├── detect.py             # 检测 / 推理入口
└── run_experiments.py    # 实验汇总入口
```

## 当前状态

这个仓库现在不是“只做过单测”的状态，而是已经做过真实数据端到端验证。

- 数据：只下载了 2 份真实 LUNA16 CT 配对，没有拉整包。
- 训练：GPU 已可用，真实 1 个 epoch 已跑通。
- 预训练：本地 MONAI 官方 SSL 权重已验证可读，严格模式预训练 1 个 epoch 已跑通。
- 检测：同一批真实 CT 已跑通目录级推理并输出热图。
- 实验：健康统计与 LUNA16 弱标注评估脚本都已接入检测输出。
- 依赖：当前环境使用 CUDA 版 PyTorch，GPU 可见。

最近一次验证结果：

- `162 passed, 5 warnings`
- 真实训练 checkpoint 已生成
- 严格模式预训练 checkpoint 已生成
- 真实检测输出已落盘
- 实验摘要已生成
- LUNA16 弱标注病例/结节级指标已可导出

## 运行环境

建议环境：

- Python 3.10+
- CUDA 版 PyTorch
- NVIDIA GPU 优先

当前仓库已经验证的实际环境：

- `torch 2.5.1+cu121`
- `torch.cuda.is_available() == True`
- `NVIDIA GeForce RTX 3070 Ti Laptop GPU`

如果你本机还是 CPU 版 PyTorch，训练和检测会退回 CPU，速度会明显慢很多。

## 配置文件

### `configs/train_autoencoder.yaml`

训练配置，控制：

- 数据目录
- patch 大小
- batch size
- 预训练开关
- 预训练权重路径和严格模式
- 最大 epoch
- 优化器和调度器

### `configs/detect.yaml`

检测配置，控制：

- 测试 CT 目录
- checkpoint 路径
- 滑窗 patch / stride
- 输出目录
- 指标列表

### `configs/experiments.yaml`

实验配置，控制：

- 健康统计分位数
- 合成敏感度注入参数
- 输出目录

## 流程 1: 下载数据

现在的下载器不是“整包下载 LUNA16”，而是“按文件级别下载少量真实 CT 配对”。

### 推荐命令

```bash
python scripts/download_data.py --subset 2
```

会下载：

- `annotations.csv`
- 前 2 份真实 `.mhd/.raw` CT 配对

默认输出目录：

- `data/raw/LUNA16/`

### 说明

- `--subset` 表示下载多少份真实 CT 配对。
- 默认行为是“小样本验证优先”，不是盲目拉整包。
- `LIDC-IDRI` 入口仍保留，但当前仓库里没有实际跑通它。

## 流程 2: 训练

训练入口是 `scripts/train_autoencoder.py`，实际吃的是 Hydra 配置。

训练前先决定预训练策略：

- `model.encoder_pretrained=false`：随机初始化，适合 smoke test 和无预训练基线。
- `model.encoder_pretrained=true model.pretrained_strict=false`：尝试自动下载 MONAI 官方 SSL 权重，失败时退回随机初始化。
- `model.encoder_pretrained=true model.pretrained_strict=true model.pretrained_checkpoint_path=<local_path>`：论文复现推荐模式；本地权重不存在、下载失败或坏文件都会直接报错。

### GPU 优先训练

```bash
python scripts/train_autoencoder.py \
  data.dataset_dir=data/raw/LUNA16 \
  data.luna16_raw_dir=data/raw/LUNA16 \
  data.batch_size=2 \
  data.num_workers=2 \
  model.encoder_pretrained=false \
  model.use_checkpoint=true \
  training.max_epochs=1 \
  training.precision=16-mixed \
  training.accelerator=gpu \
  training.devices=1 \
  training.checkpoint_dir=./data/tmp-checkpoints/gpu-smoke
```

### 预训练训练

```bash
python scripts/train_autoencoder.py \
  data.dataset_dir=data/raw/LUNA16 \
  data.luna16_raw_dir=data/raw/LUNA16 \
  data.batch_size=2 \
  data.num_workers=0 \
  model.encoder_pretrained=true \
  model.pretrained_checkpoint_path=./.cache/monai/pretrained/ssl_pretrained_weights.pth \
  model.pretrained_strict=true \
  model.use_checkpoint=true \
  training.max_epochs=1 \
  training.precision=16-mixed \
  training.accelerator=gpu \
  training.devices=1 \
  training.checkpoint_dir=./data/tmp-checkpoints/gpu-pretrained
```

### 输入

- `data.dataset_dir`：CT 目录，目录下应有 `.mhd/.raw`
- `data.luna16_raw_dir`：`annotations.csv` 所在目录

### 输出

- `training.checkpoint_dir` 下保存 checkpoint
- 当前验证路径：
  - `data/tmp-checkpoints/gpu-smoke/final.ckpt`

### 已验证结论

- 真实 CT 数据可训练
- GPU 可用
- 1 个 epoch 可以正常结束
- 本地官方预训练权重可在严格模式下完整加载
- 严格模式预训练 1 个 epoch 已跑通
- 严格模式预训练检测与健康统计已跑通
- 预训练失败根因已经定位：
  - 主要问题不是官方权重格式天然不兼容
  - 主要问题是网络慢或中断时会留下半截 `.pth`
  - 半截文件会触发 `torch.load(...): failed finding central directory`
  - 当前代码已改成先下载到 `.part`，成功后再原子替换正式文件
  - 如果读到坏缓存，会自动删除并重试一次

### 预训练排障

- 先跑通链路：用 `model.encoder_pretrained=false`
- 做论文级预训练实验：必须用 `model.pretrained_strict=true`
- 自动下载不稳定时：手工把 `ssl_pretrained_weights.pth` 放到 `./.cache/monai/pretrained/`
- 或者显式指定：`model.pretrained_checkpoint_path=/abs/path/to/ssl_pretrained_weights.pth`
- 如果报 `failed finding central directory`：删除坏的 `.pth` 后重下

## 流程 3: 检测 / 推理

检测入口是 `scripts/detect.py`，支持目录级 CT 推理。

### 推荐命令

```bash
python scripts/detect.py \
  data.test_ct_dir=data/raw/LUNA16 \
  data.output_dir=./data/tmp-detect \
  model.checkpoint_path=./data/tmp-checkpoints/gpu-smoke/final.ckpt
```

### 输入

- `data.test_ct_dir`：真实 CT 目录
- `model.checkpoint_path`：训练得到的 checkpoint

### 输出

会在输出目录里生成：

- `*_anomaly.nii.gz`
- `*_viz.png`
- 如果有标注，还会生成评估结果和报告

### 已修复的关键兼容点

- Lightning checkpoint 的 `model.` 前缀会自动去掉
- Hydra 的 `stride` 列表会自动收敛成可用的滑窗步长
- 目录推理的可视化切片索引已对齐真实 CT 深度轴
- `detect.py` 会优先使用 `torch.load(..., weights_only=True)` 安全加载 checkpoint，旧格式再兼容回退
- 检测输出的 `anomaly.nii.gz` 现在会继承原 CT 的 `origin/spacing`，避免后续评估坐标漂移

## 流程 4: 实验

实验入口是 `scripts/run_experiments.py`，用于把检测输出做健康统计、合成敏感度和 ablation 汇总。

### 健康统计

```bash
python scripts/run_experiments.py health \
  --input-dir ./data/tmp-detect \
  --output ./data/tmp-experiments/health_summary.json
```

### LUNA16 弱标注评估

```bash
python scripts/run_experiments.py luna16 \
  --input-dir ./data/tmp-detect \
  --ct-dir ./data/raw/LUNA16 \
  --annotations ./data/raw/LUNA16/annotations.csv \
  --output ./data/tmp-experiments/luna16_weak_eval.json
```

如果要导出 FROC 风格工作点并启用连通域后处理，可以直接加：

```bash
python scripts/run_experiments.py luna16 \
  --input-dir ./data/tmp-detect \
  --ct-dir ./data/raw/LUNA16 \
  --annotations ./data/raw/LUNA16/annotations.csv \
  --score-percentiles 99.0 99.5 99.9 \
  --component-min-size-voxels 16 \
  --output ./data/tmp-experiments/luna16_weak_eval_post16.json
```

这个命令不假装有体素级真值，而是基于 LUNA16 的 `seriesuid + (coordX, coordY, coordZ, diameter_mm)` 输出更诚实的论文指标，并自动导出默认阈值扫描：

- 病例级 `case_auc`
- 病例级 `case_ap`
- 结节级 `lesion_recall`
- 正例峰值命中率 `peak_localization_rate`
- 每病例 / 每阴性病例假阳性连通域数
- `sweep[]` 中的 `threshold_percentile -> lesion_recall / fp_per_case`
- `recommended` 中的自动推荐工作点，优先保留更高结节召回，其次更低假阳性

如果要显式控制阈值扫描：

```bash
python scripts/run_experiments.py luna16 \
  --input-dir ./data/tmp-detect \
  --ct-dir ./data/raw/LUNA16 \
  --annotations ./data/raw/LUNA16/annotations.csv \
  --score-percentiles 99.0 99.5 99.9 \
  --output ./data/tmp-experiments/luna16_weak_eval.json
```

### 合成敏感度

```bash
python scripts/run_experiments.py synthetic \
  --volume <volume_file> \
  --output ./data/tmp-experiments/synthetic_summary.json
```

### ablation 汇总

```bash
python scripts/run_experiments.py ablation \
  --reports <report1.json> <report2.json> \
  --output ./data/tmp-experiments/ablation_summary.json
```

### 基线对比

```bash
python scripts/run_experiments.py compare \
  --baseline <baseline_report.json> \
  --variant <variant_report.json> \
  --output ./data/tmp-experiments/compare_summary.json
```

这个命令适合做主方法和单个基线的两两对比，输出会包含：

- baseline
- variant
- delta
- delta_pct

### 消融汇总

`ablation` 更适合把多份报告合并成表格，比如：

- `AE`
- `AE + Sliding Window`
- `AE + Pretrained Encoder`
- `w/o smoothing`
- `w/o checkpoint`

先用 `compare` 做两两对比，再用 `ablation` 汇总多组结果，最不容易乱。

### 实验协议导出

```bash
python scripts/run_experiments.py plan \
  --format md \
  --output ./outputs/experiments/experiment_plan.md
```

这个命令会把完整实验协议导出成可读的 Markdown，里面包含：

- 研究目标
- 数据划分规则
- 对照组
- 指标
- 按顺序执行的步骤
- 重复性 seed 扫描
- 基线对比与消融汇总
- 每一步的命令和预期产物

### 已验证结论

- `health` 命令已经在真实检测输出上跑通
- `luna16` 弱标注评估命令已接入真实 LUNA16 标注格式
- `luna16` 会自动从阈值扫描里选一个推荐工作点，适合直接进实验表
- `luna16` 现在额外导出 `froc_curve`，并支持 `--component-min-size-voxels` / `--keep-largest-component`
- 已生成：
  - `data/tmp-experiments/health_summary.json`
  - `data/tmp-experiments/health_summary_pretrained_strict.json`

## 测试

```bash
pytest -q
```

当前仓库验证结果：

- `162 passed, 5 warnings`

warnings 主要来自第三方库的未来兼容提示，不是当前项目的功能错误。

## 教程

这一节不是背景介绍，是给别人直接照着复现用的。

### 教程 1：先看懂目录

- `configs/`：训练、检测、实验默认参数，改流程先改这里。
- `scripts/`：四个命令行入口，只负责组装流程。
- `src/data/`：下载、预处理、Dataset、patch 采样。
- `src/models/`：自编码器主体、预训练权重加载、扩散骨架。
- `src/training/`：LightningModule、loss、callback。
- `src/detection/`：滑窗推理、异常图、融合。
- `src/evaluation/`：指标和报告。
- `src/experiments/`：health、synthetic、compare、ablation、plan。
- `tests/`：权重缓存、入口脚本、实验协议和核心逻辑的回归保护。

### 教程 2：最短复现路径

1. 下载少量真实 CT：`python scripts/download_data.py --subset 2`
2. 跑 1 个 epoch：Windows 下建议 `data.num_workers=0`
3. 跑目录级检测：`python scripts/detect.py ...`
4. 汇总健康统计：`python scripts/run_experiments.py health ...`
5. 导出实验协议：`python scripts/run_experiments.py plan --output outputs/experiments/experiment_plan.md`

### 教程 3：论文复现路径

1. 固定 `training.seed`
2. 明确 train/val/test 病例级划分
3. 每个设置至少重复 3 次
4. 预训练实验启用 `model.pretrained_strict=true`
5. 用 `compare` 做成对对比，用 `ablation` 汇总多组结果
6. 保留命令、checkpoint、检测输出和 JSON 摘要

## 实验方案

这个项目如果作为完整科研工作，不能只停留在“跑通流程”，还要有一套能回答研究问题的实验设计。下面是当前推荐的完整方案。

### 研究目标

核心问题只有一个：

- 仅用正常 CT 训练出的重建模型，是否能稳定区分异常区域

对应的实验目标分成四层：

- 验证方法是否可用
- 验证方法是否优于简单基线
- 验证方法对超参数和数据扰动是否稳健
- 验证方法在哪些异常类型上更有效，在哪些情况下失效

### 数据划分

如果使用完整 LUNA16 或扩展 CT 数据集，建议按病例级划分，避免切片泄漏。

- 训练集：仅正常 CT，学习“正常结构”
- 验证集：正常 CT + 少量异常 CT，用于调参
- 测试集：完全独立病例，不参与任何调参

原则：

- 病例不能跨训练、验证、测试重复
- 3D CT 不能按切片随机打散后再划分
- 若有多中心数据，最好按中心做外部测试

### 对照组

至少保留这些基线，才能证明方法不是“看起来能用”：

- `Random`：随机异常图，作为下限参考
- `Intensity Threshold`：固定 HU 阈值检测
- `AE`：普通自编码器重建误差
- `AE + Sliding Window`：当前主方法的最小实现
- `AE + Pretrained Encoder`：验证预训练编码器是否真的带来收益
- `Ablation without smoothing`：去掉后处理平滑，检查方法稳定性

如果后续引入分割监督或弱监督模块，还应补充：

- `Supervised U-Net`
- `Weakly supervised detector`

### 评估指标

建议分成“定位能力”和“检测能力”两层。

#### 检测层

- ROC-AUC
- PR-AUC
- Accuracy
- Precision
- Recall
- F1

#### 定位层

- Dice
- IoU
- lesion-wise sensitivity
- volume-wise false positive rate

#### 运行层

- 单病例推理时间
- 单 epoch 训练时间
- 显存占用

### 主实验

主实验应该回答“方法是否有效”。

建议流程：

1. 只用正常 CT 训练自编码器
2. 在独立测试集上生成异常图
3. 对异常图做病例级和病灶级评估
4. 和基线方法做横向对比

主表建议报告：

- 方法名称
- 是否使用预训练编码器
- 是否使用滑窗重建
- 是否使用后处理
- AUC / Dice / F1 / 推理时间

### 消融实验

消融实验用来回答“到底是哪一部分起作用”。

建议至少做这些：

- 去掉预训练编码器
- 去掉 checkpoint
- 改变 patch size
- 改变 stride
- 改变重建误差聚合方式
- 改变后处理平滑强度

如果只做一个消融表，优先保留：

- `w/o pretrained`
- `w/o sliding window`
- `w/o smoothing`
- `w/ smaller patch`

### 鲁棒性实验

科研上不能只看平均结果，还要看稳定性。

建议加入：

- 噪声扰动
- 重采样间隔变化
- 不同 CT window
- 不同扫描中心
- 不同异常体积大小

如果方法对这些变化特别敏感，就说明它更像工程技巧，不像稳健方法。

### 统计规范

为了避免一次结果偶然好看，建议：

- 固定随机种子
- 每个设置至少重复 3 次
- 报告均值和标准差
- 记录最优 checkpoint 的选择规则
- 当前代码已支持通过 `training.seed=0/1/2` 做重复性扫描

如果样本量足够，还可以做：

- bootstrap 置信区间
- 配对显著性检验

### 推荐实验顺序

如果现在要按最小代价逐步补全，顺序应该是：

1. 真实数据上的单次端到端验证
2. 小规模基线对照
3. 主实验表
4. 消融实验
5. 鲁棒性实验
6. 外部测试集验证

### 当前仓库的实验边界

现在这套仓库已经证明了：

- 下载、训练、检测、实验流程都能跑通
- GPU 端到端链路是正常的

但它还没有替你完成“完整论文级实验”的全部内容，因为：

- 数据规模还很小
- 真实科研需要更大的训练/测试划分
- 还需要正式的基线对照表和统计表

换句话说：

- 代码链路已经通了
- 科研方案还需要按上面这套表继续补齐

## 最近更新

最近几轮不是改一小块文案，而是把真实运行链路收口了。

- `3c8ac3f refactor: harden pretrained experiment pipeline`
  - 预训练权重严格模式、本地权重路径、坏缓存重试已收口
  - `detect.py` 已优先安全加载 checkpoint
- 当前未提交改动
  - 新增 `scripts/run_experiments.py luna16` 子命令
  - 新增 LUNA16 弱标注病例/结节级评估与默认阈值扫描
  - 检测输出现在保留原 CT 空间元数据
  - 全量测试基线更新到 `162 passed, 5 warnings`
  - 真实 2 例 CT 的 `luna16` 弱标注评估已跑通，当前现象是 `case_auc=1.0` 但 `lesion_recall=0.0`
- `d8703f2 refactor: validate gpu real-data end-to-end`
  - 下载 2 份真实 CT 配对
  - CUDA 版 PyTorch 可用
  - 训练 1 epoch 跑通
  - 检测跑通
  - 实验跑通
- `6b6daac refactor: split metrics test concerns`
  - 指标测试拆分成更窄的职责
- `ca48a73 refactor: split data preprocessing responsibilities`
  - `src/data` 预处理职责拆分
- `e27d4d6 refactor: stabilize package and test runtime`
  - 包结构和测试运行环境收口

## 如何维护这份 README

这部分是给未来的维护者看，也就是给你自己看。

- 新增或删除入口脚本时，更新 `scripts/` 这一节。
- 新增或拆分 `src/` 模块时，更新项目地图。
- 训练、检测、实验命令变了，就同步更新对应流程命令。
- 真实验证结果变了，就更新“当前状态”和“最近更新”。

原则很简单：

- 代码变了，README 也要变。
- 流程变了，命令也要变。
- 验证结果变了，状态也要变。

## 引用
