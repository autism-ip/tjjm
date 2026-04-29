# Lung-Diffusion-Anomaly - 3D 肺部异常检测系统
PyTorch + MONAI + PyTorch Lightning + Hydra/OmegaConf

<directory>
src/ - 源码根，承载 `src.*` 包导入体系与 data/models/training/detection/evaluation/utils/experiments 七层业务模块
src/data/ - 数据层，负责 LUNA16 下载、CT 预处理、Dataset、健康 patch 采样
src/models/ - 模型层，负责 SwinUNETR 自编码器、预训练权重加载与扩散骨架
src/training/ - 训练层，负责 LightningModule、损失函数、训练回调
src/detection/ - 检测层，负责滑窗重建、异常热图、目录级推理与融合
src/evaluation/ - 评估层，负责 Dice/AUC 等指标与报告生成
src/utils/ - 工具层，负责配置加载、日志、可视化输出
src/experiments/ - 实验层，负责健康统计、合成异常、实验协议生成与 ablation 汇总
configs/ - Hydra 配置文件，包含训练、检测、实验和协议默认值
scripts/ - 可执行入口脚本，包含下载、训练、检测、实验协议导出
tests/ - 单测与集成测试，保证入口、核心模块与实验协议可复现
docs/ - 补充文档、论文草稿、培训材料
.cache/ - 项目内缓存目录，承接 matplotlib、fontconfig、MONAI pretrained 等运行态缓存
</directory>

<config>
setup.py - 对齐 `src.*` 包结构的打包配置
requirements.txt - 运行依赖
pytest.ini - pytest 配置，关闭会污染沙箱的默认缓存行为
.gitignore - 忽略数据、输出与本地缓存产物
</config>

<status>
当前仓库已经验证过真实数据端到端流程：
- 只下载了少量真实 LUNA16 CT 配对
- CUDA 版 PyTorch 可用，GPU 可见
- 真实 1 epoch 训练已跑通
- 严格模式预训练 1 epoch 训练已跑通，且本地官方权重可完整加载 159 层
- 真实目录级检测已跑通
- 严格模式预训练目录级检测与 health 摘要已跑通
- 健康统计与实验协议导出已跑通
- `pytest -q` 当前基线为 `157 passed, 5 warnings`
- 预训练权重问题已定位：当前环境到 GitHub Release 下载慢或中断时，可能留下半截 `.pth` 缓存
- 当前权重加载器已支持显式结果返回、严格模式、本地权重路径、`.part` 临时文件下载与坏缓存自动重试
- `luna16` 弱标注评估、阈值扫描与推荐工作点选择已接入真实检测输出，当前真实 2 例 CT 可完整跑通
</status>

<rules>
- 代码变了，文档必须同步。
- 新增目录或模块，必须更新对应 CLAUDE.md。
- 实验方案要能被代码导出，不能只写在 README 里。
- 任何真实验证都要留下命令、产物与成功标准。
- 预训练实验若要写入论文，必须使用显式本地权重或 `model.pretrained_strict=true`，禁止把随机初始化伪装成预训练结论。
</rules>
