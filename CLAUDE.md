# Lung-Diffusion-Anomaly - 3D lung CT anomaly detection and experiment system
PyTorch + MONAI + PyTorch Lightning + Hydra/OmegaConf + Slurm

<directory>
src/ - 核心业务层，包含数据、模型、训练、检测、评估、实验与工具。
configs/ - Hydra 配置层，统一管理训练、检测、实验参数。
scripts/ - CLI 启动层，承载本地命令入口与集群提交入口。
scripts/slurm/ - Slurm 启动层，只负责资源绑定、环境激活和命令转发。
tests/ - 单测与集成测试，覆盖主流程、评估与回归。
data/ - 原始数据、临时 checkpoint、检测产物与中间实验结果。
outputs/ - 对外输出的实验协议、摘要和可复现文档。
docs/ - 外部资料、培训材料与集群说明。
</directory>

<config>
requirements.txt - 运行依赖清单。
pytest.ini - 测试配置。
setup.py - 包装 `src.*` 导入路径。
README.md - 面向使用者的总说明、教程和复现入口。
</config>

<status>
- 真实 LUNA16 CT 下载、GPU 训练、严格预训练加载、真实检测、弱标注评估、FROC 风格阈值扫描都已经跑通。
- 当前回归基线是 `162 passed, 5 warnings`。
- 真实数据的主要瓶颈仍是结节级定位能力，而不是工程链路。
- Slurm 启动层已经补上，后续可以直接迁移到远端集群做训练和实验。
- 完整 LUNA16 数据集（888 CT，111GB）已下载到 `/root/autodl-tmp/data/raw/LUNA16`。
- 多维度数据收集系统已设计，支持训练/检测/评估/实验全流程数据持久化。
</status>

<rules>
- 任何 `scripts/` 或 `configs/` 变更，都要同步检查 README 和各级 CLAUDE。
- 新增的启动入口必须保持"薄壳"原则：Slurm 只管资源，Python CLI 承担业务。
- 真实产物统一放在 `data/tmp-*` 或 `outputs/experiments`，不要再分裂出第二套事实来源。
- 如果目录结构变化了，先更新文档，再算任务完成。
- 实验数据必须持久化到 `outputs/metrics/`，支持后续可视化而无需重跑实验。
</rules>

<design>
BDD/SDD/TDD 设计文档位于 `docs/design/`：
- `bdd_scenarios.md` - 行为驱动场景定义
- `sdd_architecture.md` - 系统架构设计
- `tdd_strategy.md` - 测试驱动策略
- `data_collection.md` - 多维度数据收集方案
</design>
