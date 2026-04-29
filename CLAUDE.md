# Lung-Diffusion-Anomaly - 3D 肺部 CT 无监督异常检测科研工程
PyTorch + MONAI + PyTorch Lightning + Hydra/OmegaConf

<directory>
src/ - 核心源码层，按 data/models/training/detection/evaluation/experiments/utils 分层
configs/ - Hydra 配置层，统一管理训练、检测、实验参数
scripts/ - CLI 入口层，负责下载、训练、检测、实验编排
tests/ - 回归验证层，覆盖单测与集成测试
data/ - 本地真实数据、临时 checkpoint、临时检测输出
outputs/ - 实验计划与导出报告
</directory>

<config>
requirements.txt - 依赖清单
pytest.ini - pytest 配置
setup.py - `src.*` 包安装入口
</config>

<status>
- 真实 LUNA16 2 例 CT 已完成下载、GPU 训练、真实检测、弱标注评估闭环
- 严格预训练权重链路已跑通，`159` 层编码器权重成功加载
- `detect.py` 已使用安全 checkpoint 加载，并保留 `anomaly.nii.gz` 的空间元数据
- `luna16` 评估支持 `sweep`、`froc_curve`、`recommended`，并支持连通域后处理
- 当前真实结果仍显示 `case_auc=1.0` 但 `lesion_recall=0.0`，说明工程闭环成立，方法定位能力仍待优化
- 当前测试基线：`162 passed, 5 warnings`
</status>

<rules>
- 代码变更后同步更新对应 README/CLAUDE 文档，保持代码与文档同构
- 真实实验优先走 `scripts/` 入口，不绕过 CLI 直接拼临时脚本
- 论文对比优先使用 `outputs` 与 `data/tmp-experiments` 中的可复现实验产物
- 预训练实验必须显式区分随机初始化、宽松加载、严格加载三种模式
</rules>
