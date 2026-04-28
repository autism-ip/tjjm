# Lung-Diffusion-Anomaly - 基于3D自编码重建的无监督肺部异常检测
PyTorch + MONAI + PyTorch Lightning + Hydra/OmegaConf

<directory>
src/ - 源码根，承载 src.* 包导入体系与 data/models/training/detection/evaluation/utils/experiments 七层业务模块
src/data/ - 数据层：下载、预处理、Dataset、健康patch筛选 (4业务文件 + CLAUDE.md)
src/models/ - 模型层：SwinUNETR自编码器、权重迁移、DDPM/DDIM工具 (4业务文件 + CLAUDE.md)
src/training/ - 训练层：Lightning Module、损失函数、回调 (4业务文件 + CLAUDE.md)
src/detection/ - 检测层：目录推理、滑动窗口重建、异常热图、融合 (5业务文件 + CLAUDE.md)
src/evaluation/ - 评估层：Dice/AUC等指标、报告生成 (3业务文件 + CLAUDE.md)
src/utils/ - 工具层：配置、日志、可视化 (4业务文件 + CLAUDE.md)
src/experiments/ - 实验层：健康样本统计、合成异常敏感性、ablation 汇总 (4业务文件 + CLAUDE.md)
configs/ - Hydra 配置文件 (train_autoencoder.yaml, detect.yaml, experiments.yaml, CLAUDE.md)
scripts/ - 可执行入口脚本 (train_autoencoder.py, detect.py, download_data.py, run_experiments.py, CLAUDE.md)
tests/ - 测试套件 (unit/, integration/ + CLAUDE.md)
docs/ - 外部资料与论文/培训文档
.cache/ - 本地运行期缓存目录，承载 matplotlib/fontconfig 临时状态，不纳入版本契约
</directory>

<config>
setup.py - src.* 包定义与运行时依赖声明
requirements.txt - 开发/测试环境依赖约束
pytest.ini - pytest 运行配置，关闭内建 tmpdir 插件并改走 tests/conftest.py 的自定义 tmp_path
.gitignore - Python 项目标准忽略规则
</config>

法则: 极简·稳定·导航·版本精确
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
