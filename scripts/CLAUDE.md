# scripts/
> L2 | 父级: /CLAUDE.md

成员清单
download_data.py: 下载 LUNA16 子集和 `annotations.csv`，用于真实数据 smoke test。
train_autoencoder.py: Hydra 训练入口，支持随机初始化和严格预训练两条链路。
detect.py: Hydra 检测入口，安全加载 checkpoint，输出 anomaly 图和可视化图。
run_experiments.py: 实验汇总入口，提供 `health / synthetic / compare / ablation / plan / luna16` 子命令。
slurm/: 集群提交层，存放 Slurm 启动包装脚本。

法则: 成员完整、一行一文件、入口薄壳、职责分层。

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
