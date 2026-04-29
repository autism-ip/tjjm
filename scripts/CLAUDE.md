# scripts/
> L2 | 父级: /CLAUDE.md

成员清单
download_data.py: 下载少量 LUNA16 CT 与 `annotations.csv`，用于真实数据 smoke test。
train_autoencoder.py: Hydra 训练入口，支持随机初始化与严格预训练两条链路。
detect.py: Hydra 检测入口，安全加载 checkpoint，输出 anomaly 图与可视化。
run_experiments.py: 实验入口，提供 `health/luna16/synthetic/compare/ablation/plan` 子命令，并把 LUNA16 后处理参数透传到评估层。

法则: 所有真实流程都应能从这里复现；CLI 参数名必须和配置、README 保持一致。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
