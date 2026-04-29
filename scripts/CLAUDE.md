# scripts/
> L2 | 父级: /CLAUDE.md

成员清单
train_autoencoder.py: Hydra 训练入口，组装 DataModule、Autoencoder3D、LightningModule、Trainer，并支持预训练权重路径、严格模式与训练 seed 固定
detect.py: Hydra 推理入口，优先用安全模式加载 checkpoint，并对 CT 目录执行滑动窗口异常检测
download_data.py: argparse 数据入口，默认只调度可用的 LUNA16 少量真实 CT 配对，保留 lidc-idri 选项但显式声明未实装
run_experiments.py: argparse 实验入口，提供 health/synthetic/compare/ablation/plan 子命令，负责健康统计、基线对比、消融汇总、合成敏感性与实验协议导出

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
