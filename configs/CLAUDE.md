# configs/
> L2 | 父级: /CLAUDE.md

成员清单
train_autoencoder.yaml: Hydra 训练配置，定义数据路径、模型参数、预训练开关、本地权重路径、严格模式、随机 seed 与输出目录
detect.yaml: Hydra 推理配置，定义测试 CT 路径、checkpoint、滑窗参数与输出目录
experiments.yaml: 轻量实验入口默认配置，定义 health 阈值、LUNA16 弱标注评估/阈值扫描参数、synthetic 参数、compare/ablation 维度、实验协议路径与重复性 seed

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
