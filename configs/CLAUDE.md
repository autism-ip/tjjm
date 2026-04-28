# configs/
> L2 | 父级: /CLAUDE.md

成员清单
train_autoencoder.yaml: Hydra 训练配置，定义数据路径、模型参数、损失权重、优化器与输出目录
detect.yaml: Hydra 推理配置，定义测试 CT 路径、checkpoint、滑动窗口参数与评估指标
experiments.yaml: 轻量实验入口默认配置，定义 health 阈值与输入、synthetic 入口与参数、ablation 对比维度与输出路径

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
