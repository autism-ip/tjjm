# training/
> L2 | 父级: /CLAUDE.md

成员清单
trainer.py: LightningModule 封装，负责训练/验证循环、优化器配置
callbacks.py: 可视化回调与 ModelCheckpoint 重导出，每 N epoch 保存重建对比图
__init__.py: 入口聚合，导出 AutoencoderLightningModule 与 ReconstructionVisualizationCallback

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
