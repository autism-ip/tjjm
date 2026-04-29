# src/
> L2 | 父级: /CLAUDE.md

成员清单
__init__.py: 包根标记与运行时守卫，维持 src.* 导入路径在源码态与安装态同构并提前收口 Windows OpenMP 环境
data/: 数据层，负责 LUNA16 下载、CT 预处理、Dataset 与健康 patch 采样
models/: 模型层，负责 SwinUNETR 自编码器、权重加载与扩散工具骨架
training/: 训练层，负责 LightningModule、损失函数、训练回调
detection/: 检测层，负责滑动窗口重建、异常热图、目录级推理与融合
evaluation/: 评估层，负责 Dice/AUC 等指标与报告生成
utils/: 工具层，负责配置加载、日志、可视化输出
experiments/: 实验层，负责健康统计、合成异常、实验协议生成与 ablation 汇总

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
