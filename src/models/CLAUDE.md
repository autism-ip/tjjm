# models/
> L2 | 父级: /CLAUDE.md

成员清单
__init__.py: 公共接口聚合层，导出自编码器、权重管理、扩散调度与采样器
autoencoder.py: SwinUNETR 编码器 + 3D U-Net 解码器，自编码重建主模型
weights.py: MONAI SSL 预训练权重加载器，负责 encoder 权重映射与冻结控制
diffusion.py: DDPM/DDIM 工具骨架，保留扩散训练与采样接口

法则: 成员完整·一行一文件·父级链接·技术词前置
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
