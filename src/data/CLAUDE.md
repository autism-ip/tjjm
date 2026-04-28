# data/
> L2 | 父级: /CLAUDE.md

成员清单
__init__.py:    公共接口聚合器, 导出所有 Dataset/预处理/下载器
preprocess.py:  3D CT 预处理引擎, 含 HU窗口/重采样/归一化/patch提取/健康筛选/世界坐标-体素坐标转换
dataset.py:     PyTorch Dataset 实现, LunaCTDataset(全CT) + LunaPatchDataset(健康patch)
download.py:    LUNA16 数据获取器与 CLI 函数 API, Kaggle API + 解压校验 + 文件清单 + LIDC 显式未实现入口, 供 CLI 诚实暴露能力边界

法则: 成员完整·一行一文件·父级链接·技术词前置

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
