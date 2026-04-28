# data/
> L2 | 父级: /CLAUDE.md

成员清单
__init__.py:    公共接口聚合器, 导出所有 Dataset/预处理/下载器
preprocess.py:  兼容门面, 维持历史预处理导入路径并转发到 intensity.py / patches.py
intensity.py:   强度与空间变换子模块, 负责 HU窗口、重采样与归一化
patches.py:     patch 与坐标子模块, 负责世界坐标-体素坐标转换、patch提取与健康筛选
dataset.py:     PyTorch Dataset 实现, LunaCTDataset(全CT) + LunaPatchDataset(健康patch)
download.py:    LUNA16 数据获取器与 CLI 函数 API, Kaggle 文件级下载 + annotations.csv + 少量 .mhd/.raw 配对选择 + LIDC 显式未实现入口, 供 CLI 诚实暴露能力边界

法则: 成员完整·一行一文件·父级链接·技术词前置

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
