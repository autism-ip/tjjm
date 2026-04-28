"""
 * [INPUT]: 依赖 src/ 下各业务子包按 Python package 方式组织
 * [OUTPUT]: 对外提供 src 包根标记与 Windows OpenMP 环境守卫，承载 src.data/src.models/src.training/src.detection/src.evaluation/src.utils/src.experiments 导入路径
 * [POS]: src/ 的包根，占位但关键，负责让仓内统一的 src.* 导入在安装态与源码态保持同构，并在重型依赖导入前收口运行时环境
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os


# ---- runtime guard ----
# Torch + MONAI + scikit-image on Windows/Anaconda can load duplicate OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
