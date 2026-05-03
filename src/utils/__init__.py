#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 聚合 src/utils/ 子模块的公共接口
 * [OUTPUT]: 对外暴露 config、logging、viz 的便捷入口
 * [POS]: src/utils/ 的聚合门面，被项目其他模块统一导入
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from src.utils.config import load_config, merge_configs
from src.utils.logging import (
    setup_logging,
    get_logger,
    TensorBoardLoggerWrapper,
)
from src.utils.viz import (
    visualize_slice,
    visualize_anomaly_map,
    plot_roc_curve,
)
from src.utils.collector import (
    BaseCollector,
    TrainingCollector,
    DetectionCollector,
    EvaluationCollector,
)
from src.utils.metrics_reader import MetricsReader

__all__ = [
    # config
    "load_config",
    "merge_configs",
    # logging
    "setup_logging",
    "get_logger",
    "TensorBoardLoggerWrapper",
    # viz
    "visualize_slice",
    "visualize_anomaly_map",
    "plot_roc_curve",
    # collector
    "BaseCollector",
    "TrainingCollector",
    "DetectionCollector",
    "EvaluationCollector",
    # metrics reader
    "MetricsReader",
]
