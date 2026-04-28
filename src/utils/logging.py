#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * [INPUT]: 依赖 logging 标准库，依赖 torch.utils.tensorboard 的 SummaryWriter
 * [OUTPUT]: 对外提供 setup_logging()、get_logger()、TensorBoardLoggerWrapper
 * [POS]: src/utils/ 的日志中枢，被 training/ 和 scripts/ 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


# ============================================================
# 全局常量
# ============================================================
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> None:
    """
    初始化根日志器。

    Args:
        level: 日志级别，默认 INFO。
        log_file: 若提供，则同时写入该文件。
        fmt: 日志格式字符串。
        datefmt: 时间格式字符串。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取命名日志器。

    Args:
        name: 通常是 __name__。

    Returns:
        配置好的 logging.Logger 实例。
    """
    return logging.getLogger(name)


class TensorBoardLoggerWrapper:
    """
    对 torch.utils.tensorboard.SummaryWriter 的轻量包装。

    提供训练过程的标准化日志接口，自动处理 step 计数。
    """

    def __init__(self, log_dir: str, comment: str = "") -> None:
        """
        Args:
            log_dir: TensorBoard 日志目录。
            comment: 附加到目录名的注释。
        """
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
        self.global_step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """记录标量值。"""
        s = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, s)

    def log_scalars(self, tag: str, values: dict, step: Optional[int] = None) -> None:
        """批量记录标量字典。"""
        s = step if step is not None else self.global_step
        self.writer.add_scalars(tag, values, s)

    def log_histogram(self, tag: str, values, step: Optional[int] = None) -> None:
        """记录张量直方图。"""
        s = step if step is not None else self.global_step
        self.writer.add_histogram(tag, values, s)

    def log_image(self, tag: str, image_tensor, step: Optional[int] = None) -> None:
        """记录单张图像。"""
        s = step if step is not None else self.global_step
        self.writer.add_image(tag, image_tensor, s)

    def increment_step(self) -> None:
        """全局 step +1。"""
        self.global_step += 1

    def close(self) -> None:
        """关闭 writer，释放资源。"""
        self.writer.close()
