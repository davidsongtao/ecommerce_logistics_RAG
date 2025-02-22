"""
Description: 日志配置文件，用于配置日志管理系统
             使用示例 -->
                from log_config import logger, get_logger
                # 使用默认日志记录器（显示日志）
                logger.info("这是一条日志信息")

                # 获取特定类型的日志记录器（显示日志）
                debug_logger = get_logger("debug", show_log=True)
                debug_logger.debug("这是一条调试日志")

                # 获取只显示错误的日志记录器
                silent_logger = get_logger("default", show_log=False)
                silent_logger.info("这条信息不会显示")
                silent_logger.error("这条错误信息会显示")

                # 配置全局日志显示
                from log_config import configure_logging
                configure_logging(show_log=False)  # 全局禁用非错误日志
                configure_logging(show_log=True)   # 重新启用所有日志

-*- Encoding: UTF-8 -*-
@File     ：log_config.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午12:31
@Contact  ：king.songtao@gmail.com
"""
import os
import sys
from pathlib import Path
from loguru import logger
from typing import Union, Optional


class LogConfig:
    """日志配置管理类，用于集中管理项目日志配置"""

    def __init__(
            self,
            base_log_dir: Optional[Union[str, Path]] = None,
            log_level: Optional[str] = None
    ):
        """
        初始化日志配置
        :param base_log_dir: 基础日志目录，默认为项目根目录下的logs文件夹
        :param log_level: 日志级别，默认从环境变量读取，若无则为INFO
        """
        # 确定项目根目录
        # 假设log_config.py在configs目录下，项目根目录就是configs的父目录
        project_root = Path(__file__).resolve().parent.parent

        # 设置日志基础目录
        self.base_log_dir = (
            Path(base_log_dir) if base_log_dir
            else project_root / "logs"
        )

        # 确保日志目录存在
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

        # 日志级别，优先使用传入参数，其次使用环境变量，最后默认为INFO
        self.log_level = (
                log_level or
                os.getenv("LOG_LEVEL", "DEBUG")  # 默认改为DEBUG
        ).upper()

        # 日志格式配置
        self.log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # 不同类型日志的子目录
        self.log_subdirs = {
            "default": self.base_log_dir / "default",
            "error": self.base_log_dir / "error",
            "info": self.base_log_dir / "info",
            "debug": self.base_log_dir / "debug",
        }

        # 创建子目录
        for subdir in self.log_subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

    def configure_logger(
            self,
            log_type: str = "default",
            show_log: bool = True,
            additional_rotation: Optional[str] = "2 hours",
            retention: str = "10 days",
            compression: str = "zip"
    ):
        """
        配置特定类型的日志记录器
        :param log_type: 日志类型（default/error/info/debug）
        :param show_log: 是否显示日志
        :param additional_rotation: 日志文件轮转间隔
        :param retention: 日志保留时间
        :param compression: 日志压缩方式
        :return: logger对象
        """
        # 确保log_type存在，默认使用default
        log_type = log_type.lower()
        log_type = log_type if log_type in self.log_subdirs else "default"

        # 日志文件路径
        log_file_path = (
                self.log_subdirs[log_type] /
                f"{log_type}_logs_{{time:YYYY-MM-DD_HH-MM}}.log"
        )

        # 移除之前的logger，防止重复添加
        logger.remove()

        if show_log:
            # 添加控制台输出处理器
            logger.add(
                sys.stderr,
                level=self.log_level,
                format=self.log_format,
            )

            # 添加文件日志处理器
            logger.add(
                log_file_path,
                rotation=additional_rotation,
                retention=retention,
                compression=compression,
                level=self.log_level,
                format=self.log_format,
                enqueue=True,
            )
        else:
            # 不显示日志时，只添加ERROR级别的处理器
            logger.add(
                sys.stderr,
                level="ERROR",
                format="<red>{message}</red>"
            )

            # 错误日志仍然记录到文件
            logger.add(
                log_file_path,
                level="ERROR",
                format=self.log_format,
                enqueue=True,
            )

        return logger


def configure_logging(show_log: bool = True):
    """
    配置日志显示的快捷函数
    :param show_log: 是否显示日志
    """
    log_config = LogConfig()
    return log_config.configure_logger(show_log=show_log)


def get_logger(log_type: str = "default", show_log: bool = True):
    """
    获取特定类型的日志记录器
    :param log_type: 日志类型
    :param show_log: 是否显示日志
    :return: logger对象
    """
    log_config = LogConfig()
    logger_instance = log_config.configure_logger(log_type, show_log)
    return logger_instance.bind(name=log_type)


# 创建全局日志配置实例
log_config = LogConfig()

# 默认配置日志记录器
default_logger = log_config.configure_logger()

# 为了兼容之前的导入方式，直接导出logger
logger = default_logger
