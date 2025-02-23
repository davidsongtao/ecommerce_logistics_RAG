"""
Description: 统一的日志管理系统

-*- Encoding: UTF-8 -*-
@File     ：log_manager.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

import sys
from typing import Dict, Optional
from loguru import logger
from configs.config import config


class LogManager:
    """日志管理器"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not LogManager._initialized:
            # 创建日志目录
            self.log_dir = config.project_root / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # 初始化处理器映射
            self.handlers: Dict[str, int] = {}

            # 移除默认处理器
            logger.remove()

            LogManager._initialized = True

    def get_logger(
            self,
            name: str,
            level: Optional[str] = None,
            show_console: Optional[bool] = None,
            **kwargs
    ) -> logger:
        """
        获取或创建logger

        Args:
            name: logger名称
            level: 日志级别,覆盖默认配置
            show_console: 是否显示控制台输出,覆盖默认配置
            **kwargs: 其他配置参数

        Returns:
            配置好的logger实例
        """
        # 清理同名logger的现有处理器
        if name in self.handlers:
            for handler_id in self.handlers[name]:
                logger.remove(handler_id)

        # 使用配置
        log_level = level or config.logging.level
        show_log = show_console if show_console is not None else config.logging.show_console

        handlers = []

        # 添加控制台处理器
        if show_log:
            handler_id = logger.add(
                sys.stderr,
                level=log_level,
                format=config.logging.format,
                filter=lambda record: record["extra"].get("name") == name
            )
            handlers.append(handler_id)

        # 添加文件处理器
        log_file = self.log_dir / name / f"{name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        handler_id = logger.add(
            str(log_file),
            level=log_level,
            format=config.logging.format,
            rotation=config.logging.rotation,
            retention=config.logging.retention,
            compression=config.logging.compression,
            filter=lambda record: record["extra"].get("name") == name
        )
        handlers.append(handler_id)

        # 保存处理器ID
        self.handlers[name] = handlers

        # 返回带有name标记的logger
        return logger.bind(name=name)

    def shutdown(self):
        """关闭所有日志处理器"""
        for handler_ids in self.handlers.values():
            for handler_id in handler_ids:
                logger.remove(handler_id)
        self.handlers.clear()


# 创建全局日志管理器实例
log_manager = LogManager()


# 导出获取logger的函数
def get_logger(name: str, **kwargs) -> logger:
    """获取logger的便捷函数"""
    return log_manager.get_logger(name, **kwargs)
