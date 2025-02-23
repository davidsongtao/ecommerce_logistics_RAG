"""
Description: 项目全集参数配置文件
    
-*- Encoding: UTF-8 -*-
@File     ：config.py.py
@Author   ：King Songtao
@Time     ：2025/2/23 上午8:51
@Contact  ：king.songtao@gmail.com
"""
import os
from pathlib import Path
import torch
from typing import Dict, Any, Optional
import logging


class BaseConfig:
    """配置基类"""

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
            if not key.startswith('_')
        }


class LogConfig(BaseConfig):
    """日志配置"""

    def __init__(self):
        # 日志格式
        self.log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # 日志基础设置
        self.default_level = "DEBUG"
        self.show_log = True

        # 日志文件设置
        self.rotation = "2 hours"
        self.retention = "10 days"
        self.compression = "zip"


class ModelConfig(BaseConfig):
    """模型配置"""

    def __init__(self):
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

        # 生成配置默认值
        self.generation_config = {
            "max_new_tokens": 2048,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
        }

        # 资源监控阈值
        self.gpu_memory_threshold = 0.85  # GPU使用率阈值(85%)
        self.cpu_memory_threshold = 0.30  # CPU空闲内存阈值(30%)


class ErrorConfig(BaseConfig):
    """错误配置"""

    def __init__(self):
        # 错误码
        self.error_codes = {
            "MODEL_LOAD_ERROR": "模型加载错误",
            "MODEL_GENERATE_ERROR": "模型生成错误",
            "MODEL_RESOURCE_ERROR": "模型资源错误"
        }

        # 错误详情字段截断
        self.prompt_max_length = 100  # 错误信息中prompt字段最大长度


class LLMConfig:
    """LLM配置管理"""

    def __init__(self, env: str = "development"):
        # 基础配置
        self.env = env
        self.is_development = env == "development"
        self.is_production = env == "production"

        # 初始化子配置
        self._log_config = LogConfig()
        self._model_config = ModelConfig()
        self._error_config = ErrorConfig()

        # 根据环境调整配置
        self._setup_env_config()

    def _setup_env_config(self):
        """根据环境配置参数"""
        if self.is_production:
            # 生产环境日志配置
            self._log_config.default_level = "INFO"

            # 生产环境模型配置
            self._model_config.generation_config.update({
                "temperature": 0.5,
                "max_new_tokens": 1024
            })

    @property
    def log(self) -> LogConfig:
        """获取日志配置"""
        return self._log_config

    @property
    def model(self) -> ModelConfig:
        """获取模型配置"""
        return self._model_config

    @property
    def error(self) -> ErrorConfig:
        """获取错误配置"""
        return self._error_config

    def to_dict(self) -> Dict[str, Any]:
        """导出所有配置"""
        return {
            "environment": self.env,
            "log_config": self.log.to_dict(),
            "model_config": self.model.to_dict(),
            "error_config": self.error.to_dict()
        }


# 创建全局配置实例
default_config = LLMConfig()

# 创建生产环境配置实例
production_config = LLMConfig(env="production")
