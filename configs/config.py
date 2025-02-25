"""
Description: 统一的配置管理系统

-*- Encoding: UTF-8 -*-
@File     ：config.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import torch


@dataclass
class LoggingConfig:
    """日志配置"""
    # 日志级别
    level: str = "INFO"
    # 日志格式
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    # 文件配置
    rotation: str = "2 hours"
    retention: str = "10 days"
    compression: str = "zip"
    # 控制台输出
    show_console: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    # 设备配置
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.bfloat16

    # 生成参数
    generation_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 2048,
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.5,
        "repetition_penalty": 1.6,
    })

    # 资源阈值
    gpu_memory_threshold: float = 0.85  # GPU使用率阈值(85%)
    cpu_memory_threshold: float = 0.30  # CPU空闲内存阈值(30%)


class AppConfig:
    """应用配置管理器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化配置"""
        # 项目根目录
        self.project_root = Path(__file__).resolve().parent.parent

        # 子配置
        self.logging = LoggingConfig()
        self.model = ModelConfig()

        # 加载环境变量
        self._load_env_vars()

    def _load_env_vars(self):
        """从环境变量加载配置"""
        # 日志配置
        if log_level := os.getenv("LOG_LEVEL"):
            self.logging.level = log_level

        # 模型配置
        if device := os.getenv("MODEL_DEVICE"):
            self.model.device = device

        # 评估配置
        if min_score := os.getenv("MIN_TOTAL_SCORE"):
            self.evaluation.thresholds["min_total_score"] = float(min_score)

    def to_dict(self) -> Dict[str, Any]:
        """导出配置"""
        return {
            "project_root": str(self.project_root),
            "logging": self.logging.__dict__,
            "model": {
                "device": self.model.device,
                "dtype": str(self.model.dtype),
                "generation_params": self.model.generation_params,
                "thresholds": {
                    "gpu_memory": self.model.gpu_memory_threshold,
                    "cpu_memory": self.model.cpu_memory_threshold
                }
            },
            "evaluation": {
                "weights": self.evaluation.weights,
                "thresholds": self.evaluation.thresholds
            }
        }


# 创建全局配置实例
config = AppConfig()
