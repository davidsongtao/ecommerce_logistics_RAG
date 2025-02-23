"""
Description: 模型相关异常类定义

-*- Encoding: UTF-8 -*-
@File     ：exceptions.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午12:31
@Contact  ：king.songtao@gmail.com
"""

import traceback
from datetime import datetime
from typing import Optional, Any, Dict
from configs.config import default_config as cfg


class ModelError(Exception):
    """
    模型相关错误的基类

    Attributes:
        message: 错误消息
        error_code: 错误代码
        timestamp: 错误发生的时间戳
        details: 详细错误信息
        traceback: 错误追踪信息
    """

    def __init__(
            self,
            message: str,
            error_code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
            *args: object
    ) -> None:
        super().__init__(message, *args)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.timestamp = datetime.now()
        self.details = details or {}
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """将错误信息转换为字典格式"""
        return {
            "error_code": self.error_code,
            "error_desc": cfg.error.error_codes.get(self.error_code, self.error_code),
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "traceback": self.traceback
        }

    def __str__(self) -> str:
        """提供友好的字符串表示"""
        error_desc = cfg.error.error_codes.get(self.error_code, self.error_code)
        return (
            f"[{self.error_code}] {error_desc}: {self.message}\n"
            f"Timestamp: {self.timestamp}\n"
            f"Details: {self.details}"
        )


class ModelLoadError(ModelError):
    """
    模型加载错误

    用于处理模型加载过程中的错误，如：
    - 模型文件不存在
    - 模型格式不正确
    - 设备不兼容
    - 内存不足
    """

    def __init__(
            self,
            message: str,
            model_path: Optional[str] = None,
            device: Optional[str] = None,
            memory_info: Optional[Dict[str, Any]] = None,
            *args: object
    ) -> None:
        details = {
            "model_path": model_path,
            "device": device,
            "memory_info": memory_info
        }
        super().__init__(
            message,
            error_code="MODEL_LOAD_ERROR",
            details=details,
            *args
        )


class ModelGenerateError(ModelError):
    """
    模型生成错误

    用于处理模型生成过程中的错误，如：
    - 生成超时
    - 输入格式错误
    - 生成结果无效
    - 资源耗尽
    """

    def __init__(
            self,
            message: str,
            prompt: Optional[str] = None,
            generation_config: Optional[Dict[str, Any]] = None,
            generation_info: Optional[Dict[str, Any]] = None,
            *args: object
    ) -> None:
        # 截断过长的prompt
        if prompt and len(prompt) > cfg.error.prompt_max_length:
            prompt = prompt[:cfg.error.prompt_max_length] + "..."

        details = {
            "prompt": prompt,
            "generation_config": generation_config,
            "generation_info": generation_info
        }
        super().__init__(
            message,
            error_code="MODEL_GENERATE_ERROR",
            details=details,
            *args
        )


class ModelResourceError(ModelError):
    """
    模型资源错误

    用于处理资源相关的错误，如：
    - GPU 内存不足
    - CPU 内存不足
    - 设备不可用
    """

    def __init__(
            self,
            message: str,
            resource_type: str,
            resource_info: Optional[Dict[str, Any]] = None,
            *args: object
    ) -> None:
        details = {
            "resource_type": resource_type,
            "resource_info": resource_info
        }
        super().__init__(
            message,
            error_code="MODEL_RESOURCE_ERROR",
            details=details,
            *args
        )
