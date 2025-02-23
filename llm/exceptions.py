"""
Description: 模型相关异常定义

-*- Encoding: UTF-8 -*-
@File     ：exceptions.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

from datetime import datetime
from typing import Any, Dict
import traceback
from dataclasses import dataclass, field


@dataclass
class ModelError(Exception):
    """模型错误基类"""
    message: str
    details: Dict[str, Any]
    error_code: str = field(default="UNKNOWN_ERROR")
    timestamp: datetime = field(default_factory=datetime.now)
    traceback_info: str = field(default_factory=lambda: traceback.format_exc())

    def __post_init__(self):
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "traceback": self.traceback_info
        }


@dataclass
class ModelLoadError(ModelError):
    """模型加载错误"""

    def __init__(
            self,
            message: str,
            model_path: str,
            device: str,
            memory_info: Dict[str, Any]
    ):
        details = {
            "model_path": model_path,
            "device": device,
            "memory_info": memory_info
        }
        super().__init__(
            message=message,
            details=details,
            error_code="MODEL_LOAD_ERROR"
        )


@dataclass
class ModelGenerateError(ModelError):
    """模型生成错误"""

    def __init__(
            self,
            message: str,
            prompt: str,
            generation_config: Dict[str, Any],
            generation_info: Dict[str, Any]
    ):
        # 截断过长的prompt
        if len(prompt) > 100:
            prompt = prompt[:100] + "..."

        details = {
            "prompt": prompt,
            "generation_config": generation_config,
            "generation_info": generation_info
        }
        super().__init__(
            message=message,
            details=details,
            error_code="MODEL_GENERATE_ERROR"
        )


@dataclass
class ModelResourceError(ModelError):
    """资源错误"""

    def __init__(
            self,
            message: str,
            resource_type: str,  # "gpu_memory" 或 "system_memory"
            resource_info: Dict[str, Any]
    ):
        details = {
            "resource_type": resource_type,
            "resource_info": resource_info
        }
        super().__init__(
            message=message,
            details=details,
            error_code="MODEL_RESOURCE_ERROR"
        )
