"""
Description: 模型评估指标数据类定义

-*- Encoding: UTF-8 -*-
@File     ：evaluation_metrics.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PerformanceMetrics:
    """性能指标"""
    inference_speed: float  # 毫秒/token
    memory_usage: float  # GB
    gpu_memory: float  # GB
    throughput: float  # tokens/秒
    latency: float  # 毫秒

    def to_dict(self) -> Dict:
        return {
            "inference_speed": f"{self.inference_speed:.2f} ms/token",
            "memory_usage": f"{self.memory_usage:.2f} GB",
            "gpu_memory": f"{self.gpu_memory:.2f} GB",
            "throughput": f"{self.throughput:.2f} tokens/s",
            "latency": f"{self.latency:.2f} ms"
        }


@dataclass
class StabilityMetrics:
    """稳定性指标"""
    output_variance: float  # 输出方差
    length_variance: float  # 长度方差
    semantic_variance: float  # 语义方差

    def to_dict(self) -> Dict:
        return {
            "output_variance": f"{self.output_variance:.3f}",
            "length_variance": f"{self.length_variance:.3f}",
            "semantic_variance": f"{self.semantic_variance:.3f}"
        }


@dataclass
class QualityMetrics:
    """质量指标"""
    fluency: float  # 流畅度
    grammar: float  # 语法
    diversity: Dict[str, float]  # 多样性指标
    relevance: float  # 相关性
    coverage: float  # 覆盖度
    consistency: float  # 一致性
    coherence: float  # 连贯性

    def to_dict(self) -> Dict:
        return {
            "fluency": f"{self.fluency:.3f}",
            "grammar": f"{self.grammar:.3f}",
            "diversity": {k: f"{v:.3f}" for k, v in self.diversity.items()},
            "relevance": f"{self.relevance:.3f}",
            "coverage": f"{self.coverage:.3f}",
            "consistency": f"{self.consistency:.3f}",
            "coherence": f"{self.coherence:.3f}"
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    performance: PerformanceMetrics
    stability: Dict[str, StabilityMetrics]
    quality: Dict[str, QualityMetrics]
    final_score: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict:
        return {
            "performance": self.performance.to_dict(),
            "stability": {k: v.to_dict() for k, v in self.stability.items()},
            "quality": {k: v.to_dict() for k, v in self.quality.items()},
            "final_score": self.final_score
        }
