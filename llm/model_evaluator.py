"""
Description: 模型评估系统

-*- Encoding: UTF-8 -*-
@File     ：model_evaluator.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import asdict
import numpy as np

from configs.config import config
from configs.log_config import get_logger
from base_evaluator import BaseEvaluator
from evaluator_components import (
    PerformanceEvaluator,
    StabilityEvaluator,
    QualityEvaluator
)
from evaluation_metrics import EvaluationResult


class ModelEvaluator:
    """模型评估系统"""

    def __init__(
            self,
            bert_model_name: str = "bert-base-chinese",
            num_stability_runs: int = 5,
            show_log: bool = False
    ):
        """
        初始化评估系统

        Args:
            bert_model_name: BERT模型名称
            num_stability_runs: 稳定性测试运行次数
            show_log: 是否显示日志
        """
        self.logger = get_logger("evaluator", show_log=show_log)
        self.num_stability_runs = num_stability_runs

        # 初始化基础评估器
        self.logger.info("初始化评估器基础组件...")
        self.base_evaluator = BaseEvaluator(bert_model_name)

        # 初始化专门评估器组件，共享基础评估器的资源
        self.performance_evaluator = PerformanceEvaluator(self.base_evaluator)
        self.stability_evaluator = StabilityEvaluator(self.base_evaluator)
        self.quality_evaluator = QualityEvaluator(self.base_evaluator)

        self.logger.info("模型评估系统初始化完成")

    def evaluate_model(
            self,
            model,
            test_cases: List[Dict],
            output_dir: Optional[str] = None
    ) -> EvaluationResult:
        """
        全面评估模型

        Args:
            model: 待评估的模型
            test_cases: 测试用例列表
            output_dir: 结果输出目录

        Returns:
            评估结果
        """
        self.logger.info("开始模型评估...")

        # 1. 性能评测
        self.logger.info("正在评估性能指标...")
        performance = self.performance_evaluator.evaluate_performance(model, test_cases)

        # 检查性能是否达标
        if not self._check_performance(performance):
            self.logger.warning("模型性能未达到要求，建议考虑其他模型")
            return EvaluationResult(
                performance=performance,
                stability={},
                quality={},
                final_score={"total": 0.0}
            )

        # 2. 稳定性评测
        self.logger.info("正在评估稳定性...")
        stability = self.stability_evaluator.evaluate_stability(
            model,
            test_cases,
            self.num_stability_runs
        )

        # 3. 质量评测
        self.logger.info("正在评估生成质量...")
        quality = self.quality_evaluator.evaluate_quality(model, test_cases)

        # 4. 计算综合得分
        self.logger.info("正在计算综合得分...")
        final_score = self._calculate_final_score(
            performance=performance,
            stability=stability,
            quality=quality
        )

        # 5. 整合评估结果
        result = EvaluationResult(
            performance=performance,
            stability=stability,
            quality=quality,
            final_score=final_score
        )

        # 6. 保存结果
        if output_dir:
            self._save_results(result, output_dir)

        self.logger.info("模型评估完成!")
        return result

    def _check_performance(self, metrics: 'PerformanceMetrics') -> bool:
        """检查性能是否达标"""
        thresholds = config.evaluation.thresholds
        return (
                metrics.inference_speed <= thresholds["max_inference_time"] and
                metrics.memory_usage <= thresholds["max_memory_usage"] and
                metrics.gpu_memory <= thresholds["max_gpu_memory"] and
                metrics.throughput >= thresholds["min_throughput"] and
                metrics.latency <= thresholds["max_latency"]
        )

    def _calculate_final_score(
            self,
            performance: 'PerformanceMetrics',
            stability: Dict[str, 'StabilityMetrics'],
            quality: Dict[str, 'QualityMetrics']
    ) -> Dict[str, Dict[str, float]]:
        """计算综合得分"""
        weights = config.evaluation.weights
        thresholds = config.evaluation.thresholds
        scores = {}

        for case_id in quality.keys():
            # 1. 性能得分 (30%)
            perf_score = (
                    (1 - performance.inference_speed / thresholds["max_inference_time"]) * 0.1 +
                    (1 - performance.memory_usage / thresholds["max_memory_usage"]) * 0.1 +
                    (performance.throughput / thresholds["min_throughput"]) * 0.1
            )

            # 2. 稳定性得分 (20%)
            stability_metric = stability.get(case_id, stability[next(iter(stability))])
            stability_score = (1 - stability_metric.output_variance) * 0.2

            # 3. 质量得分 (50%)
            quality_metrics = quality[case_id]
            quality_score = (
                    quality_metrics.relevance * weights["relevance"] +
                    quality_metrics.coverage * weights["coverage"] +
                    quality_metrics.consistency * weights["consistency"] +
                    quality_metrics.fluency * weights["fluency"] +
                    quality_metrics.grammar * weights["grammar"] +
                    np.mean(list(quality_metrics.diversity.values())) * weights["diversity"] +
                    quality_metrics.coherence * weights["coherence"]
            )

            # 计算总分
            final_score = perf_score + stability_score + quality_score
            scores[case_id] = {
                "final_score": final_score,
                "performance_score": perf_score,
                "stability_score": stability_score,
                "quality_score": quality_score
            }

        # 计算总体平均分
        avg_score = np.mean([s["final_score"] for s in scores.values()])
        scores["average"] = {
            "final_score": avg_score,
            "performance_score": perf_score,  # 性能分数对所有case都一样
            "stability_score": np.mean([s["stability_score"] for s in scores.values()]),
            "quality_score": np.mean([s["quality_score"] for s in scores.values()])
        }

        return scores

    def _save_results(self, results: EvaluationResult, output_dir: str):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果 (JSON)
        result_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)

        # 生成评测报告 (TXT)
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        self._generate_report(results, report_file)

        self.logger.info(f"评测结果已保存到: {output_dir}")

    def _generate_report(self, results: EvaluationResult, report_file: str):
        """生成评测报告"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("基座模型评测报告\n")
            f.write("=" * 50 + "\n\n")

            # 1. 性能指标
            f.write("1. 性能指标\n")
            f.write("-" * 30 + "\n")
            perf = results.performance
            f.write(f"推理速度: {perf.inference_speed:.2f} ms/token\n")
            f.write(f"内存使用: {perf.memory_usage:.2f} GB\n")
            f.write(f"GPU内存: {perf.gpu_memory:.2f} GB\n")
            f.write(f"吞吐量: {perf.throughput:.2f} tokens/s\n")
            f.write(f"延迟: {perf.latency:.2f} ms\n\n")

            # 2. 稳定性指标
            f.write("2. 稳定性指标\n")
            f.write("-" * 30 + "\n")
            for case_id, metrics in results.stability.items():
                f.write(f"\n测试用例 {case_id}:\n")
                f.write(f"输出方差: {metrics.output_variance:.3f}\n")
                f.write(f"长度方差: {metrics.length_variance:.3f}\n")
                f.write(f"语义方差: {metrics.semantic_variance:.3f}\n")

            # 3. 质量指标
            f.write("\n3. 质量指标\n")
            f.write("-" * 30 + "\n")
            for case_id, metrics in results.quality.items():
                f.write(f"\n测试用例 {case_id}:\n")
                f.write(f"流畅度: {metrics.fluency:.3f}\n")
                f.write(f"语法: {metrics.grammar:.3f}\n")
                f.write(f"相关性: {metrics.relevance:.3f}\n")
                f.write(f"覆盖度: {metrics.coverage:.3f}\n")
                f.write(f"一致性: {metrics.consistency:.3f}\n")
                f.write(f"连贯性: {metrics.coherence:.3f}\n")
                f.write("多样性指标:\n")
                for k, v in metrics.diversity.items():
                    f.write(f"  - {k}: {v:.3f}\n")

            # 4. 综合得分
            f.write("\n4. 综合得分\n")
            f.write("-" * 30 + "\n")
            for case_id, scores in results.final_score.items():
                if case_id == "average":
                    f.write("\n总体平均分:\n")
                else:
                    f.write(f"\n测试用例 {case_id}:\n")
                f.write(f"总分: {scores['final_score']:.3f}\n")
                f.write(f"性能得分: {scores['performance_score']:.3f}\n")
                f.write(f"稳定性得分: {scores['stability_score']:.3f}\n")
                f.write(f"质量得分: {scores['quality_score']:.3f}\n")

            # 5. 评估建议
            f.write("\n5. 评估建议\n")
            f.write("-" * 30 + "\n")
            avg_score = results.final_score["average"]["final_score"]
            if avg_score >= 0.8:
                f.write("✓ 模型性能优秀，建议采用\n")
            elif avg_score >= 0.6:
                f.write("△ 模型性能一般，建议进一步优化\n")
            else:
                f.write("✗ 模型性能不佳，建议选择其他模型\n")

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'performance_evaluator'):
                del self.performance_evaluator
            if hasattr(self, 'stability_evaluator'):
                del self.stability_evaluator
            if hasattr(self, 'quality_evaluator'):
                del self.quality_evaluator
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")
