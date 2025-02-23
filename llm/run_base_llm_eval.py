"""
Description: 模型评测运行脚本
    
-*- Encoding: UTF-8 -*-
@File     ：run_base_llm_eval.py
@Author   ：King Songtao
@Time     ：2025/2/23 上午9:30
@Contact  ：king.songtao@gmail.com
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from llm.local_model import LocalLLM
from llm.base_llm_eval import ModelEvaluator
from configs.log_config import get_logger, configure_logging

# 获取专门的评测日志记录器
logger = get_logger("evaluator", show_log=True)


class ModelEvaluationRunner:
    """模型评测运行器"""

    def __init__(
            self,
            model_path: str,
            eval_config: Optional[Dict] = None,
            output_dir: str = "evaluation_results"
    ):
        """
        初始化评测运行器

        Args:
            model_path: 模型路径
            eval_config: 评测配置
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.output_dir = output_dir

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 评测配置
        self.eval_config = eval_config or {
            "weights": {
                "relevance": 0.20,
                "coverage": 0.15,
                "consistency": 0.15,
                "fluency": 0.15,
                "grammar": 0.15,
                "diversity": 0.10,
                "coherence": 0.10
            },
            "thresholds": {
                "min_total_score": 0.6,
                "min_relevance": 0.7,
                "min_coverage": 0.7
            }
        }

        # 初始化模型和评测器
        try:
            logger.info(f"正在加载模型: {model_path}")
            self.model = LocalLLM(model_path, show_log=False)
            self.evaluator = ModelEvaluator(
                performance_thresholds=self.eval_config.get("thresholds")
            )
            logger.info("模型和评测器初始化完成")
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

        # 测试样例
        self.test_cases = [
            {
                "category": "知识解答",
                "prompt": "请详细介绍人工智能的发展历史和主要里程碑。",
                "context": "",
                "expected_length": 500
            },
            {
                "category": "开放讨论",
                "prompt": "如何看待人工智能对就业市场的影响？",
                "context": "",
                "expected_length": 400
            },
            {
                "category": "多轮对话",
                "prompt": "深度学习和机器学习有什么区别？",
                "context": "上文中我们讨论了AI的基础概念。",
                "expected_length": 300
            },
            {
                "category": "创意写作",
                "prompt": "请写一篇关于未来智能城市的短文。",
                "context": "",
                "expected_length": 600
            },
            {
                "category": "代码生成",
                "prompt": "用Python实现一个简单的神经网络模型。",
                "context": "",
                "expected_length": 400
            }
        ]

    def evaluate_parameters(
            self,
            params_list: List[Dict],
            save_format: str = "both"
    ) -> Optional[Dict]:
        """
        评估不同参数组合的效果

        Args:
            params_list: 参数组合列表
            save_format: 保存格式，可选 'json', 'txt', 'both'

        Returns:
            最佳参数组合的评测结果
        """
        try:
            results = []
            logger.info(f"开始评估{len(params_list)}组参数...")

            for params in params_list:
                logger.info(f"\n测试参数组合: {params}")
                scores = []

                # 对每个测试用例进行评估
                for case in self.test_cases:
                    try:
                        # 生成回复
                        response = self.model.generate(
                            case["prompt"],
                            **params
                        )

                        # 评估生成质量
                        quality_metrics = self.evaluator.evaluate_text(
                            text=response,
                            prompt=case["prompt"],
                            context=case["context"]
                        )

                        # 计算加权总分
                        total_score = self._calculate_weighted_score(quality_metrics)

                        # 构建评估结果
                        eval_result = {
                            "category": case["category"],
                            "prompt": case["prompt"],
                            "response": response,
                            "total_score": total_score,
                            "fluency": quality_metrics.fluency,
                            "grammar": quality_metrics.grammar,
                            "relevance": quality_metrics.relevance,
                            "coverage": quality_metrics.coverage,
                            "consistency": quality_metrics.consistency,
                            "coherence": quality_metrics.coherence,
                            "diversity": quality_metrics.diversity
                        }
                        scores.append(eval_result)

                        # 打印当前评估结果
                        self._print_eval_result(eval_result)

                    except Exception as e:
                        logger.error(f"评估错误: {e}")
                        continue

                if scores:
                    # 计算这组参数的平均得分
                    avg_score = {
                        "parameters": params,
                        "average_total_score": np.mean([s["total_score"] for s in scores]),
                        "scores_by_category": self._aggregate_by_category(scores),
                        "detailed_scores": scores
                    }
                    results.append(avg_score)

                    logger.info(f"参数组合平均得分: {avg_score['average_total_score']:.3f}")

            if not results:
                logger.warning("没有有效的评估结果")
                return None

            # 按总分排序
            results.sort(key=lambda x: x["average_total_score"], reverse=True)

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if save_format in ["json", "both"]:
                json_file = os.path.join(self.output_dir, f"evaluation_results_{timestamp}.json")
                self._save_json_results(results, json_file)

            if save_format in ["txt", "both"]:
                txt_file = os.path.join(self.output_dir, f"evaluation_report_{timestamp}.txt")
                self._save_txt_report(results, txt_file)

            logger.info(f"评估完成! 结果已保存到: {self.output_dir}")
            return results[0]

        except Exception as e:
            logger.error(f"评测过程出错: {e}")
            return None

    def _calculate_weighted_score(self, metrics) -> float:
        """计算加权总分"""
        weights = self.eval_config["weights"]

        return (
                metrics.relevance * weights["relevance"] +
                metrics.coverage * weights["coverage"] +
                metrics.consistency * weights["consistency"] +
                metrics.fluency * weights["fluency"] +
                metrics.grammar * weights["grammar"] +
                np.mean(list(metrics.diversity.values())) * weights["diversity"] +
                metrics.coherence * weights["coherence"]
        )

    def _aggregate_by_category(self, scores: List[Dict]) -> Dict:
        """按类别汇总分数"""
        category_scores = {}
        for score in scores:
            category = score["category"]
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score["total_score"])

        return {
            cat: np.mean(scores) for cat, scores in category_scores.items()
        }

    def _print_eval_result(self, result: Dict):
        """打印评估结果"""
        logger.info(f"\n类别: {result['category']}")
        logger.info(f"总分: {result['total_score']:.3f}")
        logger.info("详细评分:")
        logger.info(f"- 流畅度: {result['fluency']:.3f}")
        logger.info(f"- 语法: {result['grammar']:.3f}")
        logger.info(f"- 相关性: {result['relevance']:.3f}")
        logger.info(f"- 覆盖度: {result['coverage']:.3f}")
        logger.info(f"- 一致性: {result['consistency']:.3f}")
        logger.info(f"- 连贯性: {result['coherence']:.3f}")

    def _save_json_results(self, results: List[Dict], file_path: str):
        """保存JSON格式结果"""
        try:
            logger.info("正在生成文本报告...")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON结果已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存JSON结果失败: {e}")

    def _save_txt_report(self, results: List[Dict], file_path: str):
        """保存文本格式报告"""
        try:
            logger.info("正在生成文本报告...")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("基座模型评测报告\n")
                f.write("=" * 50 + "\n\n")

                # 写入评测配置
                f.write("评测配置\n")
                f.write("-" * 30 + "\n")
                f.write(f"模型路径: {self.model_path}\n")
                f.write(f"评分权重: {json.dumps(self.eval_config['weights'], indent=2, ensure_ascii=False)}\n\n")

                # 写入参数评估结果
                f.write("参数评估结果\n")
                f.write("-" * 30 + "\n")
                for idx, result in enumerate(results, 1):
                    f.write(f"\n{idx}. 参数组合: {result['parameters']}\n")
                    f.write(f"   平均得分: {result['average_total_score']:.3f}\n")
                    f.write("   各类别得分:\n")
                    for category, score in result['scores_by_category'].items():
                        f.write(f"   - {category}: {score:.3f}\n")

                # 写入最佳组合
                best_result = results[0]
                f.write("\n最佳参数组合\n")
                f.write("-" * 30 + "\n")
                f.write(f"参数: {json.dumps(best_result['parameters'], indent=2, ensure_ascii=False)}\n")
                f.write(f"得分: {best_result['average_total_score']:.3f}\n")

                # 写入建议
                f.write("\n建议\n")
                f.write("-" * 30 + "\n")
                if best_result['average_total_score'] >= 0.8:
                    f.write("✓ 模型性能优秀，建议采用此参数配置\n")
                elif best_result['average_total_score'] >= 0.6:
                    f.write("△ 模型性能一般，建议进一步优化参数\n")
                else:
                    f.write("✗ 模型性能不佳，建议重新选择模型或参数\n")

            logger.info(f"评测报告已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存评测报告失败: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="模型评测运行器")
    parser.add_argument("--model-path", type=str, help="模型路径")
    parser.add_argument("--output-dir", default="evaluation_results", help="输出目录")
    parser.add_argument("--save-format", choices=["json", "txt", "both"], default="both", help="保存格式")

    args = parser.parse_args()

    # 如果未指定模型路径，使用默认路径
    if not args.model_path:
        args.model_path = ("/root/autodl-tmp/ecommerce_logistics_RAG/models/DeepSeek_R1_Distill_Qwen_7B"
                           if os.name == 'posix'
                           else r"D:\ecommerce_logistics_RAG\models\DeepSeek_R1_Distill_Qwen_1_5B")

    # 待测试的参数组合
    params_list = [
        {"temperature": 0.5, "top_p": 0.7, "repetition_penalty": 1.2},
        {"temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.3},
        {"temperature": 0.7, "top_p": 0.85, "repetition_penalty": 1.4},
        {"temperature": 0.2, "top_p": 0.6, "repetition_penalty": 1.5},
        {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.1},
        {"temperature": 0.4, "top_p": 0.8, "repetition_penalty": 1.0},
        {"temperature": 0.1, "top_p": 1.0, "repetition_penalty": 1.3},
        {"temperature": 0.8, "top_p": 0.7, "repetition_penalty": 1.2},
        {"temperature": 0.9, "top_p": 0.85, "repetition_penalty": 1.4},
        {"temperature": 0.6, "top_p": 0.9, "repetition_penalty": 1.5},
        {"temperature": 0.3, "top_p": 0.6, "repetition_penalty": 1.1},
        {"temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.3},
        {"temperature": 0.4, "top_p": 1.0, "repetition_penalty": 1.2},
        {"temperature": 0.2, "top_p": 0.8, "repetition_penalty": 1.0},
        {"temperature": 0.5, "top_p": 0.85, "repetition_penalty": 1.4},
        {"temperature": 0.9, "top_p": 0.7, "repetition_penalty": 1.5},
        {"temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.2},
        {"temperature": 0.8, "top_p": 0.6, "repetition_penalty": 1.3},
        {"temperature": 0.6, "top_p": 1.0, "repetition_penalty": 1.1},
        {"temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.4},
        {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.5},
        {"temperature": 0.3, "top_p": 0.85, "repetition_penalty": 1.0},
        {"temperature": 0.5, "top_p": 0.6, "repetition_penalty": 1.3},
        {"temperature": 0.2, "top_p": 0.95, "repetition_penalty": 1.2},
        {"temperature": 0.9, "top_p": 1.0, "repetition_penalty": 1.1},
        {"temperature": 0.1, "top_p": 0.8, "repetition_penalty": 1.4},
        {"temperature": 0.8, "top_p": 0.85, "repetition_penalty": 1.5},
        {"temperature": 0.6, "top_p": 0.7, "repetition_penalty": 1.0},
        {"temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.3},
        {"temperature": 0.7, "top_p": 1.0, "repetition_penalty": 1.2},
        {"temperature": 0.3, "top_p": 0.6, "repetition_penalty": 1.4},
        {"temperature": 0.5, "top_p": 0.95, "repetition_penalty": 1.5},
        {"temperature": 0.2, "top_p": 0.8, "repetition_penalty": 1.1},
        {"temperature": 0.9, "top_p": 0.85, "repetition_penalty": 1.0},
        {"temperature": 0.1, "top_p": 0.7, "repetition_penalty": 1.3},
        {"temperature": 0.8, "top_p": 1.0, "repetition_penalty": 1.4},
        {"temperature": 0.6, "top_p": 0.85, "repetition_penalty": 1.2},
        {"temperature": 0.4, "top_p": 0.6, "repetition_penalty": 1.5},
        {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1},
        {"temperature": 0.3, "top_p": 0.95, "repetition_penalty": 1.0},
        {"temperature": 0.5, "top_p": 1.0, "repetition_penalty": 1.3},
        {"temperature": 0.2, "top_p": 0.7, "repetition_penalty": 1.4},
        {"temperature": 0.9, "top_p": 0.8, "repetition_penalty": 1.5},
        {"temperature": 0.1, "top_p": 0.85, "repetition_penalty": 1.2},
        {"temperature": 0.8, "top_p": 0.6, "repetition_penalty": 1.1},
        {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.0},
        {"temperature": 0.4, "top_p": 0.7, "repetition_penalty": 1.3},
        {"temperature": 0.7, "top_p": 1.0, "repetition_penalty": 1.4},
        {"temperature": 0.3, "top_p": 0.8, "repetition_penalty": 1.5},
        {"temperature": 0.5, "top_p": 0.9, "repetition_penalty": 1.1},
        {"temperature": 0.2, "top_p": 0.85, "repetition_penalty": 1.2}
    ]

    try:
        # 创建评测运行器
        runner = ModelEvaluationRunner(
            model_path=args.model_path,
            output_dir=args.output_dir
        )

        # 运行评估
        best_result = runner.evaluate_parameters(
            params_list=params_list,
            save_format=args.save_format
        )

        if best_result:
            logger.info("\n最佳参数组合:")
            logger.info(json.dumps(best_result["parameters"], indent=2, ensure_ascii=False))
            logger.info("\n各类别得分:")
            logger.info(json.dumps(best_result["scores_by_category"], indent=2, ensure_ascii=False))
        else:
            logger.error("评测失败，未能找到最佳参数组合")

    except Exception as e:
        logger.error(f"运行评测时出错: {e}")


if __name__ == "__main__":
    main()
