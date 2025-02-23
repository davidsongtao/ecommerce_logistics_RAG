"""
Description: 模型评估运行系统

-*- Encoding: UTF-8 -*-
@File     ：run_model_eval.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from configs.config import config
from configs.log_config import get_logger
from local_model import LocalLLM
from model_evaluator import ModelEvaluator
from evaluation_metrics import EvaluationResult


@dataclass
class TestCase:
    """测试用例"""
    category: str
    prompt: str
    context: str = ""
    expected_length: int = 0

    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "prompt": self.prompt,
            "context": self.context,
            "expected_length": self.expected_length
        }


@dataclass
class GenerationParams:
    """生成参数"""
    temperature: float
    top_p: float
    repetition_penalty: float

    def to_dict(self) -> Dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty
        }


class ModelEvaluationRunner:
    """模型评估运行器"""

    def __init__(
            self,
            model_path: str,
            output_dir: str = "evaluation_results"
    ):
        """
        初始化评估运行器

        Args:
            model_path: 模型路径
            output_dir: 输出目录
        """
        self.logger = get_logger("runner")
        self.model_path = model_path
        self.output_dir = Path(output_dir)

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化测试用例
        self.test_cases = self._init_test_cases()

        self.logger.info(f"评估运行器初始化完成，模型路径: {model_path}")

    def _init_test_cases(self) -> List[TestCase]:
        """初始化测试用例"""
        return [
            TestCase(
                category="知识解答",
                prompt="请详细介绍人工智能的发展历史和主要里程碑。",
                expected_length=500
            ),
            TestCase(
                category="开放讨论",
                prompt="如何看待人工智能对就业市场的影响？",
                expected_length=400
            ),
            TestCase(
                category="多轮对话",
                prompt="深度学习和机器学习有什么区别？",
                context="上文中我们讨论了AI的基础概念。",
                expected_length=300
            ),
            TestCase(
                category="创意写作",
                prompt="请写一篇关于未来智能城市的短文。",
                expected_length=600
            ),
            TestCase(
                category="代码生成",
                prompt="用Python实现一个简单的神经网络模型。",
                expected_length=400
            )
        ]

    def _generate_param_combinations(self) -> List[GenerationParams]:
        """生成参数组合"""
        param_combinations = []

        # 温度参数范围
        temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # top_p参数范围
        top_ps = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
        # 重复惩罚参数范围
        repetition_penalties = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

        # 生成组合
        for temp in temperatures:
            for top_p in top_ps:
                for rep_pen in repetition_penalties:
                    param_combinations.append(
                        GenerationParams(
                            temperature=temp,
                            top_p=top_p,
                            repetition_penalty=rep_pen
                        )
                    )

        return param_combinations

    def evaluate_parameters(
            self,
            save_format: str = "both"
    ) -> Optional[Dict]:
        """
        评估不同参数组合的效果

        Args:
            save_format: 保存格式，可选 'json', 'txt', 'both'

        Returns:
            最佳参数组合的评测结果
        """
        try:
            self.logger.info("开始参数评估...")

            # 加载模型
            self.logger.info(f"正在加载模型: {self.model_path}")
            model = LocalLLM(self.model_path)

            # 初始化评估器
            evaluator = ModelEvaluator()

            # 获取参数组合
            param_combinations = self._generate_param_combinations()
            self.logger.info(f"共生成 {len(param_combinations)} 组参数组合")

            # 评估结果列表
            results = []

            # 对每组参数进行评估
            for params in param_combinations:
                self.logger.info(f"\n测试参数组合: {params.to_dict()}")

                # 配置模型参数
                model_config = config.model.generation_params.copy()
                model_config.update(params.to_dict())

                # 运行评估
                evaluation_result = evaluator.evaluate_model(
                    model=model,
                    test_cases=[case.to_dict() for case in self.test_cases]
                )

                # 记录结果
                result = {
                    "parameters": params.to_dict(),
                    "evaluation": evaluation_result.to_dict()
                }
                results.append(result)

                # 打印当前评估结果
                avg_score = evaluation_result.final_score["average"]["final_score"]
                self.logger.info(f"参数组合平均得分: {avg_score:.3f}")

            if not results:
                self.logger.warning("没有有效的评估结果")
                return None

            # 按总分排序
            results.sort(
                key=lambda x: x["evaluation"]["final_score"]["average"]["final_score"],
                reverse=True
            )

            # 保存结果
            self._save_evaluation_results(results, save_format)

            return results[0]  # 返回最佳结果

        except Exception as e:
            self.logger.error(f"评估过程出错: {e}")
            return None

    def _save_evaluation_results(
            self,
            results: List[Dict],
            save_format: str
    ):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_format in ["json", "both"]:
            json_file = self.output_dir / f"evaluation_results_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"JSON结果已保存到: {json_file}")

        if save_format in ["txt", "both"]:
            txt_file = self.output_dir / f"evaluation_report_{timestamp}.txt"
            self._generate_text_report(results, txt_file)
            self.logger.info(f"文本报告已保存到: {txt_file}")

    def _generate_text_report(self, results: List[Dict], file_path: Path):
        """生成文本格式报告"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("基座模型评测报告\n")
            f.write("=" * 50 + "\n\n")

            # 写入评测配置
            f.write("评测配置\n")
            f.write("-" * 30 + "\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"评分权重: {json.dumps(config.evaluation.weights, indent=2, ensure_ascii=False)}\n\n")

            # 写入参数评估结果
            f.write("参数评估结果\n")
            f.write("-" * 30 + "\n")
            for idx, result in enumerate(results, 1):
                f.write(f"\n{idx}. 参数组合: {result['parameters']}\n")
                final_score = result["evaluation"]["final_score"]["average"]["final_score"]
                f.write(f"   平均得分: {final_score:.3f}\n")
                f.write("   各类别得分:\n")
                for case in self.test_cases:
                    case_id = f"case_{case.category}"
                    if case_id in result["evaluation"]["final_score"]:
                        score = result["evaluation"]["final_score"][case_id]["final_score"]
                        f.write(f"   - {case.category}: {score:.3f}\n")

            # 写入最佳组合
            best_result = results[0]
            f.write("\n最佳参数组合\n")
            f.write("-" * 30 + "\n")
            f.write(f"参数: {json.dumps(best_result['parameters'], indent=2, ensure_ascii=False)}\n")
            f.write(f"得分: {best_result['evaluation']['final_score']['average']['final_score']:.3f}\n")

            # 写入建议
            f.write("\n建议\n")
            f.write("-" * 30 + "\n")
            best_score = best_result['evaluation']['final_score']['average']['final_score']
            if best_score >= 0.8:
                f.write("✓ 模型性能优秀，建议采用此参数配置\n")
            elif best_score >= 0.6:
                f.write("△ 模型性能一般，建议进一步优化参数\n")
            else:
                f.write("✗ 模型性能不佳，建议重新选择模型或参数\n")


def main():
    """主函数"""
    # 获取主程序logger
    main_logger = get_logger("main")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="模型评测运行器")
    parser.add_argument("--model-path", type=str, help="模型路径")
    parser.add_argument("--output-dir", default="evaluation_results", help="输出目录")
    parser.add_argument(
        "--save-format",
        choices=["json", "txt", "both"],
        default="both",
        help="保存格式"
    )

    args = parser.parse_args()

    # 如果未指定模型路径，使用默认路径
    if not args.model_path:
        args.model_path = (
            "/root/autodl-tmp/ecommerce_logistics_RAG/models/DeepSeek_R1_Distill_Qwen_7B"
            if os.name == 'posix'
            else r"D:\ecommerce_logistics_RAG\models\DeepSeek_R1_Distill_Qwen_1_5B"
        )

    try:
        # 创建运行器
        runner = ModelEvaluationRunner(
            model_path=args.model_path,
            output_dir=args.output_dir
        )

        # 运行评估
        best_result = runner.evaluate_parameters(
            save_format=args.save_format
        )

        if best_result:
            main_logger.info("\n最佳参数组合:")
            main_logger.info(json.dumps(best_result["parameters"], indent=2, ensure_ascii=False))
            main_logger.info("\n各类别得分:")
            main_logger.info(json.dumps(
                best_result["evaluation"]["final_score"],
                indent=2,
                ensure_ascii=False
            ))
        else:
            main_logger.error("评测失败，未能找到最佳参数组合")

    except Exception as e:
        main_logger.error(f"运行评测时出错: {e}")


if __name__ == "__main__":
    main()
