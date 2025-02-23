"""
Description: 
    
-*- Encoding: UTF-8 -*-
@File     ：run_base_llm_eval.py
@Author   ：King Songtao
@Time     ：2025/2/23 上午9:30
@Contact  ：king.songtao@gmail.com
"""

import os
import json
from typing import Dict, List
from llm.local_model import LocalLLM
from llm.base_llm_eval import ProfessionalEvaluator


class ModelEvaluationRunner:
    """模型评测运行器"""

    def __init__(self, model_path: str):
        """初始化评测运行器"""
        self.model = LocalLLM(model_path, show_log=False)
        self.evaluator = ProfessionalEvaluator()

        # 测试样例
        self.test_cases = [
            {
                "category": "知识解答",
                "prompt": "请详细介绍人工智能的发展历史和主要里程碑。",
                "context": ""
            },
            {
                "category": "开放讨论",
                "prompt": "如何看待人工智能对就业市场的影响？",
                "context": ""
            },
            {
                "category": "多轮对话",
                "prompt": "深度学习和机器学习有什么区别？",
                "context": "上文中我们讨论了AI的基础概念。"
            },
            {
                "category": "创意写作",
                "prompt": "请写一篇关于未来智能城市的短文。",
                "context": ""
            },
            {
                "category": "代码生成",
                "prompt": "用Python实现一个简单的神经网络模型。",
                "context": ""
            }
        ]

    def evaluate_parameters(
            self,
            params_list: List[Dict],
            output_file: str = "evaluation_results.json"
    ) -> Dict:
        """评估不同参数组合的效果"""
        results = []

        print(f"\n开始评估{len(params_list)}组参数...")

        for params in params_list:
            print(f"\n测试参数组合: {params}")
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
                    eval_result = self.evaluator.evaluate_text(
                        text=response,
                        prompt=case["prompt"],
                        context=case["context"]
                    )

                    # 添加测试用例信息
                    eval_result["category"] = case["category"]
                    eval_result["prompt"] = case["prompt"]
                    eval_result["response"] = response
                    scores.append(eval_result)

                    print(f"\n类别: {case['category']}")
                    print(f"总分: {eval_result['total_score']:.3f}")
                    print("详细评分:")
                    print(f"- 流畅度: {eval_result['fluency']:.3f}")
                    print(f"- 语法: {eval_result['grammar']:.3f}")
                    print(f"- 相关性: {eval_result['relevance']:.3f}")
                    print(f"- 一致性: {eval_result['consistency']:.3f}")

                except Exception as e:
                    print(f"评估错误: {e}")
                    continue

            if scores:
                # 计算这组参数的平均分
                avg_score = {
                    "parameters": params,
                    "average_total_score": sum(s["total_score"] for s in scores) / len(scores),
                    "scores_by_category": self._aggregate_by_category(scores),
                    "detailed_scores": scores
                }
                results.append(avg_score)

                print(f"\n参数组合平均得分: {avg_score['average_total_score']:.3f}")

        # 按总分排序
        results.sort(key=lambda x: x["average_total_score"], reverse=True)

        # 保存结果
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n评估完成! 结果已保存到: {output_file}")

        # 返回最佳参数组合
        return results[0] if results else None

    def _aggregate_by_category(self, scores: List[Dict]) -> Dict:
        """按类别汇总分数"""
        category_scores = {}
        for score in scores:
            category = score["category"]
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score["total_score"])

        return {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }


def main():
    # 模型路径
    model_path = "/root/autodl-tmp/ecommerce_logistics_RAG/models/DeepSeek_R1_Distill_Qwen_7B" if os.name == 'posix' else r"D:\ecommerce_logistics_RAG\models\DeepSeek_R1_Distill_Qwen_1_5B"

    # 创建评测运行器
    runner = ModelEvaluationRunner(model_path)

    # 待测试的参数组合
    params_list = [
        {
            "temperature": 0.3,
            "top_p": 0.8,
            "repetition_penalty": 1.1
        },
        {
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.0
        },
        {
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2
        }
    ]

    # 运行评估
    best_result = runner.evaluate_parameters(params_list)

    if best_result:
        print("\n最佳参数组合:")
        print(json.dumps(best_result["parameters"], indent=2, ensure_ascii=False))
        print("\n各类别得分:")
        print(json.dumps(best_result["scores_by_category"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
