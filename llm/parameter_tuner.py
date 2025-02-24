import json
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from loguru import logger
from dataclasses import dataclass
from llm.local_model import LocalLLM
from configs.config import config
from configs.log_config import get_logger


@dataclass
class EvaluationMetrics:
    """评估指标"""
    response_quality: float  # 响应质量得分
    response_time: float  # 响应时间(ms)
    memory_usage: float  # 内存使用率

    def total_score(self, weights: Dict[str, float]) -> float:
        """计算总分"""
        return (
                weights.get('quality', 0.5) * self.response_quality +
                weights.get('time', 0.3) * (1 - self.response_time / 5000) +  # 假设5000ms为基准
                weights.get('memory', 0.2) * (1 - self.memory_usage)
        )


class ModelParamTuner:
    """模型参数优化器"""

    def __init__(
            self,
            model_path: str,
            test_cases: List[Dict[str, Any]],
            param_grid: Optional[Dict[str, List[Any]]] = None
    ):
        """
        初始化参数优化器
        
        Args:
            model_path: 模型路径
            test_cases: 测试用例列表
            param_grid: 参数搜索空间
        """
        self.logger = get_logger("param_tuner")
        self.model_path = model_path
        self.test_cases = test_cases

        # 默认参数搜索空间
        self.param_grid = param_grid or {
            "temperature": [0.1, 0.3, 0.6, 0.9],
            "top_p": [0.85, 0.9, 0.95],
            "repetition_penalty": [1.0, 1.1, 1.2],
            "max_new_tokens": [512, 1024],  # 减小token数以降低内存占用
        }

        # 评估权重
        self.eval_weights = {
            "quality": 0.5,  # 响应质量权重
            "time": 0.3,  # 响应时间权重
            "memory": 0.2  # 内存使用权重
        }

        self.best_params = None
        self.best_score = float('-inf')

        # 初始化模型(只加载一次)
        self.model = None
        self._init_model()

        self.logger.info("初始化参数优化器完成")
        self.logger.debug(f"参数搜索空间: {json.dumps(self.param_grid, indent=2)}")

    def _init_model(self):
        """初始化模型"""
        try:
            self.model = LocalLLM(
                model_path=self.model_path,
                device=config.model.device,
                dtype=config.model.dtype
            )
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
            raise

    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        """生成参数组合"""
        param_list = []
        for temp in self.param_grid["temperature"]:
            for top_p in self.param_grid["top_p"]:
                for rep_pen in self.param_grid["repetition_penalty"]:
                    for max_tokens in self.param_grid["max_new_tokens"]:
                        params = {
                            "temperature": temp,
                            "top_p": top_p,
                            "repetition_penalty": rep_pen,
                            "max_new_tokens": max_tokens,
                            "do_sample": True  # 保持采样方式一致
                        }
                        param_list.append(params)
        return param_list

    def evaluate_params(self, params: Dict[str, Any]) -> EvaluationMetrics:
        """
        评估单组参数
        
        Args:
            params: 待评估的参数组合
            
        Returns:
            评估指标对象
        """
        try:
            quality_scores = []
            response_times = []
            memory_usages = []

            # 运行测试用例
            for case in self.test_cases:
                start_time = time.time()

                # 生成响应
                response = self.model.generate(
                    prompt=case["input"],
                    **params
                )

                # 计算响应时间
                response_time = (time.time() - start_time) * 1000  # 转换为毫秒
                response_times.append(response_time)

                # 计算响应质量
                quality_score = self._calculate_quality_score(
                    response=response if isinstance(response, str) else "",
                    expected=case.get("expected"),
                    keywords=case.get("keywords", [])
                )
                quality_scores.append(quality_score)

                # 获取内存使用情况
                memory_info = self.model._get_memory_info()
                if self.model.device == "cuda":
                    memory_usage = float(
                        memory_info["gpu_memory"]["gpu_0"]["percent"].strip('%')
                    ) / 100
                else:
                    memory_usage = memory_info["cpu_memory"]["percent"] / 100
                memory_usages.append(memory_usage)

                # 记录详细信息
                self.logger.debug(f"测试用例结果:")
                self.logger.debug(f"输入: {case['input']}")
                self.logger.debug(f"输出: {response}")
                self.logger.debug(f"响应时间: {response_time:.2f}ms")
                self.logger.debug(f"质量得分: {quality_score:.2f}")
                self.logger.debug(f"内存使用: {memory_usage:.2%}")

            # 汇总指标
            metrics = EvaluationMetrics(
                response_quality=np.mean(quality_scores) if quality_scores else 0.0,
                response_time=np.mean(response_times) if response_times else 5000.0,
                memory_usage=np.mean(memory_usages) if memory_usages else 1.0
            )

            return metrics

        except Exception as e:
            self.logger.error(f"参数评估失败: {str(e)}")
            # 返回一个极低分的评估结果
            return EvaluationMetrics(
                response_quality=0.0,
                response_time=5000.0,
                memory_usage=1.0
            )

    def _calculate_quality_score(
            self,
            response: str,
            expected: Optional[str] = None,
            keywords: Optional[List[str]] = None
    ) -> float:
        """
        计算响应质量得分
        
        Args:
            response: 模型响应
            expected: 期望响应
            keywords: 关键词列表
            
        Returns:
            质量得分(0-1)
        """
        score = 0.0

        # 检查关键词覆盖率
        if keywords and len(keywords) > 0:
            matched = sum(1 for kw in keywords if kw in response)
            score += 0.5 * (matched / len(keywords))

        # 与期望响应的相似度
        if expected and response:
            # 使用简单的词重叠率作为相似度度量
            expected_words = set(expected.split())
            response_words = set(response.split())
            if expected_words:
                overlap = len(expected_words & response_words)
                score += 0.5 * (overlap / len(expected_words))
        elif not expected:
            # 如果没有期望响应，则这部分得分默认给0.25
            score += 0.25

        return score

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        执行参数优化
        
        Returns:
            (最优参数组合, 最优得分)
        """
        self.logger.info("开始参数优化...")

        # 生成所有参数组合
        param_combinations = self.generate_param_combinations()
        total_combinations = len(param_combinations)
        self.logger.info(f"共生成 {total_combinations} 种参数组合")

        # 评估每组参数
        for i, params in enumerate(param_combinations, 1):
            self.logger.info(f"评估第 {i}/{total_combinations} 组参数:")
            self.logger.info(json.dumps(params, indent=2))

            metrics = self.evaluate_params(params)
            score = metrics.total_score(self.eval_weights)

            # 更新最优结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.logger.info(f"发现更优参数组合: {json.dumps(params, indent=2)}")
                self.logger.info(f"得分: {score:.4f}")

        self.logger.info("参数优化完成")
        self.logger.info(f"最优参数组合: {json.dumps(self.best_params, indent=2)}")
        self.logger.info(f"最优得分: {self.best_score:.4f}")

        return self.best_params, self.best_score

    def __del__(self):
        """清理资源"""
        if self.model:
            del self.model


def prepare_test_cases() -> List[Dict[str, Any]]:
    """准备电商物流场景的测试用例"""
    return [
        {
            "input": "我的快递什么时候能到？订单号：JD12345678",
            "keywords": ["订单", "快递", "时间", "到达"],
            "expected": "根据订单号JD12345678的物流信息显示，您的包裹预计明天下午送达。"
        },
        {
            "input": "请问我的收货地址可以修改吗？订单还没发货。",
            "keywords": ["地址", "修改", "订单", "发货"],
            "expected": "您好，如果订单还未发货，是可以修改收货地址的。您可以在订单详情页面点击'修改地址'，或提供订单号，我来帮您处理。"
        },
        {
            "input": "你们可以寄到香港吗？运费怎么算？",
            "keywords": ["香港", "快递", "运费", "寄"],
            "expected": "我们支持寄送到香港。跨境快递的运费需要根据包裹的重量和体积来计算。请问您的包裹重量和尺寸是多少？我可以帮您估算具体费用。"
        }
    ]


# 使用示例
if __name__ == "__main__":
    # 准备测试用例
    test_cases = prepare_test_cases()

    # 初始化参数优化器
    tuner = ModelParamTuner(
        model_path="D:/ecommerce_logistics_RAG/models/DeepSeek_R1_Distill_Qwen_1_5B",
        test_cases=test_cases
    )

    # 执行参数优化
    best_params, best_score = tuner.optimize()

    # 打印结果
    print("\n最优参数配置:")
    print(json.dumps(best_params, indent=2))
    print(f"\n最优得分: {best_score:.4f}")
