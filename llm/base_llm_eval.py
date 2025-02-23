"""
Description:
            主要评估维度:
                1. 性能评测
                2. 核心能力评测
                3. 参数稳定性评测
                4. 基础语言质量评测
    
-*- Encoding: UTF-8 -*-
@File     ：base_llm_eval.py
@Author   ：King Songtao
@Time     ：2025/2/23 上午9:22
@Contact  ：king.songtao@gmail.com
"""

import os
import json
import time
import jieba
import psutil
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from scipy.stats import entropy
from nltk.util import ngrams
from tqdm import tqdm
from configs.log_config import get_logger

# 配置日志
logger = get_logger("model_eval", show_log=True)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    inference_speed: float  # 毫秒/token
    memory_usage: float  # GB
    gpu_memory: float  # GB
    throughput: float  # tokens/秒
    latency: float  # 毫秒


@dataclass
class StabilityMetrics:
    """稳定性指标数据类"""
    output_variance: float  # 输出方差
    length_variance: float  # 长度方差
    semantic_variance: float  # 语义方差


@dataclass
class QualityMetrics:
    """质量指标数据类"""
    fluency: float  # 流畅度
    grammar: float  # 语法
    diversity: Dict[str, float]  # 多样性指标
    relevance: float  # 相关性
    coverage: float  # 覆盖度
    consistency: float  # 一致性
    coherence: float  # 连贯性


class ModelEvaluator:
    """基座模型评测系统"""

    def __init__(
            self,
            bert_model_name: str = "bert-base-chinese",
            num_stability_runs: int = 5,
            batch_sizes: List[int] = [1, 4, 8, 16],
            performance_thresholds: Optional[Dict] = None
    ):
        """
        初始化评测器

        Args:
            bert_model_name: BERT模型名称
            num_stability_runs: 稳定性测试运行次数
            batch_sizes: 批处理大小列表
            performance_thresholds: 性能阈值
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_stability_runs = num_stability_runs
        self.batch_sizes = batch_sizes

        # 设置默认性能阈值
        self.performance_thresholds = performance_thresholds or {
            "max_inference_time": 100,  # ms/token
            "max_memory_usage": 16,  # GB
            "max_gpu_memory": 12,  # GB
            "min_throughput": 10,  # tokens/s
            "max_latency": 1000,  # ms
            "min_stability": 0.9  # 稳定性分数
        }

        # 初始化BERT模型
        logger.info(f"正在加载BERT模型: {bert_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.bert_model.eval()

        # 初始化jieba分词
        jieba.initialize()

        # 加载停用词
        self.stopwords = self._load_stopwords()

        logger.info("评测器初始化完成")

    def _load_stopwords(self) -> Set[str]:
        """加载停用词表"""
        return {
            "的", "了", "和", "是", "在", "我", "有", "就",
            "不", "也", "这", "着", "那", "就是", "但是", "所以"
        }

    def evaluate_model(
            self,
            model,
            test_cases: List[Dict],
            param_sets: List[Dict],
            output_dir: Optional[str] = None
    ) -> Dict:
        """
        全面评估模型

        Args:
            model: 待评估的模型
            test_cases: 测试用例列表
            param_sets: 参数组合列表
            output_dir: 结果输出目录

        Returns:
            评估结果字典
        """
        logger.info("开始模型评估...")
        start_time = time.time()

        results = {}

        # 1. 性能评测
        logger.info("正在评估性能指标...")
        performance = self.evaluate_performance(model, test_cases)
        results["performance"] = performance

        # 检查性能是否达标
        if not self._check_performance(performance):
            logger.warning("模型性能未达到要求，建议考虑其他模型")
            return results

        # 2. 参数稳定性评测
        logger.info("正在评估参数稳定性...")
        stability = self.evaluate_parameter_stability(
            model, test_cases, param_sets
        )
        results["stability"] = stability

        # 3. 质量评测
        logger.info("正在评估生成质量...")
        quality = self.evaluate_quality(model, test_cases, param_sets)
        results["quality"] = quality

        # 4. 计算综合得分
        logger.info("正在计算综合得分...")
        final_score = self.calculate_final_score(
            performance=performance,
            stability=stability,
            quality=quality
        )
        results["final_score"] = final_score

        # 5. 保存结果
        if output_dir:
            self._save_results(results, output_dir)

        duration = time.time() - start_time
        logger.info(f"评估完成! 耗时: {duration:.2f}秒")

        return results

    def evaluate_performance(
            self,
            model,
            test_cases: List[Dict]
    ) -> PerformanceMetrics:
        """评估性能指标"""
        start_time = time.time()
        total_tokens = 0
        batch_size = 1

        # 测试推理速度和吞吐量
        for case in tqdm(test_cases, desc="Testing inference speed"):
            with torch.no_grad():
                # 记录生成开始时间
                gen_start = time.time()

                # 生成文本
                output = model.generate(case["prompt"])

                # 计算延迟
                latency = (time.time() - gen_start) * 1000  # 转换为毫秒

                # 累计token数
                total_tokens += len(output.split())

        # 计算总耗时
        total_time = time.time() - start_time

        # 计算性能指标
        inference_speed = (total_time * 1000) / total_tokens  # ms/token
        throughput = total_tokens / total_time  # tokens/s

        # 获取内存使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
        else:
            gpu_memory = 0

        memory_usage = psutil.Process().memory_info().rss / 1e9  # GB

        return PerformanceMetrics(
            inference_speed=inference_speed,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            throughput=throughput,
            latency=latency
        )

    def evaluate_parameter_stability(
            self,
            model,
            test_cases: List[Dict],
            param_sets: List[Dict]
    ) -> Dict[str, List[StabilityMetrics]]:
        """评估参数稳定性"""
        stability_results = {}

        for params in tqdm(param_sets, desc="Testing parameter stability"):
            param_key = f"temp{params['temperature']}_topp{params['top_p']}"
            stability_metrics = []

            for case in test_cases:
                outputs = []
                # 多次运行同一参数组合
                for _ in range(self.num_stability_runs):
                    try:
                        output = model.generate(case["prompt"], **params)
                        outputs.append(output)
                    except Exception as e:
                        logger.error(f"生成错误: {e}")
                        continue

                if outputs:
                    # 计算稳定性指标
                    stability_metric = self._calculate_stability_metrics(outputs)
                    stability_metrics.append(stability_metric)

            stability_results[param_key] = stability_metrics

        return stability_results

    def evaluate_quality(
            self,
            model,
            test_cases: List[Dict],
            param_sets: List[Dict]
    ) -> Dict[str, QualityMetrics]:
        """评估生成质量"""
        quality_results = {}

        for params in tqdm(param_sets, desc="Testing generation quality"):
            param_key = f"temp{params['temperature']}_topp{params['top_p']}"
            all_metrics = []

            for case in test_cases:
                try:
                    # 生成文本
                    output = model.generate(case["prompt"], **params)

                    # 评估质量指标
                    metrics = self._evaluate_text_quality(
                        text=output,
                        prompt=case["prompt"],
                        context=case.get("context", "")
                    )
                    all_metrics.append(metrics)

                except Exception as e:
                    logger.error(f"评估错误: {e}")
                    continue

            if all_metrics:
                # 计算平均指标
                quality_results[param_key] = self._average_quality_metrics(all_metrics)

        return quality_results

    def _evaluate_text_quality(
            self,
            text: str,
            prompt: str,
            context: str = ""
    ) -> QualityMetrics:
        """评估文本质量的各项指标"""
        # 评估流畅度
        fluency = self._evaluate_fluency(text)

        # 评估语法
        grammar = self._evaluate_grammar(text)

        # 评估多样性
        diversity = self._evaluate_diversity(text)

        # 评估相关性
        relevance = self._evaluate_relevance(text, prompt)

        # 评估覆盖度
        coverage = self._evaluate_coverage(text, prompt)

        # 评估一致性
        consistency = self._evaluate_consistency(text, prompt, context)

        # 评估连贯性
        coherence = self._evaluate_coherence(text)

        return QualityMetrics(
            fluency=fluency,
            grammar=grammar,
            diversity=diversity,
            relevance=relevance,
            coverage=coverage,
            consistency=consistency,
            coherence=coherence
        )

    def evaluate_text(self, text: str, prompt: str, context: str = "") -> QualityMetrics:
        return self._evaluate_text_quality(text, prompt, context)

    def calculate_final_score(
            self,
            performance: PerformanceMetrics,
            stability: Dict[str, List[StabilityMetrics]],
            quality: Dict[str, QualityMetrics]
    ) -> Dict[str, float]:
        """计算综合得分"""
        scores = {}

        for param_key in quality.keys():
            # 1. 性能得分 (30%)
            perf_score = (
                    (1 - performance.inference_speed / self.performance_thresholds["max_inference_time"]) * 0.1 +
                    (1 - performance.memory_usage / self.performance_thresholds["max_memory_usage"]) * 0.1 +
                    (performance.throughput / self.performance_thresholds["min_throughput"]) * 0.1
            )

            # 2. 稳定性得分 (20%)
            stability_metrics = stability[param_key]
            avg_stability = np.mean([
                1 - m.output_variance for m in stability_metrics
            ])
            stability_score = avg_stability * 0.2

            # 3. 质量得分 (50%)
            quality_metrics = quality[param_key]
            quality_score = (
                    quality_metrics.relevance * 0.15 +
                    quality_metrics.coverage * 0.15 +
                    quality_metrics.consistency * 0.1 +
                    quality_metrics.fluency * 0.05 +
                    quality_metrics.grammar * 0.05
            )

            # 计算总分
            final_score = perf_score + stability_score + quality_score
            scores[param_key] = {
                "final_score": final_score,
                "performance_score": perf_score,
                "stability_score": stability_score,
                "quality_score": quality_score
            }

        return scores

    def _check_performance(self, metrics: PerformanceMetrics) -> bool:
        """检查性能是否达标"""
        return (
                metrics.inference_speed <= self.performance_thresholds["max_inference_time"] and
                metrics.memory_usage <= self.performance_thresholds["max_memory_usage"] and
                metrics.gpu_memory <= self.performance_thresholds["max_gpu_memory"] and
                metrics.throughput >= self.performance_thresholds["min_throughput"] and
                metrics.latency <= self.performance_thresholds["max_latency"]
        )

    def _calculate_stability_metrics(
            self,
            outputs: List[str]
    ) -> StabilityMetrics:
        """计算稳定性指标"""
        # 计算长度方差
        lengths = [len(out) for out in outputs]
        length_variance = np.var(lengths)

        # 计算语义相似度方差
        semantic_similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarity = self._calculate_semantic_similarity(
                    outputs[i], outputs[j]
                )
                semantic_similarities.append(similarity)

        semantic_variance = np.var(semantic_similarities)

        # 计算整体输出方差
        output_variance = (length_variance + semantic_variance) / 2

        return StabilityMetrics(
            output_variance=output_variance,
            length_variance=length_variance,
            semantic_variance=semantic_variance
        )

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度"""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    [text1, text2],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)

                return similarity.item()
        except Exception as e:
            logger.error(f"计算语义相似度错误: {e}")
            return 0.0

    def _evaluate_fluency(self, text: str) -> float:
        """评估文本流畅度"""
        try:
            with torch.no_grad():
                sentences = [s for s in text.split("。") if s.strip()]

                if not sentences:
                    return 0.5

                sentence_embeddings = []
                for sentence in sentences:
                    inputs = self.tokenizer(
                        sentence,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    outputs = self.bert_model(**inputs)
                    last_hidden_state = outputs.last_hidden_state
                    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size())
                    sentence_embedding = torch.sum(last_hidden_state * mask, 1) / mask.sum(1)
                    sentence_embeddings.append(sentence_embedding)

                if len(sentence_embeddings) < 2:
                    return 0.8

                fluency_scores = []
                for i in range(len(sentence_embeddings) - 1):
                    similarity = F.cosine_similarity(
                        sentence_embeddings[i],
                        sentence_embeddings[i + 1],
                        dim=1
                    )
                    fluency_scores.append(similarity.item())

                return np.mean(fluency_scores)

        except Exception as e:
            logger.error(f"评估流畅度时出错: {e}")
            return 0.5

    def _evaluate_grammar(self, text: str) -> float:
        """评估文本的语法正确性"""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.bert_model(**inputs)
                last_hidden_state = outputs.last_hidden_state

                attention_weights = torch.mean(last_hidden_state, dim=-1)
                attention_scores = torch.softmax(attention_weights, dim=-1)

                grammar_score = 1 - torch.std(attention_scores).item()

                return min(max(grammar_score, 0), 1)

        except Exception as e:
            logger.error(f"评估语法时出错: {e}")
            return 0.5

    def _evaluate_diversity(self, text: str) -> Dict[str, float]:
        """评估文本的多样性"""
        try:
            words = [w for w in jieba.cut(text) if w not in self.stopwords]

            if not words:
                return {
                    "distinct_1": 0,
                    "distinct_2": 0,
                    "distinct_3": 0,
                    "vocabulary_richness": 0,
                    "information_entropy": 0
                }

            distinct_scores = {}
            for n in [1, 2, 3]:
                ngramset = set(ngrams(words, n))
                ngramlist = list(ngrams(words, n))
                if ngramlist:
                    distinct_scores[f"distinct_{n}"] = len(ngramset) / len(ngramlist)
                else:
                    distinct_scores[f"distinct_{n}"] = 0

            vocabulary_richness = len(set(words)) / len(words)

            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            word_probs = [freq / len(words) for freq in word_freq.values()]
            information_entropy = entropy(word_probs)

            max_entropy = np.log2(len(word_freq))
            normalized_entropy = information_entropy / max_entropy if max_entropy > 0 else 0

            return {
                **distinct_scores,
                "vocabulary_richness": vocabulary_richness,
                "information_entropy": normalized_entropy
            }

        except Exception as e:
            logger.error(f"评估多样性时出错: {e}")
            return {
                "distinct_1": 0,
                "distinct_2": 0,
                "distinct_3": 0,
                "vocabulary_richness": 0,
                "information_entropy": 0
            }

    def _evaluate_relevance(self, text: str, prompt: str) -> float:
        """评估文本与提示词的相关性"""
        try:
            return self._calculate_semantic_similarity(text, prompt)
        except Exception as e:
            logger.error(f"评估相关性时出错: {e}")
            return 0.5

    def _evaluate_coverage(self, text: str, prompt: str) -> float:
        """评估文本对提示词关键信息的覆盖程度"""
        try:
            prompt_words = set(jieba.cut(prompt)) - self.stopwords
            text_words = set(jieba.cut(text)) - self.stopwords

            if not prompt_words:
                return 1.0

            coverage = len(prompt_words & text_words) / len(prompt_words)
            return coverage

        except Exception as e:
            logger.error(f"评估覆盖度时出错: {e}")
            return 0.5

    def _evaluate_consistency(self, text: str, prompt: str, context: str) -> float:
        """评估文本的一致性"""
        try:
            reference = prompt + " " + context if context else prompt
            return self._calculate_semantic_similarity(text, reference)
        except Exception as e:
            logger.error(f"评估一致性时出错: {e}")
            return 0.5

    def _evaluate_coherence(self, text: str) -> float:
        """评估文本的连贯性"""
        try:
            sentences = [s.strip() for s in text.split("。") if s.strip()]
            if len(sentences) < 2:
                return 1.0

            coherence_scores = []
            for i in range(len(sentences) - 1):
                similarity = self._calculate_semantic_similarity(
                    sentences[i],
                    sentences[i + 1]
                )
                coherence_scores.append(similarity)

            return np.mean(coherence_scores)

        except Exception as e:
            logger.error(f"评估连贯性时出错: {e}")
            return 0.5

    def _average_quality_metrics(
            self,
            metrics_list: List[QualityMetrics]
    ) -> QualityMetrics:
        """计算平均质量指标"""
        avg_diversity = defaultdict(float)
        for metric in metrics_list:
            for k, v in metric.diversity.items():
                avg_diversity[k] += v

        for k in avg_diversity:
            avg_diversity[k] /= len(metrics_list)

        return QualityMetrics(
            fluency=np.mean([m.fluency for m in metrics_list]),
            grammar=np.mean([m.grammar for m in metrics_list]),
            diversity=dict(avg_diversity),
            relevance=np.mean([m.relevance for m in metrics_list]),
            coverage=np.mean([m.coverage for m in metrics_list]),
            consistency=np.mean([m.consistency for m in metrics_list]),
            coherence=np.mean([m.coherence for m in metrics_list])
        )

    def _save_results(self, results: Dict, output_dir: str):
        """保存评测结果"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        result_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 生成评测报告
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        self._generate_report(results, report_file)

        logger.info(f"评测结果已保存到: {output_dir}")

    def _generate_report(self, results: Dict, report_file: str):
        """生成评测报告"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("基座模型评测报告\n")
            f.write("=" * 50 + "\n\n")

            # 1. 性能指标
            f.write("1. 性能指标\n")
            f.write("-" * 30 + "\n")
            perf = results["performance"]
            f.write(f"推理速度: {perf.inference_speed:.2f} ms/token\n")
            f.write(f"内存使用: {perf.memory_usage:.2f} GB\n")
            f.write(f"GPU内存: {perf.gpu_memory:.2f} GB\n")
            f.write(f"吞吐量: {perf.throughput:.2f} tokens/s\n")
            f.write(f"延迟: {perf.latency:.2f} ms\n\n")

            # 2. 参数评估
            f.write("2. 参数评估\n")
            f.write("-" * 30 + "\n")
            for param_key, score in results["final_score"].items():
                f.write(f"\n参数组合: {param_key}\n")
                f.write(f"总分: {score['final_score']:.3f}\n")
                f.write(f"性能得分: {score['performance_score']:.3f}\n")
                f.write(f"稳定性得分: {score['stability_score']:.3f}\n")
                f.write(f"质量得分: {score['quality_score']:.3f}\n")

            # 3. 最佳组合
            f.write("\n3. 最佳组合\n")
            f.write("-" * 30 + "\n")
            best_param = max(
                results["final_score"].items(),
                key=lambda x: x[1]["final_score"]
            )
            f.write(f"最佳参数组合: {best_param[0]}\n")
            f.write(f"最高得分: {best_param[1]['final_score']:.3f}\n")

            # 4. 建议
            f.write("\n4. 建议\n")
            f.write("-" * 30 + "\n")
            if best_param[1]["final_score"] >= 0.8:
                f.write("✓ 模型性能良好，建议采用\n")
            elif best_param[1]["final_score"] >= 0.6:
                f.write("△ 模型性能一般，可以考虑使用但建议进一步优化\n")
            else:
                f.write("✗ 模型性能不佳，建议选择其他模型\n")
