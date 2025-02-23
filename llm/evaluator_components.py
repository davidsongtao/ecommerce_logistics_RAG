"""
Description: 评估器专门组件

-*- Encoding: UTF-8 -*-
@File     ：evaluator_components.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
from nltk.util import ngrams
from typing import List, Dict, Any

from configs.config import config
from base_evaluator import BaseEvaluator
from evaluation_metrics import (
    PerformanceMetrics,
    StabilityMetrics,
    QualityMetrics
)


class PerformanceEvaluator:
    """性能评估器"""

    def __init__(self, base_evaluator: BaseEvaluator):
        self.base = base_evaluator
        self.logger = self.base.logger
        self.device = self.base.device

    def evaluate_performance(
            self,
            model,
            test_cases: List[Dict]
    ) -> PerformanceMetrics:
        """评估性能指标"""
        self.logger.info("开始评估性能指标...")
        start_time = time.time()
        total_tokens = 0
        total_latency = 0

        # 测试推理速度和吞吐量
        for case in test_cases:
            with torch.no_grad():
                # 记录生成开始时间
                gen_start = time.time()

                # 生成文本
                output = model.generate(case["prompt"])

                # 计算延迟
                latency = (time.time() - gen_start) * 1000  # 转换为毫秒
                total_latency += latency

                # 累计token数
                total_tokens += len(output.split())

        # 计算总耗时
        total_time = time.time() - start_time

        # 计算性能指标
        inference_speed = (total_time * 1000) / total_tokens  # ms/token
        throughput = total_tokens / total_time  # tokens/s
        avg_latency = total_latency / len(test_cases)

        # 获取内存使用情况
        memory_usage = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_memory = memory_usage

        self.logger.info("性能评估完成")
        return PerformanceMetrics(
            inference_speed=inference_speed,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            throughput=throughput,
            latency=avg_latency
        )


class StabilityEvaluator:
    """稳定性评估器"""

    def __init__(self, base_evaluator: BaseEvaluator):
        self.base = base_evaluator
        self.logger = self.base.logger
        self.device = self.base.device

    def evaluate_stability(
            self,
            model,
            test_cases: List[Dict],
            num_runs: int = 5
    ) -> Dict[str, List[StabilityMetrics]]:
        """评估参数稳定性"""
        self.logger.info("开始评估稳定性...")
        stability_results = {}

        for case in test_cases:
            outputs = []
            # 多次运行
            for _ in range(num_runs):
                try:
                    output = model.generate(case["prompt"])
                    outputs.append(output)
                except Exception as e:
                    self.logger.error(f"生成错误: {e}")
                    continue

            if outputs:
                # 计算稳定性指标
                stability_metric = self._calculate_stability_metrics(outputs)
                param_key = f"case_{len(stability_results)}"
                stability_results[param_key] = stability_metric

        self.logger.info("稳定性评估完成")
        return stability_results

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
                similarity = self.base.calculate_semantic_similarity(
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


class QualityEvaluator:
    """质量评估器"""

    def __init__(self, base_evaluator: BaseEvaluator):
        self.base = base_evaluator
        self.logger = self.base.logger
        self.device = self.base.device
        self.tokenizer = self.base.tokenizer
        self.bert_model = self.base.bert_model

    def evaluate_quality(
            self,
            model,
            test_cases: List[Dict]
    ) -> Dict[str, QualityMetrics]:
        """评估生成质量"""
        self.logger.info("开始评估质量...")
        quality_results = {}

        for case in test_cases:
            try:
                # 生成文本
                output = model.generate(case["prompt"])

                # 评估质量指标
                metrics = self._evaluate_text_quality(
                    text=output,
                    prompt=case["prompt"],
                    context=case.get("context", "")
                )
                param_key = f"case_{len(quality_results)}"
                quality_results[param_key] = metrics

            except Exception as e:
                self.logger.error(f"评估错误: {e}")
                continue

        self.logger.info("质量评估完成")
        return quality_results

    def _evaluate_text_quality(
            self,
            text: str,
            prompt: str,
            context: str = ""
    ) -> QualityMetrics:
        """评估文本质量的各项指标"""
        self.logger.debug(f"评估文本质量，文本长度: {len(text)}")

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

    def _evaluate_fluency(self, text: str) -> float:
        """评估文本流畅度"""
        try:
            sentences = self.base.split_sentences(text)
            if len(sentences) < 2:
                return 0.8  # 单句默认较高流畅度

            sentence_embeddings = []
            for sentence in sentences:
                embedding = self.base.get_sentence_embedding(sentence)
                sentence_embeddings.append(embedding)

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
            self.logger.error(f"评估流畅度时出错: {e}")
            return 0.5

    def _evaluate_grammar(self, text: str) -> float:
        """评估文本的语法正确性"""
        try:
            # 使用BERT的注意力分数评估语法
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.bert_model(**inputs)
                last_hidden_state = outputs.last_hidden_state

                # 计算注意力权重的均匀性作为语法得分
                attention_weights = torch.mean(last_hidden_state, dim=-1)
                attention_scores = torch.softmax(attention_weights, dim=-1)
                grammar_score = 1 - torch.std(attention_scores).item()

                return min(max(grammar_score, 0), 1)

        except Exception as e:
            self.logger.error(f"评估语法时出错: {e}")
            return 0.5

    def _evaluate_diversity(self, text: str) -> Dict[str, float]:
        """评估文本的多样性"""
        try:
            words = self.base.cut_words(text)

            if not words:
                return {
                    "distinct_1": 0,
                    "distinct_2": 0,
                    "distinct_3": 0,
                    "vocabulary_richness": 0,
                    "information_entropy": 0
                }

            # 计算n-gram多样性
            distinct_scores = {}
            for n in [1, 2, 3]:
                ngramset = set(ngrams(words, n))
                ngramlist = list(ngrams(words, n))
                if ngramlist:
                    distinct_scores[f"distinct_{n}"] = len(ngramset) / len(ngramlist)
                else:
                    distinct_scores[f"distinct_{n}"] = 0

            # 计算词汇丰富度
            vocabulary_richness = len(set(words)) / len(words)

            # 计算信息熵
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            word_probs = [freq / len(words) for freq in word_freq.values()]
            information_entropy = entropy(word_probs)

            # 归一化熵值
            max_entropy = np.log2(len(word_freq))
            normalized_entropy = information_entropy / max_entropy if max_entropy > 0 else 0

            return {
                **distinct_scores,
                "vocabulary_richness": vocabulary_richness,
                "information_entropy": normalized_entropy
            }

        except Exception as e:
            self.logger.error(f"评估多样性时出错: {e}")
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
            return self.base.calculate_semantic_similarity(text, prompt)
        except Exception as e:
            self.logger.error(f"评估相关性时出错: {e}")
            return 0.5

    def _evaluate_coverage(self, text: str, prompt: str) -> float:
        """评估文本对提示词关键信息的覆盖程度"""
        try:
            prompt_words = set(self.base.cut_words(prompt))
            text_words = set(self.base.cut_words(text))

            if not prompt_words:
                return 1.0

            coverage = len(prompt_words & text_words) / len(prompt_words)
            return coverage

        except Exception as e:
            self.logger.error(f"评估覆盖度时出错: {e}")
            return 0.5

    def _evaluate_consistency(self, text: str, prompt: str, context: str) -> float:
        """评估文本的一致性"""
        try:
            reference = prompt + " " + context if context else prompt
            return self.base.calculate_semantic_similarity(text, reference)
        except Exception as e:
            self.logger.error(f"评估一致性时出错: {e}")
            return 0.5

    def _evaluate_coherence(self, text: str) -> float:
        """评估文本的连贯性"""
        try:
            sentences = self.base.split_sentences(text)
            if len(sentences) < 2:
                return 1.0

            coherence_scores = []
            for i in range(len(sentences) - 1):
                similarity = self.base.calculate_semantic_similarity(
                    sentences[i],
                    sentences[i + 1]
                )
                coherence_scores.append(similarity)

            return np.mean(coherence_scores)

        except Exception as e:
            self.logger.error(f"评估连贯性时出错: {e}")
            return 0.5
