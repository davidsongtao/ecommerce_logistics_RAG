"""
Description: 
    
-*- Encoding: UTF-8 -*-
@File     ：base_llm_eval.py
@Author   ：King Songtao
@Time     ：2025/2/23 上午9:22
@Contact  ：king.songtao@gmail.com
"""
"""
Description: 专业的LLM模型评测实现
主要评估维度:
1. 语言质量评估
2. 文本多样性评估
3. 主题相关性评估
4. 一致性与连贯性评估
"""

import os
import json
import jieba
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.stats import entropy
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
import torch.nn.functional as F


class ProfessionalEvaluator:
    """专业的模型评测器"""

    def __init__(self, bert_model_name: str = "bert-base-chinese"):
        """
        初始化评测器

        Args:
            bert_model_name: BERT模型名称
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化BERT模型
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.bert_model.eval()

        # 初始化jieba分词
        jieba.initialize()

        # 加载停用词
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> Set[str]:
        """加载停用词表"""
        stopwords = set()
        try:
            # 这里可以加载你的自定义停用词表
            stopwords = {"的", "了", "和", "是", "在", "我", "有", "就", "不", "也", "这", "着"}
        except Exception as e:
            print(f"加载停用词失败: {e}")
        return stopwords

    def evaluate_text(self, text: str, prompt: str, context: str = "") -> Dict[str, float]:
        """
        全面评估生成文本的质量

        Args:
            text: 生成的文本
            prompt: 原始提示词
            context: 上下文信息(可选)

        Returns:
            包含各项评分的字典
        """
        # 1. 语言质量评估
        fluency_score = self._evaluate_fluency(text)
        grammar_score = self._evaluate_grammar(text)

        # 2. 文本多样性评估
        diversity_scores = self._evaluate_diversity(text)

        # 3. 主题相关性评估
        relevance_score = self._evaluate_relevance(text, prompt)
        coverage_score = self._evaluate_coverage(text, prompt)

        # 4. 一致性评估
        consistency_score = self._evaluate_consistency(text, prompt, context)
        coherence_score = self._evaluate_coherence(text)

        # 5. 计算加权总分
        total_score = self._calculate_weighted_score({
            "fluency": fluency_score,
            "grammar": grammar_score,
            "diversity": np.mean(list(diversity_scores.values())),
            "relevance": relevance_score,
            "coverage": coverage_score,
            "consistency": consistency_score,
            "coherence": coherence_score
        })

        return {
            "total_score": total_score,
            "fluency": fluency_score,
            "grammar": grammar_score,
            "diversity": diversity_scores,
            "relevance": relevance_score,
            "coverage": coverage_score,
            "consistency": consistency_score,
            "coherence": coherence_score
        }

    def _evaluate_fluency(self, text: str) -> float:
        """
        评估文本流畅度

        使用BERT模型计算困惑度(perplexity)，评估语言模型对文本的预测概率
        """
        try:
            with torch.no_grad():
                # 对文本进行分句
                sentences = [s for s in text.split("。") if s.strip()]

                perplexities = []
                for sentence in sentences:
                    inputs = self.tokenizer(
                        sentence,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    outputs = self.bert_model(**inputs)
                    logits = outputs.logits

                    # 计算困惑度
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()

                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)

                # 将困惑度转换为0-1分数(困惑度越低越好)
                avg_perplexity = np.mean(perplexities)
                fluency_score = 1 / (1 + avg_perplexity)

                return min(max(fluency_score, 0), 1)  # 确保分数在0-1之间

        except Exception as e:
            print(f"评估流畅度时出错: {e}")
            return 0.5

    def _evaluate_grammar(self, text: str) -> float:
        """
        评估文本的语法正确性

        使用BERT模型的预测概率来评估语法正确性
        """
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.bert_model(**inputs)

                # 使用BERT的预测来评估语法
                prediction_scores = outputs.prediction_logits
                confidence_scores = torch.softmax(prediction_scores, dim=-1).max(dim=-1)[0]

                # 计算平均置信度
                grammar_score = confidence_scores.mean().item()

                return min(max(grammar_score, 0), 1)

        except Exception as e:
            print(f"评估语法时出错: {e}")
            return 0.5

    def _evaluate_diversity(self, text: str) -> Dict[str, float]:
        """
        评估文本的多样性

        计算:
        1. Distinct-n scores (n=1,2,3)
        2. 词汇丰富度
        3. 信息熵
        """
        try:
            # 分词
            words = [w for w in jieba.cut(text) if w not in self.stopwords]

            if not words:
                return {
                    "distinct_1": 0,
                    "distinct_2": 0,
                    "distinct_3": 0,
                    "vocabulary_richness": 0,
                    "information_entropy": 0
                }

            # 计算distinct-n
            distinct_scores = {}
            for n in [1, 2, 3]:
                ngramset = set(ngrams(words, n))
                ngramlist = list(ngrams(words, n))
                if ngramlist:
                    distinct_scores[f"distinct_{n}"] = len(ngramset) / len(ngramlist)
                else:
                    distinct_scores[f"distinct_{n}"] = 0

            # 计算词汇丰富度(Type-Token Ratio)
            vocabulary_richness = len(set(words)) / len(words)

            # 计算信息熵
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            word_probs = [freq / len(words) for freq in word_freq.values()]
            information_entropy = entropy(word_probs)

            # 归一化信息熵
            max_entropy = np.log2(len(word_freq))
            normalized_entropy = information_entropy / max_entropy if max_entropy > 0 else 0

            return {
                **distinct_scores,
                "vocabulary_richness": vocabulary_richness,
                "information_entropy": normalized_entropy
            }

        except Exception as e:
            print(f"评估多样性时出错: {e}")
            return {
                "distinct_1": 0,
                "distinct_2": 0,
                "distinct_3": 0,
                "vocabulary_richness": 0,
                "information_entropy": 0
            }

    def _evaluate_relevance(self, text: str, prompt: str) -> float:
        """
        评估文本与提示词的相关性

        使用BERT计算语义相似度
        """
        try:
            with torch.no_grad():
                # 编码文本和提示词
                inputs = self.tokenizer(
                    [text, prompt],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记的embedding

                # 计算余弦相似度
                similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)

                return similarity.item()

        except Exception as e:
            print(f"评估相关性时出错: {e}")
            return 0.5

    def _evaluate_coverage(self, text: str, prompt: str) -> float:
        """
        评估文本对提示词关键信息的覆盖程度
        """
        try:
            # 提取提示词中的关键词
            prompt_words = set(jieba.cut(prompt)) - self.stopwords
            text_words = set(jieba.cut(text)) - self.stopwords

            if not prompt_words:
                return 1.0

            # 计算覆盖率
            coverage = len(prompt_words & text_words) / len(prompt_words)

            return coverage

        except Exception as e:
            print(f"评估覆盖度时出错: {e}")
            return 0.5

    def _evaluate_consistency(self, text: str, prompt: str, context: str) -> float:
        """
        评估文本的一致性
        包括与提示词和上下文的一致性
        """
        try:
            # 如果有上下文，将其加入评估
            reference = prompt + " " + context if context else prompt

            with torch.no_grad():
                inputs = self.tokenizer(
                    [text, reference],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]

                consistency = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)

                return consistency.item()

        except Exception as e:
            print(f"评估一致性时出错: {e}")
            return 0.5

    def _evaluate_coherence(self, text: str) -> float:
        """
        评估文本的连贯性
        分析句子间的语义连贯性
        """
        try:
            sentences = [s.strip() for s in text.split("。") if s.strip()]
            if len(sentences) < 2:
                return 1.0

            with torch.no_grad():
                # 计算相邻句子间的语义相似度
                coherence_scores = []
                for i in range(len(sentences) - 1):
                    inputs = self.tokenizer(
                        [sentences[i], sentences[i + 1]],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    outputs = self.bert_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
                    coherence_scores.append(similarity.item())

                return np.mean(coherence_scores)

        except Exception as e:
            print(f"评估连贯性时出错: {e}")
            return 0.5

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        计算加权总分

        权重分配:
        - 流畅度: 15%
        - 语法: 15%
        - 多样性: 15%
        - 相关性: 20%
        - 覆盖度: 15%
        - 一致性: 10%
        - 连贯性: 10%
        """
        weights = {
            "fluency": 0.15,
            "grammar": 0.15,
            "diversity": 0.15,
            "relevance": 0.20,
            "coverage": 0.15,
            "consistency": 0.10,
            "coherence": 0.10
        }

        total_score = sum(score * weights[metric] for metric, score in scores.items())

        return min(max(total_score, 0), 1)  # 确保分数在0-1之间
