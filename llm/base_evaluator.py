"""
Description: 评估器基础组件

-*- Encoding: UTF-8 -*-
@File     ：base_evaluator.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

import jieba
import torch
import torch.nn.functional as F
from typing import Set, List
from transformers import AutoTokenizer, AutoModel

from configs.config import config
from configs.log_config import get_logger
from exceptions import ModelLoadError


class BaseEvaluator:
    """评估器基础类"""

    def __init__(self, bert_model_name: str = "bert-base-chinese"):
        """
        初始化评估器基础组件

        Args:
            bert_model_name: BERT模型名称
        """
        self.logger = get_logger("evaluator")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化BERT模型
        try:
            self.logger.info(f"正在加载BERT模型: {bert_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_model = AutoModel.from_pretrained(bert_model_name).to(self.device)
            self.bert_model.eval()
        except Exception as e:
            raise ModelLoadError(
                message=f"BERT模型加载失败: {str(e)}",
                model_path=bert_model_name,
                device=self.device,
                memory_info={}
            )

        # 初始化jieba分词
        jieba.initialize()

        # 加载停用词
        self.stopwords = self._load_stopwords()

        self.logger.info("评估器基础组件初始化完成")

    def _load_stopwords(self) -> Set[str]:
        """加载停用词表"""
        return {
            "的", "了", "和", "是", "在", "我", "有", "就",
            "不", "也", "这", "着", "那", "就是", "但是", "所以"
        }

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
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
            self.logger.error(f"计算语义相似度错误: {e}")
            return 0.0

    def get_sentence_embedding(self, text: str) -> torch.Tensor:
        """获取文本的句子嵌入"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            outputs = self.bert_model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            )
            return torch.sum(
                outputs.last_hidden_state * mask, 1
            ) / mask.sum(1)

    def split_sentences(self, text: str) -> List[str]:
        """分割文本为句子"""
        return [s.strip() for s in text.split("。") if s.strip()]

    def cut_words(self, text: str) -> List[str]:
        """分词并移除停用词"""
        return [w for w in jieba.cut(text) if w not in self.stopwords]

    def __del__(self):
        """清理资源"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")
