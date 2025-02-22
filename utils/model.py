"""
Description: 自定义大模型
    
-*- Encoding: UTF-8 -*-
@File     ：model.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午12:52
@Contact  ：king.songtao@gmail.com
"""
import os
from typing import Optional, List
from configs.log_config import logger, get_logger
import torch
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
from langchain.llms.utils import enforce_stop_tokens


class DeepSeek(LLM):
    max_token: int = 4096
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history = []

    def __init__(self):
        super().__init__()

    @property
    def llm_type(self) -> str:
        return "DeepSeek-R1-Distill-Qwen-32B"

    # 定义load_model方法，进行模型的加载
    def load_model(self, model_path=None):
        # 检查模型路径是否存在
        try:
            # 检查模型路径是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在：{model_path}")
                raise FileNotFoundError(f"模型路径不存在：{model_path}")
            # 记录模型加载信息
            logger.info(f"开始加载模型：{model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # 使用半精度
                device_map='auto'  # 自动分配显存
            ).cuda()
            logger.success(f"模型加载成功：{model_path}")
        except Exception as e:
            logger.error(f"模型加载失败：{model_path}")
            logger.error(f"错误信息：{str(e)}")
            raise

    # 定义_call方法，进行模型的推理
    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        try:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=self.history,
                max_length=self.max_token,
                top_p=self.top_p,
                temperature=self.temperature,
            )

            if stop is not None:
                response = enforce_stop_tokens(response, stop)

            self.history = self.history + [[None, response]]
            return response
        except Exception as e:
            logger.error(f"模型推理失败：{prompt}")
            raise
