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
from langchain_community.llms.utils import enforce_stop_tokens


class DeepSeek(LLM):
    max_token: int = 4096
    temperature: float = 0.8
    top_p: float = 0.9
    tokenizer: object = None
    model: object = None
    history: List[List[Optional[str]]] = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "DeepSeek-R1-Distill-Qwen-32B"

    # 定义load_model方法，进行模型的加载
    def load_model(self, model_path: Optional[str] = None):
        """
        加载模型

        Args:
            model_path (Optional[str], optional): 模型路径. Defaults to None.

        Raises:
            TypeError: 模型路径不是字符串
            FileNotFoundError: 模型路径不存在
        """
        # 检查模型路径是否为None或空字符串
        if model_path is None:
            logger.error("模型路径不能为None")
            raise TypeError("模型路径不能为None")

        # 检查模型路径是否为字符串
        if not isinstance(model_path, str):
            logger.error(f"模型路径应为字符串，当前类型为：{type(model_path)}")
            raise TypeError(f"模型路径应为字符串，当前类型为：{type(model_path)}")

        # 检查模型路径是否为空字符串
        if model_path.strip() == "":
            logger.error("模型路径不能为空")
            raise FileNotFoundError("模型路径不能为空")

        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型路径不存在：{model_path}")
            raise FileNotFoundError(f"模型路径不存在：{model_path}")

        try:
            # 记录模型加载信息
            logger.info(f"开始加载模型：{model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # 使用半精度
                device_map='auto'  # 自动分配显存
            ).cuda()
            logger.info(f"模型加载成功：{model_path}")
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
