"""
Description: 自定义大模型
    
-*- Encoding: UTF-8 -*-
@File     ：model.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午12:52
@Contact  ：king.songtao@gmail.com
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from typing import Optional, List


class DeepSeekLLM(LLM):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self._device = device
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(self._device)

        # 显式设置 pad_token_id 和 eos_token_id
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = self._model.config.eos_token_id

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                num_beams=5,
                repetition_penalty=1.2,
                do_sample=True,
                early_stopping=True
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "deepseek_r1_distill_qwen_1.5b"


if __name__ == '__main__':
    # 使用自定义的 LLM 类
    model_name = r"D:\ecommerce_logistics_RAG\models\DeepSeek_R1_Distill_Qwen_1_5B"  # 替换为你的本地模型路径
    llm = DeepSeekLLM(model_name=model_name, device="cuda")

    # 进行推理
    prompt = """你好。"""
    response = llm.invoke(prompt)
    print(response)
