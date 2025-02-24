"""
Description: 本地大语言模型封装类

-*- Encoding: UTF-8 -*-
@File     ：local_llm.py
@Author   ：King Songtao
@Time     ：2025/2/23
@Contact  ：king.songtao@gmail.com
"""

import psutil
import torch
from threading import Thread
from typing import Generator, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from configs.config import config
from configs.log_config import get_logger
from llm.exceptions import ModelLoadError, ModelGenerateError, ModelResourceError


class LocalLLM:
    """本地大语言模型封装类"""

    def __init__(
            self,
            model_path: str,
            device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        """
        初始化本地大语言模型

        Args:
            model_path: 模型路径
            device: 设备类型 ('cpu', 'cuda', 'cuda:0' 等)
            dtype: 模型精度类型
        """
        self.logger = None

        try:
            self.logger = get_logger("model")
            self.model_path = model_path
            self.device = device or config.model.device
            self.dtype = dtype or config.model.dtype

            self.logger.info(f"初始化LocalLLM，模型路径: {model_path}")
            self._log_system_info()
            self._load_model()
        except Exception as e:
            if self.logger:
                self.logger.error(f"模型初始化失败: {str(e)}")
            raise ModelLoadError(
                message=f"模型初始化失败: {str(e)}",
                model_path=model_path,
                device=self.device,
                memory_info=self._get_memory_info()
            ) from e

    def _log_system_info(self):
        """记录系统信息"""
        self.logger.info(f"使用设备: {self.device}")
        if self.device == "cuda":
            self.logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
            self.logger.info(f"当前CUDA设备: {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                self.logger.info(f"CUDA设备 {i}: {prop.name}, {prop.total_memory / 1e9:.2f}GB 内存")

        self.logger.info(
            "系统信息: "
            f"CPU核心数={psutil.cpu_count()}, "
            f"总内存={psutil.virtual_memory().total / 1e9:.2f}GB, "
            f"可用内存={psutil.virtual_memory().available / 1e9:.2f}GB"
        )

    def _get_memory_info(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        memory_info = {
            "cpu_memory": {
                "total": f"{psutil.virtual_memory().total / 1e9:.2f}GB",
                "available": f"{psutil.virtual_memory().available / 1e9:.2f}GB",
                "percent": psutil.virtual_memory().percent
            }
        }

        if self.device == "cuda":
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory
                reserved = torch.cuda.memory_reserved(i)
                allocated = torch.cuda.memory_allocated(i)
                gpu_info[f"gpu_{i}"] = {
                    "total": f"{total / 1e9:.2f}GB",
                    "reserved": f"{reserved / 1e9:.2f}GB",
                    "allocated": f"{allocated / 1e9:.2f}GB",
                    "percent": f"{(allocated / total) * 100:.2f}%"
                }
            memory_info["gpu_memory"] = gpu_info

        return memory_info

    def _check_resources(self):
        """检查资源使用情况"""
        # 检查GPU资源
        if self.device == "cuda":
            device_id = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            if allocated / total > config.model.gpu_memory_threshold:
                raise ModelResourceError(
                    message="GPU内存使用率过高",
                    resource_type="gpu_memory",
                    resource_info=self._get_memory_info()
                )

        # 检查系统内存
        vm = psutil.virtual_memory()
        available_memory_percent = vm.available / vm.total
        if available_memory_percent < config.model.cpu_memory_threshold:
            raise ModelResourceError(
                message=f"系统可用内存不足(当前可用: {available_memory_percent:.1%})",
                resource_type="system_memory",
                resource_info=self._get_memory_info()
            )

    def _load_model(self):
        """加载模型和分词器"""
        try:
            self.logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='right'
            )

            if self.tokenizer.pad_token is None:
                self.logger.debug("设置pad_token...")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            # 检查资源
            self._check_resources()

            self.logger.info("加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

            self.logger.debug("调整token嵌入大小...")
            self.model.resize_token_embeddings(len(self.tokenizer))

            if self.device != "cuda":
                self.logger.debug(f"将模型移动到设备: {self.device}")
                self.model = self.model.to(self.device)

            self.model.eval()
            self.logger.info("模型加载成功")

        except Exception as e:
            raise ModelLoadError(
                message=f"模型加载错误: {str(e)}",
                model_path=self.model_path,
                device=self.device,
                memory_info=self._get_memory_info()
            ) from e

    def generate(
            self,
            prompt: str,
            stream: bool = False,
            **kwargs
    ) -> str | Generator[str, None, None]:
        """
        生成响应

        Args:
            prompt: 输入提示
            stream: 是否使用流式输出
            **kwargs: 生成参数

        Returns:
            如果 stream=True，返回生成器；否则返回完整响应字符串

        Raises:
            ModelGenerateError: 生成过程中的错误
            ModelResourceError: 资源不足错误
        """
        self.logger.info(f"生成响应，提示: {prompt[:50]}...")
        try:
            # 检查资源
            self._check_resources()

            # 合并生成参数
            generation_config = {
                **config.model.generation_params,
                **kwargs
            }

            # 构造输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            # 创建 streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
                timeout=10.0
            )

            # 生成参数
            gen_kwargs = {
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **generation_config,
                **inputs
            }

            # 在后台线程中运行生成
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            if stream:
                return self._stream_output(streamer, thread)
            else:
                return "".join(self._stream_output(streamer, thread))

        except Exception as e:
            raise ModelGenerateError(
                message=f"生成错误: {str(e)}",
                prompt=prompt,
                generation_config=kwargs,
                generation_info=self._get_memory_info()
            ) from e

    def _stream_output(
            self,
            streamer: TextIteratorStreamer,
            thread: Thread
    ) -> Generator[str, None, None]:
        """处理流式输出"""
        try:
            full_response = ""
            for new_text in streamer:
                full_response += new_text
                yield new_text

            # 检查回答是否完整
            if not full_response.strip().endswith(("。", "!", "?", "！", "？", ".", "…")):
                # 构造后续生成的输入
                continue_inputs = self.tokenizer(
                    full_response,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.model.device)

                gen_kwargs = {
                    **continue_inputs,
                    "max_new_tokens": 512,
                    "streamer": streamer
                }

                thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
                thread.start()

                for new_text in streamer:
                    yield new_text

        finally:
            thread.join()

    def __del__(self):
        """清理资源"""
        try:
            # 检查 logger 是否存在
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("清理模型资源...")
            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            # 只在 logger 存在时记录错误
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"清理过程错误: {str(e)}")
