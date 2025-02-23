"""
Description: 本地大语言模型封装类

-*- Encoding: UTF-8 -*-
@File     ：local_model.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午12:31
@Contact  ：king.songtao@gmail.com
"""

import os
import sys
import psutil
import warnings
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Generator, Optional, Dict, Any, Union

# 将项目根目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from configs.log_config import get_logger, configure_logging
from llm.exceptions import ModelLoadError, ModelGenerateError, ModelResourceError
from configs.config import default_config as cfg

warnings.filterwarnings("ignore")


class LocalLLM:
    def __init__(
            self,
            model_path: str,
            device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None,
            show_log: bool = True,
            **kwargs
    ):
        """
        初始化本地大语言模型

        Args:
            model_path: 模型路径
            device: 设备类型 ('cpu', 'cuda', 'cuda:0' 等)
            dtype: 模型精度类型
            show_log: 是否显示日志
            **kwargs: 其他参数，如生成配置
        """
        # 基础配置
        self.model_path = model_path
        self.device = device or cfg.model.device
        self.dtype = dtype or cfg.model.dtype

        # 生成配置
        self.generation_config = cfg.model.generation_config.copy()
        self.generation_config.update(kwargs)

        # 配置日志
        configure_logging(show_log)
        self.logger = get_logger("model", show_log=show_log)

        try:
            if show_log:
                self.logger.info(f"初始化LocalLLM，模型路径: {model_path}")
                self._log_system_info()

            # 加载模型
            self._load_model()

        except Exception as e:
            error_msg = f"模型初始化失败: {str(e)}"
            self.logger.error(error_msg)
            raise ModelLoadError(
                message=error_msg,
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

        system_info = {
            "CPU核心数": psutil.cpu_count(),
            "总内存": f"{psutil.virtual_memory().total / 1e9:.2f}GB",
            "可用内存": f"{psutil.virtual_memory().available / 1e9:.2f}GB"
        }
        self.logger.info(f"系统信息: {system_info}")

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
            if allocated / total > cfg.model.gpu_memory_threshold:
                raise ModelResourceError(
                    message="GPU内存使用率过高",
                    resource_type="gpu_memory",
                    resource_info=self._get_memory_info()
                )

        # 检查系统内存
        vm = psutil.virtual_memory()
        available_memory_percent = vm.available / vm.total
        if available_memory_percent < cfg.model.cpu_memory_threshold:
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
            error_msg = f"模型加载错误: {str(e)}"
            self.logger.error(error_msg)
            raise ModelLoadError(
                message=error_msg,
                model_path=self.model_path,
                device=self.device,
                memory_info=self._get_memory_info()
            ) from e

    def generate_stream(
            self,
            prompt: str,
            **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成回复

        Args:
            prompt: 输入提示
            **kwargs: 生成参数，会覆盖默认配置

        Yields:
            生成的文本片段
        """
        self.logger.debug(f"生成流式响应，提示: {prompt[:50]}...")

        try:
            # 检查资源
            self._check_resources()

            # 更新生成参数
            generation_config = self.generation_config.copy()
            generation_config.update(kwargs)

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
                **generation_config
            }
            gen_kwargs.update(inputs)  # 将inputs合并到gen_kwargs中

            # 在后台线程中运行生成
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

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

                gen_kwargs.update(continue_inputs)
                gen_kwargs["max_new_tokens"] = 512

                thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
                thread.start()

                for new_text in streamer:
                    yield new_text

        except Exception as e:
            error_msg = f"流式生成错误: {str(e)}"
            self.logger.error(error_msg)
            raise ModelGenerateError(
                message=error_msg,
                prompt=prompt,
                generation_config=generation_config,
                generation_info=self._get_memory_info()
            ) from e
        finally:
            thread.join()

    def generate(
            self,
            prompt: str,
            stream: bool = False,
            **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        生成回复的便捷方法

        Args:
            prompt: 输入提示
            stream: 是否使用流式输出
            **kwargs: 生成参数

        Returns:
            如果 stream=True，返回生成器；否则返回完整回复字符串

        Raises:
            ModelGenerateError: 生成过程中的错误
            ModelResourceError: 资源不足错误
        """
        self.logger.info(f"生成响应，提示: {prompt[:50]}...")
        try:
            if stream:
                return self.generate_stream(prompt, **kwargs)
            else:
                return "".join(self.generate_stream(prompt, **kwargs))
        except Exception as e:
            error_msg = f"生成错误: {str(e)}"
            self.logger.error(error_msg)
            raise ModelGenerateError(
                message=error_msg,
                prompt=prompt,
                generation_config=kwargs,
                generation_info=self._get_memory_info()
            ) from e

    def __del__(self):
        """清理资源"""
        try:
            self.logger.info("清理模型资源...")
            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"清理过程错误: {str(e)}")
