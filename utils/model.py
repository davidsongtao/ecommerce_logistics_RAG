"""
Description: 本地大语言模型封装类

-*- Encoding: UTF-8 -*-
@File     ：model.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午12:31
@Contact  ：king.songtao@gmail.com
"""

import os
import sys

# 将项目根目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import warnings
from typing import Generator, Optional, Dict, Any, Union
import time
import psutil
from configs.log_config import get_logger, configure_logging
from utils.exceptions import ModelError, ModelLoadError, ModelGenerateError, ModelResourceError

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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.bfloat16

        # 生成配置
        self.generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 2048),
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.8),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }

        # 配置日志显示
        configure_logging(show_log)
        self.logger = get_logger("model", show_log=show_log)

        try:
            if show_log:
                self.logger.info(f"Initializing LocalLLM with model path: {model_path}")
                self._log_system_info()

            # 加载模型
            self._load_model()

        except Exception as e:
            error_msg = f"Failed to initialize model: {str(e)}"
            self.logger.error(error_msg)
            raise ModelLoadError(
                message=error_msg,
                model_path=model_path,
                device=self.device,
                memory_info=self._get_memory_info()
            ) from e

    def _log_system_info(self):
        """记录系统信息"""
        self.logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            self.logger.info(f"CUDA current device: {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                self.logger.info(f"CUDA device {i}: {prop.name}, {prop.total_memory / 1e9:.2f}GB memory")

        system_info = {
            "CPU cores": psutil.cpu_count(),
            "Memory total": f"{psutil.virtual_memory().total / 1e9:.2f}GB",
            "Memory available": f"{psutil.virtual_memory().available / 1e9:.2f}GB"
        }
        self.logger.info(f"System info: {system_info}")

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
        if self.device == "cuda":
            device_id = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            if allocated / total > 0.9:  # GPU 使用率超过 90%
                raise ModelResourceError(
                    message="GPU memory usage too high",
                    resource_type="gpu_memory",
                    resource_info=self._get_memory_info()
                )

        if psutil.virtual_memory().percent > 90:  # CPU 内存使用率超过 90%
            raise ModelResourceError(
                message="System memory usage too high",
                resource_type="system_memory",
                resource_info=self._get_memory_info()
            )

    def _load_model(self):
        """加载模型和分词器"""
        try:
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='right'
            )

            if self.tokenizer.pad_token is None:
                self.logger.debug("Setting pad_token...")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            # 检查资源
            self._check_resources()

            self.logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

            self.logger.debug("Resizing token embeddings...")
            self.model.resize_token_embeddings(len(self.tokenizer))

            if self.device != "cuda":
                self.logger.debug(f"Moving model to device: {self.device}")
                self.model = self.model.to(self.device)

            self.model.eval()
            self.logger.info("Model loaded successfully")

        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
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

        Raises:
            ModelGenerateError: 生成过程中的错误
            ModelResourceError: 资源不足错误
        """
        self.logger.debug(f"Generating stream response for prompt: {prompt[:50]}...")

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
                **inputs,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **generation_config
            }

            # 在后台线程中运行生成
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            full_response = ""
            for new_text in streamer:
                full_response += new_text
                yield new_text

            # 检查回答是否完整
            if not full_response.strip().endswith(("。", "!", "?", "！", "？", ".", "…")):
                gen_kwargs["inputs"] = self.tokenizer(
                    full_response,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.model.device)
                gen_kwargs["max_new_tokens"] = 512

                thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
                thread.start()

                for new_text in streamer:
                    yield new_text

        except Exception as e:
            error_msg = f"Error during stream generation: {str(e)}"
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
        self.logger.info(f"Generating response for prompt: {prompt[:50]}...")
        try:
            if stream:
                return self.generate_stream(prompt, **kwargs)
            else:
                return "".join(self.generate_stream(prompt, **kwargs))
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
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
            self.logger.info("Cleaning up model resources...")
            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    try:
        # 测试代码
        model_path = "/root/autodl-tmp/ecommerce_logistics_RAG/models/DeepSeek_R1_Distill_Qwen_7B"
        llm = LocalLLM(model_path)

        prompt = "你好，地球是圆的么？"
        print(f"User: {prompt}")
        print("Assistant: ", end="", flush=True)

        # 流式输出测试
        for text in llm.generate_stream(prompt):
            print(text, end="", flush=True)
            sys.stdout.flush()
            time.sleep(0.02)
        print("\n")

    except ModelLoadError as e:
        print(f"Model failed to load: {e}")
        print(f"Details: {e.to_dict()}")

    except ModelGenerateError as e:
        print(f"Generation failed: {e}")
        print(f"Details: {e.to_dict()}")

    except ModelResourceError as e:
        print(f"Resource error: {e}")
        print(f"Details: {e.to_dict()}")

    except Exception as e:
        print(f"Unexpected error: {e}")
