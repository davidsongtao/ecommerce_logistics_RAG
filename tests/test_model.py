"""
Description: 对/utils/model.py脚本进行测试
    
-*- Encoding: UTF-8 -*-
@File     ：test_model.py.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午1:51
@Contact  ：king.songtao@gmail.com
"""
"""
Description: DeepSeek模型测试文件

-*- Encoding: UTF-8 -*-
@File     ：test_model.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午12:52
@Contact  ：king.songtao@gmail.com
"""

import os
import pytest
import torch
from typing import Optional
from utils.model import DeepSeek

# 配置测试模型路径（请替换为实际可用的模型路径）
MODEL_PATH = os.environ.get(
    'DEEPSEEK_MODEL_PATH',
    'models/DeepSeek_R1_Distill_Qwen_32B'
)


# 辅助函数：检查模型路径是否可用
def is_model_path_valid(path: str) -> bool:
    """
    检查模型路径是否有效

    :param path: 模型路径
    :return: 路径是否有效
    """
    return os.path.exists(path) and os.path.isdir(path)


@pytest.fixture(scope="module")
def deepseek_model():
    """
    模型实例fixture，确保每个测试模块只加载一次模型
    """
    if not is_model_path_valid(MODEL_PATH):
        pytest.skip(f"无效的模型路径：{MODEL_PATH}")

    model = DeepSeek()
    model.load_model(MODEL_PATH)
    return model


def test_model_initialization():
    """
    测试模型初始化参数
    """
    model = DeepSeek()

    # 检查默认参数
    assert model.max_token == 4096, "默认最大token数不正确"
    assert model.temperature == 0.8, "默认温度参数不正确"
    assert model.top_p == 0.9, "默认top_p参数不正确"
    assert model.history == [], "初始对话历史应为空"


def test_load_model_invalid_path():
    """
    测试加载无效模型路径时的异常处理
    """
    model = DeepSeek()
    invalid_paths = [
        "/path/to/non_existent_model",
        "",
        None
    ]

    for path in invalid_paths:
        with pytest.raises((FileNotFoundError, TypeError),
                           message=f"应对无效路径 {path} 抛出异常"):
            model.load_model(path)


def test_model_inference(deepseek_model):
    """
    测试模型推理基本功能

    :param deepseek_model: 模型实例fixture
    """
    # 测试用例
    test_prompts = [
        "你好",
        "介绍下人工智能",
        "解释量子计算",
        "讲一个有趣的故事"
    ]

    for prompt in test_prompts:
        # 推理
        response = deepseek_model._call(prompt)

        # 检查推理结果
        assert response is not None, f"prompt '{prompt}' 推理结果不应为None"
        assert isinstance(response, str), f"推理结果应为字符串，实际为 {type(response)}"
        assert len(response) > 0, f"prompt '{prompt}' 推理结果不应为空"


def test_conversation_history(deepseek_model):
    """
    测试对话历史管理

    :param deepseek_model: 模型实例fixture
    """
    # 清空历史
    deepseek_model.history = []

    # 多轮对话测试
    conversation = [
        "你好，今天心情如何？",
        "能给我讲个笑话吗？",
        "我们聊点serious的话题"
    ]

    responses = []
    for prompt in conversation:
        response = deepseek_model._call(prompt)
        responses.append(response)

    # 检查历史记录
    assert len(deepseek_model.history) == len(conversation), "对话历史记录长度不匹配"

    # 检查历史记录内容
    for idx, (prompt, response) in enumerate(zip(conversation, responses)):
        assert deepseek_model.history[idx][1] == response, f"第 {idx + 1} 轮对话历史记录不正确"


def test_stop_tokens(deepseek_model):
    """
    测试停止标记功能

    :param deepseek_model: 模型实例fixture
    """
    prompt = "介绍人工智能的发展历程"
    stop_words = ["1990", "2000"]

    response = deepseek_model._call(prompt, stop=stop_words)

    # 检查是否被停止标记截断
    for word in stop_words:
        assert word not in response, f"响应未正确处理停止标记：{word}"


def test_performance(deepseek_model):
    """
    简单性能测试

    :param deepseek_model: 模型实例fixture
    """
    import time

    prompt = "用20字总结人工智能的未来发展"

    start_time = time.time()
    response = deepseek_model._call(prompt)
    inference_time = time.time() - start_time

    assert inference_time < 10, f"推理时间过长：{inference_time}秒"
    assert len(response) > 0, "性能测试推理结果为空"


def test_special_characters(deepseek_model):
    """
    测试特殊字符处理

    :param deepseek_model: 模型实例fixture
    """
    special_prompts = [
        "🤖 人工智能是什么？",
        "こんにちは AI について教えて",
        "   空白开头的prompt   ",
        "包含\n换行\n的prompt",
        "💡 与 🚀 结合的技术"
    ]

    for prompt in special_prompts:
        response = deepseek_model._call(prompt)
        assert response is not None, f"特殊prompt处理失败：{prompt}"


# 边界测试
def test_edge_cases(deepseek_model):
    """
    边界条件测试

    :param deepseek_model: 模型实例fixture
    """
    # 超长输入测试
    long_prompt = "a" * 10000
    with pytest.raises(Exception):
        deepseek_model._call(long_prompt)

    # None输入测试
    with pytest.raises(TypeError):
        deepseek_model._call(None)


# 主函数，用于独立运行测试
if __name__ == "__main__":
    pytest.main([__file__])
