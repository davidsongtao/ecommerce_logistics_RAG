"""
Description: 
    
-*- Encoding: UTF-8 -*-
@File     ：app.py
@Author   ：King Songtao
@Time     ：2025/2/22 下午12:48
@Contact  ：king.songtao@gmail.com
"""
import os
import sys
import time

from utils.exceptions import ModelLoadError, ModelGenerateError, ModelResourceError

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
from utils.model import LocalLLM


def main():
    try:

        model_path = "/root/autodl-tmp/ecommerce_logistics_RAG/models/DeepSeek_R1_Distill_Qwen_7B"
        llm = LocalLLM(model_path, show_log=False)
        while True:
            prompt = input("User >>> :")
            print(f"User >>> : {prompt}")
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


if __name__ == '__main__':
    main()
