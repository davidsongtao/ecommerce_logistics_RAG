"""
Description: 对原始语料进行tokenize,将每段对话处理成如下形式：“<｜begin▁of▁sentence｜>sentence_ids_1[<｜end▁of▁sentence｜>]sentence_ids_2<｜end▁of▁sentence｜>”
    
-*- Encoding: UTF-8 -*-
@File     ：preprocess.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午4:34
@Contact  ：king.songtao@gmail.com
"""
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer
import json
from configs.log_config import get_logger
from configs.train_config import TrainConfig

param = TrainConfig()


def data_preprocess(train_path, train_pkl_path):
    logger = get_logger("preprocess")

    try:
        # 1. 加载tokenizer
        logger.info("开始加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(r"D:\ecommerce_logistics_RAG\models\DeepSeek_R1_Distill_Qwen_1_5B", trust_remote_code=True)
        logger.info("tokenizer加载成功...")

        # 2. 加载数据集
        with open(train_path, "rb") as f:
            data = f.read().decode("utf-8")
        if "\r\n" in data:
            train_data = data.split("\r\n")
            logger.info("数据集加载成功...")
        else:
            train_data = data.split("\n")
            logger.info("数据集加载成功...")

        dialogue_len = []
        dialogue_list = []

        for i, json_data in enumerate(tqdm(train_data)):
            data = json.loads(json_data)
            question = data['问题内容']
            answer = data['答案']

            question_ids = tokenizer.encode(question, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)

            input_ids = [param.bos_token_id]

            input_ids += question_ids
            input_ids.append(param.eos_token_id)
            input_ids += answer_ids
            input_ids.append(param.eos_token_id)

            dialogue_len.append(len(input_ids))
            dialogue_list.append(input_ids)

        logger.info("数据预处理完成...")

        with open(train_pkl_path, "wb") as f:
            pickle.dump(dialogue_list, f)
            logger.info("数据保存成功...")
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")


if __name__ == '__main__':
    data_preprocess(param.txt_data_path, param.pkl_data_path)
