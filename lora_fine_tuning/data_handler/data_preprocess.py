"""
Description: source_ids + [gMASK] + <sop> + target_ids + <eop>
    
-*- Encoding: UTF-8 -*-
@File     ：data_preprocess.py
@Author   ：King Songtao
@Time     ：2025/2/25 上午10:06
@Contact  ：king.songtao@gmail.com
"""
import json

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from configs.log_config import get_logger
from configs.lora_config import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_samples(samples: dict, tokenizer, max_source_seq_len: int, max_target_seq_len: int):
    """转换数据格式，输出为模型接收的格式"""
    tokenized_output = {
        'input_ids': [],
        'labels': []
    }
    max_seq_len = max_source_seq_len + max_target_seq_len

    for sample in samples['text']:
        try:
            sample = json.loads(sample)
            context = sample['context']
            target = sample['target']

            prompts_ids = tokenizer.encode(context, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)

            if len(prompts_ids) >= max_source_seq_len:
                prompts_ids = prompts_ids[:max_source_seq_len - 1]

            if len(target_ids) >= max_target_seq_len:
                target_ids = target_ids[:max_target_seq_len - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)
            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1:]
            pad_len = max_seq_len - len(input_ids)

            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

            labels = labels + [-100] * pad_len

            tokenized_output['input_ids'].append(input_ids)
            tokenized_output['labels'].append(labels)
            print(f"input_ids --> {input_ids}")
            print(f"input_ids_length --> {len(input_ids)}")
            print(f"labels --> {labels}")
            print(f"labels_length --> {len(labels)}")

            break

        except Exception as e:
            print(e)
            continue

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


def get_max_length(tokenizer, dataset_file: str):
    source_seq_len_list = []
    target_seq_len_list = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)

            source_len = tokenizer.encode(line['context'])
            source_seq_len_list.append(len(source_len))

            target_len = tokenizer.encode(line['target'])
            target_seq_len_list.append(len(target_len))

    print(dataset_file)
    print(f'【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.')
    print(f'【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.')


if __name__ == '__main__':
    param = LoraConfig()
    train_dataset = load_dataset("text", data_files={'train': param.train_dataset})['train']
    # print(type(train_dataset))
    tokenizer = AutoTokenizer.from_pretrained(r"D:\ecommerce_logistics_RAG\models\chatglm_6b_4int", trust_remote_code=True, revision='main')
    tokenized_output = convert_samples(train_dataset, tokenizer, 63, 55)
    get_max_length(tokenizer, param.train_dataset)
