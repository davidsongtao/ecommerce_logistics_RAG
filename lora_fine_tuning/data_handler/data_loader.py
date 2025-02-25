"""
Description: 将数据组织成一个batch一个batch的形式
    
-*- Encoding: UTF-8 -*-
@File     ：data_loader.py
@Author   ：King Songtao
@Time     ：2025/2/25 上午11:52
@Contact  ：king.songtao@gmail.com
"""
from functools import partial

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

from data_preprocess import convert_samples
from datasets import load_dataset
from configs.lora_config import LoraConfig

tokenizer = AutoTokenizer.from_pretrained(r"D:\ecommerce_logistics_RAG\models\chatglm_6b_4int", trust_remote_code=True, revision='main')


def get_data(train_data, valid_data):
    train_dataset = load_dataset('text', data_files={'train': train_data,
                                                     'dev': valid_data})

    new_func = partial(convert_samples, tokenizer=tokenizer, max_source_seq_len=63, max_target_seq_len=55)

    dataset = train_dataset.map(new_func, batched=True)
    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True, collate_fn=default_data_collator)
    dev_dataloader = DataLoader(dev_dataset, batch_size=param.batch_size, shuffle=False, collate_fn=default_data_collator)

    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    param = LoraConfig()
    train_dataloader, dev_dataloader = get_data(param.train_dataset, param.valid_dataset)
