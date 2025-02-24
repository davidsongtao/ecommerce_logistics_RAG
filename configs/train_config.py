"""
Description: 模型训练配置文件
    
-*- Encoding: UTF-8 -*-
@File     ：train_config.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午7:52
@Contact  ：king.songtao@gmail.com
"""


class TrainConfig():

    def __init__(self):
        self.txt_data_path = r"D:\ecommerce_logistics_RAG\dataset\dataset.txt"
        self.pkl_data_path = r"D:\ecommerce_logistics_RAG\dataset\dataset.pkl"
        self.batch_size = 8
        self.max_len = 512
        self.bos_token_id = 151646
        self.eos_token_id = 151643
