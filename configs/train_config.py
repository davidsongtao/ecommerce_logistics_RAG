"""
Description: 模型训练配置文件
    
-*- Encoding: UTF-8 -*-
@File     ：train_config.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午7:52
@Contact  ：king.songtao@gmail.com
"""
import torch


class TrainConfig():

    def __init__(self):
        self.txt_data_path = r"D:\ecommerce_logistics_RAG\dataset\dataset.txt"
        self.pkl_data_path = r"D:\ecommerce_logistics_RAG\dataset\dataset.pkl"
        self.max_len = 512
        self.bos_token_id = 151646
        self.eos_token_id = 151643
        self.deepseek_1_5B = r"D:\ecommerce_logistics_RAG\models\DeepSeek_R1_Distill_Qwen_1_5B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qwen2 = r"D:\ecommerce_logistics_RAG\models\Qwen2_5_0_5_B"

        # 模型训练超参数
        self.batch_size = 1
        self.gradient_accumulation_steps = 4  # 梯度累积步数
        self.epochs = 10
        self.learning_rate = 0.00000000000001
        self.eps = 1e-8
        self.max_grad_norm = 1.0
        self.warm_up_ratio = 0.08
        self.init_val_loss = 10000
        self.ignore_index = -100
        self.loss_step = 1
        self.save_model_path = r"D:\ecommerce_logistics_RAG\models\saved_models"


if __name__ == '__main__':
    param = TrainConfig()
    print(param.device)
