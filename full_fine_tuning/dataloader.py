import pickle

import torch
from torch.utils.data import random_split, DataLoader
from torch.nn.utils import rnn
from configs.train_config import TrainConfig

from full_fine_tuning.dataset import MyDataset

param = TrainConfig()


def load_dataset(dataset_path):
    # 1. 加载数据集
    with open(dataset_path, "rb") as f:
        dataset_input_list = pickle.load(f)

    my_dataset = MyDataset(dataset_input_list, 512)

    # 2. 划分训练集和验证机
    train_size = int(0.8 * len(my_dataset))
    valid_size = len(my_dataset) - train_size

    train_dataset, valid_dataset = random_split(
        my_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )

    return train_dataset, valid_dataset


def collate_fn(batch):
    input_ids = rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn.pad_sequence(batch, batch_first=True, padding_value=param.ignore_index)

    return input_ids, labels


def get_dataloader(dataset_path):
    train_dataset, valid_dataset = load_dataset(dataset_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=param.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=param.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True
    )

    return train_dataloader, valid_dataloader


if __name__ == '__main__':

    train_dataloader, valid_dataloader = get_dataloader(param.pkl_data_path)
    for input_ids, labels in train_dataloader:
        print(f"input_ids.shape--> {input_ids.shape}")
        print(f"input_ids--> {input_ids}")
        print(f"labels--> {labels}")
        break
