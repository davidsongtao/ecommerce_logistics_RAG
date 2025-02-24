from torch.utils.data import Dataset
import torch
import pickle
from configs.log_config import get_logger
from configs.train_config import TrainConfig

param = TrainConfig()


class MyDataset(Dataset):

    def __init__(self, input_list, max_len):
        super().__init__()
        self.logger = get_logger("dataset")
        self.input_list = input_list
        self.max_len = max_len

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        """获取数据集中每一条样本"""
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids


if __name__ == '__main__':
    with open(param.pkl_data_path, "rb") as f:
        train_input_list = pickle.load(f)

    my_dataset = MyDataset(train_input_list, param.max_len)

    print(len(my_dataset))
    print(my_dataset[1])
