import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class CustomWorkloadDataset(Dataset):
    def __init__(self, csv_path: str, window_size: int):
        self.data = pd.read_csv(csv_path)[['request_numbers']]
        self.data_tensor: torch.Tensor = torch.tensor(data=self.data.values)
        self.window_size: int = window_size

    def __getitem__(self, index: int):
        return self.data_tensor[index:index + self.window_size]

    def __len__(self):
        return self.data_tensor.size()[0] - self.window_size + 1
