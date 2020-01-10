import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class RecurrentSequentialWorkloadDataset(Dataset):
    def __init__(self, csv_path: str, window_size: int):
        self.data = pd.read_csv(csv_path)[['normalized_request_rate']]
        self.data_tensor: torch.Tensor = torch.tensor(data=self.data.values, dtype=torch.float)
        self.data_tensor = self.data_tensor.contiguous()
        self.window_size: int = window_size

    def __getitem__(self, index: int):
        return self.data_tensor[index:index + self.window_size, :]

    def __len__(self):
        return self.data_tensor.size()[0] - self.window_size + 1
