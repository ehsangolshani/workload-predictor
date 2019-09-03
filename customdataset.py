import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class CustomWorkloadDataset(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)[['request_numbers']]
        self.data_tensor = torch.tensor(data=self.data.values)

    def __getitem__(self, index: int):
        return self.data.iloc[index]

    def __len__(self):
        return len(self.data)
