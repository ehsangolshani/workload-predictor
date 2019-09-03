from model import TCN
import pandas as pd
from torch.utils import data
from customdataset import CustomWorkloadDataset

workload_dataset_august = CustomWorkloadDataset(
    'dataset/nasa-http/nasa_temporal_request_number_dataset_August95_30s.csv')
workload_dataset_july = CustomWorkloadDataset(
    'dataset/nasa-http/nasa_temporal_request_number_dataset_July95_30s.csv')

dataloader_august = data.DataLoader(dataset=workload_dataset_august, shuffle=False)
dataloader_july = data.DataLoader(dataset=workload_dataset_august, shuffle=False)

data_iterator = iter(dataloader_august)

hidden_units_per_layer = 1
levels = 8
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
kernel_size = 8
dropout = 0.0

model = TCN(input_size=input_channels, num_channels=channel_sizes, kernel_size=kernel_size, dropout=dropout)
