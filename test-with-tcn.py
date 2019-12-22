import torch
from torch import nn
from TCN.model import TCNModel
from torch.utils import data
from windoweddataset import WindowedWorkloadDataset

window_Size = 17

workload_dataset_july = WindowedWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_rps_July95_1m.csv',
    window_size=window_Size
)

workload_dataset_august = WindowedWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_rps_August95_1m.csv',
    window_size=window_Size
)

dataloader_july: data.DataLoader = data.DataLoader(dataset=workload_dataset_august, batch_size=1, shuffle=False)
dataloader_august: data.DataLoader = data.DataLoader(dataset=workload_dataset_august, batch_size=1, shuffle=False)

hidden_units_per_layer = 1  # channel
levels = 5
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
output_size = 1
kernel_size = 5
dropout = 0.0

model: TCNModel = TCNModel(input_size=input_channels, output_size=output_size, num_channels=channel_sizes,
                           kernel_size=kernel_size, dropout=dropout, sequence_length=window_Size - 1)

model.eval()
criterion = nn.MSELoss()

sum_of_loss = 0
print(len(dataloader_august))
for i, data in enumerate(dataloader_august, 0):
    previous_sequence: torch.Tensor = data[:, :, :-1]
    current_value: torch.Tensor = data[:, :, -1]
    current_value = current_value.view(-1)

    outputs, hidden = model(previous_sequence)
    loss = criterion(outputs, current_value)

    sum_of_loss += loss.item()
    print('real: ', str(current_value.item()), '----- got: ', str(outputs.item()))

print("average total loss: ", sum_of_loss / len(dataloader_august))
