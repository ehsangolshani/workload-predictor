import torch
from torch import nn
from TCN.model import TCNModel
from torch.utils import data
from custom_datasets.windowed_dataset import WindowedWorkloadDataset

window_size = 49

workload_dataset_july = WindowedWorkloadDataset(
    csv_path='raw_dataset/nasa_http/nasa_temporal_rps_July95_1m.csv',
    window_size=window_size
)

workload_dataset_august = WindowedWorkloadDataset(
    csv_path='raw_dataset/nasa_http/nasa_temporal_rps_August95_1m.csv',
    window_size=window_size
)

dataset = data.ConcatDataset([workload_dataset_july, workload_dataset_august])

data_loader: data.DataLoader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)

hidden_units_per_layer = 1  # channel
levels = 5
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
output_size = 1
kernel_size = 3
dropout = 0.0

model: TCNModel = TCNModel(input_size=input_channels, output_size=output_size, num_channels=channel_sizes,
                           kernel_size=kernel_size, dropout=dropout, sequence_length=window_size - 1)
model.load_state_dict(torch.load('trained_models/TCN_workload_model_nasa_dataset.pt'))
model.eval()

mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()

sum_of_mse_loss = 0
sum_of_l1_loss = 0

for i, data in enumerate(data_loader, 0):
    previous_sequence: torch.Tensor = data[:, :, :-1]
    current_value: torch.Tensor = data[:, :, -1]
    current_value = current_value.view(-1)

    outputs = model(previous_sequence)
    mse_loss = mse_criterion(outputs, current_value)
    l1_loss = l1_criterion(outputs, current_value)

    sum_of_mse_loss += mse_loss.item()
    sum_of_l1_loss += l1_loss.item()

print("average total MSE loss: ", sum_of_mse_loss / len(data_loader))
print("average total L1 loss: ", sum_of_l1_loss / len(data_loader))
