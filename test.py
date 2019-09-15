import torch
from torch import nn
import numpy as np
from model import TCN
from torch.utils import data
from customdataset import CustomWorkloadDataset
import torch.optim as optim

epoch_number = 1
window_Size = 129

workload_dataset_july = CustomWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_request_number_dataset_July95_30s.csv',
    window_size=window_Size
)

workload_dataset_august = CustomWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_request_number_dataset_August95_30s.csv',
    window_size=window_Size
)

dataloader_july: data.DataLoader = data.DataLoader(dataset=workload_dataset_august, shuffle=False)
dataloader_august: data.DataLoader = data.DataLoader(dataset=workload_dataset_august, shuffle=False)

# data_iterator_july = iter(dataloader_july)
# data_iterator_august = iter(dataloader_august)

hidden_units_per_layer = 1  # channel
levels = 8
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
output_size = 1
kernel_size = 3
dropout = 0.0

model: TCN = TCN(input_size=input_channels, output_size=output_size, num_channels=channel_sizes,
                 kernel_size=kernel_size, dropout=dropout, sequence_length=window_Size - 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
model.train(mode=True)

for epoch in range(epoch_number):
    running_loss = 0.0
    for i, data in enumerate(dataloader_july, 0):
        # a = data.size()
        # data.squeeze()
        # b = data.size()
        previous_sequence: torch.Tensor = data[:, :, :-1]
        current_value: torch.Tensor = data[:, :, -1]
        current_value = current_value.view(-1)
        # current_value.long()
        # current_value=current_value.squeeze(dim=)

        # print(previous_sequence.size())
        # print(current_value.size())

        optimizer.zero_grad()
        outputs = model(previous_sequence)
        loss = criterion(outputs, current_value)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 1:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), "model_nasa_dataset.pt")
print('Trained Model Saved')
