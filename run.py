import torch
from torch import nn, autograd
from model import TCN
from torch.utils import data
from customdataset import CustomWorkloadDataset
import torch.optim as optim

epoch_number = 1
window_Size = 33

workload_dataset_july = CustomWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_request_number_dataset_July95_30s.csv',
    window_size=window_Size
)

workload_dataset_august = CustomWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_request_number_dataset_August95_30s.csv',
    window_size=window_Size
)

dataloader_july: data.DataLoader = data.DataLoader(dataset=workload_dataset_july, batch_size=1, shuffle=False)
dataloader_august: data.DataLoader = data.DataLoader(dataset=workload_dataset_august, batch_size=1, shuffle=False)

# data_iterator_july = iter(dataloader_july)
# data_iterator_august = iter(dataloader_august)

hidden_units_per_layer = 1  # channel
levels = 6
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
output_size = 1
kernel_size = 3
dropout = 0.0

model: TCN = TCN(input_size=input_channels, output_size=output_size, num_channels=channel_sizes,
                 kernel_size=kernel_size, dropout=dropout, sequence_length=window_Size - 1)
criterion = nn.L1Loss()
# optimizer = optim.Adam(params=model.parameters())
optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.0)
model.train(mode=True)

with autograd.detect_anomaly():
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
            # previous_sequence = previous_sequence.squeeze(dim=0)

            # print(previous_sequence.size())
            # print(current_value.size())
            # print()

            # print(current_value)
            # print()

            optimizer.zero_grad()
            outputs = model(previous_sequence)
            loss = criterion(outputs, current_value)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 0 and i > 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                print('real: ', str(current_value.item()), '----- got: ', str(outputs.item()))
                print()

                if i > 3000 and i % 10000 == 0 and (loss < 2.0 or i > 20000):
                    torch.save(model.state_dict(), "model_nasa_dataset_sample" +
                               str(i) + "_loss" + str(loss.item()) + ".pt")

                running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), "final_model_nasa_dataset.pt")
print('Trained Model Saved')
