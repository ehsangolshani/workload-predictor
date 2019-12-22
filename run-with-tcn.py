import torch
from torch import nn
from TCN.model import TCNModel
from torch.utils import data
from windoweddataset import WindowedWorkloadDataset
import torch.optim as optim

epoch_number = 2
window_Size = 17

workload_dataset_july = WindowedWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_rps_July95_1m.csv',
    window_size=window_Size
)

workload_dataset_august = WindowedWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_rps_August95_1m.csv',
    window_size=window_Size
)

dataset = data.ConcatDataset([workload_dataset_july, workload_dataset_august])

train_set_size = int((6 / 10) * len(dataset))
test_set_size = len(dataset) - train_set_size

train_dataset, test_dataset = data.random_split(dataset=dataset, lengths=[train_set_size, test_set_size])

data_loader_july: data.DataLoader = data.DataLoader(dataset=workload_dataset_july,
                                                    batch_size=1, shuffle=True)
data_loader_august: data.DataLoader = data.DataLoader(dataset=workload_dataset_august,
                                                      batch_size=1, shuffle=True)

data_loader: data.DataLoader = data.DataLoader(dataset=dataset, batch_size=1,
                                               num_workers=4, shuffle=True)
train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset, batch_size=1,
                                                     num_workers=4, shuffle=True)
test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset, batch_size=1,
                                                    num_workers=4, shuffle=True)

hidden_units_per_layer = 1  # channel
levels = 5
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
output_size = 1
kernel_size = 5
dropout = 0.25

model: TCNModel = TCNModel(input_size=input_channels, output_size=output_size, num_channels=channel_sizes,
                           kernel_size=kernel_size, dropout=dropout, sequence_length=window_Size - 1)

mse_criterion = nn.MSELoss()  # this is used for training phase
l1_criterion = nn.L1Loss()

optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
# optimizer = optim.SGD(params=model.parameters(), lr=1e-4, momentum=0.3)
model.train(mode=True)

# with autograd.detect_anomaly():
for epoch in range(epoch_number):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        previous_sequence: torch.Tensor = data[:, :, :-1]
        current_value: torch.Tensor = data[:, :, -1]
        current_value = current_value.view(-1)
        optimizer.zero_grad()
        outputs = model(previous_sequence)
        mse_loss = mse_criterion(outputs, current_value)
        mse_loss.backward()
        optimizer.step()

        running_loss += mse_loss.item()
        if i % 500 == 0 and i > 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            print('real: ', str(current_value.item()), '----- got: ', str(outputs.item()))
            running_loss = 0.0
            print()

print('Finished Training')
torch.save(model.state_dict(), "TCN_final_model_nasa_dataset.pt")
print('Trained Model Saved')

print('\n\n\n')
print('start evaluation')
model.eval()

sum_of_mse_loss = 0
sum_of_l1_loss = 0
for i, data in enumerate(test_data_loader, 0):
    previous_sequence: torch.Tensor = data[:, :, :-1]
    current_value: torch.Tensor = data[:, :, -1]
    current_value = current_value.view(-1)

    outputs = model(previous_sequence)
    mse_loss = mse_criterion(outputs, current_value)
    l1_loss = l1_criterion(outputs, current_value)

    sum_of_mse_loss += mse_loss.item()
    sum_of_l1_loss += l1_loss.item()

print("average total MSE loss: ", sum_of_mse_loss / len(test_data_loader))
print("average total L1 loss: ", sum_of_l1_loss / len(test_data_loader))
