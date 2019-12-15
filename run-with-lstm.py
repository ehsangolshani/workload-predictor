from customdataset import CustomWorkloadDataset
import torch
from torch import nn
from torch.utils import data
import torch.optim as optim
from sklearn.model_selection import KFold

epoch_number = 2

workload_dataset_july = CustomWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_rps_July95_1m.csv'
)

workload_dataset_august = CustomWorkloadDataset(
    csv_path='dataset/nasa-http/nasa_temporal_rps_August95_1m.csv',
)

dataset = data.ConcatDataset([workload_dataset_july, workload_dataset_august])

train_set_size = int((6 / 10) * len(dataset))
test_set_size = int((4 / 10) * len(dataset)) + 1

train_dataset, test_dataset = data.random_split(dataset=dataset, lengths=[train_set_size, test_set_size])

data_loader_july: data.DataLoader = data.DataLoader(dataset=workload_dataset_july, batch_size=1, shuffle=False)
data_loader_august: data.DataLoader = data.DataLoader(dataset=workload_dataset_august, batch_size=1, shuffle=False)

data_loader: data.DataLoader = data.DataLoader(dataset=dataset, batch_size=1, num_workers=4, shuffle=False)
train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=4, shuffle=False)
test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=4, shuffle=False)

model: nn.LSTM = nn.LSTM(input_size=1, hidden_size=1)
criterion = nn.MSELoss()

optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
# optimizer = optim.SGD(params=model.parameters(), lr=1e-4, momentum=0.3)
model.train(mode=True)

# with autograd.detect_anomaly():
for epoch in range(epoch_number):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        current_value: torch.Tensor = data[:, :, 0: 1]
        future_value: torch.Tensor = data[:, :, 1:2]

        optimizer.zero_grad()
        output, hidden = model(current_value, None)
        loss = criterion(output, current_value)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 0 and i > 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            print('real: ', str(current_value.item()), '----- got: ', str(output.item()))
            print()
            if (i > 3000 or epoch > 0) and i % 10000 == 0 and (loss < 1.0 or i > 20000):
                torch.save(model.state_dict(), "model_nasa_dataset" +
                           "_epoch" + str(epoch) +
                           "_sample" + str(i) +
                           "_loss" + str(running_loss / 500) +
                           ".pt")

            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), "final_model_nasa_dataset.pt")
print('Trained Model Saved')

print('\n\n\n')
print('start evaluation')
model.eval()

sum_of_loss = 0
for i, data in enumerate(test_data_loader, 0):
    current_value: torch.Tensor = data[:, :, 0: 1]
    future_value: torch.Tensor = data[:, :, 1:2]

    output, hidden = model(current_value, None)
    loss = criterion(output, future_value)

    sum_of_loss += loss.item()

print("average total loss: ", sum_of_loss / len(test_data_loader))
