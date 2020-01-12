from RNN.model import RNNModel
from custom_datasets.recurrent_sequential_dataset import RecurrentSequentialWorkloadDataset
import torch
from torch import nn
from torch.utils import data
import torch.optim as optim

dropout = 0.1
epoch_number = 2
hidden_dim = 1
input_size = 1
output_size = 1
batch_size = 1
num_layers = 1
window_size = 17

workload_dataset = RecurrentSequentialWorkloadDataset(
    csv_path='raw_dataset/nasa_http/nasa_temporal_rps_1m.csv',
    window_size=window_size
)

train_set_size = int((6 / 10) * len(workload_dataset))
test_set_size = len(workload_dataset) - train_set_size

# train_dataset, test_dataset = data.random_split(raw_dataset=raw_dataset,
#                                                 lengths=[train_set_size, test_set_size])

train_dataset = data.Subset(dataset=workload_dataset, indices=[i for i in range(0, train_set_size)])
test_dataset = data.Subset(dataset=workload_dataset, indices=[i for i in range(train_set_size, len(workload_dataset))])

train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset,
                                                     batch_size=batch_size,
                                                     num_workers=4,
                                                     shuffle=False)
test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=4,
                                                    shuffle=False)

model: RNNModel = RNNModel(input_size=input_size,
                           output_size=output_size,
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           batch_size=batch_size,
                           dropout=dropout)

mse_criterion = nn.MSELoss()  # this is used for training phase
l1_criterion = nn.L1Loss()

optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
model.train(mode=True)

for epoch in range(epoch_number):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        current_value: torch.Tensor = data[:, :-1, :]
        future_value: torch.Tensor = data[:, -1:, :]

        optimizer.zero_grad()
        output, hidden = model(current_value)
        last_prediction = output[:, -1:, :]

        mse_loss = mse_criterion(last_prediction, future_value)
        mse_loss.backward()
        optimizer.step()

        running_loss += mse_loss.item()
        if i % 500 == 0 and i > 0:
            print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 500))
            print('real: ', str(future_value.item()), '----- got: ', str(last_prediction.item()))
            running_loss = 0.0
            print()

print('Finished Training')
torch.save(model.state_dict(), "trained_models/RNN_workload_model_nasa_dataset.pt")
print('Trained Model Saved')

print('\n\n\n')
print('start evaluation')

model.eval()

sum_of_mse_loss = 0
sum_of_l1_loss = 0
for i, data in enumerate(test_data_loader, 0):
    current_value: torch.Tensor = data[:, :-1, :]
    future_value: torch.Tensor = data[:, -1:, :]

    output, hidden = model(current_value)
    last_prediction = output[:, -1:, :]

    mse_loss = mse_criterion(last_prediction, future_value)
    l1_loss = l1_criterion(last_prediction, future_value)

    sum_of_mse_loss += mse_loss.item()
    sum_of_l1_loss += l1_loss.item()

    if i % 500 == 0 and i > 0:
        print('[%5d] mse loss: %.5f' % (i + 1, sum_of_mse_loss / i))
        print('real: ', str(future_value.item()), '----- got: ', str(last_prediction.item()))
        running_loss = 0.0
        print()

print("average total MSE loss: ", sum_of_mse_loss / len(test_data_loader))
print("average total L1 loss: ", sum_of_l1_loss / len(test_data_loader))
