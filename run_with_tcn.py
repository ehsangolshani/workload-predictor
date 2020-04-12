from datetime import datetime

import torch
from torch import nn
from TCN.model import TCNModel
from torch.utils import data
from custom_datasets.windowed_dataset import WindowedWorkloadDataset
import torch.optim as optim
import matplotlib.pyplot as plt

window_size = 24

workload_dataset = WindowedWorkloadDataset(
    csv_path='raw_dataset/nasa_http/nasa_temporal_rps_1m.csv',
    window_size=window_size + 1
)

train_set_size = int((6 / 10) * len(workload_dataset))
test_set_size = len(workload_dataset) - train_set_size

# train_dataset, test_dataset = data.random_split(dataset=workload_dataset, lengths=[train_set_size, test_set_size])
train_dataset = data.Subset(dataset=workload_dataset, indices=[i for i in range(0, train_set_size)])
test_dataset = data.Subset(dataset=workload_dataset, indices=[i for i in range(train_set_size, len(workload_dataset))])

train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset, batch_size=1,
                                                     num_workers=4, shuffle=True)
test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset, batch_size=1,
                                                    num_workers=4, shuffle=True)
epoch_number = 2
hidden_units_per_layer = 1  # channel
levels = 4
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
output_size = 1
kernel_size = 3
dropout = 0.25

model: TCNModel = TCNModel(input_size=input_channels, output_size=output_size, num_channels=channel_sizes,
                           kernel_size=kernel_size, dropout=dropout, sequence_length=window_size)

mse_criterion = nn.MSELoss()  # this is used for training phase
l1_criterion = nn.L1Loss()

optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
model.train(mode=True)

train_avg_loss_x = list()
train_avg_mse_loss_y = list()
train_avg_l1_loss_y = list()

test_avg_loss_x = list()
test_avg_mse_loss_y = list()
test_avg_l1_loss_y = list()

mse_loss_sum_for_plot = 0
l1_loss_sum_for_plot = 0

train_workload_sample_x = list()
train_real_workload_sample_y = list()
train_predicted_workload_sample_y = list()

test_workload_sample_x = list()
test_real_workload_sample_y = list()
test_predicted_workload_sample_y = list()

plot_x_counter = 0
iteration = 0

train_iterations_num = epoch_number * train_set_size
train_workload_sample_num = 250
test_workload_sample_num = 250

for epoch in range(epoch_number):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        iteration += 1
        previous_workload_sequence: torch.Tensor = data[:, :, :-1]
        real_future_workload: torch.Tensor = data[:, :, -1]
        real_future_workload = real_future_workload.view(-1)
        optimizer.zero_grad()
        predicted_future_workload = model(previous_workload_sequence)

        mse_loss = mse_criterion(predicted_future_workload, real_future_workload)
        l1_loss = l1_criterion(predicted_future_workload, real_future_workload)

        real_future_workload_value = real_future_workload.item()
        predicted_future_workload_value = predicted_future_workload.item()

        mse_loss.backward()
        # l1_loss.backward()
        optimizer.step()

        mse_loss_value = mse_loss.item()
        l1_loss_value = l1_loss.item()

        running_loss += mse_loss_value

        mse_loss_sum_for_plot += mse_loss_value
        l1_loss_sum_for_plot += l1_loss_value

        if i % 1000 == 0 and i > 0:
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            print('real: ', str(real_future_workload.item()), '----- got: ', str(predicted_future_workload.item()),
                  '\n')
            running_loss = 0.0

        if iteration % 100 == 0:
            plot_x_counter += 1
            train_avg_loss_x.append(plot_x_counter)

            denominator = 100

            train_avg_mse_loss_y.append(mse_loss_sum_for_plot / denominator)
            train_avg_l1_loss_y.append(l1_loss_sum_for_plot / denominator)

            mse_loss_sum_for_plot = 0
            l1_loss_sum_for_plot = 0

        if iteration > train_iterations_num - train_workload_sample_num:
            train_workload_sample_x.append(iteration)
            train_real_workload_sample_y.append(real_future_workload_value)
            train_predicted_workload_sample_y.append(predicted_future_workload_value)

plot_x_where_train_stopped = iteration / 100
iteration_where_train_stopped = iteration
train_avg_loss_x.append(plot_x_where_train_stopped)

train_avg_mse_loss_y.append(mse_loss_sum_for_plot / (iteration % 100))
train_avg_l1_loss_y.append(l1_loss_sum_for_plot / (iteration % 100))

test_avg_loss_x.append(plot_x_where_train_stopped)

test_avg_mse_loss_y.append(mse_loss_sum_for_plot / (iteration % 100))
test_avg_l1_loss_y.append(l1_loss_sum_for_plot / (iteration % 100))

mse_loss_sum_for_plot = 0
l1_loss_sum_for_plot = 0

print('Finished Training')
torch.save(model.state_dict(), 'trained_models/TCN_workload_model_nasa_dataset.pt')
print('Trained Model Saved')

print('\n\n\n')
print('start evaluation')
model.load_state_dict(torch.load('trained_models/TCN_workload_model_nasa_dataset.pt'))
model.eval()

sum_of_mse_loss = 0
sum_of_l1_loss = 0

first_plot_x_test_count = True

response_time_sum = 0
response_time_counter = 0

for i, data in enumerate(test_data_loader, 0):
    iteration += 1

    start_timestamp = datetime.now().timestamp()

    previous_workload_sequence: torch.Tensor = data[:, :, :-1]
    real_future_workload: torch.Tensor = data[:, :, -1]
    real_future_workload = real_future_workload.view(-1)

    predicted_future_workload = model(previous_workload_sequence)

    finish_timestamp = datetime.now().timestamp()
    diff_in_seconds = finish_timestamp - start_timestamp
    response_time_counter += 1
    response_time_sum += diff_in_seconds

    mse_loss = mse_criterion(predicted_future_workload, real_future_workload)
    l1_loss = l1_criterion(predicted_future_workload, real_future_workload)

    real_future_workload_value = real_future_workload.item()
    predicted_future_workload_value = predicted_future_workload.item()

    mse_loss_value = mse_loss.item()
    l1_loss_value = l1_loss.item()

    sum_of_mse_loss += mse_loss_value
    sum_of_l1_loss += l1_loss_value

    mse_loss_sum_for_plot += mse_loss_value
    l1_loss_sum_for_plot += l1_loss_value

    if i % 1000 == 0 and i > 0:
        print('[%5d] mse loss: %.5f' % (i + 1, sum_of_mse_loss / i))
        print('real: ', str(real_future_workload.item()), '----- got: ', str(predicted_future_workload.item()))
        print()

    if iteration % 100 == 0:
        plot_x_counter += 1
        test_avg_loss_x.append(plot_x_counter)

        if first_plot_x_test_count:
            denominator = 100 - (iteration_where_train_stopped % 100)
            first_plot_x_test_count = False
        else:
            denominator = 100

        test_avg_mse_loss_y.append(mse_loss_sum_for_plot / denominator)
        test_avg_l1_loss_y.append(l1_loss_sum_for_plot / denominator)

        mse_loss_sum_for_plot = 0
        l1_loss_sum_for_plot = 0

    if i < test_workload_sample_num:
        test_workload_sample_x.append(i)
        test_real_workload_sample_y.append(real_future_workload_value)
        test_predicted_workload_sample_y.append(predicted_future_workload_value)

test_stopping_plot_x = iteration / 100
test_avg_loss_x.append(test_stopping_plot_x)

test_avg_mse_loss_y.append(mse_loss_sum_for_plot / (iteration % 100))
test_avg_l1_loss_y.append(l1_loss_sum_for_plot / (iteration % 100))

mse_loss_sum_for_plot = 0
l1_loss_sum_for_plot = 0

print("average total MSE loss: ", sum_of_mse_loss / len(test_data_loader))
print("average total L1 loss: ", sum_of_l1_loss / len(test_data_loader))
print('average response time of model: ',
      (response_time_sum * 1000) / response_time_counter)

# draw loss plots
plt.figure(figsize=[12.0, 8.0])
plt.title('Prediction Error (MSE Loss)')
plt.xlabel("Time")
plt.ylabel("Total Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.08])
plt.plot(train_avg_loss_x, train_avg_mse_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, test_avg_mse_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('mse_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Prediction Error (L1 Loss)')
plt.xlabel("Time")
plt.ylabel("Total Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.5])
plt.plot(train_avg_loss_x, train_avg_l1_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, test_avg_l1_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('l1_loss_plot.png')
plt.show()
plt.close()

# draw workload plots
plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future normalized workload (Training)')
plt.xlabel("Samples")
plt.ylabel("Normalized Workload")
plt.axis([train_iterations_num - train_workload_sample_num, train_iterations_num + 1, 0, 1])
plt.plot(train_workload_sample_x, train_real_workload_sample_y, 'r-', label='Real workload')
plt.plot(train_workload_sample_x, train_predicted_workload_sample_y, 'g-', label='Predicted workload')
plt.legend(loc='upper left')
plt.savefig('train_real_predicted_workload_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future normalized workload (Testing)')
plt.xlabel("Samples")
plt.ylabel("Normalized Workload")
plt.axis([0, test_workload_sample_num + 1, 0, 1])
plt.plot(test_workload_sample_x, test_real_workload_sample_y, 'r-', label='Real workload')
plt.plot(test_workload_sample_x, test_predicted_workload_sample_y, 'g-', label='Predicted workload')
plt.legend(loc='upper left')
plt.savefig('test_real_predicted_workload_plot.png')
plt.show()
plt.close()
