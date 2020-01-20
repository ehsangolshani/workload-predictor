import torch
from torch import nn
from LSTM.model import LSTMModel
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

####################### create tcn model

hidden_units_per_layer = 1  # channel
levels = 4
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
output_size = 1
kernel_size = 3
tcn_dropout = 0.25

tcn_model: TCNModel = TCNModel(input_size=input_channels, output_size=output_size, num_channels=channel_sizes,
                               kernel_size=kernel_size, dropout=tcn_dropout, sequence_length=window_size)

tcn_mse_criterion = nn.MSELoss()  # this is used for training phase
tcn_l1_criterion = nn.L1Loss()

tcn_optimizer = optim.Adam(params=tcn_model.parameters(), lr=1e-4)

####################### create one-to-one LSTM model

one_to_one_lstm_dropout = 0.25
hidden_dim = 1
input_size = 1
output_size = 1
batch_size = 1
num_layers = 1

one_to_one_lstm_model: LSTMModel = LSTMModel(input_size=input_size,
                                             output_size=output_size,
                                             hidden_dim=hidden_dim,
                                             num_layers=num_layers,
                                             batch_size=batch_size,
                                             dropout=one_to_one_lstm_dropout)

one_to_one_lstm_mse_criterion = nn.MSELoss()  # this is used for training phase
one_to_one_lstm_l1_criterion = nn.L1Loss()

one_to_one_lstm_optimizer = optim.Adam(params=one_to_one_lstm_model.parameters(), lr=1e-4)

####################### create many-to-one LSTM model
many_to_one_lstm_dropout = 0.25
hidden_dim = 1
input_size = 1
output_size = 1
batch_size = 1
num_layers = 1

many_to_one_lstm_model: LSTMModel = LSTMModel(input_size=input_size,
                                              output_size=output_size,
                                              hidden_dim=hidden_dim,
                                              num_layers=num_layers,
                                              batch_size=batch_size,
                                              dropout=one_to_one_lstm_dropout)

many_to_one_lstm_mse_criterion = nn.MSELoss()  # this is used for training phase
many_to_one_lstm_l1_criterion = nn.L1Loss()

many_to_one_lstm_optimizer = optim.Adam(params=many_to_one_lstm_model.parameters(), lr=1e-4)

#######################

train_avg_loss_x = list()

one_to_one_lstm_train_avg_mse_loss_y = list()
many_to_one_lstm_train_avg_mse_loss_y = list()
tcn_train_avg_mse_loss_y = list()

one_to_one_lstm_train_avg_l1_loss_y = list()
many_to_one_lstm_train_avg_l1_loss_y = list()
tcn_train_avg_l1_loss_y = list()

test_avg_loss_x = list()
one_to_one_lstm_test_avg_mse_loss_y = list()
many_to_one_lstm_test_avg_mse_loss_y = list()
tcn_test_avg_mse_loss_y = list()

one_to_one_lstm_test_avg_l1_loss_y = list()
many_to_one_lstm_test_avg_l1_loss_y = list()
tcn_test_avg_l1_loss_y = list()

one_to_one_lstm_mse_loss_sum_for_plot = 0
many_to_one_lstm_mse_loss_sum_for_plot = 0
tcn_mse_loss_sum_for_plot = 0

one_to_one_lstm_l1_loss_sum_for_plot = 0
many_to_one_lstm_l1_loss_sum_for_plot = 0
tcn_l1_loss_sum_for_plot = 0

train_workload_sample_x = list()
train_real_workload_sample_y = list()
train_one_to_one_lstm_predicted_workload_sample_y = list()
train_many_to_one_lstm_predicted_workload_sample_y = list()
train_tcn_predicted_workload_sample_y = list()

test_workload_sample_x = list()
test_real_workload_sample_y = list()
test_one_to_one_lstm_predicted_workload_sample_y = list()
test_many_to_one_lstm_predicted_workload_sample_y = list()
test_tcn_predicted_workload_sample_y = list()

plot_x_counter = 0
iteration = 0

train_iterations_num = epoch_number * train_set_size
train_workload_sample_num = 250
test_workload_sample_num = 250

tcn_model.train(mode=True)
one_to_one_lstm_model.train(mode=True)
many_to_one_lstm_model.train(mode=True)

for epoch in range(epoch_number):
    one_to_one_lstm_running_loss = 0.0
    many_to_one_lstm_running_loss = 0.0
    tcn_running_loss = 0.0

    for i, data in enumerate(train_data_loader, 0):
        iteration += 1

        ####
        tcn_previous_workload_sequence: torch.Tensor = data[:, :, :-1]
        tcn_real_future_workload: torch.Tensor = data[:, :, -1]
        tcn_real_future_workload = tcn_real_future_workload.view(-1)

        one_to_one_lstm_previous_workload_sequence: torch.Tensor = data[:, :, -2:-1]
        one_to_one_lstm_real_future_workload: torch.Tensor = data[:, :, -1:]

        data_for_many_to_one_lstm = data.permute(0, 2, 1)
        many_to_one_lstm_previous_workload_sequence: torch.Tensor = data_for_many_to_one_lstm[:, :-1, :]
        many_to_one_lstm_real_future_workload: torch.Tensor = data_for_many_to_one_lstm[:, -1:, :]

        ####
        tcn_optimizer.zero_grad()
        one_to_one_lstm_optimizer.zero_grad()
        many_to_one_lstm_optimizer.zero_grad()

        ####
        tcn_output = tcn_model(tcn_previous_workload_sequence)
        one_to_one_lstm_output, _ = one_to_one_lstm_model(one_to_one_lstm_previous_workload_sequence)

        many_to_one_lstm_whole_output, _ = many_to_one_lstm_model(many_to_one_lstm_previous_workload_sequence)
        many_to_one_lstm_output = many_to_one_lstm_whole_output[:, -1:, :]

        ####
        tcn_mse_loss = tcn_mse_criterion(tcn_output, tcn_real_future_workload)

        one_to_one_lstm_mse_loss = one_to_one_lstm_mse_criterion(
            one_to_one_lstm_output,
            one_to_one_lstm_real_future_workload
        )

        many_to_one_lstm_mse_loss = many_to_one_lstm_mse_criterion(
            many_to_one_lstm_output,
            many_to_one_lstm_real_future_workload
        )

        ####
        tcn_l1_loss = tcn_l1_criterion(
            tcn_output,
            tcn_real_future_workload
        )

        one_to_one_lstm_l1_loss = one_to_one_lstm_l1_criterion(
            one_to_one_lstm_output,
            one_to_one_lstm_real_future_workload
        )

        many_to_one_lstm_l1_loss = many_to_one_lstm_l1_criterion(
            many_to_one_lstm_output,
            many_to_one_lstm_real_future_workload
        )

        ####
        real_future_workload_value = tcn_real_future_workload.item()

        tcn_predicted_future_workload_value = tcn_output.item()
        one_to_one_lstm_predicted_future_workload_value = one_to_one_lstm_output.item()
        many_to_one_lstm_predicted_future_workload_value = many_to_one_lstm_output.item()

        ####
        tcn_mse_loss.backward()
        one_to_one_lstm_mse_loss.backward()
        many_to_one_lstm_mse_loss.backward()

        ####
        tcn_optimizer.step()
        one_to_one_lstm_optimizer.step()
        many_to_one_lstm_optimizer.step()

        ####
        tcn_mse_loss_value = tcn_mse_loss.item()
        one_to_one_lstm_mse_loss_value = one_to_one_lstm_mse_loss.item()
        many_to_one_lstm_mse_loss_value = many_to_one_lstm_mse_loss.item()

        tcn_l1_loss_value = tcn_l1_loss.item()
        one_to_one_lstm_l1_loss_value = one_to_one_lstm_l1_loss.item()
        many_to_one_lstm_l1_loss_value = many_to_one_lstm_l1_loss.item()

        ####
        tcn_running_loss += tcn_mse_loss_value
        one_to_one_lstm_running_loss += one_to_one_lstm_mse_loss_value
        many_to_one_lstm_running_loss += many_to_one_lstm_mse_loss_value

        tcn_mse_loss_sum_for_plot += tcn_mse_loss_value
        one_to_one_lstm_mse_loss_sum_for_plot += one_to_one_lstm_mse_loss_value
        many_to_one_lstm_mse_loss_sum_for_plot += many_to_one_lstm_mse_loss_value

        tcn_l1_loss_sum_for_plot += tcn_l1_loss_value
        one_to_one_lstm_l1_loss_sum_for_plot += one_to_one_lstm_l1_loss_value
        many_to_one_lstm_l1_loss_sum_for_plot += many_to_one_lstm_l1_loss_value

        if i % 500 == 0 and i > 0:
            print('[%d, %5d] MSE loss (tcn, 1_to_1_lstm, n_to_1_lstm) --> : %.5f  %.5f  %.5f' %
                  (epoch + 1, i + 1,
                   tcn_running_loss / 500,
                   one_to_one_lstm_running_loss / 500,
                   many_to_one_lstm_running_loss / 500)
                  )

            print('real: {}  ---  got: {}  {}  {}\n'.format(
                real_future_workload_value,
                tcn_predicted_future_workload_value,
                one_to_one_lstm_predicted_future_workload_value,
                many_to_one_lstm_predicted_future_workload_value)
            )

            tcn_running_loss = 0.0
            one_to_one_lstm_running_loss = 0.0
            many_to_one_lstm_running_loss = 0.0

        if iteration % 100 == 0:
            plot_x_counter += 1
            train_avg_loss_x.append(plot_x_counter)

            denominator = 100

            tcn_train_avg_mse_loss_y.append(tcn_mse_loss_sum_for_plot / denominator)
            one_to_one_lstm_train_avg_mse_loss_y.append(one_to_one_lstm_mse_loss_sum_for_plot / denominator)
            many_to_one_lstm_train_avg_mse_loss_y.append(many_to_one_lstm_mse_loss_sum_for_plot / denominator)

            tcn_train_avg_l1_loss_y.append(tcn_l1_loss_sum_for_plot / denominator)
            one_to_one_lstm_train_avg_l1_loss_y.append(one_to_one_lstm_l1_loss_sum_for_plot / denominator)
            many_to_one_lstm_train_avg_l1_loss_y.append(many_to_one_lstm_l1_loss_sum_for_plot / denominator)

            tcn_mse_loss_sum_for_plot = 0
            one_to_one_lstm_mse_loss_sum_for_plot = 0
            many_to_one_lstm_mse_loss_sum_for_plot = 0

            tcn_l1_loss_sum_for_plot = 0
            one_to_one_lstm_l1_loss_sum_for_plot = 0
            many_to_one_lstm_l1_loss_sum_for_plot = 0

        if iteration > train_iterations_num - train_workload_sample_num:
            train_workload_sample_x.append(iteration)
            train_real_workload_sample_y.append(real_future_workload_value)
            train_tcn_predicted_workload_sample_y.append(tcn_predicted_future_workload_value)
            train_one_to_one_lstm_predicted_workload_sample_y.append(one_to_one_lstm_predicted_future_workload_value)
            train_many_to_one_lstm_predicted_workload_sample_y.append(many_to_one_lstm_predicted_future_workload_value)

plot_x_where_train_stopped = iteration / 100
iteration_where_train_stopped = iteration
train_avg_loss_x.append(plot_x_where_train_stopped)

tcn_train_avg_mse_loss_y.append(tcn_mse_loss_sum_for_plot / (iteration % 100))
one_to_one_lstm_train_avg_mse_loss_y.append(one_to_one_lstm_mse_loss_sum_for_plot / (iteration % 100))
many_to_one_lstm_train_avg_mse_loss_y.append(many_to_one_lstm_mse_loss_sum_for_plot / (iteration % 100))

tcn_train_avg_l1_loss_y.append(tcn_l1_loss_sum_for_plot / (iteration % 100))
one_to_one_lstm_train_avg_l1_loss_y.append(one_to_one_lstm_l1_loss_sum_for_plot / (iteration % 100))
many_to_one_lstm_train_avg_l1_loss_y.append(many_to_one_lstm_l1_loss_sum_for_plot / (iteration % 100))

test_avg_loss_x.append(plot_x_where_train_stopped)

tcn_test_avg_mse_loss_y.append(tcn_mse_loss_sum_for_plot / (iteration % 100))
one_to_one_lstm_test_avg_mse_loss_y.append(one_to_one_lstm_mse_loss_sum_for_plot / (iteration % 100))
many_to_one_lstm_test_avg_mse_loss_y.append(many_to_one_lstm_mse_loss_sum_for_plot / (iteration % 100))

tcn_test_avg_l1_loss_y.append(tcn_l1_loss_sum_for_plot / (iteration % 100))
one_to_one_lstm_test_avg_l1_loss_y.append(one_to_one_lstm_l1_loss_sum_for_plot / (iteration % 100))
many_to_one_lstm_test_avg_l1_loss_y.append(many_to_one_lstm_l1_loss_sum_for_plot / (iteration % 100))

tcn_mse_loss_sum_for_plot = 0
one_to_one_lstm_mse_loss_sum_for_plot = 0
many_to_one_lstm_mse_loss_sum_for_plot = 0

tcn_l1_loss_sum_for_plot = 0
one_to_one_lstm_l1_loss_sum_for_plot = 0
many_to_one_lstm_l1_loss_sum_for_plot = 0

print('Finished Training')
torch.save(tcn_model.state_dict(), 'trained_models/top_3_models_TCN_workload_model_nasa_dataset.pt')
torch.save(one_to_one_lstm_model.state_dict(), 'trained_models/one_to_one_LSTM_workload_model_nasa_dataset.pt')
torch.save(many_to_one_lstm_model.state_dict(), 'trained_models/many_to_one_LSTM_workload_model_nasa_dataset.pt')
print('Trained Models Saved')

print('\n\n\n')
print('start evaluation')
tcn_model.eval()
one_to_one_lstm_model.eval()
many_to_one_lstm_model.eval()

tcn_sum_of_mse_loss = 0
one_to_one_lstm_sum_of_mse_loss = 0
many_to_one_lstm_sum_of_mse_loss = 0

tcn_sum_of_l1_loss = 0
one_to_one_lstm_sum_of_l1_loss = 0
many_to_one_lstm_sum_of_l1_loss = 0

first_plot_x_test_count = True

for i, data in enumerate(test_data_loader, 0):
    iteration += 1

    ####
    tcn_previous_workload_sequence: torch.Tensor = data[:, :, :-1]
    tcn_real_future_workload: torch.Tensor = data[:, :, -1]
    tcn_real_future_workload = tcn_real_future_workload.view(-1)

    one_to_one_lstm_previous_workload_sequence: torch.Tensor = data[:, :, -2:-1]
    one_to_one_lstm_real_future_workload: torch.Tensor = data[:, :, -1:]

    data_for_many_to_one_lstm = data.permute(0, 2, 1)
    many_to_one_lstm_previous_workload_sequence: torch.Tensor = data_for_many_to_one_lstm[:, :-1, :]
    many_to_one_lstm_real_future_workload: torch.Tensor = data_for_many_to_one_lstm[:, -1:, :]

    ####
    tcn_output = tcn_model(tcn_previous_workload_sequence)
    one_to_one_lstm_output, _ = one_to_one_lstm_model(one_to_one_lstm_previous_workload_sequence)

    many_to_one_lstm_whole_output, _ = many_to_one_lstm_model(many_to_one_lstm_previous_workload_sequence)
    many_to_one_lstm_output = many_to_one_lstm_whole_output[:, -1:, :]

    ####
    tcn_mse_loss = tcn_mse_criterion(tcn_output, tcn_real_future_workload)

    one_to_one_lstm_mse_loss = one_to_one_lstm_mse_criterion(
        one_to_one_lstm_output,
        one_to_one_lstm_real_future_workload
    )

    many_to_one_lstm_mse_loss = many_to_one_lstm_mse_criterion(
        many_to_one_lstm_output,
        many_to_one_lstm_real_future_workload
    )

    ####
    tcn_l1_loss = tcn_l1_criterion(
        tcn_output,
        tcn_real_future_workload
    )

    one_to_one_lstm_l1_loss = one_to_one_lstm_l1_criterion(
        one_to_one_lstm_output,
        one_to_one_lstm_real_future_workload
    )

    many_to_one_lstm_l1_loss = many_to_one_lstm_l1_criterion(
        many_to_one_lstm_output,
        many_to_one_lstm_real_future_workload
    )

    ####
    real_future_workload_value = tcn_real_future_workload.item()

    tcn_predicted_future_workload_value = tcn_output.item()
    one_to_one_lstm_predicted_future_workload_value = one_to_one_lstm_output.item()
    many_to_one_lstm_predicted_future_workload_value = many_to_one_lstm_output.item()

    ####
    tcn_mse_loss_value = tcn_mse_loss.item()
    one_to_one_lstm_mse_loss_value = one_to_one_lstm_mse_loss.item()
    many_to_one_lstm_mse_loss_value = many_to_one_lstm_mse_loss.item()

    tcn_l1_loss_value = tcn_l1_loss.item()
    one_to_one_lstm_l1_loss_value = one_to_one_lstm_l1_loss.item()
    many_to_one_lstm_l1_loss_value = many_to_one_lstm_l1_loss.item()

    ####

    tcn_sum_of_mse_loss += tcn_mse_loss_value
    one_to_one_lstm_sum_of_mse_loss += one_to_one_lstm_mse_loss_value
    many_to_one_lstm_sum_of_mse_loss += many_to_one_lstm_mse_loss_value

    tcn_sum_of_l1_loss += tcn_l1_loss_value
    one_to_one_lstm_sum_of_l1_loss += one_to_one_lstm_l1_loss_value
    many_to_one_lstm_sum_of_l1_loss += many_to_one_lstm_l1_loss_value

    tcn_mse_loss_sum_for_plot += tcn_mse_loss_value
    one_to_one_lstm_mse_loss_sum_for_plot += one_to_one_lstm_mse_loss_value
    many_to_one_lstm_mse_loss_sum_for_plot += many_to_one_lstm_mse_loss_value

    tcn_l1_loss_sum_for_plot += tcn_l1_loss_value
    one_to_one_lstm_l1_loss_sum_for_plot += one_to_one_lstm_l1_loss_value
    many_to_one_lstm_l1_loss_sum_for_plot += many_to_one_lstm_l1_loss_value

    if i % 500 == 0 and i > 0:
        print('[%5d] MSE loss (tcn, 1_to_1_lstm, n_to_1_lstm) --> : %.5f  %.5f  %.5f' %
              (i + 1,
               tcn_sum_of_mse_loss / 500,
               one_to_one_lstm_sum_of_mse_loss / 500,
               many_to_one_lstm_sum_of_mse_loss / 500)
              )

        print('real: {}  ---  got: {}  {}  {}\n'.format(
            real_future_workload_value,
            tcn_predicted_future_workload_value,
            one_to_one_lstm_predicted_future_workload_value,
            many_to_one_lstm_predicted_future_workload_value)
        )

    if iteration % 100 == 0:
        plot_x_counter += 1
        test_avg_loss_x.append(plot_x_counter)

        if first_plot_x_test_count:
            denominator = 100 - (iteration_where_train_stopped % 100)
            first_plot_x_test_count = False
        else:
            denominator = 100

        tcn_test_avg_mse_loss_y.append(tcn_mse_loss_sum_for_plot / denominator)
        one_to_one_lstm_test_avg_mse_loss_y.append(one_to_one_lstm_mse_loss_sum_for_plot / denominator)
        many_to_one_lstm_test_avg_mse_loss_y.append(many_to_one_lstm_mse_loss_sum_for_plot / denominator)

        tcn_test_avg_l1_loss_y.append(tcn_l1_loss_sum_for_plot / denominator)
        one_to_one_lstm_test_avg_l1_loss_y.append(one_to_one_lstm_l1_loss_sum_for_plot / denominator)
        many_to_one_lstm_test_avg_l1_loss_y.append(many_to_one_lstm_l1_loss_sum_for_plot / denominator)

        tcn_mse_loss_sum_for_plot = 0
        one_to_one_lstm_mse_loss_sum_for_plot = 0
        many_to_one_lstm_mse_loss_sum_for_plot = 0

        tcn_l1_loss_sum_for_plot = 0
        one_to_one_lstm_l1_loss_sum_for_plot = 0
        many_to_one_lstm_l1_loss_sum_for_plot = 0

    if i < test_workload_sample_num:
        test_workload_sample_x.append(i)
        test_real_workload_sample_y.append(real_future_workload_value)
        test_tcn_predicted_workload_sample_y.append(tcn_predicted_future_workload_value)
        test_one_to_one_lstm_predicted_workload_sample_y.append(one_to_one_lstm_predicted_future_workload_value)
        test_many_to_one_lstm_predicted_workload_sample_y.append(many_to_one_lstm_predicted_future_workload_value)

test_stopping_plot_x = iteration / 100
test_avg_loss_x.append(test_stopping_plot_x)

tcn_test_avg_mse_loss_y.append(tcn_mse_loss_sum_for_plot / (iteration % 100))
one_to_one_lstm_test_avg_mse_loss_y.append(one_to_one_lstm_mse_loss_sum_for_plot / (iteration % 100))
many_to_one_lstm_test_avg_mse_loss_y.append(many_to_one_lstm_mse_loss_sum_for_plot / (iteration % 100))

tcn_test_avg_l1_loss_y.append(tcn_l1_loss_sum_for_plot / (iteration % 100))
one_to_one_lstm_test_avg_l1_loss_y.append(one_to_one_lstm_l1_loss_sum_for_plot / (iteration % 100))
many_to_one_lstm_test_avg_l1_loss_y.append(many_to_one_lstm_l1_loss_sum_for_plot / (iteration % 100))

tcn_mse_loss_sum_for_plot = 0
one_to_one_lstm_mse_loss_sum_for_plot = 0
many_to_one_lstm_mse_loss_sum_for_plot = 0

tcn_l1_loss_sum_for_plot = 0
one_to_one_lstm_l1_loss_sum_for_plot = 0
many_to_one_lstm_l1_loss_sum_for_plot = 0

print("TCN average total MSE loss: ", tcn_sum_of_mse_loss / len(test_data_loader))
print("1 to 1 LSTM average total MSE loss: ", one_to_one_lstm_sum_of_mse_loss / len(test_data_loader))
print("n to 1 LSTM average total MSE loss: ", many_to_one_lstm_sum_of_mse_loss / len(test_data_loader))

print("TCN average total L1 loss: ", tcn_sum_of_l1_loss / len(test_data_loader))
print("1 to 1 LSTM average total L1 loss: ", one_to_one_lstm_sum_of_l1_loss / len(test_data_loader))
print("n to 1 LSTM average total L1 loss: ", many_to_one_lstm_sum_of_l1_loss / len(test_data_loader))

# draw loss plots
plt.figure(figsize=[12.0, 8.0])
plt.title('Prediction Error (MSE Loss)')
plt.xlabel("Time")
plt.ylabel("Total Loss")

a1 = len(train_avg_loss_x)
a2 = len(tcn_train_avg_mse_loss_y)
a3 = len(one_to_one_lstm_train_avg_mse_loss_y)
a4 = len(many_to_one_lstm_train_avg_mse_loss_y)

a5 = len(test_avg_loss_x)
a6 = len(tcn_test_avg_mse_loss_y)
a7 = len(one_to_one_lstm_test_avg_mse_loss_y)
a8 = len(many_to_one_lstm_test_avg_mse_loss_y)

plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.15])
plt.plot(train_avg_loss_x, tcn_train_avg_mse_loss_y, 'g-', label='TCN Training Loss')
plt.plot(train_avg_loss_x, one_to_one_lstm_train_avg_mse_loss_y, 'b-', label='One-to-One LSTM Training Loss')
plt.plot(train_avg_loss_x, many_to_one_lstm_train_avg_mse_loss_y, 'r-', label='Many-to-One LSTM Training Loss')
plt.plot(test_avg_loss_x, tcn_test_avg_mse_loss_y, 'c-', label='TCN Testing Loss')
plt.plot(test_avg_loss_x, one_to_one_lstm_test_avg_mse_loss_y, 'm-', label='One-to-One LSTM Testing Loss')
plt.plot(test_avg_loss_x, many_to_one_lstm_test_avg_mse_loss_y, 'k-', label='Many-to-One LSTM Testing Loss')
plt.legend(loc='upper left')
plt.savefig('top_3_models_mse_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Prediction Error (L1 Loss)')
plt.xlabel("Time")
plt.ylabel("Total Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.8])
plt.plot(train_avg_loss_x, tcn_train_avg_l1_loss_y, 'g-', label='TCN Training Loss')
plt.plot(train_avg_loss_x, one_to_one_lstm_train_avg_l1_loss_y, 'b-', label='One-to-One LSTM Training Loss')
plt.plot(train_avg_loss_x, many_to_one_lstm_train_avg_l1_loss_y, 'r-', label='Many-to-One LSTM Training Loss')
plt.plot(test_avg_loss_x, tcn_test_avg_l1_loss_y, 'c-', label='TCN Testing Loss')
plt.plot(test_avg_loss_x, one_to_one_lstm_test_avg_l1_loss_y, 'm-', label='One-to-One LSTM Testing Loss')
plt.plot(test_avg_loss_x, many_to_one_lstm_test_avg_l1_loss_y, 'k-', label='Many-to-One LSTM Testing Loss')
plt.legend(loc='upper left')
plt.savefig('top_3_models_l1_loss_plot.png')
plt.show()
plt.close()

# draw workload plots
plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future normalized workload (Training)')
plt.xlabel("Samples")
plt.ylabel("Normalized Workload")
plt.axis([train_iterations_num - train_workload_sample_num, train_iterations_num + 1, 0, 1])
plt.plot(train_workload_sample_x, train_real_workload_sample_y, 'r-', label='Real workload')
plt.plot(train_workload_sample_x, train_tcn_predicted_workload_sample_y,
         'g-', label='TCN Predicted workload')
plt.plot(train_workload_sample_x, train_one_to_one_lstm_predicted_workload_sample_y,
         'b-', label='One-to-One LSTM Predicted workload')
plt.plot(train_workload_sample_x, train_many_to_one_lstm_predicted_workload_sample_y,
         'k-', label='Many-to-One LSTM Predicted workload')
plt.legend(loc='upper left')
plt.savefig('top_3_models_train_real_predicted_workload_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future normalized workload (Testing)')
plt.xlabel("Samples")
plt.ylabel("Normalized Workload")
plt.axis([0, test_workload_sample_num + 1, 0, 1])
plt.plot(test_workload_sample_x, test_real_workload_sample_y, 'r-', label='Real workload')
plt.plot(test_workload_sample_x, test_tcn_predicted_workload_sample_y,
         'g-', label='TCN Predicted workload')
plt.plot(test_workload_sample_x, test_one_to_one_lstm_predicted_workload_sample_y,
         'b-', label='One-to-One LSTM Predicted workload')
plt.plot(test_workload_sample_x, test_many_to_one_lstm_predicted_workload_sample_y,
         'k-', label='Many-to-One LSTM Predicted workload')
plt.legend(loc='upper left')
plt.savefig('top_3_models_test_real_predicted_workload_plot.png')
plt.show()
plt.close()
