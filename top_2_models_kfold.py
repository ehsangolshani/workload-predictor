import torch
from torch import nn
from LSTM.model import LSTMModel
from TCN.model import TCNModel
from torch.utils import data
from custom_datasets.windowed_dataset import WindowedWorkloadDataset
import torch.optim as optim
from sklearn.model_selection import RepeatedKFold

tcn_window_size = 24
many_to_one_lstm_tcn_window_size = 20

workload_dataset = WindowedWorkloadDataset(
    csv_path='raw_dataset/nasa_http/nasa_temporal_rps_1m.csv',
    window_size=tcn_window_size + 1
)

number_of_splits = 3
number_of_repeats = 3

tcn_mse_loss_avg_sum = 0
many_to_one_lstm_mse_loss_avg_sum = 0
tcn_l1_loss_avg_sum = 0
many_to_one_lstm_l1_loss_avg_sum = 0

repeated_k_fold = RepeatedKFold(n_splits=number_of_splits, n_repeats=number_of_repeats)

# TODO: make LSTM usable for data sampled non-sequentially

k = 0

for train_indices, test_indices in repeated_k_fold.split(workload_dataset):

    train_dataset = data.Subset(dataset=workload_dataset, indices=train_indices)
    test_dataset = data.Subset(dataset=workload_dataset, indices=test_indices)

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
                                   kernel_size=kernel_size, dropout=tcn_dropout, sequence_length=tcn_window_size)

    tcn_mse_criterion = nn.MSELoss()  # this is used for training phase
    tcn_l1_criterion = nn.L1Loss()

    tcn_optimizer = optim.Adam(params=tcn_model.parameters(), lr=1e-4)

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
                                                  dropout=many_to_one_lstm_dropout)

    many_to_one_lstm_mse_criterion = nn.MSELoss()  # this is used for training phase
    many_to_one_lstm_l1_criterion = nn.L1Loss()

    many_to_one_lstm_optimizer = optim.Adam(params=many_to_one_lstm_model.parameters(), lr=1e-4)

    #######################

    tcn_model.train(mode=True)
    many_to_one_lstm_model.train(mode=True)

    repeat_number = (k // number_of_splits) + 1
    fold_number = (k % number_of_splits) + 1

    print("*** repeat: {}, fold: {} ***\n".format(repeat_number, fold_number))

    for epoch in range(epoch_number):
        tcn_running_loss = 0.0
        many_to_one_lstm_running_loss = 0.0

        for i, train_data in enumerate(train_data_loader, 0):

            ####
            tcn_previous_workload_sequence: torch.Tensor = train_data[:, :, :-1]
            tcn_real_future_workload: torch.Tensor = train_data[:, :, -1]
            tcn_real_future_workload = tcn_real_future_workload.view(-1)

            data_for_many_to_one_lstm = train_data.permute(0, 2, 1)
            many_to_one_lstm_previous_workload_sequence: torch.Tensor = data_for_many_to_one_lstm[:,
                                                                        -1 - many_to_one_lstm_tcn_window_size:-1, :]
            many_to_one_lstm_real_future_workload: torch.Tensor = data_for_many_to_one_lstm[:, -1:, :]

            ####
            tcn_optimizer.zero_grad()
            many_to_one_lstm_optimizer.zero_grad()

            ####
            tcn_output = tcn_model(tcn_previous_workload_sequence)

            many_to_one_lstm_whole_output, _ = many_to_one_lstm_model(many_to_one_lstm_previous_workload_sequence)
            many_to_one_lstm_output = many_to_one_lstm_whole_output[:, -1:, :]

            ####
            tcn_mse_loss = tcn_mse_criterion(tcn_output, tcn_real_future_workload)

            many_to_one_lstm_mse_loss = many_to_one_lstm_mse_criterion(
                many_to_one_lstm_output,
                many_to_one_lstm_real_future_workload
            )

            ####
            tcn_l1_loss = tcn_l1_criterion(
                tcn_output,
                tcn_real_future_workload
            )

            many_to_one_lstm_l1_loss = many_to_one_lstm_l1_criterion(
                many_to_one_lstm_output,
                many_to_one_lstm_real_future_workload
            )

            ####
            real_future_workload_value = tcn_real_future_workload.item()

            tcn_predicted_future_workload_value = tcn_output.item()
            many_to_one_lstm_predicted_future_workload_value = many_to_one_lstm_output.item()

            ####
            tcn_mse_loss.backward()
            many_to_one_lstm_mse_loss.backward()

            ####
            tcn_optimizer.step()
            many_to_one_lstm_optimizer.step()

            ####
            tcn_mse_loss_value = tcn_mse_loss.item()
            many_to_one_lstm_mse_loss_value = many_to_one_lstm_mse_loss.item()

            tcn_l1_loss_value = tcn_l1_loss.item()
            many_to_one_lstm_l1_loss_value = many_to_one_lstm_l1_loss.item()

            ####
            tcn_running_loss += tcn_mse_loss_value
            many_to_one_lstm_running_loss += many_to_one_lstm_mse_loss_value

            if i % 500 == 0 and i > 0:
                print('[%d, %5d] MSE loss (tcn, n_to_1_lstm) --> : %.5f   %.5f' %
                      (epoch + 1, i + 1,
                       tcn_running_loss / 500,
                       many_to_one_lstm_running_loss / 500)
                      )

                print('real: {}  ---  got: {}  {}\n'.format(
                    real_future_workload_value,
                    tcn_predicted_future_workload_value,
                    many_to_one_lstm_predicted_future_workload_value)
                )

                tcn_running_loss = 0.0
                many_to_one_lstm_running_loss = 0.0

    print('Finished Training')

    # torch.save(tcn_model.state_dict(), 'trained_models/top_2_models_TCN_workload_model_nasa_dataset.pt')
    # torch.save(many_to_one_lstm_model.state_dict(), 'trained_models/many_to_one_LSTM_workload_model_nasa_dataset.pt')
    # print('Trained Models Saved')

    print('start evaluation\n')
    tcn_model.eval()
    many_to_one_lstm_model.eval()

    tcn_sum_of_mse_loss = 0
    many_to_one_lstm_sum_of_mse_loss = 0

    tcn_sum_of_l1_loss = 0
    many_to_one_lstm_sum_of_l1_loss = 0

    for i, test_data in enumerate(test_data_loader, 0):

        ####
        tcn_previous_workload_sequence: torch.Tensor = test_data[:, :, :-1]
        tcn_real_future_workload: torch.Tensor = test_data[:, :, -1]
        tcn_real_future_workload = tcn_real_future_workload.view(-1)

        data_for_many_to_one_lstm = test_data.permute(0, 2, 1)
        many_to_one_lstm_previous_workload_sequence: torch.Tensor = data_for_many_to_one_lstm[:,
                                                                    -1 - many_to_one_lstm_tcn_window_size:-1, :]
        many_to_one_lstm_real_future_workload: torch.Tensor = data_for_many_to_one_lstm[:, -1:, :]

        ####
        tcn_output = tcn_model(tcn_previous_workload_sequence)

        many_to_one_lstm_whole_output, _ = many_to_one_lstm_model(many_to_one_lstm_previous_workload_sequence)
        many_to_one_lstm_output = many_to_one_lstm_whole_output[:, -1:, :]

        ####
        tcn_mse_loss = tcn_mse_criterion(tcn_output, tcn_real_future_workload)

        many_to_one_lstm_mse_loss = many_to_one_lstm_mse_criterion(
            many_to_one_lstm_output,
            many_to_one_lstm_real_future_workload
        )

        ####
        tcn_l1_loss = tcn_l1_criterion(
            tcn_output,
            tcn_real_future_workload
        )

        many_to_one_lstm_l1_loss = many_to_one_lstm_l1_criterion(
            many_to_one_lstm_output,
            many_to_one_lstm_real_future_workload
        )

        ####
        real_future_workload_value = tcn_real_future_workload.item()

        tcn_predicted_future_workload_value = tcn_output.item()
        many_to_one_lstm_predicted_future_workload_value = many_to_one_lstm_output.item()

        ####
        tcn_mse_loss_value = tcn_mse_loss.item()
        many_to_one_lstm_mse_loss_value = many_to_one_lstm_mse_loss.item()

        tcn_l1_loss_value = tcn_l1_loss.item()
        many_to_one_lstm_l1_loss_value = many_to_one_lstm_l1_loss.item()

        ####

        tcn_sum_of_mse_loss += tcn_mse_loss_value
        many_to_one_lstm_sum_of_mse_loss += many_to_one_lstm_mse_loss_value

        tcn_sum_of_l1_loss += tcn_l1_loss_value
        many_to_one_lstm_sum_of_l1_loss += many_to_one_lstm_l1_loss_value

        if i % 500 == 0 and i > 0:
            print('[%5d] MSE loss (tcn, n_to_1_lstm) --> : %.5f  %.5f' %
                  (i + 1,
                   tcn_sum_of_mse_loss / 500,
                   many_to_one_lstm_sum_of_mse_loss / 500)
                  )

            print('real: {}  ---  got: {}  {}\n'.format(
                real_future_workload_value,
                tcn_predicted_future_workload_value,
                many_to_one_lstm_predicted_future_workload_value)
            )

    a1 = tcn_sum_of_mse_loss / len(test_data_loader)
    a2 = many_to_one_lstm_sum_of_mse_loss / len(test_data_loader)
    a3 = tcn_sum_of_l1_loss / len(test_data_loader)
    a4 = many_to_one_lstm_sum_of_l1_loss / len(test_data_loader)

    print("stats for repeat: {} and fold: {}".format(repeat_number, fold_number))
    print("TCN average total MSE loss: ", a1)
    print("many to 1 LSTM average total MSE loss: ", a2)
    print("TCN average total L1 loss: ", a3)
    print("many to 1 LSTM average total L1 loss: ", a4)
    print("-----------------------------------------------------\n")

    tcn_mse_loss_avg_sum += a1
    many_to_one_lstm_mse_loss_avg_sum += a2
    tcn_l1_loss_avg_sum += a3
    many_to_one_lstm_l1_loss_avg_sum += a4

    k += 1

print("*****************************")
print("*******  final stats  *******")
print("*****************************")
print("\n")

print("TCN average total MSE loss: ",
      tcn_mse_loss_avg_sum / (number_of_splits * number_of_repeats))

print("many to 1 LSTM average total MSE loss: ",
      many_to_one_lstm_mse_loss_avg_sum / (number_of_splits * number_of_repeats))

print("TCN average total L1 loss: ",
      tcn_l1_loss_avg_sum / (number_of_splits * number_of_repeats))

print("many to 1 LSTM average total L1 loss: ",
      many_to_one_lstm_l1_loss_avg_sum / (number_of_splits * number_of_repeats))
