import torch
from torch import nn
from TCN.tcn import TemporalConvNet


class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, sequence_length):
        super(TCNModel, self).__init__()
        self.tcn: TemporalConvNet = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(sequence_length, output_size)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        y1: torch.Tensor = self.tcn(x)
        y1 = y1.squeeze()
        return self.linear(y1)
