import torch
from torch import nn
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, sequence_length):
        super(TCN, self).__init__()
        self.tcn: TemporalConvNet = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(sequence_length, output_size)

    def forward(self, x):
        y1: torch.Tensor = self.tcn(x)
        y1 = y1.squeeze()
        a = y1.size()
        return self.linear(y1)
