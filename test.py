from model import TCN

hidden_units_per_layer = 10
levels = 5
channel_sizes = [hidden_units_per_layer] * levels
input_channels = 1
kernel_size = 8
dropout = 0.0

model = TCN(input_size=input_channels, num_channels=channel_sizes, kernel_size=kernel_size, dropout=dropout)
