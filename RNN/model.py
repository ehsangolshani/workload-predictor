import torch
from torch import nn


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, batch_size):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn: nn.RNN = nn.RNN(input_size=input_size,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True)

        self.hidden_state = self.init_hidden(batch_size=batch_size)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Passing in the input and hidden state into the model and obtaining outputs
        output, hidden = self.rnn(x, self.hidden_state)
        self.hidden_state = hidden.detach()

        # # Reshaping the outputs such that it can be fit into the fully connected layer
        # output = output.contiguous().view(-1, self.hidden_dim)

        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return torch.randn(self.num_layers, batch_size, self.hidden_dim)
