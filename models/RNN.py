import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_hidden_state(lstm: nn.LSTM, batch_size, device):
    c0 = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size, requires_grad=True).to(device)
    h0 = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size, requires_grad=True).to(device)
    return h0, c0


class SimpleRNN(nn.Module):
    def __init__(self, input_size=22, hidden_size=10, num_layers=2, num_classes=4, **kwargs):
        """
        A simple model consisting of an RNN, with a linear layer.  Given an input of size
        (batch_size, seq_len, feature_size), will return an output of size (batch_size, n_classes).  The entire input
        sequence will be passed through the network, then the final hidden state will be passed through a linear layer
        to create the logits.

        :param input_size: The n_features for the input
        :param hidden_size: The size of the hidden state in the RNN
        :param num_layers: The number of layers in the RNN
        """
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity='relu',
                          batch_first=True, **kwargs)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # pass through rnn
        x, _ = self.rnn(x)
        # look at hidden state at last time step
        x = x[:, -1, :]
        x = F.relu(x)
        x = self.linear(x)
        return x


class SimpleLSTM(nn.Module):
    def __init__(self):
        """
        A Simple LSTM model with a conv-filter.
        """
        super(SimpleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=22, hidden_size=50, batch_first=True)
        self.linear1 = nn.Linear(in_features=50, out_features=4)

    def forward(self, x):
        h0 = torch.zeros(1, len(x), 50).requires_grad_().to(x.device)
        c0 = torch.zeros(1, len(x), 50).requires_grad_().to(x.device)
        x, _ = self.lstm1(x, (h0, c0))
        x = x[:, -1, :]
        x = self.linear1(x)
        return x


class LSTM(nn.Module):
    def __init__(self, dropout=0, sequential_outputs=False):
        """
        A more refined LSTM model.
        """
        self.dropout = dropout
        self.sequential_targets = sequential_outputs

        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=22, hidden_size=64, dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(64, 4)

    def forward(self, x):
        batch_size = len(x)
        device = x.device
        hc1 = _init_hidden_state(self.lstm1, batch_size, device)
        x, _ = self.lstm1(x, hc1)
        hc2 = _init_hidden_state(self.lstm2, batch_size, device)
        x, _ = self.lstm2(x, hc2)
        if not self.sequential_targets:
            x = x[:, -1, :]
        x = self.linear(x)
        return x

