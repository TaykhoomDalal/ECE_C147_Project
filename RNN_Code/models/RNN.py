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
        self.rnn = nn.RNN(input_size=22, hidden_size=10, num_layers=1, nonlinearity='relu',
                          batch_first=True, **kwargs)
        self.linear = nn.Linear(10, 4)
        # self.bn = nn.BatchNorm1d(hidden_size)

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
        batch_size = len(x)
        device = x.device
        hc = _init_hidden_state(self.lstm1, batch_size, device)
        x, _ = self.lstm1(x, hc)
        x = x[:, -1, :]
        x = self.linear1(x)
        return x


class LSTM(nn.Module):
    def __init__(self, dropout):
        """
        A more refined LSTM model.
        """
        self.dropout = dropout
        self.sequential_targets = False

        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=22, hidden_size=64, num_layers=2, dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(64, 4)

    def forward(self, x):
        batch_size = len(x)
        device = x.device
        hc1 = _init_hidden_state(self.lstm1, batch_size, device)
        x, _ = self.lstm1(x, hc1)
        x = F.dropout(x, self.dropout)
        hc2 = _init_hidden_state(self.lstm2, batch_size, device)
        x, _ = self.lstm2(x, hc2)
        x = F.dropout(x, self.dropout)
        if not self.sequential_targets:
            x = x[:, -1, :]
        x = self.linear(x)
        return x


class OneLayerLSTM(nn.Module):
    def __init__(self, dropout):
        """
        A recurrent model with one layer of lstm, and 2 layers of linear.
        :param dropout: dropout in between layers
        """
        super(OneLayerLSTM, self).__init__()
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size=22, hidden_size=32, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.linear2 = nn.Linear(16, 4)

    def forward(self, x):
        batch_size = len(x)
        device = x.device
        hc1 = _init_hidden_state(self.lstm1, batch_size, device)
        x, _ = self.lstm1(x, hc1)
        x = F.dropout(x, self.dropout)
        x = x[:, -1, :]
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.linear2(x)
        return x


class LinearLSTMLinear(nn.Module):
    def __init__(self, dropout):
        super(LinearLSTMLinear, self).__init__()
        self.dropout = dropout
        self.linear1 = nn.Linear(22, 32)
        self.linear2 = nn.Linear(32, 32)
        self.lstm1 = nn.LSTM(input_size=34, hidden_size=64, num_layers=3, batch_first=True)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 4)

    def forward(self, x):
        batch_size = len(x)
        device = x.device

        # get min and max
        x_max = torch.max(x, axis=-1, keepdims=True).values
        x_min = torch.min(x, axis=-1, keepdims=True).values
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # concatenate min and max to feed to lstm
        x = torch.cat([x, x_max, x_min], axis=-1)

        # lstm
        hc = _init_hidden_state(self.lstm1, batch_size, device)
        x, _ = self.lstm1(x, hc)

        # grab last time step of lstm
        x = x[:, -1, :]

        # last 2 linear layers
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear3(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear4(x)
        return x


class SimpleGRU(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.num_layers = 1
        self.hidden_size = 32

        self.dropout = dropout
        self.linear1 = nn.Linear(22, 16)
        self.gru = nn.GRU(input_size=16, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear2 = nn.Linear(32, 4)

    def forward(self, x):
        batch_size = len(x)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).to(device)
        x = self.linear1(x)
        x = F.relu(x)
        x, _ = self.gru(x, h0)
        x = x[:, -1, :]
        x = self.linear2(x)
        return x


class TwoLayerGRU(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.num_layers = 2
        self.hidden_size = 32

        self.dropout = dropout
        self.linear1 = nn.Linear(22, 16)
        self.gru = nn.GRU(input_size=16, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear2 = nn.Linear(32, 4)

    def forward(self, x):
        batch_size = len(x)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).to(device)
        x = self.linear1(x)
        x = F.relu(x)
        x, _ = self.gru(x, h0)
        x = x[:, -1, :]
        x = self.linear2(x)
        return x


class TwoLayerBigGRU(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.num_layers = 2
        self.hidden_size = 64

        self.dropout = dropout
        self.linear1 = nn.Linear(22, 32)
        self.gru = nn.GRU(input_size=32, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear2 = nn.Linear(64, 4)

    def forward(self, x):
        batch_size = len(x)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).to(device)
        x = self.linear1(x)
        x = F.relu(x)
        x, _ = self.gru(x, h0)
        x = x[:, -1, :]
        x = self.linear2(x)
        return x


class fourLayerGRU(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.num_layers = 4
        self.hidden_size = 32

        self.dropout = dropout
        self.linear1 = nn.Linear(22, 32)
        self.gru = nn.GRU(input_size=32, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.linear2 = nn.Linear(self.hidden_size, 4)

    def forward(self, x):
        batch_size = len(x)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).to(device)
        x = self.linear1(x)
        x = F.relu(x)
        x, _ = self.gru(x, h0)
        x = x[:, -1, :]
        x = self.linear2(x)
        return x


