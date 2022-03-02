import torch
import torch.nn as nn
import torch.nn.functional as F

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
        h0 = torch.zeros(1, len(x), 50).requires_grad_()
        c0 = torch.zeros(1, len(x), 50).requires_grad_()
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
        self.sequential_targets=sequential_outputs

        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=22, hidden_size=64, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(64, 64)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, dropout=dropout, batch_first=True)
        self.linear2 = nn.Linear(64, 64)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, dropout=dropout, batch_first=True)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 4)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.linear1(x)
        x = F.dropout(x, p=self.dropout)
        x = F.relu(x)

        x, _ = self.lstm2(x)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout)
        x = F.relu(x)

        x, _ = self.lstm3(x)
        x = self.linear3(x)
        x = F.dropout(x, p=self.dropout)
        x = F.relu(x)

        if not self.sequential_targets:
            x = x[:, -1, :]

        x = self.linear4(x)
        return x


