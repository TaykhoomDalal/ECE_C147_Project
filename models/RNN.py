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
                          **kwargs)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.rnn(x)
        x = F.relu(x)
        x = self.linear(x[:, -1, :])
        return x
