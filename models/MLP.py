import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLayerMLP(nn.Module):
    def __init__(self):
        super(OneLayerMLP, self).__init__()
        self.linear = nn.Linear(125, 4)

    def forward(self, x):
        x = x.mean(axis=-1)
        x = self.linear(x)
        return x


