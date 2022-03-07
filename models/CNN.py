import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, channels):
        """
        A Basic conv block consisting of
        skip ->
        conv
        bn
        relu
        conv
        bn + skip <-
        relu
        :param channels: the number of channels for this basic block
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) + x
        out = F.relu(out)
        return out


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        """
        A simple CNN to process image data.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(22, 22, 5, stride=5)  # (B, 22, 1000) -> (B, 22, 200)
        self.bn1 = nn.BatchNorm1d(22)  # (B, 22, 200) -> (B, 22, 200)

        # basic block
        self.b1 = BasicBlock(22)  # (B, 22, 200) -> (B, 22, 200)

        # cut the number of channels
        self.conv2 = nn.Conv1d(22, 10, 1)
        self.bn2 = nn.BatchNorm1d(10)
        # relu

        self.m1 = nn.MaxPool1d(2)  # downsampling, max pooling (2) # (B, 10, 200) -> (B, 10, 100)

        self.b2 = BasicBlock(10)

        self.m2 = nn.MaxPool1d(2)  # downsampling, max pooling (2) # (B, 10, 100) -> (B, 10, 50)

        self.b3 = BasicBlock(10)

        # self.m3 = nn.MaxPool1d(2)  # downsampling, max pooling (2) # (B, 10, 100) -> (B, 10, 50)

        # flatten

        # linear layer
        self.linear = nn.Linear(10*50, 4)

    def forward(self, x):
        # downsampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # first block and 1x1 channel downscaling
        x = self.b1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.m1(x)

        # second block
        x = self.b2(x)
        x = self.m2(x)

        # third block
        x = self.b3(x)
        # x = self.m3(x)

        # flatten
        x = torch.flatten(x, start_dim=1)

        # head
        x = self.linear(x)
        return x