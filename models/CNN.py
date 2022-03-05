from re import X
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

class ShallowCNN(nn.Module):
    def __init__(self, num_classes=4):
        """
        1 Convolution layer network
        """
        super(ShallowCNN, self).__init__()
        
        #lr = 5e-4
        self.conv1d = nn.Conv1d(in_channels=22, out_channels=25, kernel_size=10, stride=1, padding = 0) 
        self.mxp1d = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        self.drp1 = nn.Dropout(0.5)
        self.flat1 = nn.Flatten()
        self.linear1 = nn.Linear(1575, 4) 

    def forward(self, x):
        
        # conv -> maxpool -> dropout -> flatten -> linear
        x = self.conv1d(x)
        x = self.mxp1d(x)
        x = self.drp1(x)
        x = self.flat1(x)
        x = self.linear1(x)

        return x