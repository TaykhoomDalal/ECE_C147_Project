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

class CNN_GRU(nn.Module):
    def __init__(self, num_classes=4):
        """
        CNN + GRU layer
        """
        super(CNN_GRU, self).__init__()
        self.conv1 = nn.Conv1d(22, 22, 10, stride=1) 
        self.bn1 = nn.BatchNorm1d(22) 
        self.mxp1 = nn.MaxPool1d(2)

        # self.conv2 = nn.Conv1d(22, 22, 10, stride=1)
        # self.bn2 = nn.BatchNorm1d(22) 
        # self.mxp2 = nn.MaxPool1d(4)

        self.gru1 = nn.GRU(input_size = 95, hidden_size = 64, num_layers=1)

        self.drp1 = nn.Dropout(0.5)
        self.flat1 = nn.Flatten()
        self.linear1 = nn.Linear(1408, 64) 
        self.drp2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(64, 4)

    def forward(self, x, model):

        # downsampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mxp1(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.mxp2(x)


        batch_size = len(x)
        device = x.device

        h0 = torch.zeros(self.gru1.num_layers, 22, self.gru1.hidden_size, requires_grad=True).to(device)
        x, _ = self.gru1(x, h0)
        x = self.drp1(x)
        x = self.flat1(x)

        x = self.linear1(x)
        x = self.drp2(x)
        x = self.linear2(x)

        # # first block and 1x1 channel downscaling
        # x = self.b1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = self.m1(x)

        # # second block
        # x = self.b2(x)
        # x = self.m2(x)

        # # third block
        # x = self.b3(x)
        # # x = self.m3(x)

        # # flatten
        # x = torch.flatten(x, start_dim=1)

        # # head
        # x = self.linear(x)
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

class DeepCNN(nn.Module):
    def __init__(self, num_classes=4):
        """
        4 Convolution layer network
        """
        super(DeepCNN, self).__init__()
        
        #layer 1
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=25, kernel_size=(10, 1), stride=1, padding = 'same') 
        self.mxp1 = nn.MaxPool2d(kernel_size=(3, 1), stride=3, padding=0)
        self.bn1 = nn.BatchNorm2d(25)
        self.drp1 = nn.Dropout(0.5)

        #layer 2
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(10, 1), stride=1, padding = 'same')
        self.mxp2 = nn.MaxPool2d(kernel_size=(3, 1), stride=3, padding=0)
        self.bn2 = nn.BatchNorm2d(50)
        self.drp2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(10, 1), stride=1, padding = 'same')
        self.mxp3 = nn.MaxPool2d(kernel_size=(3, 1), stride=3, padding=0)
        self.bn3 = nn.BatchNorm2d(100)
        self.drp3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(10, 1), stride=1, padding = 'same')
        self.mxp4 = nn.MaxPool2d(kernel_size=(3, 1), stride=3, padding=0)
        self.bn4 = nn.BatchNorm2d(200)
        self.drp4 = nn.Dropout(0.5)

        # self.conv5 = nn.Conv1d(in_channels=200, out_channels=300, kernel_size=10, stride=1, dilation = 2, padding = 'same') 
        # self.mxp5 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        # self.drp5 = nn.Dropout(0.5)

        self.flat1 = nn.Flatten()
        self.lineartrain = nn.Linear(600, 4) 

        #test mode
        self.lineartest = nn.Linear(200, 4)

    def forward(self, x, mode):
        # conv -> maxpool -> dropout -> flatten -> linear
        x = self.conv1(x)
        x = F.elu(x)
        x = self.mxp1(x)
        x = self.bn1(x)
        x = self.drp1(x)

        x = self.conv2(x)
        x = F.elu(x)
        x = self.mxp2(x)
        x = self.bn2(x)
        x = self.drp2(x)

        x = self.conv3(x)
        x = F.elu(x)
        x = self.mxp3(x)
        x = self.bn3(x)
        x = self.drp1(x)

        x = self.conv4(x)
        x = F.elu(x)
        x = self.mxp4(x)
        x = self.bn4(x)
        x = self.drp4(x)

        # x = self.conv5(x)
        # x = self.mxp5(x)
        # x = self.drp5(x)

        x = self.flat1(x)
        x = self.lineartrain(x)
        # if mode == 'test':
        #     x = self.lineartest(x)
        # else:
        #     x = self.lineartrain(x)

        return x

class MiddleCNN(nn.Module):
    def __init__(self, num_classes=4):
        """
        4 Convolution layer network
        """
        super(MiddleCNN, self).__init__()
        
        #layer 1
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=64, kernel_size=3, stride=1, padding = 'same') 
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.mxp1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)

        #layer 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding = 'same') 
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(64)
        self.mxp2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)

        #flatten for linear layers
        self.flat1 = nn.Flatten()

        #layer 3
        self.fc1 = nn.Linear(63872, 1024) 
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.drp1 = nn.Dropout(0.5)

        #layer 4
        self.fc2 = nn.Linear(1024, 512) 
        torch.nn.init.xavier_uniform(self.fc2.weight)
        self.drp2 = nn.Dropout(0.5)

        #layer 5
        self.fc3 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform(self.fc3.weight) 
        self.drp3 = nn.Dropout(0.5)

        #layer 6
        self.fc4 = nn.Linear(256, 4) 
        torch.nn.init.xavier_uniform(self.fc4.weight)

    def forward(self, x, mode):

        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mxp1(x)
        
        # layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mxp2(x)
        
        # flatten
        x = self.flat1(x)

        #layer 3
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drp1(x)

        #layer 4
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drp2(x)

        #layer 5
        x = self.fc3(x)
        x = F.relu(x)
        x = self.drp3(x)

        # layer 6
        x = self.fc4(x)

        return x