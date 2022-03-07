import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 22, out_channels = 40,kernel_size=(25,1))
        self.avgpool1 = nn.AvgPool2d(kernel_size=(75,1),stride=15)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2440,4)
        # Flatten first and do a FCN to have [128,4]
        # torch.reshape(size[0], -1)

    def forward(self,x):
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1],1))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        print("reached the end")

        return x


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 22, out_channels = 25,kernel_size=(10,1),padding = 'same')
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,1),stride = 3)
        self.conv2 = nn.Conv2d(in_channels = 25,out_channels = 50,kernel_size = (10,1),padding = 'same')
        self.conv3 = nn.Conv2d(in_channels = 50,out_channels = 100,kernel_size = (10,1),padding = 'same')
        self.conv4 = nn.Conv2d(in_channels = 100,out_channels = 200,kernel_size = (10,1),padding = 'same')
        self.bn1 = nn.BatchNorm2d(25)
        self.bn2 = nn.BatchNorm2d(50)
        self.bn3 = nn.BatchNorm2d(100)
        self.bn4 = nn.BatchNorm2d(200)
        self.dropout = nn.Dropout(p=0.5)
        self.ELU = nn.ELU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(600,4)
    def forward(self,x):

        #print("input X shape: ",x.shape)


        x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2],1))


        ## Conv Pool Block 1
        x = self.conv1(x)
        x = self.ELU(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        ## Conv Pool Block 2
        x = self.conv2(x)
        x = self.ELU(x)
        x = self.maxpool1(x)
        x = self.bn2(x)
        x = self.dropout(x)
               
        ## Conv Pool Block 3
        x = self.conv3(x)
        x = self.ELU(x)
        x = self.maxpool1(x)
        x = self.bn3(x)
        x = self.dropout(x)
        
        ## Conv Pool Block 4
        x = self.conv4(x)
        x = self.ELU(x)
        x = self.maxpool1(x)
        x = self.bn4(x)
        x = self.dropout(x)

        #Flatten
        x = self.flatten(x)
        #print("flatten output: ", x.shape)
        x = self.linear1(x)

        return x

class DeepCNNv2(nn.Module):
    def __init__(self, num_classes=4):
        """
        4 Convolution layer network
        """
        super(DeepCNNv2, self).__init__()
        
        #layer 1
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=25, kernel_size=10, stride=1, dilation = 2, padding = 'same') 
        self.mxp1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        # self.bn1 = nn.BatchNorm1d(25)
        self.drp1 = nn.Dropout(0.5)

        #layer 2
        self.conv2 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=10, stride=1, dilation = 2, padding = 'same') 
        self.mxp2 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        # self.bn2 = nn.BatchNorm1d(50)
        self.drp2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(in_channels=50, out_channels=100, kernel_size=10, stride=1, dilation = 2, padding = 'same') 
        self.mxp3 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        # self.bn3 = nn.BatchNorm1d(100)
        self.drp3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv1d(in_channels=100, out_channels=200, kernel_size=10, stride=1, dilation = 2, padding = 'same') 
        self.mxp4 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        # self.bn4 = nn.BatchNorm1d(200)
        self.drp4 = nn.Dropout(0.5)

        # self.conv5 = nn.Conv1d(in_channels=200, out_channels=300, kernel_size=10, stride=1, dilation = 2, padding = 'same') 
        # self.mxp5 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        # self.drp5 = nn.Dropout(0.5)

        self.flat1 = nn.Flatten()
        self.lineartrain = nn.Linear(400, 4) 

        #test mode
        self.lineartest = nn.Linear(600, 4)

    def forward(self, x):

        # conv -> maxpool -> dropout -> flatten -> linear
        x = self.conv1(x)
        x = self.mxp1(x)
        # x = self.bn1(x)
        x = self.drp1(x)

        x = self.conv2(x)
        x = self.mxp2(x)
        # x = self.bn2(x)
        x = self.drp2(x)

        x = self.conv3(x)
        x = self.mxp3(x)
        # x = self.bn3(x)
        x = self.drp1(x)

        x = self.conv4(x)
        x = self.mxp4(x)
        # x = self.bn4(x)
        x = self.drp4(x)

        # x = self.conv5(x)
        # x = self.mxp5(x)
        # x = self.drp5(x)

        x = self.flat1(x)
        x = self.lineartrain(x)
        #if mode == 'test':
        #    x = self.lineartest(x)
        #else:
        #    x = self.lineartrain(x)

        return x
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        