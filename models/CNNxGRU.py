import torch
import torch.nn as nn
import numpy as np

class ShallowCNN_GRU(nn.Module):
    def __init__(self):
        super(ShallowCNN_GRU, self).__init__()

        # layer 1
        self.conv1 = nn.Conv2d(in_channels = 22, out_channels = 40,kernel_size=(25,1))
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(75,1),stride=15)

        #layer 2
        self.gru = nn.GRU(input_size = 11, hidden_size = 44, num_layers=1, batch_first = True)

        # output layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1760,4)

    def forward(self,x):
        
        ## Conv Pool Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)

        # GRU layer
        x = np.squeeze(x, axis = -1)
        h0 = torch.zeros(self.gru.num_layers, x.shape[0], self.gru.hidden_size, requires_grad=True).to(x.device)
        x, _ = self.gru(x, h0)

        # Output layer
        x = self.flatten(x)
        x = self.linear1(x)

        return x

class DeepCNN_GRU(nn.Module):
    def __init__(self):
        super(DeepCNN_GRU, self).__init__()
        
        # layer 1
        self.conv1 = nn.Conv2d(in_channels = 22, out_channels = 25,kernel_size=(10,1),padding = 'same')
        self.ELU = nn.ELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,1),stride = 3)
        self.bn1 = nn.BatchNorm2d(25)
        self.dropout1 = nn.Dropout(p=0.5)

        # layer 2
        self.conv2 = nn.Conv2d(in_channels = 25,out_channels = 50,kernel_size = (10,1),padding = 'same')
        self.ELU = nn.ELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,1),stride = 3)
        self.bn2 = nn.BatchNorm2d(50)
        self.dropout2 = nn.Dropout(p=0.5)

        # layer 3
        self.conv3 = nn.Conv2d(in_channels = 50,out_channels = 100,kernel_size = (10,1),padding = 'same')
        self.ELU = nn.ELU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3,1),stride = 3)
        self.bn3 = nn.BatchNorm2d(100)
        self.dropout3 = nn.Dropout(p=0.5)

        # layer 4
        self.conv4 = nn.Conv2d(in_channels = 100,out_channels = 200,kernel_size = (10,1),padding = 'same')
        self.ELU = nn.ELU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3,1),stride = 3)
        self.bn4 = nn.BatchNorm2d(200)
        self.dropout4 = nn.Dropout(p=0.5)

        # layer 5
        self.gru = nn.GRU(input_size = 3, hidden_size = 64, num_layers=1, batch_first = True)
        
        # output layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(12800,4)


    def forward(self,x):
        ## Conv Pool Block 1
        x = self.conv1(x)
        x = self.ELU(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        ## Conv Pool Block 2
        x = self.conv2(x)
        x = self.ELU(x)
        x = self.maxpool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
               
        ## Conv Pool Block 3
        x = self.conv3(x)
        x = self.ELU(x)
        x = self.maxpool3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        ## Conv Pool Block 4
        x = self.conv4(x)
        x = self.ELU(x)
        x = self.maxpool4(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        ## GRU Layer
        x = np.squeeze(x, axis = -1) # remove dimension of length 1 from the end
        h0 = torch.zeros(self.gru.num_layers, x.shape[0], self.gru.hidden_size, requires_grad=True).to(x.device)
        x, _ = self.gru(x, h0)
        
        # Output layer
        x = self.flatten(x)
        x = self.linear1(x)

        return x