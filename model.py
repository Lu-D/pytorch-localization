import torch
import torch.nn as nn
import torch.nn.functional as F
from load.py import *
from torchvision import models


class DoubleConv(nn.Module):
    
    def __init(self, channelin, channelout):
        super(Net, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1),
            nn.BatchNorm2d(channelout),
            nn.ReLU(inplace=True),
            nn.Conv2d(channelout, channelout, 3, 1),
            nn.BatchNorm2d(channelout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, channelin, channelout):
        super().__init()
        self.pool_double = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channelin, channelout)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.dropout(x3)
        x5 = torch.flatten(x4, 1)
        x6 = self.fc1(x5)
        x7 = self.fc2(x6)
        output = self.fc3(x7, dim=1)
        return output

