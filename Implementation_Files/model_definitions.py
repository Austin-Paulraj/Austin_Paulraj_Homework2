import torch
from torch import nn

#This file contains the Torch Class definitions for all the Architectures defined in the assignment

class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, 2, 2),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),
            nn.Conv2d(8, 16, 2, 2, 2),
            nn.ReLU(),
            nn.Flatten(0, -1),
            nn.Linear(6*6*16,288),
            nn.ReLU(),
            nn.Linear(288,10),
        )
    
    def forward(self, x):
        return self.conv_layers(x)
    
class DNNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Linear(392, 196),
            nn.ReLU(),
            nn.Linear(196,10),
        )
    
    def forward(self, x):
        return self.conv_layers(x)