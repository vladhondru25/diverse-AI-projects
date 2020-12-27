import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {
    'relu': nn.ReLU(inplace=True),
    'leakyRelu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'identity': nn.Identity()
}

class Conv2dBlock(nn.Module):
    def __init__(self, in_C, out_C, k, s, p, bias=False, activation='relu'):
        super(Conv2dBlock,self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_C, out_channels=out_C, kernel_size=k, stride=s, padding=p, bias=bias),
            nn.BatchNorm2d(out_C),
            ACTIVATIONS[activation]
        )

    def forward(self, x):
        return self.conv_layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_C, activation='relu'):
        super(ResidualBlock,self).__init__()
        
        self.conv1 = Conv2dBlock(in_C=in_C,    out_C=in_C//2, k=1, s=1, p=0)
        self.conv2 = Conv2dBlock(in_C=in_C//2, out_C=in_C,    k=3, s=1, p=1, activation='identity')

        self.activation = ACTIVATIONS[activation]

    def forward(self, x):
        z1 = self.conv1(x)
        z2 = x + self.conv2(z1)
        return self.activation(z2)