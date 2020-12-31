# In progress
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Conv2dBlock, CSPBlock


class CSPDarknet53_SPP(nn.Module):
    def __init__(self):
        super(CSPDarknet53_SPP, self).__init__()

        self.downsample_layers = nn.ModuleList([
            Conv2dBlock(in_C=32,  out_C=64,   k=3, s=2, p=1),
            Conv2dBlock(in_C=64,  out_C=128,  k=3, s=2, p=1),
            Conv2dBlock(in_C=128, out_C=256,  k=3, s=2, p=1),
            Conv2dBlock(in_C=256, out_C=512,  k=3, s=2, p=1),
            Conv2dBlock(in_C=512, out_C=1024, k=3, s=2, p=1)
        ])

        self.csp_blocks = nn.ModuleList([
            CSPBlock(64,   no_of_blocks=1, activation='leakyRelu'),
            CSPBlock(128,  no_of_blocks=2, activation='leakyRelu'),
            CSPBlock(256,  no_of_blocks=8, activation='leakyRelu'),
            CSPBlock(512,  no_of_blocks=8, activation='leakyRelu'),
            CSPBlock(1024, no_of_blocks=4, activation='leakyRelu'),
        ])

        self.network = nn.Sequential(
            # Initial low-level features
            Conv2dBlock(in_C=3,  out_C=32, k=3, s=1, p=1),
            # CSP Blocks and downsample layers
            *[val for pair in zip(self.downsample_layers, self.csp_blocks) for val in pair],
            # Global average pooling
            nn.AvgPool2d(kernel_size=8),
            # Flatten
            nn.Flatten(),
            # Fully-connected layer
            nn.Linear(in_features=1024, out_features=1000)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    modelTest = CSPDarknet53_SPP()
    xTest = torch.rand((32,3,256,256))
    print(modelTest(xTest).shape)