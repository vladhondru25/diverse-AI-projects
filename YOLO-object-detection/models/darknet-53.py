import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Conv2dBlock, ResidualBlock

class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()

        self.network = nn.Sequential(
            # Initial convolution
            Conv2dBlock(in_C=3,  out_C=32, k=3, s=1, p=1),
            # Downsample -> 128x128x64
            Conv2dBlock(in_C=32, out_C=64, k=3, s=2, p=1),

            # First residual block
            ResidualBlock(in_C=64),

            # Downsample -> 64x64x128
            Conv2dBlock(in_C=64, out_C=128, k=3, s=2, p=1),

            # Second residual block
            *[ResidualBlock(in_C=128) for _ in range(2)],

            # Downsample -> 32x32x256
            Conv2dBlock(in_C=128, out_C=256, k=3, s=2, p=1),

            # Third residual block
            *[ResidualBlock(in_C=256) for _ in range(8)],

            # Downsample -> 16x16x512
            Conv2dBlock(in_C=256, out_C=512, k=3, s=2, p=1),

            # Fourth residual block
            *[ResidualBlock(in_C=512) for _ in range(8)],

            # Downsample -> 8x8x1024
            Conv2dBlock(in_C=512, out_C=1024, k=3, s=2, p=1),

            # Fifth residual block
            *[ResidualBlock(in_C=1024) for _ in range(4)],

            nn.AvgPool2d(kernel_size=8),

            nn.Flatten(),

            nn.Linear(in_features=1024, out_features=1000)
        )

    def forward(self, x):
        return self.network(x)

    
if __name__ == "__main__":
    model_test = DarkNet53()

    x_test = torch.rand((64, 3, 256, 256))

    print(model_test(x_test).shape)
