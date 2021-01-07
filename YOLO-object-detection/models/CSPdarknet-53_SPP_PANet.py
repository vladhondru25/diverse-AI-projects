# In progress
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Conv2dBlock, CSPBlock, SPP


class CSPDarknet53_SPP_PAN(nn.Module):
    def __init__(self, classes):
        super(CSPDarknet53_SPP_PAN, self).__init__()

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

        self.backbone = nn.ModuleList([
            # Initial low-level features
            Conv2dBlock(in_C=3,  out_C=32, k=3, s=1, p=1),
            # CSP Blocks and downsample layers
            *[val for pair in zip(self.downsample_layers, self.csp_blocks) for val in pair],
            
            # Convolutional layers before SPP module
            Conv2dBlock(in_C=1024, out_C=512,  k=1, s=1, p=0),
            Conv2dBlock(in_C=512,  out_C=1024, k=3, s=1, p=1),
            Conv2dBlock(in_C=1024, out_C=512,  k=1, s=1, p=0),
            
            # SPP Module
            SPP(in_C=512, out_C=512),
            
            # Final convolutional layers
            Conv2dBlock(in_C=512,  out_C=1024, k=3, s=1, p=1),
            Conv2dBlock(in_C=1024, out_C=512,  k=1, s=1, p=0),
            Conv2dBlock(in_C=512,  out_C=256,  k=1, s=1, p=0),
        ])
        
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
        ])
        
        self.lateral_connections = nn.ModuleList([
            Conv2dBlock(in_C=512, out_C=256, k=1, s=1, p=0),
            Conv2dBlock(in_C=256, out_C=128, k=1, s=1, p=0)
        ])
        
        self.p3_layers = nn.Sequential(
            Conv2dBlock(in_C=512, out_C=256, k=1, s=1, p=0),
            Conv2dBlock(in_C=256, out_C=512, k=3, s=1, p=1),
            Conv2dBlock(in_C=512, out_C=256, k=1, s=1, p=0),
            Conv2dBlock(in_C=256, out_C=512, k=3, s=1, p=1),
            Conv2dBlock(in_C=512, out_C=256, k=1, s=1, p=0),
            Conv2dBlock(in_C=256, out_C=128, k=1, s=1, p=0),
        )
        
        self.p2_layers = nn.Sequential(
            Conv2dBlock(in_C=256, out_C=128, k=1, s=1, p=0),
            Conv2dBlock(in_C=128, out_C=256, k=3, s=1, p=1),
            Conv2dBlock(in_C=256, out_C=128, k=1, s=1, p=0),
            Conv2dBlock(in_C=128, out_C=256, k=3, s=1, p=1),
            Conv2dBlock(in_C=256, out_C=128, k=1, s=1, p=0),
            
            ################################################
            Conv2dBlock(in_C=128, out_C=256, k=3, s=1, p=1),
            Conv2dBlock(in_C=256, out_C=255, k=1, s=1, p=0, activation='identity')
        )

    def forward(self, x):
        # Backbone + SPP
        for i in range(7):
            x = self.backbone[i](x)
            
        p = []
        for i in range(7,10+1,2):
            p.append(x)
            x = self.backbone[i](x)
            x = self.backbone[i+1](x)
        p.append(x)
        
        for i in range(11,len(self.backbone)):
            p[-1] = self.backbone[i](p[-1])
            
            
        # Top-down path for neck
        p[-2] = torch.cat([
            self.lateral_connections[0](p[-2]),
            self.upsample_layers[0](p[-1])
        ], dim=1)
        p[-2] = self.p3_layers(p[-2])
        
        p[-3] = torch.cat([
            self.lateral_connections[1](p[-3]),
            self.upsample_layers[1](p[-2])
        ], dim=1)
        p[-3] = self.p2_layers(p[-3])
        
        return x
    
    
if __name__ == "__main__":
    xTest = torch.rand((32,3,256,256))
    modelTest = CSPDarknet53_SPP_PAN(10)
    print(modelTest(xTest).shape)