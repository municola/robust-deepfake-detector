import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectorNet(nn.Module):
    def __init__(self):
        super(DetectorNet, self).__init__()
        print("INit DetectorNet")
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.linear1  = nn.Linear(1032256, 1)
        

    def forward(self, x):
        # [B, 3, 256, 256]
        x = self.conv1(x)
        #x = nn.ReLU(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        return x
