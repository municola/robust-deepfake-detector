import torch
import torch.nn as nn

class DetectorNet(nn.Module):
    def __init__(self):
        super(DetectorNet, self).__init__()
        print("\nInitialized DetectorNet")

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, padding="same"),
                     nn.ReLU(),
                     nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, padding="same"),
                     nn.ReLU(),
                     nn.MaxPool2d(2))                     
        self.conv3 = nn.Sequential(nn.Conv2d(16, 64, 3, 1, padding="same"),
                     nn.ReLU(),
                     nn.MaxPool2d(2))
        self.linear1  = nn.Linear(65536, 1)

    def forward(self, x):
        # [B, 3, 256, 256]
        x = self.conv1(x)
        # [B, 16, 128, 128]
        x = self.conv2(x)
        # [B, 16, 64, 64]
        x = self.conv3(x)
        # [B, 64, 32, 32]
        x = torch.flatten(x, 1)
        # [B, 65’536]
        logits = self.linear1(x)
        out = torch.sigmoid(logits)

        return out
