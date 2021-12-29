import torch.nn as nn
from torchvision import models


def Watson(finetune=False):
    print("\n Initialize Watson-architecture")
    model_ft = models.resnet18(pretrained=True)

    if finetune == False:
        # Set gradients to False
        for param in model_ft.parameters():
            param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

    return model_ft


'''
class Watson(nn.Module):
    def __init__(self, pretrained=True, finetune=False):
        super().__init__()
        
        resnet18 = models.resnet18(pretrained=pretrained)

        self.features = nn.ModuleList(resnet18.children())[:-1]
        self.features = nn.Sequential(*self.features)
    
        if finetune == False:
            # Set gradients to False
            for param in self.features.parameters():
                param.requires_grad = False

        # Add last layer
        in_features = resnet18.fc.in_features
        self.fc0 = nn.Linear(in_features, 1)

    def forward(self, input_imgs):

        # Go through pretrained Model and flatten
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)

       # Go through our last layer 
        output = self.fc0(output)
        output = torch.sigmoid(output)
                
        return output
#'''

'''
def Watson(finetune = False):
    print("\nInitialize Detective2")
    
    #model_ft = models.vgg11_bn(pretrained=True)
    model_ft = models.resnet18(pretrained=True)
    
    if finetune == False:
        # Set gradients to False
        for param in model_ft.parameters():
            param.requires_grad = False

    # Add last layer (These parameters we train)
    #num_ftrs = model_ft.classifier[6].in_features
    #model_ft.classifier[6] = nn.Linear(num_ftrs, 1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    
    return model_ft
'''


"""
# Original model with extra convolution + batch norm

class Watson(nn.Module):
    def __init__(self):
        super().__init__()
        print("\nInitialize Detective2")

        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool2d(2)
        self.out = nn.Sigmoid()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            self.activation,
            self.pooling
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            self.activation,
            self.pooling
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            self.activation,
            self.pooling
            )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            self.activation,
            self.pooling
            )

        self.linear  = nn.Sequential(
            nn.Linear(32768, 1)
            )

    def forward(self, x):

        # [B, 3, 256, 256]
        x = self.conv1(x)
        # [B, 16, 128, 128]

        x = self.conv2(x)
        # [B, 16, 64, 64]

        x = self.conv3(x)
        # [B, 64, 32, 32]

        x = self.conv4(x)
        # [B, 128, 16, 16]

        x = torch.flatten(x, 1)
        # [B, 32768]
        logits = self.linear(x)
        # [B, 1]
        probs = self.out(logits)
        # [B, 1]

        return probs

"""
"""
# Alternative model with more convolution stacking + batch norm + leaky relu

class Watson(nn.Module):
    def __init__(self):
        super().__init__()
        print("\nInitialize Detective2")

        self.activation = nn.LeakyReLU()
        self.pooling = nn.MaxPool2d(2)
        self.out = nn.Sigmoid()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            self.activation,
            self.pooling
            )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            self.activation
            )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            self.activation,
            self.pooling
            )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            self.activation
            )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            self.activation,
            self.pooling
            )

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            self.activation
            )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            self.activation,
            self.pooling
            )

        self.linear  = nn.Sequential(
            nn.Linear(32768, 1)
            )

        #self.linear  = nn.Sequential(
        #    nn.Linear(32768, 128),
        #    self.activation,
        #    nn.Linear(128, 1)
        #    )

    def forward(self, x):

        # [B, 3, 256, 256]
        x = self.conv1(x)
        # [B, 16, 128, 128]

        x = self.conv2_1(x)
        # [B, 16, 128, 128]
        x = self.conv2_2(x)
        # [B, 32, 64, 64]

        x = self.conv3_1(x)
        # [B, 32, 64, 64]
        x = self.conv3_2(x)
        # [B, 64, 32, 32]

        x = self.conv4_1(x)
        # [B, 64, 32, 32]
        x = self.conv4_2(x)
        # [B, 128, 16, 16]

        x = torch.flatten(x, 1)
        # [B, 32768]
        logits = self.linear(x)
        # [B, 1]
        probs = self.out(logits)
        # [B, 1]

        return probs

"""
"""
COMMENTS
- residual/skip connections: model may not be deep enough for real benefits
- batch normalization: faster training, better backprop
- two fully connected layers: better feature mixing but way more params
- more convolutions/more depth: wider receptive field

when using batch norm:
    - don't mix with dropout
    - use larger learning rate, accelerate learning rate decay
    - can both be applied before and after activation, but more common before
"""

