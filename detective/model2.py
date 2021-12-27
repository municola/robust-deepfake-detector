import torch
import torch.nn as nn
from torchvision import models

class Detective2(nn.Module):
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


'''
def Detective2(finetune = False):
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

