import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms
from model import DetectorNet
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *

def main(random_state=1234):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    set_seed(random_state)

    BATCH_SIZE = 10
    num_epochs = 10
    learning_rate = 1e-3
    patience = 3

    user = "Mo"
    if user=="Mo":
        data_dir_train = "/home/moritz/Documents/ETH/DL/Data/Train"
        data_dir_val = "/home/moritz/Documents/ETH/DL/Data/Validation"

    transform = transforms.Compose([
        transforms.ToTensor()
        # Maybe Normalize !!!!
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 0: Fake, 1: Real
    train_data = torchvision.datasets.ImageFolder(root=data_dir_train,transform = transform)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    val_data = torchvision.datasets.ImageFolder(root=data_dir_val,transform = transform)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = DetectorNet().to(device)
    model_summary(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='./our-detector/checkpoints/checkpoint.pt')

    for epoch in range(0, num_epochs):
        train(model, optimizer, train_data_loader, epoch, device)
        val_num_correct, val_loss = validation(model, val_data_loader, device)
        print(f"Validation Loss in Epoch {epoch}: {val_loss:.6f}")
        acc = val_num_correct/len(val_data)
        print(f"Validation Accuracy in Epoch {epoch}: {acc:.6f}")
        # Checking Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('./our-detector/checkpoints/checkpoint.pt'))
        
    
    

def train(model, optimizer, train_data_loader, epoch, device):
    model.train()
    lossSum = 0

    with tqdm(train_data_loader) as tepoch:
        for batch_idx, (data,label) in enumerate(tepoch):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(torch.sigmoid(output), torch.unsqueeze(label.to(torch.float32), dim=1))
            loss.backward()
            optimizer.step()

            lossSum += loss.item()
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss=lossSum/((batch_idx+1)))

def validation(model, val_data_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        with tqdm(val_data_loader) as tepoch:
            for batch_idx, (data, label) in enumerate(tepoch):
                data, label = data.to(device), label.to(device)
                output = model(data)
                val_loss += F.binary_cross_entropy(torch.sigmoid(output), torch.unsqueeze(label.to(torch.float32), dim=1)).item()
                tepoch.set_description("Validation")
                tepoch.set_postfix(loss=(val_loss)/(batch_idx+1))
                predictions = output>=0
                correct += (predictions.squeeze() == label).sum().item()
    return correct, (val_loss)/(batch_idx+1)   
                
                



if __name__ == "__main__":
    main(random_state=1234)
