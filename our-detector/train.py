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



def set_seed(seed):
    """Set ALL random seeds"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main(random_state=1234):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    set_seed(random_state)

    BATCH_SIZE = 10
    num_epochs = 10
    learning_rate = 1e-3
    data_dir_train = "/home/nicolas/git/robust-deepfake-detector/our-detector/data/train"
    data_dir_val = "/home/nicolas/git/robust-deepfake-detector/our-detector/data/val"

    transform = transforms.Compose([
        transforms.ToTensor()
        # Maybe Normalize !!!!
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 0: Fake, 1: Real
    train_data = torchvision.datasets.ImageFolder(root=data_dir_train,transform = transform)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    val_data = torchvision.datasets.ImageFolder(root=data_dir_val,transform = transform)
    val_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = DetectorNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(0, num_epochs):
        train(model, optimizer, train_data_loader, epoch, device)
        validation(model, val_data_loader, device)

def train(model, optimizer, train_data_loader, epoch, device):
    model.train()
    lossSum = 0

    with tqdm(train_data_loader) as tepoch:
        for batch_idx, (data,label) in enumerate(tepoch):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(F.sigmoid(output), torch.unsqueeze(label.to(torch.float32), dim=1))
            loss.backward()
            optimizer.step()

            lossSum += loss.item()

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss=(lossSum + loss.item())/(batch_idx+1)
)

def validation(model, val_data_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        with tqdm(val_data_loader) as tepoch:
            for batch_idx, (data, label) in enumerate(tepoch):
                data, label = data.to(device), label.to(device)
                output = model(data)
                val_loss += F.binary_cross_entropy(F.sigmoid(output), torch.unsqueeze(label.to(torch.float32), dim=1)).item()

                # TODO: Report Accuracy



if __name__ == "__main__":
    main(random_state=1234)
