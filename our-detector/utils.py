import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import DetectorNet


def model_summary(model):
    '''
    Print out pretty model summary including parameter count
    '''
    print("Model Summary:")
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
        print(str(i)+"\t"*3+"Parameters in Layer: "+str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params: {total_params}")


def set_seed(seed):
    """Set ALL random seeds"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def load_model(model_name, model_path, device=None):
    """"Load specified model from checkpoint onto device."""

    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if model_name == "DetectorNet":
        model = DetectorNet().to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device)
            )
    elif model_name == "PolimiNet":
        raise NotImplementedError("Missing PolimiNet")
    else:
        raise ValueError("Need to specify 'DetectorNet' or 'PolimiNet'")

    print(f"Loaded model {model_name} onto device {device}")

    return model, model_name, device


def load_data(data_path, batch_size):
    """"
    Load data from specified path and return dataloader with batch size.

    Automatic class assignments by ImageFolder function are done in order 
    of folders in specified directory, so to obtain class assignments 
    (0: fake, 1: real) one can rename FFHQ image folders as "real" and 
    StyleGAN image folders as "fake".
    """

    transform = transforms.Compose([
         transforms.ToTensor()
         # transforms.Normalize((0.1307,), (0.3081,))
     ])

    data = ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    assert len(np.unique(data.targets)) == 2, "More than two classes."
    print("\nDataset loaded")
    print("Dataset size:", len(dataloader.dataset))
    print("Class mapping:", data.class_to_idx) # 0: Fake, 1: Real
    print(f"Batch size: {batch_size}")

    return dataloader


def get_path(user_name, data_name):
    """Convenience function to get pre-specified data paths."""

    path = ""

    if user_name == "Mo":
        if data_name == "train":
            path = "/home/moritz/Documents/ETH/DL/Data/Train"
        elif data_name == "val":
            path = "/home/moritz/Documents/ETH/DL/Data/Validation"
        elif data_name == "test":
            path = "/home/moritz/Documents/ETH/DL/Data/test"
        else:
            raise ValueError("Need one of 'train', 'val', 'test'")

    elif user_name == "Nici":
        if data_name == "train":
            path = "/home/nicolas/git/robust-deepfake-detector/our-detector/data/train"
        elif data_name == "val":
            path = "/home/nicolas/git/robust-deepfake-detector/our-detector/data/val"
        elif data_name == "test":
            path = "/home/nicolas/git/robust-deepfake-detector/our-detector/data/test"
        else:
            raise ValueError("Need one of 'train', 'val', 'test'")

    elif user_name == "Alex":
        if data_name == "train":
            path = None
        elif data_name == "val":
            path = None
        elif data_name == "test":
            path = "/Users/atimans/Desktop/Deep L/Project/robust-deepfake-detector/data/test"
        else:
            raise ValueError("Need one of 'train', 'val', 'test'")

    elif user_name == "David":
        if data_name == "train":
            path = None
        elif data_name == "val":
            path = None
        elif data_name == "test":
            path = None
        else:
            raise ValueError("Need one of 'train', 'val', 'test'")

    else:
        raise ValueError("Need one of 'Mo', 'Nici', 'Alex', 'David'")

    return path




