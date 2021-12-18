import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from detective.model import Detective
from polimi.gan_vs_real_detector import Detector as PolimiNet

def model_summary(model):
    """Print out pretty model summary including parameter counts"""

    print("Model summary:")
    print("\nLayer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)

    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j, total_params = 0, 0
    print("\t" * 10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param = model_parameters[j].numel() + model_parameters[j+1].numel()
            j = j+2
        else:
            param = model_parameters[j].numel()
            j = j+1
        print(str(i) + "\t" * 3 + "Parameters in Layer: " + str(param))
        total_params += param
    print("=" * 100)
    print(f"Total Params: {total_params}\n")


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
        """Saves model when validation loss decreases."""

        if self.verbose:
            self.trace_func(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def load_model(model_name, config, device):
    """"Load specified model from checkpoint onto device."""

    if model_name == "Lestrade":
        path_model = None
        model = Detective().to(device)
    elif model_name == "Watson":
        path_model = config['path_model_watson']
        model = Detective().to(device)
        model.load_state_dict(torch.load(path_model, map_location=device))
    elif model_name == 'Sherlock':
        path_model = config['path_model_sherlock']
        model = Detective().to(device)
        model.load_state_dict(torch.load(path_model, map_location=device))
    elif model_name == "Polimi":
        path_model = None
        model = PolimiNet(device) # note: object is not a neural net
    else:
        raise ValueError("Need to specify 'Lestrade', 'Sherlock', 'Watson' or 'Polimi'")

    print(f"Loaded model: {model_name} onto device: {device}")

    return model, model_name, path_model, device


def load_data(data_path, batch_size):
    """"
    Load data from specified path and return dataloader with batch size.

    Automatic class assignments by ImageFolder function are done in order 
    of folders in specified directory, so to obtain implicit class assignments 
    (0: real, 1: fake) need to rename real FFHQ image folders as "ffhq" and 
    fake StyleGAN image folders as "stylegan" (f < s for alphabetical ordering).
    This naming is also strictly necessary to pass the assert statement.

    Data folder structure/naming:
        - train
            - ffhq
            - stylegan2
        - val
            - ffhq
            - stylegan2
        - test
            - ffhq
            - stylegan3
    """

    transform = transforms.Compose([
         transforms.ToTensor()
         # transforms.Normalize((0.1307,), (0.3081,))
     ])

    data = ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    assert data.class_to_idx == {'ffhq': 0, 'stylegan2': 1} or data.class_to_idx == {'ffhq': 0, 'stylegan3': 1}
    assert len(np.unique(data.targets)) == 2, "More than two classes."
    print("\nDataset loaded")
    print("Dataset size:", len(dataloader.dataset))
    print("Class mapping:", data.class_to_idx) # 0: Real, 1: Fake
    print(f"Batch size: {batch_size}")

    return dataloader
