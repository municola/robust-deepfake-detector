import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
import argparse
from tqdm import tqdm
from utils import load_model, model_summary, set_seed, load_data
from utils import EarlyStopping
from attacks import *


def main():
    """"Main training loop for discriminator model"""

    # Config arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    seed = config['seed']
    train_path = config['train_path']
    val_path = config['val_path']
    batch_size = config['batch_size']
    epochs = config['epochs_adversarial_training']
    learning_rate = config['learning_rate']
    patience = config['early_stopping_patience']

    # Model is always Watson (After training it becomes Sherlock)
    model_name = 'Watson'

    # Set seed
    set_seed(seed)

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    print("\nTrain:")
    train_dataloader = load_data(train_path, batch_size)
    print("\nVal:")
    val_dataloader = load_data(val_path, batch_size)

    # Model
    model, _, path_model, _ = load_model(model_name, config, device)
    model_summary(model) # nr of params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_model)

    # Loop over the Epochs
    for epoch in range(epochs):
        train(model, optimizer, train_dataloader, epoch, device)
        loss_val = validation(model, val_dataloader, epoch, device)

        # check early stopping
        if epoch >= 9:
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
            
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path_model))
        
    
def train(model, optimizer, dataloader, epoch, device):
    """"Training loop over batches for one epoch"""

    loss_sum = 0
    eps = None
    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)
            y = torch.unsqueeze(y.to(torch.float32), dim=1)
            loss_fn = F.binary_cross_entropy

            # Calculate Epsilon
            if epoch <= 9:
                eps = 0.001 * (epoch+1)

            # Generate the adversarial
            model.eval()
            X_adv, _ = FGSM_attack(X, y, model, loss_fn, eps=eps)
            model.train()

            out = model(X_adv)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))


def validation(model, dataloader, epoch, device, binary_thresh=0.5):
    """"Validation loop over batches for one epoch"""

    model.eval()
    loss_sum, correct = 0, 0
    eps = None

    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)
            loss_fn = F.binary_cross_entropy
            y = torch.unsqueeze(y.to(torch.float32), dim=1)

            # Calculate Epsilon
            if epoch <= 9:
                eps = 0.001 * (epoch+1)

            # Generate the adversarial
            X_adv, _ = FGSM_attack(X, y, model, loss_fn, eps=eps)

            out = model(X_adv)
            loss = loss_fn(out,y)

            loss_sum += loss.item()
            loss_val = loss_sum/(batch+1)
            tepoch.set_description("Validation")
            tepoch.set_postfix(loss = loss_val)

            out[out >= binary_thresh] = 1
            out[out < binary_thresh] = 0
            correct += (out == y).sum().item()

        print(f"Val loss in epoch {epoch}: {loss_val:.6f}")
        acc = correct/len(dataloader.dataset)
        print(f"Val acc in epoch {epoch}: {acc:.6f}")

    return loss_val


if __name__ == "__main__":
    main()
