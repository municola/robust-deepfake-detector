import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
import argparse
from tqdm import tqdm
from utils import load_model, model_summary, model_summary2, set_seed, load_data
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
    path_model = config['path_model_sherlock']
    epsilon = config['adversarial_eps']
    version = config['version']
    
    # Model is always Watson (After training it becomes Sherlock)
    if version == 1:
        model_name = 'Watson'
    elif version == 2:
        model_name = 'Watson2'
    else:
        raise ValueError("This version is not yet supported.")

    # Set seed
    set_seed(seed)

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    print("\nTrain:")
    train_dataloader = load_data(train_path, batch_size, model_name)
    print("\nVal:")
    val_dataloader = load_data(val_path, batch_size, model_name)

    # Model
    model, _, _, _ = load_model(model_name, config, device, finetune=True)
    model_summary2(model) # nr of params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_model)

    # Loop over the Epochs
    for epoch in range(epochs):
        train(model, optimizer, train_dataloader, epoch, device, epsilon)
        loss_val = validation(model, val_dataloader, epoch, device, epsilon)

        # check early stopping
        if epoch >= 19:
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
            
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path_model))
        
    
def train(model, optimizer, dataloader, epoch, device, epsilon):
    """"Training loop over batches for one epoch"""

    loss_sum = 0
    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)
            y = torch.unsqueeze(y.to(torch.float32), dim=1)
            loss_fn = F.binary_cross_entropy

            # Calculate Epsilon
            if epoch <= 19:
                eps = epsilon/20 * (epoch+1)
            else:
                eps = epsilon

            # Generate the adversarial
            model.eval()
            X_adv, _ = LinfPGD_Attack(X, y, model, loss_fn, eps=eps, eps_iter=eps/10, nb_iter=20)
            model.train()

            out = model(X_adv)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))


def validation(model, dataloader, epoch, device, epsilon, binary_thresh=0.5):
    """"Validation loop over batches for one epoch"""

    model.eval()
    loss_sum, correct = 0, 0

    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)
            loss_fn = F.binary_cross_entropy
            y = torch.unsqueeze(y.to(torch.float32), dim=1)

            # Calculate Epsilon
            if epoch <= 19:
                eps = epsilon/20 * (epoch+1)
            else:
                eps = epsilon

            # Generate the adversarial
            X_adv, _ = LinfPGD_Attack(X, y, model, loss_fn, eps=eps, eps_iter=eps/10, nb_iter=20)

            out = model(X_adv)
            loss = loss_fn(out,y)

            loss_sum += loss.item()
            loss_val = loss_sum/(batch+1)
            tepoch.set_description("Validation")
            tepoch.set_postfix(loss = loss_val)

            out[out >= binary_thresh] = 1
            out[out < binary_thresh] = 0
            correct += (out == torch.unsqueeze(y.to(torch.float32), dim=1)).sum().item()

        print(f"Val loss in epoch {epoch}: {loss_val:.6f}")
        acc = correct/len(dataloader.dataset)
        print(f"Val acc in epoch {epoch}: {acc:.6f}")

    return loss_val


if __name__ == "__main__":
    main()
