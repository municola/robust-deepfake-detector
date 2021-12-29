import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import argparse
from utils import EarlyStopping
from utils import model_summary, set_seed, load_data, load_model
import wandb
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
    epochs = config['epochs_training']
    learning_rate = config['learning_rate']
    patience = config['early_stopping_patience']
    finetune = config['finetune']
    test_path_adv = config['test_adv_path']
    model_name = config['model_name']
    num_workers = config['num_workers']
    
    # Wandb support
    mode = "online" if config['wandb_logging'] else "disabled"
    wandb.init(
        project="robust-deepfake-detector", 
        entity="deep-learning-eth-2021", 
        config=config, 
        mode=mode
    )

    # Set save path
    if model_name == 'Lestrade':
        save_path = config['path_model_watson']
    elif model_name == 'Watson':
        assert finetune == True
        save_path = config['path_model_watson_finetuned']
    elif model_name == 'Moriaty_untrained':
        save_path = config['path_model_moriaty']
    else:
        raise ValueError("This Model version should not be trained")
    save_path = save_path[:-3] + '_newrun.pt'

    # Set seed
    set_seed(seed)

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    print("\nTrain Dataloader:")
    train_dataloader = load_data(train_path, batch_size, model_name, seed, num_workers, False)
    print("\nVal Dataloader:")
    val_dataloader = load_data(val_path, batch_size, model_name, seed, num_workers, False)

    # Model
    model, _, _, _ = load_model(model_name, config, device, finetune=finetune)
    model_summary(model, model_name)
    wandb.watch(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path, saveEveryEpoch=True)

    # Main loop
    for epoch in range(epochs):
        try:
            train(model, optimizer, train_dataloader, epoch, device)
            loss_val = validation(model, val_dataloader, epoch, device)
        except:
            print("Exception occured. We skip to next epoch")
            continue

        # check early stopping
        early_stopping(loss_val, model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        
    
def train(model, optimizer, dataloader, epoch, device):
    """"Training loop over batches for one epoch"""

    model.train()
    loss_sum = 0

    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)

            out = model(X)
            loss = F.binary_cross_entropy(out, torch.unsqueeze(y.to(torch.float32), dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))
            wandb.log({"loss-train": loss_sum/(batch+1)})
            wandb.log({'accuracy-train': calc_accuracy(out, y)})


def validation(model, dataloader, epoch, device, binary_thresh=0.5):
    """"Validation loop over batches for one epoch"""

    model.eval()
    loss_sum, accuracy = 0, 0
    with torch.no_grad():
        with tqdm(dataloader) as tepoch:
            for batch, (X, y) in enumerate(tepoch):
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = F.binary_cross_entropy(out, torch.unsqueeze(y.to(torch.float32), dim=1))

                loss_sum += loss.item()
                loss_val = loss_sum/(batch+1)
                tepoch.set_description("Validation")
                tepoch.set_postfix(loss = loss_val)

                accuracy += calc_accuracy(out, y)

                wandb.log({'accuracy-val': calc_accuracy(out, y)})

            print(f"Val loss in epoch {epoch}: {loss_val:.6f}")
            acc = accuracy/(batch+1)
            wandb.log({"loss(end)-val": loss_val})
            print(f"Val acc end epoch {epoch}: {acc:.6f}")
            wandb.log({"accuracy(end)-val": acc})

    return loss_val



def calc_accuracy(y_pred, y_true, binary_thresh=0.5):
    """Calculate accuracy for a given decision threshold."""

    hard_pred = (y_pred>binary_thresh).float()
    correct = (torch.squeeze(hard_pred, dim=1) == y_true).float().sum()

    return correct/y_true.shape[0]


if __name__ == "__main__":
    main()
