import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import argparse
from utils import EarlyStopping
from utils import model_summary, model_summary2, set_seed, load_data, load_model
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
    version = config['version']
    finetune = config['finetune']
    test_path_adv = config['test_adv_path']

    # Wandb support
    mode = "online" if config['wandb_logging'] else "disabled"
    wandb.init(
        project="robust-deepfake-detector", 
        entity="deep-learning-eth-2021", 
        config=config, 
        mode=mode
    )

    # Model is always Lestrade (After training it becomes Watson)
    if version == 1:
        model_name = 'Lestrade'
        path_model = config['path_model_watson']
    elif version == 2:
        if finetune:
            model_name = 'Watson2'
            path_model = config['path_model_watson2_finetuned']
        else:
            model_name = 'Lestrade2'
            path_model = config['path_model_watson2']
    else:
        raise ValueError("No such version exist currently")

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
    print("\ntest")
    test_dataloader = load_data(test_path_adv, batch_size, model_name)


    # Model
    print("Modified adversarial training in the train.py file!!!!")
    model, _, _, _ = load_model(model_name, config, device, finetune=finetune)
    model_summary2(model)
    wandb.watch(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5, eps=1e-5,
    #                                             patience=patience-1)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_model)
    #print(f"Scheduled LR reduction after {patience-1} epochs without improvement")

    # Main loop
    for epoch in range(epochs):
        train(model, optimizer, train_dataloader, epoch, device)
        loss_val = validation(model, val_dataloader, epoch, device)

        #scheduler.step(loss_val)
        _ = validation_test(model, test_dataloader, epoch, device)

        # check early stopping
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path_model))
        
    
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

def validation_test(model, dataloader, epoch, device, binary_thresh=0.5):
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
                tepoch.set_description("Test")
                tepoch.set_postfix(loss = loss_val)

                accuracy += calc_accuracy(out, y)

                wandb.log({'accuracy-test': calc_accuracy(out, y)})

            print(f"Test loss end epoch {epoch}: {loss_val:.6f}")
            wandb.log({"loss(end)-test": loss_val})

            acc = accuracy/(batch+1)
            print(f"Test acc end epoch {epoch}: {acc:.6f}")
            wandb.log({"accuracy(end)-test": acc})

    return loss_val


def calc_accuracy(y_pred, y_true, binary_thresh=0.5):
    """Accuracy for a given decision threshold."""

    hard_pred = (y_pred>binary_thresh).float()
    correct = (torch.squeeze(hard_pred, dim=1) == y_true).float().sum()

    return correct/y_true.shape[0]


if __name__ == "__main__":
    main()
