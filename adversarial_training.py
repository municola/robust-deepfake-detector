import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
import argparse
from tqdm import tqdm
from utils import load_model, model_summary, model_summary2, set_seed, load_data
from utils import EarlyStopping
from attacks import *
from advertorch.context import ctx_noparamgrad_and_eval
import wandb
from torchvision import transforms

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
    epsilon = config['adversarial_eps']
    version = config['version']
    test_path_adv = config['test_adv_path']
    finetune = config['finetune']
    num_workers = config['num_workers']

    # Wandb support
    mode = "online" if config['wandb_logging'] else "disabled"
    wandb.init(
        project="robust-deepfake-detector", 
        entity="deep-learning-eth-2021", 
        config=config, 
        mode=mode
    )
    
    # Model is always Watson (After training it becomes Sherlock)
    if version == 1:
        model_name = 'Watson'
        path_model = config['path_model_sherlock']
    elif version == 2:
        model_name = 'Watson2'
        path_model = config['path_model_sherlock2']
    else:
        raise ValueError("This version is not yet supported.")

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    print("\nTrain:")
    train_dataloader = load_data(train_path, batch_size, model_name, seed, num_workers)
    print("\nVal:")
    val_dataloader = load_data(val_path, batch_size, model_name, seed, num_workers)
    print("\ntest")
    test_dataloader = load_data(test_path_adv, batch_size, model_name, seed, num_workers)

    # Model
    model, _, _, _ = load_model(model_name, config, device, finetune=finetune)
    model_summary2(model)
    wandb.watch(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_model)

    # Make one validation run
    _ = validation(model, val_dataloader, 0, device, epsilon)
    _ = validation_test(model, test_dataloader, 0, device, epsilon)

    # Loop over the Epochs
    for epoch in range(epochs):
        train(model, optimizer, train_dataloader, epoch, device, epsilon, model_name, config)
        loss_val = validation(model, val_dataloader, epoch, device, epsilon)
        _ = validation_test(model, test_dataloader, epoch, device, epsilon)

        # check early stopping
        if epoch >= 19:
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save every checkpoint
        savepath = path_model + "_epoch_"  + str(epoch) + "_" + ".pt"
        print('Saving model to:', savepath)
        torch.save(model.state_dict(), savepath)
            
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path_model))
        
    
def train(model, optimizer, dataloader, epoch, device, epsilon, model_name, config):
    """"Training loop over batches for one epoch"""
    model.train()

    loss_sum = 0
    accuracy = 0
    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y = torch.unsqueeze(y.to(torch.float32), dim=1)
            loss_fn = F.binary_cross_entropy

            # Calculate Epsilon
            if epoch <= 19:
                eps = epsilon/20 * (epoch+1)
            else:
                eps = epsilon

            # Generate the adversarial with non-normalized X
            with ctx_noparamgrad_and_eval(model):
                X_adv, _ = FGSM_attack(X, y, model, loss_fn, eps=eps)

            # Normalize X_adv
            normalizeTransformation = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            X_adv = normalizeTransformation(X_adv)

            # Calculate loss
            out = model(X_adv)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            accuracy += calc_accuracy(out, y)

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))
            wandb.log({"loss-train": loss_sum/(batch+1)})
            wandb.log({'accuracy-train': calc_accuracy(out, y)})

        acc = accuracy/(batch+1)
        wandb.log({"accuracy-train(epoch end)": acc})
        print('accuracy-train(epoch end):', acc)


def validation(model, dataloader, epoch, device, epsilon, binary_thresh=0.5):
    """"Validation loop over batches for one epoch"""

    model.eval()
    loss_sum, accuracy = 0, 0

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
            with ctx_noparamgrad_and_eval(model):
                X_adv, _ = LinfPGD_Attack(X, y, model, loss_fn, eps=eps, eps_iter=eps/10, nb_iter=20)

            # Normalize X_adv
            normalizeTransformation = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            X_adv = normalizeTransformation(X_adv)

            out = model(X_adv)
            loss = loss_fn(out,y)

            loss_sum += loss.item()
            loss_val = loss_sum/(batch+1)
            accuracy += calc_accuracy(out, y)

            tepoch.set_description("Validation")
            tepoch.set_postfix(loss = loss_val)
            wandb.log({'accuracy-val': calc_accuracy(out, y)})
            wandb.log({"loss-val": loss_val})

        acc = accuracy/(batch+1)
        wandb.log({"accuracy(end)-val": acc})
        print(f"Val acc end epoch {epoch}: {acc:.6f}")

    return loss_val


def validation_test(model, dataloader, epoch, device, epsilon, binary_thresh=0.5):
    """"Validation loop over batches for one epoch"""

    model.eval()
    loss_sum, accuracy = 0, 0

    i = 0
    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)
            loss_fn = F.binary_cross_entropy
            y = torch.unsqueeze(y.to(torch.float32), dim=1)

            # Normalize X_adv
            normalizeTransformation = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            X = normalizeTransformation(X)

            with torch.no_grad():
                out = model(X)

            loss = loss_fn(out,y)

            loss_sum += loss.item()
            loss_val = loss_sum/(batch+1)
            accuracy += calc_accuracy(out, y)

            tepoch.set_description("Validation Test")
            tepoch.set_postfix(loss = loss_val)
            wandb.log({'accuracy-test': calc_accuracy(out, y)})
            wandb.log({"loss-test": loss_val})

        acc = accuracy/(batch+1)
        wandb.log({"accuracy(end)-test": acc})
        print(f"Test acc end epoch {epoch}: {acc:.6f}")

    return loss_val


def calc_accuracy(y_pred, y_true, binary_thresh=0.5):
    """Accuracy for a given decision threshold."""

    hard_pred = (y_pred>binary_thresh).float()
    correct = (hard_pred == y_true).float().sum()

    return correct/y_true.shape[0]


if __name__ == "__main__":
    main()
