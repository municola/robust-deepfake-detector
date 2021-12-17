import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_model, model_summary, set_seed, EarlyStopping, load_data, get_path
from attacks import *


def main(random_seed=1234, device=None):
    """"Main training loop for discriminator model"""

    # Setup
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    set_seed(random_seed)
    print(f"Set random seed: {random_seed}")

    user = "Nici"
    path_model='./our-detector/checkpoints/checkpoint.pt'
    path_adversarial_model = './our-detector/checkpoints/checkpoint_adversarial.pt'

    # HYPERPARAMS
    batch_size = 10
    epochs = 3
    learning_rate = 1e-3
    patience = 3 # early stopping wait time

    # Load data
    print("\nTrain:")
    train_path = get_path(user, "train")
    train_dataloader = load_data(train_path, batch_size)
    print("\nVal:")
    val_path = get_path(user, "val")
    val_dataloader = load_data(val_path, batch_size)

    # Model
    model, _, _ = load_model('DetectorNet', path_model)
    model_summary(model) # nr of params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_adversarial_model)

    # Main loop
    for epoch in range(epochs):
        train(model, optimizer, train_dataloader, epoch, device)
        loss_val = validation(model, val_dataloader, epoch, device)

        # check early stopping
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path_adversarial_model))
        
    
def train(model, optimizer, dataloader, epoch, device):
    """"Training loop over batches for one epoch"""

    model.train()
    loss_sum = 0
    with tqdm(dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)
            loss_fn = F.binary_cross_entropy()

            # Generate the adversarial
            X_adv, _ = PGD_attack(X, y, model, loss_fn, eps=0.15, eps_iter=0.03, nb_iter=10)

            out = model(X_adv)
            loss = loss_fn(out, torch.unsqueeze(y.to(torch.float32), dim=1))

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
    with torch.no_grad():
        with tqdm(dataloader) as tepoch:
            for batch, (X, y) in enumerate(tepoch):
                X, y = X.to(device), y.to(device)
                loss_fn = F.binary_cross_entropy()

                # Generate the adversarial
                X_adv, _ = PGD_attack(X, y, model, loss_fn, eps=0.15, eps_iter=0.03, nb_iter=10)

                out = model(X_adv)
                loss = loss_fn(out, torch.unsqueeze(y.to(torch.float32), dim=1))

                loss_sum += loss.item()
                loss_val = loss_sum/(batch+1)
                tepoch.set_description("Validation")
                tepoch.set_postfix(loss = loss_val)

                out[out >= binary_thresh] = 1
                out[out < binary_thresh] = 0
                correct += (out.squeeze() == y).sum().item()

            print(f"Val loss in epoch {epoch}: {loss_val:.6f}")
            acc = correct/len(dataloader.dataset)
            print(f"Val acc in epoch {epoch}: {acc:.6f}")

    return loss_val


if __name__ == "__main__":
    main()
