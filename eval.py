import numpy as np
import torch
import yaml
import argparse
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from utils import load_model, load_data
from tqdm import tqdm

def main():
    """Evaluate a given model with AUC and other metrics on the test data"""

    # Config arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    test_path = config['test_path']
    batch_size = config['batch_size']
    model_name = config['model_name']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load Model and data
    model, model_name, _, _ = load_model(model_name, config, device)
    test_dataloader = load_data(test_path, batch_size)

    # Evaluate
    evaluate(model, model_name, test_dataloader, device)


def evaluate(model, model_name, dataloader, device):
    """Obtain predictions for given dataset/loader and compute relevant metrics."""

    model.eval()
    y_true, y_pred = np.array([]), np.array([])
    print("\nCollecting predictions...")

    with torch.no_grad():
        with tqdm(dataloader) as tepoch:
            for batch, (X, y) in enumerate(tepoch):
                X, y = X.to(device), y.to(device)
                out = model(X)
                y_pred = np.append(y_pred, out.cpu().numpy()) #proba class 1
                y_true = np.append(y_true, y.cpu().numpy())

    assert y_true.shape == y_pred.shape, "y_true, y_pred of unequal length."
    print(f"Performance metrics for {model_name}:")
    roc_auc(y_true, y_pred, model_name)
    accuracy(y_true, y_pred, binary_thresh=0.5)

    return y_true, y_pred


def roc_auc(y_true, y_pred, model_name, plot=True):
    """ROC AUC and optional ROC curve plot."""

    auc = roc_auc_score(y_true, y_pred, labels=[0,1])
    print(f"AUC: {auc:.6f}")
    if plot:
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        roc_plot = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
                                   estimator_name=model_name
                                   )
        roc_plot.plot()


def accuracy(y_true, y_pred, binary_thresh=0.5):
    """Accuracy for a given decision threshold."""

    print(f"Assign class 1 for probabilities >={binary_thresh}")
    y_pred[y_pred >= binary_thresh] = 1
    y_pred[y_pred < binary_thresh] = 0
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
