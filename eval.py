import os
import numpy as np
import torch
import yaml
import argparse
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from utils import load_model, load_data
from tqdm import tqdm


def main():
    """Evaluate a given model with metrics on normal/adversarial test data"""

    # Config arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))

    batch_size = config['batch_size']
    model_name = config['model_name']
    eval_res_path = config['eval_res_path']
    test_adv_bool = config['test_adv_bool'] # Important to correctly specify!
    test_path = config['test_adv_path'] if test_adv_bool else config['test_path']
    seed = config['seed']
    num_workers = config['num_workers']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load Model and data
    model, _, _, _ = load_model(model_name, config, device)
    test_dataloader = load_data(test_path, batch_size, model_name, seed, num_workers, False)
    print(f"Evaluating on adversarial test set: {test_adv_bool}")
    print(f"Evaluating of test set: {test_path}")
    
    # Evaluate
    if model_name == "Polimi":
        y_true, y_pred = evaluate_polimi(model, model_name, test_path, test_adv_bool)
    else:
        y_true, y_pred = evaluate(model, model_name, test_dataloader, device)

    # Save model results to file (save ROC plot manually)
    testset = os.path.basename(os.path.normpath(test_path))
    np.savetxt(eval_res_path + "/y_true_" + model_name + "_" + testset + ".csv", y_true, delimiter=",")
    np.savetxt(eval_res_path + "/y_pred_" + model_name + "_" + testset +".csv", y_pred, delimiter=",")


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
                y_pred = np.append(y_pred, out.cpu().numpy()) # proba class 1
                y_true = np.append(y_true, y.cpu().numpy())

    assert y_true.shape == y_pred.shape, "y_true, y_pred of unequal length."
    print(f"\nPerformance metrics for {model_name}...")
    roc_auc(y_true, y_pred, model_name)
    accuracy(y_true, y_pred, binary_thresh=0.5)

    return y_true, y_pred


def evaluate_polimi(model, model_name, test_path, test_adv_bool):
    """
    Obtain predictions for given dataset/loader and compute relevant metrics.
    Modifications to specifically address PolimiNet external code.

    PolimiNet output is an averaged logit score 
    (sklearn.decision_function equivalent, see e.g.
     https://www.kite.com/python/docs/sklearn.linear_model.SGDClassifier.decision_function)
    A hard threshold is given at 0: out<0 is real (class 0), out>=0 is fake (class 1)
    The value itself quantifies a sort of mean confidence in the prediction 
    across the five nets of the ensemble (larger abs. value = more confident).
    """

    # manually create str paths for all test set images (normal/adversarial)
    if test_adv_bool:
        test_img_ffhq = [os.path.join(test_path + "/ffhq", f"{str(img).zfill(5)}.png") for img in range(0, 20000)]
        test_img_stylegan3 = [os.path.join(test_path + "/stylegan3", f"{img}.png") for img in range(20000, 40000)]
    else:
        test_img_ffhq = [os.path.join(test_path + "/ffhq", f"{img}.jpg") for img in range(50000, 70000)]
        test_img_stylegan3 = [os.path.join(test_path + "/stylegan3", f"seed{str(img).zfill(4)}.png") for img in range(0, 20000)]

    test_img = test_img_ffhq + test_img_stylegan3
    size = len(test_img)
    print("\nSample path from ffhq:", test_img_ffhq[0])
    print("Sample path from stylegan3:", test_img_stylegan3[0])
    print("Total nr of test paths:", size)

    y_true, y_pred = np.array([]), np.array([])
    print("\nCollecting predictions, this will take a long time...")

    for _, img in enumerate(tqdm(test_img)):
        out = model.synth_real_detector(img)
        y_pred = np.append(y_pred, out)

    y_true = np.append(y_true, np.repeat(0, len(test_img_ffhq)))
    y_true = np.append(y_true, np.repeat(1, len(test_img_stylegan3)))

    assert y_true.shape == y_pred.shape, "y_true, y_pred of unequal length."
    print(f"\nPerformance metrics for {model_name}:")
    roc_auc(y_true, y_pred, model_name)
    accuracy(y_true, y_pred, binary_thresh=0)

    return y_true, y_pred


def roc_auc(y_true, y_pred, model_name):
    """ROC AUC and ROC curve plot."""

    auc = roc_auc_score(y_true, y_pred, labels=[0,1])
    print(f"AUC: {auc:.6f}")

    fpr, tpr, thr = roc_curve(y_true, y_pred)
    roc_plot = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr, 
        roc_auc=auc,
        estimator_name=model_name
    )
    roc_plot.plot()


def accuracy(y_true, y_pred, binary_thresh=0.5):
    """Accuracy for a given decision threshold."""

    print(f"Assign class 1 for probabilities >={binary_thresh}")
    hard_pred = np.empty(len(y_true))
    hard_pred[np.flatnonzero(y_pred >= binary_thresh)] = 1
    hard_pred[np.flatnonzero(y_pred < binary_thresh)] = 0
    acc = accuracy_score(y_true, hard_pred)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
