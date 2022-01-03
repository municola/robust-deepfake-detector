import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def get_data(model_path, model_name):
    """
    Load y_true, y_pred for given model for all test sets: 
    normal, adv_v1, adv_v2, adv_v3.
    """

    path_pred = model_path + "/y_pred_" + model_name + "_test"
    path_true = model_path + "/y_true_" + model_name + "_test"

    return {
        "pred" : np.loadtxt(path_pred + ".csv"),
        "pred_adv1" : np.loadtxt(path_pred + "_adv_v1.csv"),
        "pred_adv2" : np.loadtxt(path_pred + "_adv_v2.csv"),
        "pred_adv3" : np.loadtxt(path_pred + "_adv_v3.csv"),
        "true" : np.loadtxt(path_true + ".csv"),
        "true_adv1" : np.loadtxt(path_true + "_adv_v1.csv"),
        "true_adv2" : np.loadtxt(path_true + "_adv_v2.csv"),
        "true_adv3" : np.loadtxt(path_true + "_adv_v3.csv")
            }


def plot_model(model_name, data, color, ax):
    """Plot ROC curves for given model for all test sets."""

    y_true, y_pred = data["true"], data["pred"]
    auc = roc_auc_score(y_true, y_pred, labels=[0,1])
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    ax.plot(fpr, tpr, label=f"normal, AUC: {str(auc)[:6]}",
            color=color)
    
    y_true, y_pred = data["true_adv1"], data["pred_adv1"]
    auc = roc_auc_score(y_true, y_pred, labels=[0,1])
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    ax.plot(fpr, tpr, label=f"adv1, AUC: {str(auc)[:6]}",
            color=color, linestyle="--")
    
    y_true, y_pred = data["true_adv2"], data["pred_adv2"]
    auc = roc_auc_score(y_true, y_pred, labels=[0,1])
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    ax.plot(fpr, tpr, label=f"adv2, AUC: {str(auc)[:6]}",
            color=color, linestyle=":")
    
    y_true, y_pred = data["true_adv3"], data["pred_adv3"]
    auc = roc_auc_score(y_true, y_pred, labels=[0,1])
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    ax.plot(fpr, tpr, label=f"adv3, AUC: {str(auc)[:6]}",
            color=color, linestyle="-.")


def main(fig_save=False, fig_name=None):

    #moriarty = get_data("eval_results/moriarty", "Moriaty")
    polimi = get_data("eval_results/polimi", "Polimi")
    sherlock = get_data("eval_results/sherlock", "Sherlock")
    watson = get_data("eval_results/watson", "Watson")

    fig = plt.figure(figsize=(7,6))
    #fig = plt.figure()
    ax = fig.gca()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.grid()

    #ax.plot(1, 0, label="Moriarty", color="white")
    #plot_model("Moriarty", moriarty, "firebrick", ax)
    ax.plot(1, 0, label="Polimi", color="white")
    plot_model("Polimi", polimi, "orchid", ax)
    ax.plot(1, 0, label="Watson", color="white")
    plot_model("Watson", watson, "mediumblue", ax)
    ax.plot(1, 0, label="Sherlock", color="white")
    plot_model("Sherlock", sherlock, "seagreen", ax)

    plt.legend(loc='lower right')
    
    if fig_save:
        fig_path = 'eval_results/{}.pdf'.format(fig_name)
        plt.savefig(fig_path, bbox_inches='tight')


if __name__ == "__main__":
    main(True, "ROC_polimi_sherlock_watson")


# colors: https://matplotlib.org/stable/gallery/color/named_colors.html