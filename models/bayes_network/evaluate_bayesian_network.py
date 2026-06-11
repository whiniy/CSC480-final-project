import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

def plot_confusion_matrix(cm, output_path):
    """
    plot and save a confusion matrix 
    cm format:
        [[TN, FP],
         [FN, TP]]
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # white background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # blue heatmap with white text
    im = ax.imshow(cm, cmap="Blues")

    labels = ["no", "yes"]

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    # add numbers inside cells
    threshold = cm.max() / 2

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                color="white" if cm[i, j] > threshold else "#4C6F91",
            )

    # remove top and right border to match the clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="white")
    plt.close()


def plot_precision_recall_curve(y_true, y_score, output_path):
    # plot and save a precision-recall curve for the Bayesian Network.
    # y_true = actual labels, 0 or 1
    # y_score = predicted probability of class 1 / anomalous

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # baseline = proportion of positive/anomalous examples
    no_skill = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(7, 5))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(
        recall,
        precision,
        alpha=0.15,
        color="tab:orange"
    )

    ax.plot(
        recall,
        precision,
        color="tab:orange",
        linewidth=2,
        label=f"Bayesian Network (AP = {pr_auc:.3f})"
    )

    ax.axhline(
        y=no_skill,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"No-skill baseline ({no_skill:.2f})"
    )

    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{int(x * 100)}%" for x in ticks])
    ax.set_yticklabels([f"{int(y * 100)}%" for y in ticks])

    ax.legend(loc="upper right", fontsize=9)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="white")
    plt.close()

    return pr_auc


def evaluate_from_predictions(csv_path, label_col, score_col, pred_col=None, threshold=0.5):
    # evaluate model using row-by-row predictions
    # need a CSV with columns:
    #    label_col = true class labels, 0 or 1
    #    score_col = predicted probability of anomalous, between 0 and 1
    df = pd.read_csv(csv_path)

    y_true = df[label_col].astype(int).values
    y_score = df[score_col].astype(float).values

    if pred_col is not None and pred_col in df.columns:
        y_pred = df[pred_col].astype(int).values
    else:
        y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    print("Bayesian Network Metrics")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    os.makedirs("outputs/network_output", exist_ok=True)

    plot_confusion_matrix(cm, "outputs/network_output/bayesian_network_confusion_matrix.png")
    plot_precision_recall_curve(
        y_true,
        y_score,
        "outputs/network_output/bayesian_network_precision_recall_curve.png",
    )

    print("\nSaved graphs:")
    print("outputs/network_output/bayesian_network_confusion_matrix.png")
    print("outputs/network_output/bayesian_network_precision_recall_curve.png")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bayesian Network results.")
    parser.add_argument("--csv", type=str, default=None, help="CSV file with predictions.")
    parser.add_argument("--label-col", type=str, default="y_true", help="Column with true labels.")
    parser.add_argument("--score-col", type=str, default="y_score", help="Column with predicted probabilities.")
    parser.add_argument("--pred-col", type=str, default=None, help="Optional column with predicted labels.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting probability to class.")

    args = parser.parse_args()

    if args.csv is None:
        print("Error: CSV file with predictions is required.")
    else:
        evaluate_from_predictions(
            csv_path=args.csv,
            label_col=args.label_col,
            score_col=args.score_col,
            pred_col=args.pred_col,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()