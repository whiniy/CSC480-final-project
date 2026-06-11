import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

OUT_DIR = "outputs/hierarchical_output"
os.makedirs(OUT_DIR, exist_ok=True)


# Load saved outputs
y_true = np.load(os.path.join(OUT_DIR, "y_true.npy"))
y_pred = np.load(os.path.join(OUT_DIR, "y_pred.npy"))
mean_probs = np.load(os.path.join(OUT_DIR, "mean_probs.npy"))

posterior_alpha = np.load(os.path.join(OUT_DIR, "posterior_alpha.npy"))
posterior_beta = np.load(os.path.join(OUT_DIR, "posterior_beta.npy"))

vendor_names = joblib.load(os.path.join(OUT_DIR, "vendor_names.joblib"))
feature_names = joblib.load(os.path.join(OUT_DIR, "feature_names.joblib"))
config = joblib.load(os.path.join(OUT_DIR, "config.joblib"))

decision_threshold = config["decision_threshold"]

class_names = np.array(["no", "yes"])
pos_label = 1

PALETTE = {
    "primary": "#4C6EF5",
    "accent": "#F76707",
    "pos": "#2F9E44",
    "neg": "#C92A2A",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# -----------------------------
# Metrics report
# -----------------------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
roc_auc = roc_auc_score(y_true, mean_probs)
avg_prec = average_precision_score(y_true, mean_probs)

report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

metrics_text = f"""
Hierarchical Bayesian Logistic Regression Evaluation Metrics
   Decision Threshold : {decision_threshold:.2f}
   Accuracy           : {acc:.4f}
   Precision          : {prec:.4f}
   Recall             : {rec:.4f}
   F1 Score           : {f1:.4f}
   ROC-AUC            : {roc_auc:.4f}
   Avg Precision (PR) : {avg_prec:.4f}

- Per-class report -
{report}
"""

with open(os.path.join(OUT_DIR, "metrics_report.txt"), "w") as fh:
    fh.write(metrics_text)

print(metrics_text)


# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix (threshold = {decision_threshold:.2f})",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close(fig)


# ROC curve
fpr, tpr, _ = roc_curve(y_true, mean_probs, pos_label=pos_label)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2,
        label=f"Hierarchical Bayes LogReg (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
plt.close(fig)


# Precision-recall curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true, mean_probs, pos_label=pos_label)
baseline = y_true.sum() / len(y_true)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(recall_vals, precision_vals, color=PALETTE["accent"], lw=2,
        label=f"Hierarchical Bayes LogReg (AP = {avg_prec:.3f})")
ax.axhline(baseline, color="k", lw=1, linestyle="--", alpha=0.5,
           label=f"No-skill baseline ({baseline:.2f})")
ax.fill_between(recall_vals, precision_vals, alpha=0.08, color=PALETTE["accent"])
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "precision_recall_curve.png"))
plt.close(fig)


# Threshold sweep
thresholds = np.linspace(0.05, 0.95, 91)

rows = []
for t in thresholds:
    y_pred_t = (mean_probs >= t).astype(int)
    rows.append({
        "threshold": t,
        "precision": precision_score(y_true, y_pred_t, zero_division=0),
        "recall": recall_score(y_true, y_pred_t, zero_division=0),
        "f1": f1_score(y_true, y_pred_t, zero_division=0),
    })

thresh_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision")
ax.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall")
ax.plot(thresh_df["threshold"], thresh_df["f1"], label="F1", linewidth=2)

best_idx = thresh_df["f1"].idxmax()
best_thresh = thresh_df.loc[best_idx, "threshold"]
best_f1 = thresh_df.loc[best_idx, "f1"]

ax.axvline(best_thresh, color="black", linestyle="--", alpha=0.6,
           label=f"Best F1 threshold = {best_thresh:.2f}")
ax.axvline(decision_threshold, color=PALETTE["accent"], linestyle=":", alpha=0.8,
           label=f"Chosen threshold = {decision_threshold:.2f}")

ax.set_title("Threshold Sweep for Hierarchical Bayesian Logistic Regression",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "threshold_sweep.png"))
plt.close(fig)


# Vendor intercept plot
alpha_mean = posterior_alpha.mean(axis=0)
alpha_low = np.percentile(posterior_alpha, 2.5, axis=0)
alpha_high = np.percentile(posterior_alpha, 97.5, axis=0)

vendor_df = pd.DataFrame({
    "vendor": vendor_names,
    "mean": alpha_mean,
    "low": alpha_low,
    "high": alpha_high,
}).sort_values("mean", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    vendor_df["mean"],
    vendor_df["vendor"],
    xerr=[
        vendor_df["mean"] - vendor_df["low"],
        vendor_df["high"] - vendor_df["mean"],
    ],
    fmt="o",
    capsize=3,
)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Vendor-Level Intercepts with 95% Credible Intervals",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Vendor intercept (log-odds scale)")
ax.set_ylabel("Vendor")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "vendor_intercepts.png"))
plt.close(fig)


# Posterior coefficient plot
beta_mean = posterior_beta.mean(axis=0)
beta_low = np.percentile(posterior_beta, 2.5, axis=0)
beta_high = np.percentile(posterior_beta, 97.5, axis=0)

coef_df = pd.DataFrame({
    "feature": feature_names,
    "mean": beta_mean,
    "low": beta_low,
    "high": beta_high,
})

coef_df["feature"] = coef_df["feature"].astype(str).str.replace("cat__", "", regex=False)
coef_df["abs_mean"] = coef_df["mean"].abs()
coef_df = coef_df.sort_values("abs_mean", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
ax.errorbar(
    coef_df["mean"],
    coef_df["feature"],
    xerr=[
        coef_df["mean"] - coef_df["low"],
        coef_df["high"] - coef_df["mean"],
    ],
    fmt="o",
    capsize=3,
)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Posterior Feature Effects with 95% Credible Intervals",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Coefficient (log-odds scale)")
ax.set_ylabel("Encoded feature")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "posterior_coefficients.png"))
plt.close(fig)

print(f"All outputs written to ./{OUT_DIR}/")