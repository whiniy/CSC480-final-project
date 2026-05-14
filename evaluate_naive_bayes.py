"""
Loads the saved Naive Bayes pipeline and generates evaluation metrics + charts.

Outputs (written to ./output/):
   metrics_report.txt
   confusion_matrix.png
   roc_curve.png
   precision_recall_curve.png
   feature_log_probs.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

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

MODEL_PATH    = "data/naive_bayes_pipeline.joblib"
ENCODER_PATH  = "data/target_label_encoder.joblib"
X_TEST_PATH   = "data/X_test.csv"
Y_TEST_PATH   = "data/y_test.csv"
OUT_DIR       = "output"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model and data …")
model         = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()          # encoded (0/1)

class_names = label_encoder.classes_.astype(str)     # ["no", "yes"]
pos_label   = 1                                      # encoded "anomalous = yes"

# predictions
y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, pos_label]

# scalar metrics
acc       = accuracy_score(y_test, y_pred)
prec      = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
rec       = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
f1        = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_prob)
avg_prec  = average_precision_score(y_test, y_prob, pos_label=pos_label)

report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

metrics_text = f"""
   Naive Bayes — Evaluation Metrics
      Accuracy          : {acc:.4f}
      Precision         : {prec:.4f}
      Recall            : {rec:.4f}
      F1 Score          : {f1:.4f}
      ROC-AUC           : {roc_auc:.4f}
      Avg Precision (PR): {avg_prec:.4f}

- Per-class report -
{report}
"""

print(metrics_text)
with open(os.path.join(OUT_DIR, "metrics_report.txt"), "w") as fh:
   fh.write(metrics_text)

# styling
PALETTE = {"primary": "#4C6EF5", "accent": "#F76707", "pos": "#2F9E44", "neg": "#C92A2A"}
plt.rcParams.update({
   "figure.dpi": 150,
   "axes.spines.top": False,
   "axes.spines.right": False,
   "font.size": 11,
})

# confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close(fig)
print("Saved confusion_matrix.png")

# roc curve
fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=pos_label)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2,
        label=f"Naive Bayes (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
plt.close(fig)
print("Saved roc_curve.png")

# precision-recall curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob, pos_label=pos_label)
baseline = y_test.sum() / len(y_test)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(recall_vals, precision_vals, color=PALETTE["accent"], lw=2,
        label=f"Naive Bayes (AP = {avg_prec:.3f})")
ax.axhline(baseline, color="k", lw=1, linestyle="--", alpha=0.5,
           label=f"No-skill baseline ({baseline:.2f})")
ax.fill_between(recall_vals, precision_vals, alpha=0.08, color=PALETTE["accent"])
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "precision_recall_curve.png"))
plt.close(fig)
print("Saved precision_recall_curve.png")

# feature log-probabilities (naive bayes "importance")
'''
CategoricalNB stores feature_log_prob_ with shape (n_classes, n_features_total).
After OrdinalEncoding each original feature maps to one encoded column, so the
total number of encoded columns equals len(categorical_features).
We take log P(feature | anomalous) − log P(feature | normal)
as a signed importance: positive → feature pushes toward "anomalous".
'''

categorical_features = [
   "vendor_id", "amt_deviation", "duplicate_entry", "new_vendor",
   "missing_po", "entered_after_hrs", "manual_entry", "unusual_accts",
   "desc_quality",
]

nb_classifier = model.named_steps["classifier"]

# CategoricalNB stores feature_log_prob_ as a list of arrays.
# We average the log-prob difference across all categories per feature.
feat_importance = {}
for feat, lp_array in zip(categorical_features, nb_classifier.feature_log_prob_):
    # lp_array: (n_classes, n_categories)
    # log P(cat|anomalous) − log P(cat|normal)
    diff = lp_array[pos_label] - lp_array[0]
    feat_importance[feat] = diff.mean()

feat_series = pd.Series(feat_importance).sort_values()
colors = [PALETTE["pos"] if v >= 0 else PALETTE["neg"] for v in feat_series]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(feat_series.index, feat_series.values, color=colors, edgecolor="none", height=0.6)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Mean log-prob difference\n(anomalous − normal)", fontsize=10)
ax.set_title("Feature Influence on Anomaly Prediction\n(Naive Bayes log-probability difference)",
             fontsize=12, fontweight="bold", pad=12)
# Annotate bar values
for bar, val in zip(bars, feat_series.values):
   xpos = val + 0.01 if val >= 0 else val - 0.01
   ha   = "left" if val >= 0 else "right"
   ax.text(xpos, bar.get_y() + bar.get_height() / 2,
           f"{val:+.3f}", va="center", ha=ha, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "feature_log_probs.png"))
plt.close(fig)
print("Saved feature_log_probs.png")

print(f"\nAll outputs written to ./{OUT_DIR}/")