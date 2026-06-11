"""
non-hierarchical Bayesian Logistic Regression version.

it uses:
- one global set of feature weights
- one global intercept
- posterior uncertainty over the weights
- posterior uncertainty over fraud/anomaly probabilities
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import pyro
import pyro.distributions as dist

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
   accuracy_score,
   precision_score,
   recall_score,
   f1_score,
   roc_auc_score,
   average_precision_score,
   confusion_matrix,
)

from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam


# paths
X_TRAIN_PATH = "data/X_train.csv"
X_TEST_PATH = "data/X_test.csv"
Y_TRAIN_PATH = "data/y_train.csv"
Y_TEST_PATH = "data/y_test.csv"

OUTPUT_DIR = "output"
DATA_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# loading data
X_train_df = pd.read_csv(X_TRAIN_PATH)
X_test_df = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH)["anomalous"].values.astype(np.float32)
y_test = pd.read_csv(Y_TEST_PATH)["anomalous"].values.astype(np.float32)

categorical_features = X_train_df.columns.tolist()


# preprocess categorical data
# use one-hot encoding for categorical variables
# avoids incorrectly treating categories as ordered numbers
one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
   transformers=[
      ("cat", one_hot, categorical_features),
   ]
)

X_train_np = preprocessor.fit_transform(X_train_df).astype(np.float32)
X_test_np = preprocessor.transform(X_test_df).astype(np.float32)

feature_names = preprocessor.get_feature_names_out()

joblib.dump(preprocessor, "data/bayesian_logistic_preprocessor.joblib")


# convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)


# bayesian Logistic Regression model
def bayesian_logistic_regression_model(X, y=None):
   """
   X: tensor with shape [num_transactions, num_features]
   y: tensor with shape [num_transactions], values 0 or 1

   weights and intercept are sampled from prior distributions instead of being fixed parameters.
   """

   num_transactions, num_features = X.shape

   # prior over feature weights
   w = pyro.sample(
      "w",
      dist.Normal(
         torch.zeros(num_features, device=X.device),
         torch.ones(num_features, device=X.device),
      ).to_event(1),
   )

   # prior over intercept
   b = pyro.sample(
      "b",
      dist.Normal(
         torch.tensor(0.0, device=X.device),
         torch.tensor(1.0, device=X.device),
      ),
   )

   # logistic regression equation
   logits = X @ w + b
   probs = torch.sigmoid(logits)

   pyro.deterministic("fraud_probability", probs)

   with pyro.plate("transactions", num_transactions):
      pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


# train with SVI
def train_model(num_steps=3000, learning_rate=0.01):
   pyro.clear_param_store()

   guide = AutoDiagonalNormal(bayesian_logistic_regression_model)

   svi = SVI(
      model=bayesian_logistic_regression_model,
      guide=guide,
      optim=Adam({"lr": learning_rate}),
      loss=Trace_ELBO(),
   )

   losses = []

   for step in range(num_steps):
      loss = svi.step(X_train, y_train_tensor)
      losses.append(loss)

      if step % 500 == 0:
         print(f"step {step}: loss = {loss:.2f}")

   pyro.get_param_store().save("data/bayesian_logistic_param_store.pt")

   loss_df = pd.DataFrame({"step": np.arange(len(losses)), "loss": losses})
   loss_df.to_csv("output/bayesian_logistic_training_loss.csv", index=False)

   return guide


# predict with posterior samples
def predict_with_uncertainty(guide, num_samples=1000):
   predictive = Predictive(
      bayesian_logistic_regression_model,
      guide=guide,
      num_samples=num_samples,
      return_sites=("fraud_probability", "w", "b"),
   )

   samples = predictive(X_test, y=None)

   # shape: [num_samples, num_test_rows]
   prob_samples = samples["fraud_probability"].detach().cpu()

   mean_probs = prob_samples.mean(dim=0).numpy()
   lower_probs = prob_samples.quantile(0.05, dim=0).numpy()
   upper_probs = prob_samples.quantile(0.95, dim=0).numpy()

   y_pred = (mean_probs >= 0.5).astype(int)

   return mean_probs, lower_probs, upper_probs, y_pred, samples


# save feature weight summary
def save_feature_weights(samples):
   # shape: [num_samples, num_features]
   weight_samples = samples["w"].detach().cpu().numpy()

   weight_mean = weight_samples.mean(axis=0)
   weight_lower = np.quantile(weight_samples, 0.05, axis=0)
   weight_upper = np.quantile(weight_samples, 0.95, axis=0)

   weights_df = pd.DataFrame(
      {
         "feature": feature_names,
         "posterior_mean_weight": weight_mean,
         "posterior_5_percent": weight_lower,
         "posterior_95_percent": weight_upper,
         "abs_mean_weight": np.abs(weight_mean),
      }
   )

   weights_df = weights_df.sort_values("abs_mean_weight", ascending=False)
   weights_df.to_csv("output/bayesian_logistic_feature_weights.csv", index=False)


# evaluate and save outputs
def evaluate_and_save(mean_probs, lower_probs, upper_probs, y_pred):
   y_true = y_test.astype(int)

   accuracy = accuracy_score(y_true, y_pred)
   precision = precision_score(y_true, y_pred, zero_division=0)
   recall = recall_score(y_true, y_pred, zero_division=0)
   f1 = f1_score(y_true, y_pred, zero_division=0)

   try:
      roc_auc = roc_auc_score(y_true, mean_probs)
   except ValueError:
      roc_auc = np.nan

   try:
      pr_auc = average_precision_score(y_true, mean_probs)
   except ValueError:
      pr_auc = np.nan

   cm = confusion_matrix(y_true, y_pred)

   predictions_df = X_test_df.copy()
   predictions_df["true_anomalous"] = y_true
   predictions_df["predicted_anomalous"] = y_pred
   predictions_df["mean_anomaly_probability"] = mean_probs
   predictions_df["lower_90_percent_interval"] = lower_probs
   predictions_df["upper_90_percent_interval"] = upper_probs

   predictions_df.to_csv("output/bayesian_logistic_predictions.csv", index=False)

   with open("output/bayesian_logistic_metrics.txt", "w") as f:
      f.write("Bayesian Logistic Regression Metrics\n")
      f.write("------------------------------------\n")
      f.write(f"Accuracy: {accuracy:.4f}\n")
      f.write(f"Precision: {precision:.4f}\n")
      f.write(f"Recall: {recall:.4f}\n")
      f.write(f"F1-score: {f1:.4f}\n")
      f.write(f"ROC-AUC: {roc_auc:.4f}\n")
      f.write(f"PR-AUC: {pr_auc:.4f}\n")
      f.write("\nConfusion Matrix:\n")
      f.write(str(cm))


   print("\nEvaluation complete.")
   print(f"Accuracy:  {accuracy:.4f}")
   print(f"Precision: {precision:.4f}")
   print(f"Recall:    {recall:.4f}")
   print(f"F1-score:  {f1:.4f}")
   print(f"ROC-AUC:   {roc_auc:.4f}")
   print(f"PR-AUC:    {pr_auc:.4f}")
   print("\nSaved:")
   print("- output/bayesian_logistic_metrics.txt")
   print("- output/bayesian_logistic_predictions.csv")
   print("- output/bayesian_logistic_feature_weights.csv")
   print("- output/bayesian_logistic_training_loss.csv")
   print("- data/bayesian_logistic_preprocessor.joblib")
   print("- data/bayesian_logistic_param_store.pt")


if __name__ == "__main__":
   print("Training Bayesian Logistic Regression model with Pyro...")
   print(f"Training rows: {X_train.shape[0]}")
   print(f"Testing rows: {X_test.shape[0]}")
   print(f"Number of encoded features: {X_train.shape[1]}")

   guide = train_model(num_steps=3000, learning_rate=0.01)

   mean_probs, lower_probs, upper_probs, y_pred, samples = predict_with_uncertainty(
      guide,
      num_samples=1000,
   )

   save_feature_weights(samples)
   evaluate_and_save(mean_probs, lower_probs, upper_probs, y_pred)
