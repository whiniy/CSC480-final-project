import os
import joblib
import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")["anomalous"]
y_test = pd.read_csv("data/y_test.csv")["anomalous"]

vendor_train_raw = X_train["vendor_id"].copy()
vendor_test_raw = X_test["vendor_id"].copy()

X_train_features = X_train.drop(columns=["vendor_id"])
X_test_features = X_test.drop(columns=["vendor_id"])

categorical_cols = [
    "amt_deviation",
    "duplicate_entry",
    "new_vendor",
    "missing_po",
    "entered_after_hrs",
    "manual_entry",
    "unusual_accts",
    "desc_quality",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols,),
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train_features)
X_test_encoded = preprocessor.transform(X_test_features)

vendor_encoder = LabelEncoder()
vendor_train_idx = vendor_encoder.fit_transform(vendor_train_raw)
vendor_test_idx = vendor_encoder.transform(vendor_test_raw)

num_vendors = len(vendor_encoder.classes_)

y_train_arr = y_train.to_numpy()
y_test_arr = y_test.to_numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train_encoded, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_encoded, dtype=torch.float32, device=device)

vendor_train_tensor = torch.tensor(vendor_train_idx, dtype=torch.long, device=device)
vendor_test_tensor = torch.tensor(vendor_test_idx, dtype=torch.long, device=device)

y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test_arr, dtype=torch.float32, device=device)

num_features = X_train_encoded.shape[1]

def model(X, vendor_idx, y=None, num_vendors=None):
    N, P = X.shape

    mu_alpha = pyro.sample("mu_alpha", dist.Normal(0.0, 1.0))
    sigma_alpha = pyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_beta = pyro.sample("sigma_beta", dist.HalfNormal(1.0))

    with pyro.plate("vendors", num_vendors):
        alpha = pyro.sample("alpha", dist.Normal(mu_alpha, sigma_alpha))

    beta = pyro.sample(
        "beta",
        dist.Normal(
            torch.zeros(P, device=X.device),
            sigma_beta * torch.ones(P, device=X.device)
        ).to_event(1)
    )

    logits = alpha[vendor_idx] + (X * beta).sum(dim=1)

    with pyro.plate("data", N):
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

# fit with SVI (stochastic variational inference)
pyro.clear_param_store()

guide = AutoNormal(model)
svi = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

num_steps = 3000
for step in range(num_steps):
    loss = svi.step(X_train_tensor, vendor_train_tensor, y_train_tensor, num_vendors=num_vendors)
    if step % 250 == 0:
        print(f"Step {step}, Loss: {loss:.2f}")

# posterior predictive probabilities
predictive = Predictive(
    model,
    guide=guide,
    num_samples=1000,
    return_sites=["alpha", "beta"]
)

samples = predictive(X_test_tensor, vendor_test_tensor, None, num_vendors=num_vendors)
alpha_samples = samples["alpha"]
beta_samples = samples["beta"].squeeze()
print("alpha_samples shape:", alpha_samples.shape)
print("beta_samples shape :", beta_samples.shape)
print("X_test_tensor shape:", X_test_tensor.shape)
print("vendor_test_tensor shape:", vendor_test_tensor.shape)

logits_samples = alpha_samples[:, vendor_test_tensor] + torch.matmul(beta_samples, X_test_tensor.T)
probs_samples = torch.sigmoid(logits_samples)
mean_probs = probs_samples.mean(dim=0)
y_pred = (mean_probs >= 0.25).float()

# metrics
y_true = y_test_tensor.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()
mean_probs_np = mean_probs.detach().cpu().numpy()

print("Accuracy :", accuracy_score(y_true, y_pred_np))
print("Precision:", precision_score(y_true, y_pred_np))
print("Recall   :", recall_score(y_true, y_pred_np))
print("F1       :", f1_score(y_true, y_pred_np))
print("ROC AUC  :", roc_auc_score(y_true, mean_probs_np))

train_posterior = Predictive(
    model,
    guide=guide,
    num_samples=1000,
    return_sites=["mu_alpha", "sigma_alpha", "alpha", "beta"],
)

posterior = train_posterior(X_train_tensor, vendor_train_tensor, None, num_vendors=num_vendors)

posterior_alpha = posterior["alpha"]
posterior_beta = posterior["beta"]

if posterior_alpha.dim() == 3:
    posterior_alpha = posterior_alpha.squeeze(1)

if posterior_beta.dim() == 3:
    posterior_beta = posterior_beta.squeeze(1)

alpha_mean = posterior_alpha.mean(dim=0).cpu().numpy()
beta_mean = posterior_beta.mean(dim=0).cpu().numpy()

print("alpha_mean shape:", alpha_mean.shape)
print("beta_mean shape:", beta_mean.shape)

OUT_DIR = "outputs/hierarchical_output"
os.makedirs(OUT_DIR, exist_ok=True)

np.save(os.path.join(OUT_DIR, "y_true.npy"), y_true)
np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred_np)
np.save(os.path.join(OUT_DIR, "mean_probs.npy"), mean_probs_np)

np.save(os.path.join(OUT_DIR, "posterior_alpha.npy"), posterior_alpha.detach().cpu().numpy())
np.save(os.path.join(OUT_DIR, "posterior_beta.npy"), posterior_beta.detach().cpu().numpy())

joblib.dump(vendor_encoder.classes_, os.path.join(OUT_DIR, "vendor_names.joblib"))
joblib.dump(preprocessor.get_feature_names_out(), os.path.join(OUT_DIR, "feature_names.joblib"))
joblib.dump({"decision_threshold": 0.25}, os.path.join(OUT_DIR, "config.joblib"))

print(f"Saved model outputs to ./{OUT_DIR}/")