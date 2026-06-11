"""
bayesian network: models dependence among red flags

does not assume all red flags are independent
creates relationships between red flags
some examples:
    new_vendor -> missing_po
   entered_after_hrs -> manual_entry
   manual_entry -> duplicate_entry
   manual_entry -> desc_quality

red flags are used to estimate P(anomalous | observed red flags)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetworkModel
except ImportError:
    try:
        from pgmpy.models import BayesianNetwork as BayesianNetworkModel
    except ImportError as exc:
        raise ImportError(
            "pgmpy is not installed. Install it with:\n\n"
            "pip install pgmpy\n"
        ) from exc

from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# paths
X_TRAIN_PATH = "data/X_train.csv"
X_TEST_PATH = "data/X_test.csv"
Y_TRAIN_PATH = "data/y_train.csv"
Y_TEST_PATH = "data/y_test.csv"

OUTPUT_DIR = "output"
DATA_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# helper functions
def normalize_target_value(value):

    # normalize representations of yes/true/anomalous and no/false/normal to "yes" and "no"
    value_str = str(value).strip().lower()
    if value_str in {"1", "yes", "true", "anomalous"}:
        return "yes"
    if value_str in {"0", "no", "false", "normal"}:
        return "no"

    return value_str

# helper function to load data, merge X and y, and convert all columns to strings for pgmpy
def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["anomalous"]
    y_test = pd.read_csv(Y_TEST_PATH)["anomalous"]

    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df["anomalous"] = y_train.apply(normalize_target_value)
    test_df["anomalous"] = y_test.apply(normalize_target_value)

    # pgmpy works best with discrete/categorical variables
    # convert everything to strings so states are consistent
    for col in train_df.columns:
        train_df[col] = train_df[col].astype(str)

    for col in test_df.columns:
        test_df[col] = test_df[col].astype(str)

    return train_df, test_df


def get_probability_of_yes(query_result):
    # extract P(anomalous = yes) from pgmpy's query result.
    
    states = list(query_result.state_names["anomalous"])
    values = query_result.values

    # find index of "yes" or "1" in states to get the corresponding probability value
    if "yes" in states:
        yes_index = states.index("yes")
    elif "1" in states:
        yes_index = states.index("1")
    else:
        raise ValueError(
            f"Could not find a positive anomalous state in states: {states}"
        )

    return float(values[yes_index])

EDGES = [
    # dependencies among red flags
    ("new_vendor", "missing_po"),
    ("manual_entry", "entered_after_hrs"),
    ("manual_entry", "duplicate_entry"),
    ("manual_entry", "desc_quality"),

    # red flags that directly influence anomalous risk
    ("amt_deviation", "anomalous"),
    ("duplicate_entry", "anomalous"),
    ("new_vendor", "anomalous"),
    ("missing_po", "anomalous"),
    ("entered_after_hrs", "anomalous"),
    ("manual_entry", "anomalous"),
    ("unusual_accts", "anomalous"),
    ("desc_quality", "anomalous"),
]


def build_and_train_bayesian_network(train_df):
    # build a Bayesian Network and learn its conditional probability tables.
    
    model = BayesianNetworkModel(EDGES)

    model_columns = list(model.nodes())
    train_model_df = train_df[model_columns].copy()

    estimator = BayesianEstimator(model, train_model_df)

    # use BDeu prior with equivalent sample size of 10 to learn CPDs
    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=10,
    )

    model.add_cpds(*cpds)
    model.check_model()

    return model


def predict_with_bayesian_network(model, test_df):
    # predict P(anomalous = yes | observed red flags) for each test row.
    inference = VariableElimination(model)

    # grab all columns except the target "anomalous" as evidence columns
    evidence_columns = [
        col for col in model.nodes()
        if col != "anomalous"
    ]

    probabilities = []

    # cache repeated red-flag combinations so prediction runs faster
    probability_cache = {}

    for _, row in test_df.iterrows():
        evidence = {
            col: str(row[col])
            for col in evidence_columns
            if col in test_df.columns
        }

        # remove evidence values that the model has never seen during training
        safe_evidence = {}
        for col, value in evidence.items():
            cpd = model.get_cpds(col)
            known_states = set(map(str, cpd.state_names[col]))
            if value in known_states:
                safe_evidence[col] = value

        cache_key = tuple(sorted(safe_evidence.items()))

        # if we've already computed probability for this combination, use cached value
        if cache_key not in probability_cache:
            query_result = inference.query(
                variables=["anomalous"],
                evidence=safe_evidence,
                show_progress=False,
            )
            probability_cache[cache_key] = get_probability_of_yes(query_result)

        probabilities.append(probability_cache[cache_key])

    return np.array(probabilities)


def save_model_outputs(model, test_df, anomaly_probabilities):
    # save metrics, predictions, network edges, and learned CPDs.
    y_true_labels = test_df["anomalous"].values
    y_true = np.array([1 if label == "yes" else 0 for label in y_true_labels])

    y_pred = (anomaly_probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # handle cases where only one class is present in y_true
    try:
        roc_auc = roc_auc_score(y_true, anomaly_probabilities)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, anomaly_probabilities)
    except ValueError:
        pr_auc = np.nan

    cm = confusion_matrix(y_true, y_pred)

    # save predictions
    predictions_df = test_df.copy()
    predictions_df["true_anomalous_binary"] = y_true
    predictions_df["predicted_anomalous_binary"] = y_pred
    predictions_df["anomaly_probability"] = anomaly_probabilities
    predictions_df.to_csv("outputs/network_output/bayesian_network_predictions.csv", index=False)

    # save metrics report
    with open("outputs/network_output/bayesian_network_metrics.txt", "w") as f:
        f.write("Bayesian Network Metrics\n")
        f.write("------------------------\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # save network edges
    edges_df = pd.DataFrame(list(model.edges()), columns=["parent", "child"])
    edges_df.to_csv("outputs/network_output/bayesian_network_edges.csv", index=False)

    # save CPDs as readable text
    with open("outputs/network_output/bayesian_network_cpds.txt", "w") as f:
        for cpd in model.get_cpds():
            f.write(str(cpd))
            f.write("\n\n")

    # save model as pickle
    # pickle is a python library that allows us to save a model to an object
    with open("data/bayesian_network_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # build graph visualization of the Bayesian Network structure and save as image
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_edges_from(model.edges())

        plt.figure(figsize=(12, 7))
        pos = nx.spring_layout(graph, seed=42)

        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=2500,
            font_size=9,
            arrows=True,
            arrowsize=20,
        )

        plt.title("Bayesian Network Structure: Dependencies Among Red Flags")
        plt.tight_layout()
        plt.savefig("outputs/network_output/bayesian_network_structure.png", dpi=300)
        plt.close()

    except Exception as exc:
        print(f"Could not save graph image: {exc}")

    print("\nEvaluation complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("\nSaved:")
    print("- outputs/network_output/bayesian_network_metrics.txt")
    print("- outputs/network_output/bayesian_network_predictions.csv")
    print("- outputs/network_output/bayesian_network_edges.csv")
    print("- outputs/network_output/bayesian_network_cpds.txt")
    print("- outputs/network_output/bayesian_network_structure.png")
    print("- data/bayesian_network_model.pkl")


def main():
    print("Loading data...")
    train_df, test_df = load_data()

    print("Building Bayesian Network...")
    model = build_and_train_bayesian_network(train_df)

    print("\nNetwork edges:")
    for parent, child in model.edges():
        print(f"{parent} -> {child}")

    print("\nPredicting anomaly probabilities...")
    anomaly_probabilities = predict_with_bayesian_network(model, test_df)

    save_model_outputs(model, test_df, anomaly_probabilities)


if __name__ == "__main__":
    main()
