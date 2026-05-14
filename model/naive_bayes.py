import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

DATA_PATH = "data/transactions.csv"

df = pd.read_csv(DATA_PATH)

TARGET_COL = "anomalous"

X = df.drop(columns=[TARGET_COL, "amt"])
y = df[TARGET_COL].copy()

# Encode target if it is yes/no strings
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_features = [
    "vendor_id",
    "amt_deviation",
    "duplicate_entry",
    "new_vendor",
    "missing_po",
    "entered_after_hrs",
    "manual_entry",
    "unusual_accts",
    "desc_quality",
]

# numeric_features = [
#     "amt",
# ]

# One-hot encode categorical features
# Standardize numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
    ]
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", CategoricalNB()),
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "naive_bayes_pipeline.joblib")
joblib.dump(label_encoder, "target_label_encoder.joblib")

# Save train/test splits
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

pd.DataFrame({"anomalous": y_train}).to_csv("y_train.csv", index=False)
pd.DataFrame({"anomalous": y_test}).to_csv("y_test.csv", index=False)

print("Model training complete.")
print("Saved:")
print("- naive_bayes_pipeline.joblib")
print("- target_label_encoder.joblib")
print("- X_train.csv, X_test.csv, y_train.csv, y_test.csv")