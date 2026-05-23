import json
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ── 1. Create folders if they don't exist ──────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 2. Load the Adult Income dataset ──────────────────────────────────────────
# Column names for the Adult dataset from UCI
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

print("Loading Adult Income dataset...")
df = pd.read_csv(
    "adult.csv", names=COLUMNS, header=None, skipinitialspace=True  # removes leading spaces in values
)

# ── 3. Cleaning the data ──────────────────────────────────────────────────────────

# Replace missing values (marked as '?') with NaN and drop them
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# Strip any extra whitespace from string columns
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# Save the cleaned dataset to CSV so audit.py can load it
df.to_csv("data/adult.csv", index=False)
print(f"Dataset saved to data/adult.csv ({len(df)} rows)")

# ── 4. Preparing features for the training ──────────────────────────────────────────
# Fairness analysis will still use the categorical sensitive attribute (sex).
FEATURE_COLUMNS = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

X = df[FEATURE_COLUMNS]

# Encode the label column: ">50K" → 1, "<=50K" → 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["income"])

print(f"Label classes: {label_encoder.classes_}")  # shows which is 0 and which is 1
print(f"Feature columns used: {FEATURE_COLUMNS}")

# ── 5. Split into train and test sets ─────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Save the test split so audit.py evaluates on unseen data
X_test_df = df.loc[X_test.index].copy()
X_test_df.to_csv("data/adult_test.csv", index=False)
print("Test split saved to data/adult_test.csv")

print("\nTraining logistic regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("Training complete.")

# ── 6. Save the model ─────────────────────────────────────────────────────────
joblib.dump(model, "models/model.joblib")
print("Model saved to models/model.joblib")

# ── 7. Save a config.json ─────────────────────────────────────────────────────
config = {
    "model_path": "models/model.joblib",
    "dataset_path": "data/adult_test.csv",
    "label_column": "income",
    "positive_label": ">50K",
    "sensitive_attribute": "sex",
    "feature_columns": FEATURE_COLUMNS,
    "threshold": 0.1
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Config saved to config.json")
print("\nDone! You are ready to run audit.py")
