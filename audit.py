# audit.py

# This is the main auditing script. It loads a trained model and dataset,
# runs predictions, computes fairness metrics per group, flags any
# big differences between groups, and saves everything to a report file.

import json
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

# load the config file so I know what model, dataset, and settings to use
with open("config.json", "r") as f:
    config = json.load(f)

print("Config loaded")
print(f"  Sensitive attribute: {config['sensitive_attribute']}")
print(f"  Threshold: {config['threshold']}")

# load the dataset and the trained model
df = pd.read_csv(config["dataset_path"])
model = joblib.load(config["model_path"])

print(f"\nDataset loaded — {len(df)} rows")
print(f"Model loaded — {type(model).__name__}")

# run predictions on every row in the dataset
X = df[config["feature_columns"]]
predictions = model.predict(X)
df["prediction"] = predictions

print(f"\nPredictions done — {len(predictions)} total")
print("\nSample (first 10 rows):")
print(df[[config["sensitive_attribute"], config["label_column"], "prediction"]].head(10))

# convert the income label to 1s and 0s so the confusion matrix works
# >50K = 1, <=50K = 0
positive_label = config["positive_label"]
df["true_label"] = (df[config["label_column"]] == positive_label).astype(int)

# --- metric functions ---
# each one takes TP, TN, FP, FN and returns a single number
# returns None if there's a division by zero situation

def compute_accuracy(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    if total == 0:
        return None
    return (TP + TN) / total

def compute_tpr(TP, TN, FP, FN):
    # how many actual positives did the model catch?
    denominator = TP + FN
    if denominator == 0:
        return None
    return TP / denominator

def compute_fpr(TP, TN, FP, FN):
    # how many actual negatives did the model wrongly flag?
    denominator = FP + TN
    if denominator == 0:
        return None
    return FP / denominator

def compute_ppr(TP, TN, FP, FN):
    # out of all predictions, how many were labeled positive?
    total = TP + TN + FP + FN
    if total == 0:
        return None
    return (TP + FP) / total

# --- compute metrics for each group ---
print(f"\nMetrics by group ({config['sensitive_attribute']})")
print("-" * 60)

sensitive_attr = config["sensitive_attribute"]
groups = df[sensitive_attr].unique()
group_metrics = {}

for group in groups:
    # filter down to just this group
    group_df = df[df[sensitive_attr] == group]
    y_true = group_df["true_label"]
    y_pred = group_df["prediction"]

    # get the confusion matrix values for this group
    # confusion_matrix gives back [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # compute all four metrics
    accuracy = compute_accuracy(TP, TN, FP, FN)
    tpr      = compute_tpr(TP, TN, FP, FN)
    fpr      = compute_fpr(TP, TN, FP, FN)
    ppr      = compute_ppr(TP, TN, FP, FN)

    # save results so I can use them in the disparity analysis below
    group_metrics[group] = {
        "group_size": len(group_df),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "tpr":      round(tpr, 4)      if tpr      is not None else None,
        "fpr":      round(fpr, 4)      if fpr      is not None else None,
        "ppr":      round(ppr, 4)      if ppr      is not None else None,
    }

    print(f"\nGroup: {group} (n={len(group_df)})")
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"  Accuracy:                 {accuracy:.4f}")
    print(f"  True Positive Rate:       {tpr:.4f}")
    print(f"  False Positive Rate:      {fpr:.4f}")
    print(f"  Positive Prediction Rate: {ppr:.4f}")

# --- disparity analysis ---
# compare each pair of groups and flag metrics where the gap is too big
print("\n" + "-" * 60)
print("Disparity Analysis")
print("-" * 60)

threshold = config["threshold"]
group_list = list(group_metrics.keys())
metrics_to_check = ["accuracy", "tpr", "fpr", "ppr"]
flagged_results = []

for i in range(len(group_list)):
    for j in range(i + 1, len(group_list)):
        group_a = group_list[i]
        group_b = group_list[j]

        print(f"\nComparing: {group_a} vs {group_b}")

        for metric in metrics_to_check:
            val_a = group_metrics[group_a][metric]
            val_b = group_metrics[group_b][metric]

            # skip if one of the values is missing
            if val_a is None or val_b is None:
                print(f"  {metric.upper():<12} skipped (missing value)")
                continue

            disparity = round(abs(val_a - val_b), 4)
            flagged = disparity > threshold
            flag_marker = " <- FLAGGED" if flagged else ""

            print(f"  {metric.upper():<12} {group_a}={val_a:.4f}  {group_b}={val_b:.4f}  diff={disparity:.4f}{flag_marker}")

            # save flagged results so I can put them in the report later
            if flagged:
                flagged_results.append({
                    "metric": metric,
                    "group_a": group_a,
                    "group_b": group_b,
                    "value_a": val_a,
                    "value_b": val_b,
                    "disparity": disparity,
                    "threshold": threshold
                })

# --- flagged results summary ---
print("\n" + "-" * 60)
print("Flagged Results")
print("-" * 60)

if flagged_results:
    for flag in flagged_results:
        print(f"  [{flag['metric'].upper()}] {flag['group_a']} vs {flag['group_b']} "
              f"- diff={flag['disparity']:.4f} (threshold={flag['threshold']})")
else:
    print("  Nothing flagged — no big disparities found.")

# --- generate audit report ---

# pull everything together into one dictionary and save it as report.json
# this is the final output of the tool

report = {
    # identifiers so we know exactly what was audited
    "model_name": config["model_name"],
    "model_path": config["model_path"],
    "dataset_name": config["dataset_name"],
    "dataset_path": config["dataset_path"],

    # audit settings that were used
    "audit_config": {
        "sensitive_attribute": config["sensitive_attribute"],
        "label_column": config["label_column"],
        "positive_label": config["positive_label"],
        "threshold": config["threshold"],
        "feature_columns": config["feature_columns"]
    },

    # per group metric results
    "group_metrics": group_metrics,

    # flagged disparities
    "flagged_results": flagged_results,

    # short disclaimer so nobody misreads the results
    "disclaimer": (
        "These results are descriptive only. They show differences in model "
        "performance across groups but do not imply legal or causal conclusions."
    )
}

# save the report to a file
with open("report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\n" + "-" * 60)
print("Audit report saved to report.json")
print("Done.")
