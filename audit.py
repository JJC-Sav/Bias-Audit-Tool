# audit.py
# Main auditing script. Loads a trained model and dataset, runs predictions,
# computes fairness metrics per group for multiple sensitive attributes,
# flags disparities, and saves a full audit report.

import json
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

# load the config file
with open("config.json", "r") as f:
    config = json.load(f)

print("Config loaded")
print(f"  Sensitive attributes: {config['sensitive_attributes']}")
print(f"  Threshold: {config['threshold']}")

# load the dataset and model
df = pd.read_csv(config["dataset_path"])
model = joblib.load(config["model_path"])

print(f"\nDataset loaded — {len(df)} rows")
print(f"Model loaded — {type(model).__name__}")

# run predictions on every row
X = df[config["feature_columns"]]
predictions = model.predict(X)
df["prediction"] = predictions

print(f"\nPredictions done — {len(predictions)} total")

# convert label column to 1s and 0s so confusion matrix works
positive_label = config["positive_label"]
df["true_label"] = (df[config["label_column"]] == positive_label).astype(int)

# ── Metric functions ───────────────────────────────────────────────────────────

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
    # out of all predictions, how often did the model predict positive?
    total = TP + TN + FP + FN
    if total == 0:
        return None
    return (TP + FP) / total

def compute_fnr(TP, TN, FP, FN):
    # how many actual positives did the model miss?
    denominator = TP + FN
    if denominator == 0:
        return None
    return FN / denominator

def compute_precision(TP, TN, FP, FN):
    # out of all predicted positives, how many were actually positive?
    denominator = TP + FP
    if denominator == 0:
        return None
    return TP / denominator

def compute_f1(TP, TN, FP, FN):
    # balance between precision and TPR
    precision = compute_precision(TP, TN, FP, FN)
    tpr = compute_tpr(TP, TN, FP, FN)
    if precision is None or tpr is None:
        return None
    if precision + tpr == 0:
        return None
    return 2 * (precision * tpr) / (precision + tpr)

def compute_equalized_odds(tpr_a, fpr_a, tpr_b, fpr_b):
    # max gap across TPR and FPR between two groups
    if any(v is None for v in [tpr_a, fpr_a, tpr_b, fpr_b]):
        return None
    tpr_gap = abs(tpr_a - tpr_b)
    fpr_gap = abs(fpr_a - fpr_b)
    return round(max(tpr_gap, fpr_gap), 4)

def compute_group_metrics(group_df):
    """Compute all metrics for a single group dataframe."""
    y_true = group_df["true_label"]
    y_pred = group_df["prediction"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    accuracy  = compute_accuracy(TP, TN, FP, FN)
    tpr       = compute_tpr(TP, TN, FP, FN)
    fpr       = compute_fpr(TP, TN, FP, FN)
    ppr       = compute_ppr(TP, TN, FP, FN)
    fnr       = compute_fnr(TP, TN, FP, FN)
    precision = compute_precision(TP, TN, FP, FN)
    f1        = compute_f1(TP, TN, FP, FN)

    return {
        "group_size": len(group_df),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "accuracy":  round(accuracy,  4) if accuracy  is not None else None,
        "tpr":       round(tpr,       4) if tpr       is not None else None,
        "fpr":       round(fpr,       4) if fpr       is not None else None,
        "ppr":       round(ppr,       4) if ppr       is not None else None,
        "fnr":       round(fnr,       4) if fnr       is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "f1":        round(f1,        4) if f1        is not None else None,
    }

def run_disparity_analysis(group_metrics, threshold):
    """Compare every pair of groups and flag metrics over the threshold."""
    metrics_to_check = ["accuracy", "tpr", "fpr", "ppr", "fnr", "precision", "f1"]
    group_list = list(group_metrics.keys())
    flagged_results = []

    for i in range(len(group_list)):
        for j in range(i + 1, len(group_list)):
            group_a = group_list[i]
            group_b = group_list[j]

            print(f"\n  Comparing: {group_a} vs {group_b}")

            for metric in metrics_to_check:
                val_a = group_metrics[group_a][metric]
                val_b = group_metrics[group_b][metric]

                if val_a is None or val_b is None:
                    print(f"    {metric.upper():<12} skipped (missing value)")
                    continue

                disparity = round(abs(val_a - val_b), 4)
                flagged = disparity > threshold
                flag_marker = " <- FLAGGED" if flagged else ""
                print(f"    {metric.upper():<12} {group_a}={val_a:.4f}  {group_b}={val_b:.4f}  diff={disparity:.4f}{flag_marker}")

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

            # equalized odds check
            eq_odds = compute_equalized_odds(
                group_metrics[group_a]["tpr"], group_metrics[group_a]["fpr"],
                group_metrics[group_b]["tpr"], group_metrics[group_b]["fpr"]
            )
            if eq_odds is not None:
                flagged = eq_odds > threshold
                flag_marker = " <- FLAGGED" if flagged else ""
                print(f"    {'EQ_ODDS':<12} max gap={eq_odds:.4f}{flag_marker}")
                if flagged:
                    flagged_results.append({
                        "metric": "equalized_odds",
                        "group_a": group_a,
                        "group_b": group_b,
                        "value_a": None,
                        "value_b": None,
                        "disparity": eq_odds,
                        "threshold": threshold
                    })

    return flagged_results

# ── Main audit loop — runs for each sensitive attribute ───────────────────────
threshold = config["threshold"]
sensitive_attributes = config["sensitive_attributes"]
all_results = {}  # stores results for every attribute

for attr in sensitive_attributes:
    print(f"\n{'='*60}")
    print(f"AUDITING ATTRIBUTE: {attr}")
    print(f"{'='*60}")

    # check the column exists in the dataset
    if attr not in df.columns:
        print(f"  WARNING: column '{attr}' not found in dataset — skipping")
        continue

    groups = df[attr].unique()
    group_metrics = {}

    # compute metrics for each group
    for group in groups:
        group_df = df[df[attr] == group]
        group_metrics[group] = compute_group_metrics(group_df)

        m = group_metrics[group]
        print(f"\nGroup: {group} (n={m['group_size']})")
        print(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
        print(f"  Accuracy:                 {m['accuracy']:.4f}")
        print(f"  True Positive Rate:       {m['tpr']:.4f}")
        print(f"  False Positive Rate:      {m['fpr']:.4f}")
        print(f"  False Negative Rate:      {m['fnr']:.4f}")
        print(f"  Positive Prediction Rate: {m['ppr']:.4f}")
        print(f"  Precision:                {m['precision']:.4f}")
        print(f"  F1 Score:                 {m['f1']:.4f}")

    # run disparity analysis for this attribute
    print(f"\nDisparity Analysis — {attr}")
    print("-" * 60)
    flagged = run_disparity_analysis(group_metrics, threshold)

    # flagged summary for this attribute
    print(f"\nFlagged Results — {attr}")
    print("-" * 60)
    if flagged:
        for flag in flagged:
            print(f"  [{flag['metric'].upper()}] {flag['group_a']} vs {flag['group_b']} "
                  f"- diff={flag['disparity']:.4f} (threshold={flag['threshold']})")
    else:
        print("  Nothing flagged.")

    # store results for this attribute
    all_results[attr] = {
        "group_metrics": group_metrics,
        "flagged_results": flagged
    }

# ── Save full audit report ─────────────────────────────────────────────────────
report = {
    "model_name":   config["model_name"],
    "model_path":   config["model_path"],
    "dataset_name": config["dataset_name"],
    "dataset_path": config["dataset_path"],
    "audit_config": {
        "sensitive_attributes": config["sensitive_attributes"],
        "label_column":         config["label_column"],
        "positive_label":       config["positive_label"],
        "threshold":            config["threshold"],
        "feature_columns":      config["feature_columns"],
        "metrics_computed": [
            "accuracy", "tpr", "fpr", "ppr",
            "fnr", "precision", "f1", "equalized_odds"
        ]
    },
    "results_by_attribute": all_results,
    "disclaimer": (
        "These results are descriptive only. They show differences in model "
        "performance across groups but do not imply legal or causal conclusions."
    )
}

with open("report.json", "w") as f:
    json.dump(report, f, indent=4)

print(f"\n{'='*60}")
print("Audit complete. Report saved to report.json")
print("Done.")
