# bias_detector/dataset_auditor.py
# Analyzes a dataset for potential bias before a model is even trained.
# Checks group representation, label distribution, sample size, and imbalance ratios.

import pandas as pd


# threshold for flagging small groups (fewer than this many records)
MIN_GROUP_SIZE = 100

# threshold for flagging imbalance ratio between largest and smallest group
MAX_IMBALANCE_RATIO = 5.0

# threshold for flagging label distribution differences between groups
LABEL_DISPARITY_THRESHOLD = 0.1


def check_representation(df, sensitive_attr):
    """
    Checks how evenly groups are represented in the dataset.
    Flags groups that are very small or heavily imbalanced relative to others.
    """
    total = len(df)
    group_counts = df[sensitive_attr].value_counts()
    results = {}
    flags = []

    print(f"\n  Group Representation ({sensitive_attr})")
    print(f"  {'Group':<30} {'Count':<10} {'Percentage':<12} {'Flag'}")
    print(f"  {'-'*60}")

    for group, count in group_counts.items():
        percentage = round((count / total) * 100, 2)
        flagged = bool(count < MIN_GROUP_SIZE)
        flag_marker = "<- SMALL GROUP" if flagged else ""

        print(f"  {str(group):<30} {count:<10} {str(percentage)+'%':<12} {flag_marker}")

        results[group] = {
            "count": int(count),
            "percentage": percentage,
            "small_group_flag": flagged
        }

        if flagged:
            flags.append({
                "check": "small_group",
                "group": group,
                "count": int(count),
                "message": f"Group '{group}' has only {count} records — metrics may be unreliable."
            })

    # check imbalance ratio between largest and smallest group
    largest = int(group_counts.max())
    smallest = int(group_counts.min())
    ratio = round(largest / smallest, 2)
    ratio_flagged = bool(ratio > MAX_IMBALANCE_RATIO)

    print(f"\n  Imbalance ratio (largest / smallest): {ratio}x", end="")
    print(f" <- FLAGGED" if ratio_flagged else "")

    if ratio_flagged:
        flags.append({
            "check": "imbalance_ratio",
            "ratio": ratio,
            "message": f"Group size imbalance ratio is {ratio}x — the largest group is {ratio}x bigger than the smallest."
        })

    return results, flags


def check_label_distribution(df, sensitive_attr, label_column, positive_label):
    """
    Checks whether the positive label rate is consistent across groups.
    A big difference in positive rates between groups means the model
    will have very different training signal for each group.
    """
    results = {}
    flags = []

    print(f"\n  Label Distribution ({sensitive_attr} vs {label_column})")
    print(f"  {'Group':<30} {'Positive Rate':<18} {'Count Positive':<18} {'Flag'}")
    print(f"  {'-'*70}")

    positive_rates = {}

    for group in df[sensitive_attr].unique():
        group_df = df[df[sensitive_attr] == group]
        total_in_group = len(group_df)
        positive_count = int((group_df[label_column] == positive_label).sum())
        positive_rate = round(positive_count / total_in_group, 4)
        positive_rates[group] = positive_rate

        results[group] = {
            "positive_rate": positive_rate,
            "positive_count": positive_count,
            "total": total_in_group
        }

    # compare each group against the overall positive rate
    overall_positive_rate = round(
        (df[label_column] == positive_label).sum() / len(df), 4
    )

    for group, rate in positive_rates.items():
        disparity = round(abs(rate - overall_positive_rate), 4)
        flagged = bool(disparity > LABEL_DISPARITY_THRESHOLD)
        flag_marker = "<- FLAGGED" if flagged else ""

        print(f"  {str(group):<30} {rate:<18} {results[group]['positive_count']:<18} {flag_marker}")

        results[group]["disparity_from_overall"] = disparity
        results[group]["flagged"] = flagged

        if flagged:
            flags.append({
                "check": "label_distribution",
                "group": group,
                "positive_rate": rate,
                "overall_positive_rate": overall_positive_rate,
                "disparity": disparity,
                "message": (
                    f"Group '{group}' has a positive label rate of {rate:.1%} "
                    f"vs overall rate of {overall_positive_rate:.1%} "
                    f"(disparity: {disparity:.4f})."
                )
            })

    print(f"\n  Overall positive rate: {overall_positive_rate:.1%}")

    return results, flags


def check_missing_values(df, sensitive_attr, label_column):
    """
    Checks for missing values in the sensitive attribute and label columns.
    Missing values can silently skew results.
    """
    flags = []

    attr_missing = int(df[sensitive_attr].isna().sum())
    label_missing = int(df[label_column].isna().sum())

    print(f"\n  Missing Values")
    print(f"  {sensitive_attr}: {attr_missing} missing", end="")
    print(f" <- FLAGGED" if attr_missing > 0 else "")
    print(f"  {label_column}: {label_missing} missing", end="")
    print(f" <- FLAGGED" if label_missing > 0 else "")

    if attr_missing > 0:
        flags.append({
            "check": "missing_values",
            "column": sensitive_attr,
            "count": attr_missing,
            "message": f"Column '{sensitive_attr}' has {attr_missing} missing values."
        })

    if label_missing > 0:
        flags.append({
            "check": "missing_values",
            "column": label_column,
            "count": label_missing,
            "message": f"Column '{label_column}' has {label_missing} missing values."
        })

    return flags


def audit_dataset(df, sensitive_attributes, label_column, positive_label):
    """
    Runs all dataset checks for each sensitive attribute.
    Returns a full dataset audit report.
    """
    print("\n" + "=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    all_results = {}

    for attr in sensitive_attributes:
        print(f"\n{'='*60}")
        print(f"CHECKING ATTRIBUTE: {attr}")
        print(f"{'='*60}")

        if attr not in df.columns:
            print(f"  WARNING: column '{attr}' not found in dataset — skipping")
            continue

        attr_results = {}
        attr_flags = []

        rep_results, rep_flags = check_representation(df, attr)
        label_results, label_flags = check_label_distribution(
            df, attr, label_column, positive_label
        )
        missing_flags = check_missing_values(df, attr, label_column)

        attr_results["representation"] = rep_results
        attr_results["label_distribution"] = label_results
        attr_flags.extend(rep_flags)
        attr_flags.extend(label_flags)
        attr_flags.extend(missing_flags)

        print(f"\n  Flagged Issues — {attr}")
        print(f"  {'-'*50}")
        if attr_flags:
            for flag in attr_flags:
                print(f"  [{flag['check'].upper()}] {flag['message']}")
        else:
            print(f"  No issues found.")

        all_results[attr] = {
            "checks": attr_results,
            "flags": attr_flags
        }

    return all_results