# api.py
# REST API for ml-bias-detector using FastAPI.
# Wraps the audit pipeline in HTTP endpoints so other developers
# can call the tool programmatically.
#
# Run with: uvicorn api:app --reload
# Then open: http://127.0.0.1:8000/app for the web interface
# Or open:   http://127.0.0.1:8000/docs for the API documentation

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from bias_detector.model_loader import load_model
from bias_detector.dataset_auditor import audit_dataset
from bias_detector.report_generator import generate_pdf
from sklearn.metrics import confusion_matrix

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Bias Detector API",
    description="A fairness auditing tool for binary classification models.",
    version="0.1.0"
)

# ── Request model ──────────────────────────────────────────────────────────────
class AuditRequest(BaseModel):
    model_path:           str
    dataset_path:         str
    label_column:         str
    positive_label:       str
    sensitive_attributes: List[str]
    feature_columns:      List[str]
    threshold:            Optional[float] = 0.1
    model_name:           Optional[str]   = "model"
    dataset_name:         Optional[str]   = "dataset"

# ── Metric functions ───────────────────────────────────────────────────────────
def compute_accuracy(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    return (TP + TN) / total if total else None

def compute_tpr(TP, TN, FP, FN):
    d = TP + FN
    return TP / d if d else None

def compute_fpr(TP, TN, FP, FN):
    d = FP + TN
    return FP / d if d else None

def compute_ppr(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    return (TP + FP) / total if total else None

def compute_fnr(TP, TN, FP, FN):
    d = TP + FN
    return FN / d if d else None

def compute_precision(TP, TN, FP, FN):
    d = TP + FP
    return TP / d if d else None

def compute_f1(TP, TN, FP, FN):
    p = compute_precision(TP, TN, FP, FN)
    t = compute_tpr(TP, TN, FP, FN)
    if p is None or t is None or (p + t) == 0:
        return None
    return 2 * (p * t) / (p + t)

def compute_equalized_odds(tpr_a, fpr_a, tpr_b, fpr_b):
    if any(v is None for v in [tpr_a, fpr_a, tpr_b, fpr_b]):
        return None
    return round(max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b)), 4)

def compute_group_metrics(group_df):
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
    metrics_to_check = ["accuracy", "tpr", "fpr", "ppr", "fnr", "precision", "f1"]
    group_list = list(group_metrics.keys())
    flagged = []

    for i in range(len(group_list)):
        for j in range(i + 1, len(group_list)):
            group_a = group_list[i]
            group_b = group_list[j]

            for metric in metrics_to_check:
                val_a = group_metrics[group_a][metric]
                val_b = group_metrics[group_b][metric]
                if val_a is None or val_b is None:
                    continue
                disparity = round(abs(val_a - val_b), 4)
                if disparity > threshold:
                    flagged.append({
                        "metric": metric,
                        "group_a": group_a,
                        "group_b": group_b,
                        "value_a": val_a,
                        "value_b": val_b,
                        "disparity": disparity,
                        "threshold": threshold
                    })

            eq = compute_equalized_odds(
                group_metrics[group_a]["tpr"], group_metrics[group_a]["fpr"],
                group_metrics[group_b]["tpr"], group_metrics[group_b]["fpr"]
            )
            if eq is not None and eq > threshold:
                flagged.append({
                    "metric": "equalized_odds",
                    "group_a": group_a,
                    "group_b": group_b,
                    "value_a": None,
                    "value_b": None,
                    "disparity": eq,
                    "threshold": threshold
                })

    return flagged

def run_full_audit(req: AuditRequest):
    """Core audit logic shared by all endpoints."""

    if not os.path.exists(req.model_path):
        raise HTTPException(status_code=404,
            detail=f"Model file not found: {req.model_path}")
    if not os.path.exists(req.dataset_path):
        raise HTTPException(status_code=404,
            detail=f"Dataset file not found: {req.dataset_path}")

    model = load_model(req.model_path)
    df    = pd.read_csv(req.dataset_path)

    missing = [c for c in req.feature_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400,
            detail=f"Feature columns not found in dataset: {missing}")

    dataset_audit_results = audit_dataset(
        df,
        req.sensitive_attributes,
        req.label_column,
        req.positive_label
    )

    X = df[req.feature_columns]
    df["prediction"] = model.predict(X)
    df["true_label"] = (df[req.label_column] == req.positive_label).astype(int)

    all_results = {}
    for attr in req.sensitive_attributes:
        if attr not in df.columns:
            continue
        group_metrics = {}
        for group in df[attr].unique():
            group_df = df[df[attr] == group]
            group_metrics[group] = compute_group_metrics(group_df)
        flagged = run_disparity_analysis(group_metrics, req.threshold)
        all_results[attr] = {
            "group_metrics": group_metrics,
            "flagged_results": flagged
        }

    report = {
        "model_name":    req.model_name,
        "model_path":    req.model_path,
        "dataset_name":  req.dataset_name,
        "dataset_path":  req.dataset_path,
        "dataset_audit": dataset_audit_results,
        "audit_config": {
            "sensitive_attributes": req.sensitive_attributes,
            "label_column":         req.label_column,
            "positive_label":       req.positive_label,
            "threshold":            req.threshold,
            "feature_columns":      req.feature_columns,
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

    generate_pdf(report, output_path="audit_report.pdf")

    return report

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Check that the API is running."""
    return {
        "status": "ok",
        "tool": "ml-bias-detector",
        "version": "0.1.0",
        "message": "API is running. Open /app for the web interface or /docs for the API."
    }


@app.get("/app", response_class=HTMLResponse, tags=["Frontend"])
def frontend():
    """Serve the web interface."""
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404,
            detail="index.html not found. Make sure it is in the project root.")
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/audit", tags=["Audit"])
def run_audit(req: AuditRequest):
    """
    Run a full fairness audit on a trained model.

    Provide the model path, dataset path, sensitive attributes,
    feature columns, label column, positive label, and threshold.

    Returns the full audit report including dataset checks,
    per-group metrics, and flagged disparities.
    """
    report = run_full_audit(req)
    return JSONResponse(content=report)


@app.get("/report", tags=["Report"])
def get_report():
    """Return the last saved audit report as JSON."""
    if not os.path.exists("report.json"):
        raise HTTPException(status_code=404,
            detail="No report found. Run /audit first.")
    with open("report.json", "r") as f:
        return json.load(f)


@app.get("/report/pdf", tags=["Report"])
def get_pdf():
    """Download the last generated PDF audit report."""
    if not os.path.exists("audit_report.pdf"):
        raise HTTPException(status_code=404,
            detail="No PDF found. Run /audit first.")
    return FileResponse(
        "audit_report.pdf",
        media_type="application/pdf",
        filename="audit_report.pdf"
    )


@app.get("/config", tags=["Config"])
def get_config():
    """Return the current config.json settings."""
    if not os.path.exists("config.json"):
        raise HTTPException(status_code=404,
            detail="No config.json found.")
    with open("config.json", "r") as f:
        return json.load(f)