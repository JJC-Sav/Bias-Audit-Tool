# ml-bias-detector

A lightweight bias detection and audit tool for binary classification machine learning models.

Give it a trained model and a labeled dataset, and it tells you whether the model is treating different demographic groups fairly. Results are saved to a structured, reproducible audit report.

---

## What It Does

Machine learning models are used in high-stakes decisions like hiring, lending, and admissions. A model that looks accurate overall can still behave very differently across demographic groups like gender or race. This tool makes that visible.

The audit pipeline:

1. Reads settings from `config.json`
2. Loads the trained model and dataset
3. Generates predictions for every record
4. Splits predictions by the sensitive attribute (e.g. gender)
5. Computes four fairness metrics per group using a confusion matrix
6. Flags any metric where the gap between groups exceeds the threshold
7. Saves a full audit report to `report.json`

---

## Fairness Metrics

| Metric | What it measures |
|---|---|
| **Accuracy** | How often the model was correct overall for that group |
| **TPR** | Out of everyone who truly belongs to the positive class, how many did the model catch? |
| **FPR** | Out of everyone who belongs to the negative class, how many did the model wrongly flag? |
| **PPR** | Out of all predictions made, how often did the model predict the positive outcome? |

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/ml-bias-detector.git
cd ml-bias-detector
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

---

## Quick Start

**Step 1 — Set up an example model and dataset:**
```bash
python train_example_model.py
```
This downloads and cleans the Adult Income dataset, trains a logistic regression model, and generates `config.json`. Skip this step if you already have a trained model.

**Step 2 — Run the audit:**
```bash
python audit.py
```
The tool will print per-group metrics, flag any disparities over the threshold, and save a full report to `report.json`.

---

## Configuration

All settings are controlled through `config.json`:

```json
{
    "model_name": "logistic_regression_adult",
    "model_path": "models/model.joblib",
    "dataset_name": "adult_income_test",
    "dataset_path": "data/adult_test.csv",
    "label_column": "income",
    "positive_label": ">50K",
    "sensitive_attribute": "sex",
    "feature_columns": ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
    "threshold": 0.1
}
```

| Field | Description |
|---|---|
| `model_path` | Path to your saved `.joblib` model file |
| `dataset_path` | Path to your labeled CSV dataset |
| `label_column` | Column name containing the true labels |
| `positive_label` | The value that counts as the positive outcome |
| `sensitive_attribute` | Column to split groups by (e.g. `sex`, `race`) |
| `feature_columns` | List of feature columns the model was trained on |
| `threshold` | Gap size that triggers a flag (default: 0.1) |

---

## Example Results

Using the Adult Income dataset from UCI with gender as the sensitive attribute:

```
Group: Male (n=4065)
  Accuracy:                 0.7697
  True Positive Rate:       0.3978
  False Positive Rate:      0.0618
  Positive Prediction Rate: 0.1665

Group: Female (n=1968)
  Accuracy:                 0.8892
  True Positive Rate:       0.3532
  False Positive Rate:      0.0381
  Positive Prediction Rate: 0.0757

Disparity Analysis
------------------------------------------------------------
  ACCURACY     Male=0.7697  Female=0.8892  diff=0.1195 <- FLAGGED
  TPR          Male=0.3978  Female=0.3532  diff=0.0446
  FPR          Male=0.0618  Female=0.0381  diff=0.0237
  PPR          Male=0.1665  Female=0.0757  diff=0.0908
```

The model predicts high income for men more than twice as often as for women with similar backgrounds. Accuracy appears higher for women not because the model identifies high earners better, but because it defaults to predicting low income for women almost every time.

---

## Output — report.json

After every audit run the tool saves a structured report:

```json
{
    "model_name": "logistic_regression_adult",
    "dataset_name": "adult_income_test",
    "audit_config": { ... },
    "group_metrics": { ... },
    "flagged_results": [ ... ],
    "disclaimer": "These results are descriptive only..."
}
```

The report is fully reproducible — the same model, dataset, and config always produce the same output.

---

## Tech Stack

- **Python 3.12+**
- **pandas** — data loading, cleaning, and group splitting
- **scikit-learn** — model predictions and confusion matrix computation
- **joblib** — model serialization

---

## Roadmap

- [ ] Additional fairness metrics (Equalized Odds, Predictive Parity)
- [ ] Multiple sensitive attributes simultaneously
- [ ] Support for TensorFlow, PyTorch, and ONNX models
- [ ] Dataset bias detection before training
- [ ] Automated PDF report generation
- [ ] REST API endpoint
- [ ] Front end web interface
- [ ] Publish as installable package (`pip install ml-bias-detector`)

---

## Dataset

The example uses the [Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult) from the UCI Machine Learning Repository (Dua & Graff, 2019). It contains 1994 US Census data and is widely used in fairness research.

---

## Disclaimer

Audit results are descriptive only. They show differences in model performance across groups but do not imply legal or causal conclusions about fairness or discrimination.

---

## License

MIT License — free to use, modify, and distribute.