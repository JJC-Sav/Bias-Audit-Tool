# Bias-Audit-Tool
A Python tool that audits ML models for demographic bias using fairness metrics (TPR, FPR, PPR) on the Adult Income dataset.

---

# ML Bias Detection & Audit Tool

A Python-based tool that audits machine learning models for demographic bias by computing fairness metrics across population subgroups. Built as a senior capstone project at Marian University.

---

## What It Does

This tool trains a logistic regression classifier on the **Adult Income dataset** and evaluates its predictions through a fairness lens. Rather than just measuring overall accuracy, it breaks down performance by demographic group to surface hidden disparities in how the model treats different populations.

---

## Fairness Metrics Computed

| Metric | Description |
|---|---|
| **Accuracy** | Overall correctness per group |
| **TPR (True Positive Rate)** | How often the model correctly predicts income >50K |
| **FPR (False Positive Rate)** | How often the model incorrectly flags income >50K |
| **PPR (Predicted Positive Rate)** | How frequently the model predicts the positive class |

---

## Tech Stack

- **Python**
- **scikit-learn** — logistic regression model
- **pandas** — data manipulation
- **NumPy** — numerical computation

---

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/JJC-Sav/Bias-Audit-Tool.git
   cd Bias-Audit-Tool
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Run the audit:
```bash
   python main.py
```

4. The tool will output a structured audit report showing fairness metrics broken down by demographic group.

---

## Why It Matters

ML models can produce biased outcomes even when overall accuracy looks fine. This tool makes those disparities visible and measurable — a critical step toward building fairer AI systems.

---

## Author

Juan Cisneros — Computer Science, Marian University '26  
[LinkedIn](https://linkedin.com/in/juanjcisneros)
