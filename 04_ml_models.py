"""
04_ml_models.py
Trains Logistic Regression and Random Forest models for two tasks:
  1. At-risk student classification
  2. Placement probability prediction

Evaluates both, saves metrics/feature importance, and exports the
Logistic Regression coefficients as JSON so the dashboard's
"live forecast" tab can score new students instantly in-browser
(no server round-trip needed).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

out_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(out_dir, "cleaned_student_data.csv"))

FEATURES = [
    "attendance_pct", "cgpa", "backlog_count", "coding_score",
    "aptitude_score", "communication_score", "extracurricular_score",
    "internship_count", "projects_count",
]

def train_and_evaluate(target_col, model_name_prefix):
    X = df[FEATURES]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_pred = logreg.predict(X_test_scaled)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    def metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred), 3),
            "recall": round(recall_score(y_true, y_pred), 3),
            "f1": round(f1_score(y_true, y_pred), 3),
            "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
        }

    results = {
        "logistic_regression": metrics(y_test, logreg_pred, logreg_proba),
        "random_forest": metrics(y_test, rf_pred, rf_proba),
    }

    print(f"\n=== {model_name_prefix} ===")
    print(pd.DataFrame(results).T)

    # Feature importance (Random Forest)
    importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    importance.plot(kind="barh", color="#2c3e91", ax=ax)
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance — {model_name_prefix} (Random Forest)")
    fig.tight_layout()
    chart_path = os.path.join(out_dir, "charts", f"{model_name_prefix.lower().replace(' ', '_')}_feature_importance.png")
    fig.savefig(chart_path, dpi=140)
    plt.close(fig)

    # Confusion matrix for logistic regression (used in the writeup)
    cm = confusion_matrix(y_test, logreg_pred)

    return {
        "results": results,
        "feature_importance": importance.round(3).to_dict(),
        "confusion_matrix_logreg": cm.tolist(),
        "logreg_model": logreg,
        "scaler": scaler,
    }

at_risk_out = train_and_evaluate("at_risk", "At-Risk Prediction")
placement_out = train_and_evaluate("placed", "Placement Prediction")

# ---- Export Logistic Regression coefficients for live in-browser scoring ----
def export_logreg_json(bundle, label):
    scaler = bundle["scaler"]
    model = bundle["logreg_model"]
    return {
        "label": label,
        "features": FEATURES,
        "scaler_mean": scaler.mean_.round(6).tolist(),
        "scaler_scale": scaler.scale_.round(6).tolist(),
        "coefficients": model.coef_[0].round(6).tolist(),
        "intercept": round(float(model.intercept_[0]), 6),
        "metrics": bundle["results"],
        "feature_importance": bundle["feature_importance"],
    }

export = {
    "at_risk_model": export_logreg_json(at_risk_out, "At-Risk"),
    "placement_model": export_logreg_json(placement_out, "Placement"),
    "feature_ranges": {
        f: {"min": float(df[f].min()), "max": float(df[f].max()), "mean": round(float(df[f].mean()), 1)}
        for f in FEATURES
    },
}

model_json_path = os.path.join(out_dir, "model_export.json")
with open(model_json_path, "w") as f:
    json.dump(export, f, indent=2)

print(f"\nExported model coefficients for live scoring -> {model_json_path}")
