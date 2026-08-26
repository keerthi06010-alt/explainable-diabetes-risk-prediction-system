"""
02_train_models.py
-------------------
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost on
the preprocessed BRFSS data, evaluates each on Accuracy, Precision, Recall,
F1-score, and ROC-AUC, and saves the best model (by ROC-AUC, appropriate
given class imbalance) for the SHAP explainability stage.

class_weight='balanced' (or scale_pos_weight for XGBoost) is used because
the positive class (~15.8%) is a minority -- without this, all models
default toward predicting "no diabetes" and recall on the at-risk class
collapses. This should be reported explicitly in the paper as a design
choice for handling class imbalance.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# ---- Load data ----
X_train_raw = pd.read_csv("data/X_train_raw.csv")
X_test_raw = pd.read_csv("data/X_test_raw.csv")
X_train_scaled = pd.read_csv("data/X_train_scaled.csv")
X_test_scaled = pd.read_csv("data/X_test_scaled.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

os.makedirs("outputs", exist_ok=True)

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"Class imbalance ratio (neg/pos) = {scale_pos_weight:.2f}")

results = {}
fitted_models = {}

def evaluate(name, model, X_te, y_te, y_prob):
    y_pred = model.predict(X_te)
    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "ROC_AUC": roc_auc_score(y_te, y_prob),
    }
    results[name] = metrics
    print(f"\n== {name} ==")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(confusion_matrix(y_te, y_pred))
    return metrics

# ---- 1. Logistic Regression (uses scaled features) ----
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_scaled, y_train)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]
evaluate("Logistic Regression", lr, X_test_scaled, y_test, y_prob)
fitted_models["Logistic Regression"] = (lr, "scaled")

# ---- 2. Decision Tree (raw features) ----
dt = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
dt.fit(X_train_raw, y_train)
y_prob = dt.predict_proba(X_test_raw)[:, 1]
evaluate("Decision Tree", dt, X_test_raw, y_test, y_prob)
fitted_models["Decision Tree"] = (dt, "raw")

# ---- 3. Random Forest (raw features) ----
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train_raw, y_train)
y_prob = rf.predict_proba(X_test_raw)[:, 1]
evaluate("Random Forest", rf, X_test_raw, y_test, y_prob)
fitted_models["Random Forest"] = (rf, "raw")

# ---- 4. XGBoost (raw features) ----
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=42, n_jobs=-1
)
xgb.fit(X_train_raw, y_train)
y_prob = xgb.predict_proba(X_test_raw)[:, 1]
evaluate("XGBoost", xgb, X_test_raw, y_test, y_prob)
fitted_models["XGBoost"] = (xgb, "raw")

# ---- Comparison table ----
results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values("ROC_AUC", ascending=False)
print("\n\n===== MODEL COMPARISON (sorted by ROC-AUC) =====")
print(results_df)
results_df.to_csv("outputs/model_comparison.csv")

# ---- Select best model by ROC-AUC ----
# ---- Select best model by ROC-AUC, with a deterministic tie-break ----
TIE_TOLERANCE = 1e-3
PRIORITY = ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"]

top_score = results_df["ROC_AUC"].max()
tied_models = results_df[results_df["ROC_AUC"] >= top_score - TIE_TOLERANCE].index.tolist()
best_name = next(m for m in PRIORITY if m in tied_models)
best_model, best_feature_type = fitted_models[best_name]
print(f"\nModels within tie tolerance of top ROC-AUC: {tied_models}")
print(f"Best model selected: {best_name} (feature set: {best_feature_type})")

joblib.dump(best_model, "outputs/best_model.joblib")
with open("outputs/best_model_info.json", "w") as f:
    json.dump({
        "best_model_name": best_name,
        "feature_type": best_feature_type,
        "metrics": results[best_name]
    }, f, indent=2)

# Also save all models in case you want to compare SHAP across them later
joblib.dump(fitted_models, "outputs/all_models.joblib")

print("\nSaved: outputs/model_comparison.csv, outputs/best_model.joblib, outputs/best_model_info.json")
