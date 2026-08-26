"""
03_shap_explain.py
-------------------
Loads the best model (selected in 02_train_models.py) and generates SHAP
explanations:
  1. Global feature importance (mean |SHAP value|) across the test set.
  2. A SHAP summary (beeswarm) plot.
  3. Local explanation examples for a handful of individual patients
     (matches the "why did the model predict this?" output format).

This is also where you check the sharpened contribution: whether the
top SHAP features align with clinically established diabetes risk
factors (Glucose/BMI/Age-type variables), or whether the model leans on
less clinically-obvious features (e.g., Income, Education, NoDocbcCost)
-- which matters for clinical trust and deployment.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

# ---- Load best model + info ----
with open("outputs/best_model_info.json") as f:
    info = json.load(f)

best_name = info["best_model_name"]
feature_type = info["feature_type"]
print(f"Explaining best model: {best_name} (features: {feature_type})")

model = joblib.load("outputs/best_model.joblib")

X_test = pd.read_csv(f"data/X_test_{feature_type}.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Use a sample of the test set for SHAP (full 50k rows is expensive)
SAMPLE_N = 2000
sample_idx = X_test.sample(n=SAMPLE_N, random_state=42).index
X_sample = X_test.loc[sample_idx].reset_index(drop=True)

# ---- Build SHAP explainer ----
# TreeExplainer for tree-based models (Decision Tree, Random Forest, XGBoost)
# LinearExplainer / KernelExplainer fallback for Logistic Regression
tree_models = ["Decision Tree", "Random Forest", "XGBoost"]

if best_name in tree_models:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # XGBoost/RF binary classifiers with TreeExplainer return a single array
    # for the positive class in recent SHAP versions; handle both cases.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
else:
    explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

if shap_values.ndim == 3:
    # (n_samples, n_features, n_classes) -> take positive class
    shap_values = shap_values[:, :, 1]

# ---- 1. Global feature importance ----
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\n===== GLOBAL FEATURE IMPORTANCE (SHAP) =====")
print(importance_df.to_string(index=False))
importance_df.to_csv("outputs/shap_global_importance.csv", index=False)

# ---- 2. Summary (beeswarm) plot ----
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- 3. Bar plot of global importance ----
plt.figure(figsize=(8, 6))
plt.barh(importance_df["feature"][:10][::-1], importance_df["mean_abs_shap"][:10][::-1])
plt.xlabel("Mean |SHAP value|")
plt.title(f"Top 10 Feature Importances ({best_name}, SHAP)")
plt.tight_layout()
plt.savefig("outputs/shap_bar_top10.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- 4. Local explanations for a few example patients ----
# Pick: one high-risk correctly predicted, one low-risk correctly predicted,
# one misclassified case (useful discussion material for the paper).
probs = model.predict_proba(X_sample)[:, 1]
preds = (probs >= 0.5).astype(int)
y_sample_true = y_test[sample_idx.values] if hasattr(sample_idx, "values") else y_test[sample_idx]

examples = {}

# high-confidence correct positive
pos_correct = np.where((preds == 1) & (y_sample_true == 1))[0]
if len(pos_correct) > 0:
    i = pos_correct[np.argmax(probs[pos_correct])]
    examples["high_risk_correct"] = i

# high-confidence correct negative
neg_correct = np.where((preds == 0) & (y_sample_true == 0))[0]
if len(neg_correct) > 0:
    i = neg_correct[np.argmin(probs[neg_correct])]
    examples["low_risk_correct"] = i

# a misclassification
misclassified = np.where(preds != y_sample_true)[0]
if len(misclassified) > 0:
    i = misclassified[0]
    examples["misclassified_example"] = i

local_report = []
for label, i in examples.items():
    row = X_sample.iloc[i]
    contribs = pd.Series(shap_values[i], index=X_sample.columns).sort_values(
        key=np.abs, ascending=False
    )
    top5 = contribs.head(5)
    print(f"\n--- {label} (row {i}) ---")
    print(f"Predicted probability of diabetes/prediabetes: {probs[i]:.3f}")
    print(f"True label: {y_sample_true[i]}, Predicted label: {preds[i]}")
    print("Top contributing features:")
    print(top5)
    local_report.append({
        "case": label,
        "row_index": int(i),
        "predicted_probability": float(probs[i]),
        "true_label": int(y_sample_true[i]),
        "predicted_label": int(preds[i]),
        "top_features": {k: float(v) for k, v in top5.items()}
    })

with open("outputs/shap_local_examples.json", "w") as f:
    json.dump(local_report, f, indent=2)

# Save SHAP values + sample data for the Streamlit app / further analysis
np.save("outputs/shap_values_sample.npy", shap_values)
X_sample.to_csv("outputs/X_sample_for_shap.csv", index=False)

print("\nSaved: outputs/shap_global_importance.csv, shap_summary_beeswarm.png,")
print("       shap_bar_top10.png, shap_local_examples.json")
