"""
01_preprocess.py
-----------------
Loads the CDC BRFSS 2015 Diabetes Health Indicators dataset, binarizes the
target (0 = no diabetes, 1 = prediabetes or diabetes), splits into
train/test, and scales numeric features.

Dataset source: BRFSS 2015, distributed as
diabetes_012_health_indicators_BRFSS2015.csv (253,680 respondents, 21
health indicator features + 3-class target).

Original classes:
  0 = no diabetes
  1 = prediabetes
  2 = diabetes

We binarize to (0 = no diabetes) vs (1 = prediabetes or diabetes) because
our project frames this as a binary risk-screening task (matches the
"High Risk / Low Risk" framing in the project brief). This choice should
be stated explicitly in the paper's Methodology / Data section.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "brfss2015.csv")
OUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading raw BRFSS 2015 data...")
df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")

# ---- Binarize target ----
# 0 = no diabetes, 1 = prediabetes/diabetes (at-risk / diabetic)
df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)
df = df.drop(columns=["Diabetes_012"])

print("\nBinary target distribution:")
print(df["Diabetes_binary"].value_counts())
print(df["Diabetes_binary"].value_counts(normalize=True).round(3))

# ---- Feature / target split ----
FEATURE_COLS = [c for c in df.columns if c != "Diabetes_binary"]
X = df[FEATURE_COLS]
y = df["Diabetes_binary"]

# ---- Train/test split (stratified due to class imbalance) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Train class balance:", y_train.value_counts(normalize=True).round(3).to_dict())

# ---- Scale numeric features ----
# Most BRFSS features here are binary (0/1) already; BMI, GenHlth, MentHlth,
# PhysHlth, Age, Education, Income are ordinal/continuous-ish.
# We scale ALL features with StandardScaler for the linear model (Logistic
# Regression) and keep an unscaled copy for the tree-based models
# (Decision Tree, Random Forest, XGBoost), which don't require scaling and
# for which SHAP TreeExplainer works directly on raw feature values
# (more interpretable SHAP plots).
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
)

# ---- Save everything ----
X_train.to_csv(os.path.join(OUT_DIR, "X_train_raw.csv"), index=False)
X_test.to_csv(os.path.join(OUT_DIR, "X_test_raw.csv"), index=False)
X_train_scaled.to_csv(os.path.join(OUT_DIR, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(OUT_DIR, "X_test_scaled.csv"), index=False)
y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

with open(os.path.join(OUT_DIR, "feature_columns.txt"), "w") as f:
    f.write("\n".join(FEATURE_COLS))

print("\nSaved preprocessed data to /data")
print("Feature columns:", FEATURE_COLS)
