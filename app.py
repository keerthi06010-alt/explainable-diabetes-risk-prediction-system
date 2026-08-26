"""
app.py
------
Streamlit clinical decision support application.

Workflow:
  Patient enters information -> preprocessing -> best ML model ->
  diabetes risk prediction -> SHAP explanation -> prediction + important
  factors -> clinical decision support.

Run with:  streamlit run app.py
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Risk & Explainability Tool", layout="wide")

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_artifacts():
    with open("outputs/best_model_info.json") as f:
        info = json.load(f)
    model = joblib.load("outputs/best_model.joblib")
    with open("data/feature_columns.txt") as f:
        feature_cols = f.read().splitlines()
    explainer = shap.TreeExplainer(model)
    return info, model, feature_cols, explainer

info, model, feature_cols, explainer = load_artifacts()
best_name = info["best_model_name"]

st.title("🩺 Explainable AI Diabetes Risk Prediction")
st.caption(
    f"Clinical Decision Support Prototype — Best model: **{best_name}** "
    f"(ROC-AUC: {info['metrics']['ROC_AUC']:.3f} on held-out test data)"
)

st.warning(
    "⚠️ This tool is a decision-support prototype for research/educational "
    "purposes. It does not replace clinical diagnosis, laboratory testing, "
    "or physician judgment."
)

# ---------------- Sidebar: patient input form ----------------
st.sidebar.header("Patient Information")

def yn(label, key, default="No"):
    return 1 if st.sidebar.selectbox(label, ["No", "Yes"], index=0 if default == "No" else 1, key=key) == "Yes" else 0

with st.sidebar:
    st.subheader("Clinical / Screening History")
    high_bp = yn("High Blood Pressure", "highbp")
    high_chol = yn("High Cholesterol", "highchol")
    chol_check = yn("Cholesterol checked in last 5 years", "cholcheck", default="Yes")
    stroke = yn("History of Stroke", "stroke")
    heart_disease = yn("Heart Disease / Heart Attack history", "heart")
    diff_walk = yn("Difficulty walking / climbing stairs", "diffwalk")

    st.subheader("Body & Vitals")
    bmi = st.slider("BMI", 12.0, 60.0, 27.0, 0.1)
    gen_hlth = st.select_slider(
        "General Health (1=Excellent, 5=Poor)", options=[1, 2, 3, 4, 5], value=3
    )
    ment_hlth = st.slider("Days of poor mental health (past 30 days)", 0, 30, 2)
    phys_hlth = st.slider("Days of poor physical health (past 30 days)", 0, 30, 2)

    st.subheader("Lifestyle")
    smoker = yn("Smoked ≥100 cigarettes in lifetime", "smoker")
    phys_activity = yn("Physical activity in past 30 days", "physact", default="Yes")
    fruits = yn("Consumes fruit ≥1x/day", "fruits", default="Yes")
    veggies = yn("Consumes vegetables ≥1x/day", "veggies", default="Yes")
    hvy_alcohol = yn("Heavy alcohol consumption", "alcohol")

    st.subheader("Access & Demographics")
    any_healthcare = yn("Has any healthcare coverage", "healthcare", default="Yes")
    no_doc_cost = yn("Couldn't see doctor due to cost (past year)", "nodoccost")
    sex = st.selectbox("Sex", ["Female", "Male"], key="sex")
    age_group = st.select_slider(
        "Age group (BRFSS 13-level code, 1=18-24 ... 13=80+)",
        options=list(range(1, 14)), value=7
    )
    education = st.select_slider(
        "Education level (1=None ... 6=College grad)",
        options=list(range(1, 7)), value=5
    )
    income = st.select_slider(
        "Income level (1=Lowest ... 8=Highest)",
        options=list(range(1, 9)), value=5
    )

# ---------------- Build feature row ----------------
patient = {
    "HighBP": high_bp, "HighChol": high_chol, "CholCheck": chol_check,
    "BMI": bmi, "Smoker": smoker, "Stroke": stroke,
    "HeartDiseaseorAttack": heart_disease, "PhysActivity": phys_activity,
    "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": hvy_alcohol,
    "AnyHealthcare": any_healthcare, "NoDocbcCost": no_doc_cost,
    "GenHlth": gen_hlth, "MentHlth": ment_hlth, "PhysHlth": phys_hlth,
    "DiffWalk": diff_walk, "Sex": 1 if sex == "Male" else 0,
    "Age": age_group, "Education": education, "Income": income,
}
X_input = pd.DataFrame([patient])[feature_cols]

# ---------------- Predict ----------------
col1, col2 = st.columns([1, 1.4])

prob = model.predict_proba(X_input)[0, 1]
pred = int(prob >= 0.5)
risk_label = "HIGH RISK" if pred == 1 else "LOWER RISK"
risk_color = "🔴" if pred == 1 else "🟢"

with col1:
    st.subheader("Prediction")
    st.metric("Predicted Risk", f"{risk_color} {risk_label}")
    st.metric("Estimated Probability", f"{prob*100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    st.subheader("Clinical Decision Support")
    st.markdown(
        "This prediction and the contributing factors below should be "
        "reviewed **alongside** medical history, laboratory results "
        "(e.g., fasting glucose / HbA1c), and clinical examination. "
        "It is intended to support — not replace — clinician judgment."
    )

# ---------------- SHAP explanation ----------------
with col2:
    st.subheader("Why this prediction? (SHAP explanation)")
    shap_vals = explainer.shap_values(X_input)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    shap_row = pd.Series(shap_vals[0], index=feature_cols).sort_values(
        key=np.abs, ascending=False
    )
    top8 = shap_row.head(8)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top8.values[::-1]]
    ax.barh(top8.index[::-1], top8.values[::-1], color=colors)
    ax.set_xlabel("SHAP value (→ increases risk / ← decreases risk)")
    ax.set_title("Top contributing factors for this patient")
    st.pyplot(fig)

    st.caption(
        "🔴 Red bars push the prediction toward higher diabetes risk. "
        "🔵 Blue bars push it toward lower risk."
    )

st.divider()
st.subheader("Model Comparison (from training/evaluation phase)")
try:
    comp_df = pd.read_csv("outputs/model_comparison.csv", index_col=0)
    st.dataframe(comp_df.style.highlight_max(axis=0, color="lightgreen"))
except FileNotFoundError:
    st.info("Run 02_train_models.py first to generate the comparison table.")

st.divider()
st.subheader("Global Feature Importance (SHAP, test-set sample)")
try:
    global_imp = pd.read_csv("outputs/shap_global_importance.csv")
    st.bar_chart(global_imp.set_index("feature")["mean_abs_shap"].head(10))
except FileNotFoundError:
    st.info("Run 03_shap_explain.py first to generate global SHAP importance.")
