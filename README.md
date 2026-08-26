# Explainable AI-Based Diabetes Risk Prediction and Clinical Decision Support System

End-to-end pipeline: BRFSS 2015 data -> preprocessing -> 4-model comparison
-> SHAP explainability -> Streamlit clinical decision support app.

## Setup
```
pip install -r requirements.txt
```

## Run the pipeline (in order)
```
python 01_preprocess.py       # loads data/brfss2015.csv, splits, scales
python 02_train_models.py     # trains LR, Decision Tree, RF, XGBoost; compares; saves best
python 03_shap_explain.py     # SHAP global + local explanations on best model
streamlit run app.py          # interactive clinical decision support app
```

## Files
- `data/brfss2015.csv` — raw BRFSS 2015 diabetes health indicators dataset (253,680 rows)
- `data/` — preprocessed train/test splits (generated)
- `outputs/model_comparison.csv` — Accuracy/Precision/Recall/F1/ROC-AUC for all 4 models
- `outputs/best_model.joblib` — saved best model (selected by ROC-AUC)
- `outputs/shap_global_importance.csv` — ranked SHAP feature importances
- `outputs/shap_summary_beeswarm.png`, `shap_bar_top10.png` — SHAP plots for the paper
- `outputs/shap_local_examples.json` — worked local explanation examples (high-risk / low-risk / misclassified)
- `app.py` — Streamlit application

## Key experimental results (this run)
See outputs/model_comparison.csv and outputs/shap_global_importance.csv.
Best model: XGBoost (ROC-AUC ~0.821, tied closely with Random Forest).
Top SHAP features: GenHlth, HighBP, BMI, Age, HighChol (clinically established
risk factors) — followed by Income and Sex, which are worth discussing as
socioeconomic/demographic proxy features in the paper's Discussion section.

## Notes for the paper
- Target was binarized from the original 3-class BRFSS label
  (0=none, 1=prediabetes, 2=diabetes) into 0=no diabetes vs 1=at risk/diabetic.
  State this explicitly in your Methodology.
- Class imbalance (~84% / 16%) was handled via class_weight='balanced'
  (scikit-learn models) and scale_pos_weight (XGBoost). Mention this as a
  design decision, not an oversight.
- Data is self-reported survey data (BRFSS), not lab-measured — a
  legitimate limitation to discuss (contrast with e.g. NHANES).
