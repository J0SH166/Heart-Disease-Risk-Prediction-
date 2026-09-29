# Cardiovascular Risk Prediction System (CDSS)

A Clinical Decision Support System applying $L_1$-regularized Logistic Regression to assess cardiovascular disease risk using clinical examination parameters. Built with Scikit-Learn and deployed interactively using Streamlit.

---

## Key Performance Metrics

* **Test Accuracy:** 86.89% (~87%)
* **5-Fold Cross-Validation Accuracy:** 85.11% (± 2.50%)
* **ROC-AUC Score:** 0.9513
* **Clinical Screening Recall (Sensitivity at P >= 0.40):** 92.86% (Only 2 missed risk cases)

---

## System Architecture & Highlights

* **Leakage-Free Preprocessing:** Stratified 80/20 train-test split applied before fitting median/mode imputers and feature scaling.
* **Mixed-Type Transformation:** Continuous metrics and ordinal vessel counts (`ca`) normalized via `StandardScaler`; nominal diagnostics (`cp`, `restecg`, `slope`, `thal`) encoded via `OneHotEncoder`.
* **Embedded $L_1$ Lasso Feature Selection:** Automatically zeroes out non-informative clinical variables (such as fasting blood sugar) without fragmenting dummy-coded categories.
* **Clinical Sensitivity Tuning:** Interactive decision threshold control enabling physicians to prioritize recall over precision during initial cardiovascular screening.

---

## Clinical Input Parameters (13 Features)

1. `age`: Age in years
2. `sex`: Biological sex (1 = Male, 0 = Female)
3. `cp`: Chest pain type (1: Typical Angina, 2: Atypical Angina, 3: Non-anginal, 4: Asymptomatic)
4. `trestbps`: Resting blood pressure (mm Hg)
5. `chol`: Serum cholesterol (mg/dl)
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
7. `restecg`: Resting ECG results (0: Normal, 1: ST-T abnormality, 2: LV hypertrophy)
8. `thalach`: Maximum heart rate achieved (bpm)
9. `exang`: Exercise-induced angina (1 = Yes, 0 = No)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: Peak exercise ST segment slope (1: Upsloping, 2: Flat, 3: Downsloping)
12. `ca`: Number of major vessels colored by fluoroscopy (0–3)
13. `thal`: Thalassemia defect status (3: Normal, 6: Fixed defect, 7: Reversible defect)

---

## Project Structure

```text
├── app.py                      # Streamlit diagnostic web interface
├── train.py                    # Preprocessing, L1 regularized training, and evaluation
├── verify_model.py             # Archetype validation script across risk profiles
├── test_data.csv               # 20% held-out test partition for live clinical demo
├── heart_disease_pipeline.pkl  # Serialized Scikit-Learn end-to-end pipeline artifact
└── requirements.txt            # Project dependencies
