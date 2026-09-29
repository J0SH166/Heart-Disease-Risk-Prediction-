import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Cardiovascular Risk Assessment",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("heart_disease_pipeline.pkl")
    except FileNotFoundError:
        st.error("Model artifact 'heart_disease_pipeline.pkl' not found! Please run `python train.py` first.")
        st.stop()

pipeline = load_pipeline()

st.title("Heart Disease Risk Prediction System")
st.markdown(
    "A Clinical Decision Support System applying $L_1$-regularized Logistic Regression "
    "to assess cardiovascular risk based on standard clinical parameters."
)
st.divider()

# Sidebar: Patient Intake Form
with st.sidebar:
    st.header("Patient Intake Form")
    
    # Pre-filled matching Slide 15 sample patient
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=54)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", index=0)
    cp = st.selectbox(
        "Chest Pain Type",
        options=[4, 3, 2, 1],
        format_func=lambda x: {
            1: "1: Typical Angina",
            2: "2: Atypical Angina",
            3: "3: Non-anginal Pain",
            4: "4: Asymptomatic / Severe"
        }[x],
        index=0
    )
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=246)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False (<=120)" if x == 0 else "True (>120)", index=0)
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[2, 0, 1],
        format_func=lambda x: {
            0: "0: Normal",
            1: "1: ST-T Wave Abnormality",
            2: "2: Left Ventricular Hypertrophy"
        }[x],
        index=0
    )
    thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=230, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", index=0)
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.2, step=0.1)
    slope = st.selectbox(
        "Peak Exercise ST Segment Slope",
        options=[2, 1, 3],
        format_func=lambda x: {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}[x],
        index=0
    )
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", options=[1, 0, 2, 3], index=0)
    thal = st.selectbox(
        "Thalassemia Status",
        options=[7, 3, 6],
        format_func=lambda x: {3: "3: Normal", 6: "6: Fixed Defect", 7: "7: Reversible Defect"}[x],
        index=0
    )

    st.divider()
    st.subheader("Clinical Sensitivity Tuning")
    threshold = st.slider(
        "Classification Decision Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.40,
        step=0.05,
        help="Medical screening uses lower decision thresholds to reduce false negatives."
    )

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Patient Vitals Summary")
    summary_df = pd.DataFrame({
        "Clinical Parameter": [
            "Age", "Sex", "Chest Pain Type", "Resting BP", "Serum Cholesterol", 
            "Fasting Sugar > 120", "Resting ECG", "Max Heart Rate", "Exercise Angina",
            "ST Depression", "ST Slope", "Fluoroscopy Vessels (ca)", "Thalassemia"
        ],
        "Recorded Observation": [
            f"{age} yrs", "Male" if sex == 1 else "Female", f"Type {cp}",
            f"{trestbps} mm Hg", f"{chol} mg/dl", "True" if fbs == 1 else "False",
            f"Result {restecg}", f"{thalach} bpm", "Yes" if exang == 1 else "No",
            f"{oldpeak}", f"Slope {slope}", f"{ca}", f"Status {thal}"
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Cardiovascular Risk Diagnostic")
    
    patient_record = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    if st.button("Run Diagnostic Assessment", type="primary", use_container_width=True):
        risk_probability = pipeline.predict_proba(patient_record)[0][1]
        is_high_risk = risk_probability >= threshold

        st.write("")
        if is_high_risk:
            st.error("### Diagnostic Classification: HIGH RISK")
            st.metric(label="Calculated Disease Probability", value=f"{risk_probability * 100:.1f}%")
            st.warning(
                f"Flagged for clinical follow-up (Probability exceeds {threshold*100:.0f}% threshold). "
                "Secondary diagnostic validation (echocardiogram, stress imaging) is advised."
            )
        else:
            st.success("### Diagnostic Classification: LOW RISK")
            st.metric(label="Calculated Disease Probability", value=f"{risk_probability * 100:.1f}%")
            st.info(
                f"Routine monitoring recommended (Probability is below {threshold*100:.0f}% threshold). "
                "Patient profile remains within baseline parameters."
            )

        st.caption(f"Active Threshold: P >= {threshold:.2f} classified as High Risk; P < {threshold:.2f} classified as Low Risk.")

st.divider()
with st.expander("Technical Model Specification & Benchmark Performance"):
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Architecture", "L1-Logistic Regression")
    tcol2.metric("Mean CV Accuracy", "85.1%")
    tcol3.metric("Train / Test Split", "80% / 20% Stratified")
    tcol4.metric("Benchmark Target", "Cleveland Dataset (303 Records)")
    
    st.markdown(
        """
        - **Preprocessing Pipeline**: `StandardScaler` (continuous features + ordinal vessel count `ca`) and `OneHotEncoder` (nominal categoricals `cp`, `restecg`, `slope`, `thal`).
        - **Feature Selection**: Embedded $L_1$ Lasso Regularization (`penalty='l1'`, `solver='liblinear'`) shrinks non-predictive weights to zero without fragmenting dummy-coded categories.
        - **Leakage Prevention**: All scaling and imputation statistics are fitted strictly on the training partition.
        """
    )
