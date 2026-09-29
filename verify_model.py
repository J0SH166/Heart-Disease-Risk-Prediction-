import joblib
import pandas as pd

# Load saved pipeline artifact
pipeline = joblib.load("heart_disease_pipeline.pkl")

# Define diverse test profiles
test_cases = pd.DataFrame([
    # Case 1: Healthy Baseline
    {"age": 32, "sex": 0, "cp": 1, "trestbps": 110, "chol": 175, "fbs": 0, 
     "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3},
    
    # Case 2: Slide 15 Benchmark
    {"age": 54, "sex": 1, "cp": 4, "trestbps": 130, "chol": 246, "fbs": 0, 
     "restecg": 2, "thalach": 150, "exang": 1, "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 7},
     
    # Case 3: Severe Cardiac Presentation
    {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 1, 
     "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 3.2, "slope": 2, "ca": 3, "thal": 7},
     
    # Case 4: Ambiguous / Borderline Profile
    {"age": 49, "sex": 1, "cp": 3, "trestbps": 135, "chol": 230, "fbs": 0, 
     "restecg": 0, "thalach": 155, "exang": 0, "oldpeak": 0.6, "slope": 1, "ca": 0, "thal": 3}
], index=["Healthy Baseline", "Slide 15 Patient", "Severe Cardiac", "Borderline Case"])

# Compute probabilities and binary classifications
test_cases["Risk Probability"] = pipeline.predict_proba(test_cases)[:, 1]
test_cases["Predicted Class (0.50)"] = (test_cases["Risk Probability"] >= 0.50).astype(int)
test_cases["Screening Class (0.40)"] = (test_cases["Risk Probability"] >= 0.40).astype(int)

print(test_cases[["Risk Probability", "Predicted Class (0.50)", "Screening Class (0.40)"]])
