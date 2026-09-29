import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def load_data():
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", 
        "ca", "thal", "target"
    ]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    try:
        df = pd.read_csv(url, names=column_names, na_values="?")
        print("Dataset loaded from UCI ML Repository.")
    except Exception as e:
        print(f"Online download failed ({e}). Checking local 'processed.cleveland.data'...")
        if os.path.exists("processed.cleveland.data"):
            df = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
        else:
            raise FileNotFoundError(
                "Unable to fetch dataset online and local 'processed.cleveland.data' not found. "
                "Download the file manually from the UCI Repository."
            )
    return df

def train_pipeline():
    df = load_data()
    df = df.drop_duplicates()

    # Binarize target: 0 = No Disease, 1 = Disease Present (> 0)
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratified Split before transformation to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Export held-out test data to CSV for demo verification
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv("test_data.csv", index=False)
    print(f"Exported {len(test_df)} held-out test records to 'test_data.csv'.")

    # 1. 'ca' placed in numeric features so it is properly scaled as an ordinal count (0-3)
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    
    # 2. Multi-class nominal categoricals to be one-hot encoded
    categorical_features = ["cp", "restecg", "slope", "thal"]
    
    # 3. Pure binary features (0 or 1) that do not require scaling
    binary_features = ["sex", "fbs", "exang"]

    # Preprocessing pipelines
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features)
        ]
    )

    # L1 (Lasso) penalty performs embedded feature selection by driving uninformative coefficients to zero
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            random_state=42,
            max_iter=1000
        ))
    ])

    # 5-Fold Stratified Cross Validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Test set evaluation at standard cutoff (0.50)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.50).astype(int)

    # Clinical screening cutoff (0.40) to minimize false negatives
    clinical_threshold = 0.40
    y_pred_clinical = (y_prob >= clinical_threshold).astype(int)

    print("\n" + "="*55)
    print("        MODEL EVALUATION: DEFAULT THRESHOLD (0.50)")
    print("="*55)
    print(f"Accuracy            : {accuracy_score(y_test, y_pred_default)*100:.2f}%")
    print(f"Recall (Sensitivity): {recall_score(y_test, y_pred_default)*100:.2f}%")
    print(f"Precision           : {precision_score(y_test, y_pred_default)*100:.2f}%")
    print(f"F1-Score            : {f1_score(y_test, y_pred_default)*100:.2f}%")
    print(f"ROC-AUC             : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"5-Fold CV Accuracy  : {cv_scores.mean()*100:.2f}% (± {cv_scores.std()*100:.2f}%)")
    print("\nConfusion Matrix (Default):")
    print(confusion_matrix(y_test, y_pred_default))

    print("\n" + "="*55)
    print(f"      CLINICAL SCREENING THRESHOLD ({clinical_threshold:.2f})")
    print("="*55)
    print(f"Recall (Sensitivity): {recall_score(y_test, y_pred_clinical)*100:.2f}%")
    print(f"Precision           : {precision_score(y_test, y_pred_clinical)*100:.2f}%")
    print("Confusion Matrix (Screening):")
    print(confusion_matrix(y_test, y_pred_clinical))

    # Inspect Lasso zeroed-out features
    classifier = pipeline.named_steps["classifier"]
    encoded_cat_names = pipeline.named_steps["preprocessor"] \
        .named_transformers_["cat"] \
        .named_steps["onehot"] \
        .get_feature_names_out(categorical_features) \
        .tolist()
    all_feature_names = numeric_features + encoded_cat_names + binary_features
    coefficients = classifier.coef_[0]

    print("\n" + "="*55)
    print("    L1 EMBEDDED FEATURE SELECTION (COEFFICIENTS)")
    print("="*55)
    for name, coef in zip(all_feature_names, coefficients):
        status = "RETAINED" if coef != 0 else "ZEROED (EXCLUDED)"
        print(f"{name:<25} : {coef:>8.4f}  [{status}]")

    # Save artifact
    joblib.dump(pipeline, "heart_disease_pipeline.pkl")
    print("\nExported self-contained 'heart_disease_pipeline.pkl' successfully.")

if __name__ == "__main__":
    train_pipeline()
