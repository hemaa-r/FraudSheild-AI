from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        print(
            "Dataset not found.\n"
            "1. Download creditcard.csv from:\n"
            "   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "2. Place it at: data/creditcard.csv"
        )
        sys.exit(1)

    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    if "Class" not in data.columns:
        print("Error: dataset must contain a 'Class' target column.")
        sys.exit(1)

    if data.empty:
        print("Error: dataset is empty.")
        sys.exit(1)

    data = data.dropna(subset=["Class"]).copy()
    y = pd.to_numeric(data["Class"], errors="coerce")
    valid_rows = y.isin([0, 1])
    data = data.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    X = data.drop(columns=["Class"])
    X = X.select_dtypes(include=["number"]).copy()

    if X.empty:
        print("Error: no numeric feature columns were found.")
        sys.exit(1)

    # Fill missing feature values using the median for each column.
    # The imputation values are learned only from the training split.
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    from sklearn.impute import SimpleImputer

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print(f"Rows: {len(data):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Fraud rows: {int(y.sum()):,} ({y.mean():.3%})")
    print("Training logistic regression model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = 0.50
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "test_rows": int(len(y_test)),
        "threshold": threshold,
        "random_state": RANDOM_STATE,
        "model": "LogisticRegression(class_weight='balanced')",
        "feature_columns": feature_columns,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            zero_division=0,
            output_dict=True,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": feature_columns},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nEvaluation on held-out test data (threshold = 0.50)")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1-score:          {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print("\nConfusion matrix (rows = actual [legitimate, fraud], columns = predicted):")
    print(metrics["confusion_matrix"])
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("\nTraining complete. Start the dashboard with:")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()
