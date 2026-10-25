from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudShield AI")
st.write(
    "A machine-learning demonstration that scores credit-card transactions "
    "for potential fraud. Predictions are for educational use, not real banking decisions."
)

if not MODEL_PATH.exists():
    st.warning(
        "The trained model was not found. First place `creditcard.csv` in the "
        "`data` folder and run `python train_model.py` in your terminal."
    )
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Prediction settings")
    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help=(
            "Transactions at or above this predicted probability are flagged. "
            "Changing the threshold trades off catching more fraud against more false alarms."
        ),
    )
    st.caption("Lower threshold = flag more transactions; it may also increase false positives.")

st.subheader("Model evaluation")
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        c2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        c3.metric("F1-score", f"{metrics.get('f1_score', 0):.3f}")
        c4.metric("Test transactions", f"{metrics.get('test_rows', 0):,}")
        st.caption(
            "Metrics shown above were calculated on a held-out test split using "
            "the training script's default threshold of 0.50. They may differ from "
            "results at the selected threshold below."
        )
    except (OSError, json.JSONDecodeError):
        st.info("Evaluation metrics are unavailable. Re-run the training script.")
else:
    st.info("Evaluation metrics are unavailable. Re-run `python train_model.py`.")

st.divider()
st.subheader("Score transactions")
st.write(
    "Upload a CSV containing the same feature columns as the training dataset. "
    "The original Kaggle `creditcard.csv` file can be uploaded too; if it includes "
    "`Class`, the app will compare predictions with those labels."
)

uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded is not None:
    try:
        transactions = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this CSV file: {exc}")
        st.stop()

    actual_labels = None
    if "Class" in transactions.columns:
        actual_labels = transactions["Class"].copy()
        transactions = transactions.drop(columns=["Class"])

    missing = [column for column in feature_columns if column not in transactions.columns]
    if missing:
        st.error(
            "This file is missing required feature columns: "
            + ", ".join(missing[:12])
            + (" ..." if len(missing) > 12 else "")
        )
        st.write("Expected columns include:", ", ".join(feature_columns[:8]), "...")
        st.stop()

    # Extra columns are ignored; the model receives features in training order.
    X = transactions[feature_columns].copy()
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError):
        st.error("All model feature values must be numeric.")
        st.stop()

    if X.isna().any().any():
        st.error("The uploaded file contains missing values in required feature columns.")
        st.stop()

    if X.empty:
        st.error("The uploaded CSV has no transaction rows.")
        st.stop()

    try:
        probabilities = model.predict_proba(X)[:, 1]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    predictions = (probabilities >= threshold).astype(int)
    results = pd.DataFrame({
        "Predicted label": ["Potential fraud" if p == 1 else "Likely legitimate" for p in predictions],
        "Fraud probability": probabilities,
    })
    results.insert(0, "Transaction row", range(1, len(results) + 1))

    fraud_count = int(predictions.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions analysed", f"{len(X):,}")
    c2.metric("Flagged as potential fraud", f"{fraud_count:,}")
    c3.metric("Flag rate", f"{fraud_count / len(X):.1%}")

    st.dataframe(results.head(100), use_container_width=True)

    if actual_labels is not None:
        st.subheader("Comparison with supplied labels")
        try:
            y_true = pd.to_numeric(actual_labels, errors="raise").astype(int)
            if len(y_true) == len(predictions) and set(y_true.unique()).issubset({0, 1}):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy_score(y_true, predictions):.3f}")
                c2.metric("Precision", f"{precision_score(y_true, predictions, zero_division=0):.3f}")
                c3.metric("Recall", f"{recall_score(y_true, predictions, zero_division=0):.3f}")
                c4.metric("F1-score", f"{f1_score(y_true, predictions, zero_division=0):.3f}")
                st.warning(
                    "These scores are meaningful as an independent evaluation only if "
                    "the uploaded rows were not used to train the model. Do not upload "
                    "the training data and present its scores as unseen-test performance."
                )
                cm = confusion_matrix(y_true, predictions, labels=[0, 1])
                st.write("Confusion matrix (rows = actual, columns = predicted)")
                st.dataframe(
                    pd.DataFrame(
                        cm,
                        index=["Actual legitimate", "Actual fraud"],
                        columns=["Predicted legitimate", "Predicted fraud"],
                    ),
                    use_container_width=True,
                )
        except (ValueError, TypeError):
            st.info("The `Class` column could not be interpreted as binary labels.")

    output_csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction results",
        data=output_csv,
        file_name="fraudshield_predictions.csv",
        mime="text/csv",
    )
else:
    st.info(
        "No file uploaded yet. To try the app, upload a CSV with the expected feature "
        "columns. The `creditcard.csv` dataset contains the feature columns and a `Class` label."
    )

st.divider()
st.caption(
    "Educational prototype only. A model's score is not proof of fraud. "
    "Real payment systems require secure data handling, monitoring, human review, "
    "fairness checks, and validation on representative production data."
)
