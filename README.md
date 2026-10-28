# FraudShield AI

FraudShield AI is a machine learning-based credit card fraud detection application built using Python, Pandas, Scikit-learn, and Streamlit.

It trains a binary classification model on labelled transaction data, evaluates its performance on a held-out test set, and provides a dashboard to score uploaded transaction CSV files.

## Problem Statement

Fraudulent credit card transactions are rare compared with legitimate transactions. This project demonstrates how machine learning can identify potentially fraudulent transactions and how evaluation metrics help assess performance on imbalanced data.

## Features

- Train a logistic regression classifier with class weighting for imbalanced labels.
- Split data into stratified training and testing sets.
- Handle missing values and scale features using a Scikit-learn pipeline.
- Evaluate the model using precision, recall, F1-score, ROC-AUC, average precision, and a confusion matrix.
- Upload transaction CSV files to obtain fraud probabilities and predictions.
- Adjust the prediction threshold to explore the trade-off between missed fraud and false alarms.
- Download prediction results as a CSV file.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Dataset

This project uses the public [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Download `creditcard.csv` and place it inside the `data/` directory.

The dataset contains anonymized features (`V1` to `V28`), `Time`, `Amount`, and `Class`.

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

The dataset is not included in this repository because of its size and separate usage terms. Review the dataset's terms before redistributing it.

## Installation and Setup

1. Install Python 3.10 or later.
2. Open the project folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

6. Download the dataset and place `creditcard.csv` inside the `data/` directory.
7. Train and evaluate the model:

   ```powershell
   .\.venv\Scripts\python.exe train_model.py
   ```

8. Launch the dashboard:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

Streamlit will provide a local URL, usually `http://localhost:8501`.

## How It Works

1. The training script loads the labelled transaction dataset.
2. It separates the input features from the `Class` target.
3. A stratified split preserves the approximate fraud ratio in the training and test sets.
4. A Scikit-learn pipeline handles missing values, scales features, and trains a class-weighted logistic regression model.
5. The model is evaluated on held-out test data and saved locally.
6. The Streamlit application loads the trained model and predicts fraud probabilities for uploaded transactions.
7. A configurable threshold converts probabilities into predicted labels.

## Evaluation Metrics

Because fraud datasets are imbalanced, accuracy alone can be misleading.

- **Precision:** Of the transactions flagged as fraud, how many were actually fraudulent?
- **Recall:** Of the actual fraudulent transactions, how many did the model identify?
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks fraudulent transactions above legitimate ones across thresholds.
- **Average precision:** Summarizes the precision-recall curve, which is useful for imbalanced classification.

Actual results depend on the dataset, train/test split, and prediction threshold.

## Limitations

- This is an educational prototype, not a production banking system.
- The anonymized dataset may not represent current real-world transactions.
- A model prediction is not proof that a transaction is fraudulent.
- Real deployment requires secure data handling, monitoring, representative validation, human review, and careful threshold selection.
- Do not upload real cardholder or sensitive financial data.

## License

The project code is intended as an educational implementation.
