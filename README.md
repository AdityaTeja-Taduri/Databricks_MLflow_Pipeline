# Databricks MLflow Pipeline – Credit Card Fraud Detection

End-to-end machine learning pipeline built on **Databricks Free Edition** using a real-world credit card fraud dataset.  
The project covers **data ingestion**, **feature engineering**, and **MLflow experiment tracking** across multiple models  
(Logistic Regression, Random Forest, and XGBoost) with full model signatures and input examples.

---

## 🛠️ Tech Stack

- **Databricks Notebooks** – end-to-end ML development environment  
- **Unity Catalog** – governed storage for the raw Delta table  
- **Spark** – distributed data loading from Delta  
- **Pandas** – local feature preparation & train/test split  
- **Scikit-learn** – Logistic Regression & Random Forest models  
- **XGBoost** – optimized gradient boosting model  
- **MLflow** – experiment tracking (metrics, params, artifacts, signatures)  
- **Delta Lake** – ingestion storage for creditcard_raw  
- **Python 3.12** – environment used on Databricks Free Edition  

---

## 🚀 Project Overview

### ✔ Key Features
- Data ingestion into **Unity Catalog** as a Delta table  
- Feature preparation + stratified train/test split  
- Training 3 ML models for fraud detection:
  - Logistic Regression  
  - Random Forest  
  - XGBoost (best model)  
- MLflow logging of:
  - Metrics (accuracy, precision, recall, F1, AUC)
  - Hyperparameters
  - Model artifacts
  - Input example & schema signature  
- Model comparison through MLflow UI

---

## 📊 Dataset

**Source:** Kaggle — “Credit Card Fraud Detection”  
**Rows:** 284,807  
**Target Column:** `Class` (0 = normal, 1 = fraud)  
**Nature:** Extremely imbalanced (~0.17% fraud)

Stored as Unity Catalog table:
```
workspace.ml.creditcard_raw
```

---

## 🧱 Pipeline Architecture

1️⃣ Data Ingestion
  - Upload CSV via *Create or Modify Table*
  - Store as managed Delta table
  - Load with Spark:
  ```python
  df_raw = spark.table("workspace.ml.creditcard_raw")
  ```

2️⃣ Feature Prep

  - Convert Spark DataFrame → pandas
  - Define features (X) and target (y)
  - Use stratified train_test_split

3️⃣ Model Training

Models trained:

  - Logistic Regression (baseline)
  - Random Forest (tree ensemble)
  - XGBoost (optimized boosting)

4️⃣ MLflow Tracking

Each run logs:

  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC
  - Hyperparameters
  - Model binary
  - Signature + Input example

5️⃣ Model Comparison

Compared in MLflow experiment UI to select the best model (XGBoost).

```
Databricks_MLflow_Pipeline/
│
├── notebooks/
│   └── 01_credit_fraud_mlflow.py
│
├── databricks/
│   └── 01_credit_fraud_mlflow.dbc
│
├── screenshots/
│   ├── dataset_preview.png
│   ├── mlflow_runs.png
│   ├── lr_run_details.png
│   ├── rf_run_details.png
│   └── xgboost_run_details.png
│
└── README.md

```
```

---

## 📌 Results Summary

| Model               | Performance |
|---------------------|------------|
| Logistic Regression | Good precision, lower recall |
| Random Forest       | Stronger recall, better AUC |
| XGBoost             | ⭐ Best overall model |
```

Full metrics visible in MLflow experiment runs.

---

## ▶️ How to Run

  - Import 01_credit_fraud_mlflow.dbc into Databricks
  - Attach a cluster (ensure xgboost is installed)
  - Run all notebook cells
  - Open the MLflow experiment panel to review runs

---

## 📷 Screenshots

All supporting visuals are available in the screenshots/ folder:

  - Dataset preview
  - MLflow run list
  - Run details for LR, RF, XGBoost
