# Databricks notebook source
# Load raw UC table
df_raw = spark.table("workspace.ml.creditcard_raw")

print("Rows:", df_raw.count())
display(df_raw.limit(10))


# COMMAND ----------

# STEP 2.1 — Check class balance (fraud vs non-fraud)

from pyspark.sql import functions as F

class_counts = (
    df_raw.groupBy("Class")
          .agg(F.count("*").alias("cnt"))
          .orderBy("Class")
)

display(class_counts)


# COMMAND ----------

# STEP 2.2 — Convert to Pandas and create train/test split

pdf = df_raw.toPandas()

X = pdf.drop(columns=["Class"])
y = pdf["Class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train fraud ratio:", y_train.mean())
print("Test fraud ratio :", y_test.mean())


# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


with mlflow.start_run(run_name="Logistic Regression Baseline"):

    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)

    # Signature & input example
    sample_input = X_train[:5]
    sample_output = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)

    # Log model with signature + input example
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=sample_input
    )

print("Logged Logistic Regression model to MLflow with signature + example.")


# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run(run_name="Random Forest"):

    rf_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**rf_params)

    # Train
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    # Log params + metrics
    mlflow.log_params(rf_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)

    # Signature & input example
    sample_input = X_train[:5]
    sample_output = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=sample_input
    )

print("Logged Random Forest model to MLflow with signature + example.")


# COMMAND ----------

import xgboost as xgb
from mlflow.models.signature import infer_signature

# XGBoost uses optimized DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

with mlflow.start_run(run_name="XGBoost"):

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200
    )

    # Predictions
    probs = model.predict(dtest)
    preds = (probs > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    # Log params + metrics
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)

    # Signature + input example
    sample_input = X_train[:5]
    sample_output = model.predict(xgb.DMatrix(sample_input))
    signature = infer_signature(sample_input, sample_output)

    # Log full model
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=sample_input
    )

print("XGBoost model logged successfully.")
