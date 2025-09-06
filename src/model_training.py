import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from itertools import product

# Load data
df = pd.read_csv("data/processed/train.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

# Split (you already have processed train/test, but for tuning CV is better)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Hyperparam_Tuning")

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    accuracy = 1 - (rmse / np.mean(y_true))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "Accuracy": accuracy}

# --- RandomForest search ---
rf_param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

for n, d, s in product(rf_param_grid["n_estimators"], rf_param_grid["max_depth"], rf_param_grid["min_samples_split"]):
    with mlflow.start_run(run_name=f"RandomForest_n{n}_d{d}_s{s}"):
        rf = RandomForestRegressor(
            n_estimators=n, max_depth=d, min_samples_split=s, random_state=42
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)

        metrics = evaluate(y_val, preds)
        mlflow.log_params({"n_estimators": n, "max_depth": d, "min_samples_split": s})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf, artifact_path="RandomForest")

# --- XGBoost search ---
xgb_param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

for n, d, lr, sub in product(
    xgb_param_grid["n_estimators"],
    xgb_param_grid["max_depth"],
    xgb_param_grid["learning_rate"],
    xgb_param_grid["subsample"]
):
    with mlflow.start_run(run_name=f"XGBoost_n{n}_d{d}_lr{lr}_sub{sub}"):
        xgbr = xgb.XGBRegressor(
            n_estimators=n,
            max_depth=d,
            learning_rate=lr,
            subsample=sub,
            random_state=42,
            n_jobs=-1
        )
        xgbr.fit(X_train, y_train)
        preds = xgbr.predict(X_val)

        metrics = evaluate(y_val, preds)
        mlflow.log_params({
            "n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": sub
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(xgbr, artifact_path="XGBoost")

print("✅ Hyperparameter tuning complete. Check MLflow UI for comparisons.")
