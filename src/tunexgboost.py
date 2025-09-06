import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from itertools import product

# --- Load Data ---
df = pd.read_csv("data/processed/train.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("XGBoost_GPU_Tuning")

# --- Evaluation function ---
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    accuracy = 1 - (rmse / np.mean(y_true))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "Accuracy": accuracy}

# --- Load Params from YAML ---
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

xgb_param_grid = params["xgboost"]

# --- Grid Search using itertools.product ---
for n, d, lr, sub, col, ra, rl in product(
    xgb_param_grid["n_estimators"],
    xgb_param_grid["max_depth"],
    xgb_param_grid["learning_rate"],
    xgb_param_grid["subsample"],
    xgb_param_grid["colsample_bytree"],
    xgb_param_grid["reg_alpha"],
    xgb_param_grid["reg_lambda"]
):
    with mlflow.start_run(run_name=f"XGB_n{n}_d{d}_lr{lr}_sub{sub}_col{col}_ra{ra}_rl{rl}"):
        xgbr = xgb.XGBRegressor(
            n_estimators=n,
            max_depth=d,
            learning_rate=lr,
            subsample=sub,
            colsample_bytree=col,
            reg_alpha=ra,
            reg_lambda=rl,
            objective="reg:squarederror",
            tree_method="gpu_hist",   # 🚀 enable GPU
            predictor="gpu_predictor",
            random_state=42
        )
        xgbr.fit(X_train, y_train)

        preds = xgbr.predict(X_val)
        metrics = evaluate(y_val, preds)

        mlflow.log_params({
            "n_estimators": n, "max_depth": d, "learning_rate": lr,
            "subsample": sub, "colsample_bytree": col,
            "reg_alpha": ra, "reg_lambda": rl
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(xgbr, artifact_path="XGBoost_GPU")

print("✅ XGBoost GPU tuning complete. Check MLflow UI at http://localhost:5000")
