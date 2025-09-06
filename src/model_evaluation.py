import pandas as pd
import numpy as np
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("InsuranceChargesExperiment")

class ModelTraining:
    def __init__(self, train_path, test_path, params_path):
        self.train_path = train_path
        self.test_path = test_path
        self.params_path = params_path

        with open(self.params_path, "r") as f:
            self.params = yaml.safe_load(f)

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        X_train = train_df.drop("charges", axis=1)
        y_train = train_df["charges"]
        X_test = test_df.drop("charges", axis=1)
        y_test = test_df["charges"]

        return X_train, X_test, y_train, y_test

    def evaluate(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        accuracy = 1 - (rmse / np.mean(y_true))

        return {
            "R2": float(r2),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape),
            "Accuracy": float(accuracy)
        }

    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()
        results = {}

        # --- Random Forest ---
        rf_params = self.params.get("random_forest", {"n_estimators": 200, "max_depth": 10})
        with mlflow.start_run(run_name="RandomForest"):
            rf = RandomForestRegressor(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params.get("max_depth", None),
                random_state=42
            )
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)

            metrics = self.evaluate(y_test, preds)
            results["RandomForest"] = metrics
            joblib.dump(rf, "models/random_forest.pkl")

            mlflow.log_params(rf_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(rf, artifact_path="RandomForest")

        # --- XGBoost ---
        xgb_params = self.params.get("xgboost", {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1})
        with mlflow.start_run(run_name="XGBoost"):
            xgbr = xgb.XGBRegressor(
                n_estimators=xgb_params["n_estimators"],
                max_depth=xgb_params["max_depth"],
                learning_rate=xgb_params["learning_rate"],
                random_state=42
            )
            xgbr.fit(X_train, y_train)
            preds = xgbr.predict(X_test)

            metrics = self.evaluate(y_test, preds)
            results["XGBoost"] = metrics
            joblib.dump(xgbr, "models/xgboost.pkl")

            mlflow.log_params(xgb_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(xgbr, artifact_path="XGBoost")

        # --- Save metrics for DVC ---
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.yaml", "w") as f:
            yaml.dump(results, f)

        print("✅ Training complete. Metrics saved & logged to MLflow")

if __name__ == "__main__":
    trainer = ModelTraining(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
        params_path="params.yaml"
    )
    trainer.run()
