import pandas as pd
import numpy as np
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Set MLflow tracking URI (local)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LinearRegressorExperiment")


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

        # Ensure metric names are MLflow-safe (no trailing spaces)
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

        # --- Linear Regression ---
        with mlflow.start_run(run_name="LinearRegression"):
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            preds = lr.predict(X_test)

            metrics = self.evaluate(y_test, preds)
            results["LinearRegression"] = metrics
            joblib.dump(lr, "models/linear_regression.pkl")

            mlflow.log_params({})  # no hyperparams
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(lr, artifact_path="LinearRegression")

        # --- ElasticNet ---
        en_params = self.params.get("elasticnet", {"alpha": 0.1, "l1_ratio": 0.5})
        with mlflow.start_run(run_name="ElasticNet"):
            en = ElasticNet(alpha=en_params["alpha"], l1_ratio=en_params["l1_ratio"], random_state=42)
            en.fit(X_train, y_train)
            preds = en.predict(X_test)

            metrics = self.evaluate(y_test, preds)
            results["ElasticNet"] = metrics
            joblib.dump(en, "models/elasticnet.pkl")

            mlflow.log_params(en_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(en, artifact_path="ElasticNet")

        # --- Random Forest ---
        rf_params = self.params.get("random_forest", {"n_estimators": 100})
        with mlflow.start_run(run_name="RandomForest"):
            rf = RandomForestRegressor(n_estimators=rf_params["n_estimators"], random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)

            metrics = self.evaluate(y_test, preds)
            results["RandomForest"] = metrics
            joblib.dump(rf, "models/random_forest.pkl")

            mlflow.log_params(rf_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(rf, artifact_path="RandomForest")

        # --- Save metrics for DVC ---
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.yaml", "w") as f:
            yaml.dump(results, f)

        print("✅ Training complete. Metrics saved to reports/metrics.yaml and logged to MLflow")


if __name__ == "__main__":
    trainer = ModelTraining(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
        params_path="params.yaml"
    )
    trainer.run()
