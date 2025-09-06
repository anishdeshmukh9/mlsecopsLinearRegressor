import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    def evaluate(self, model_name, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        accuracy = 1 - (rmse / np.mean(y_true))

        return {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE (%)": mape,
            "Accuracy": accuracy
        }

    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()

        results = {}

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        results["LinearRegression"] = self.evaluate("LinearRegression", y_test, preds)
        joblib.dump(lr, "models/linear_regression.pkl")

        # ElasticNet
        en_params = self.params.get("elasticnet", {"alpha": 0.1, "l1_ratio": 0.5})
        en = ElasticNet(alpha=en_params["alpha"], l1_ratio=en_params["l1_ratio"], random_state=42)
        en.fit(X_train, y_train)
        preds = en.predict(X_test)
        results["ElasticNet"] = self.evaluate("ElasticNet", y_test, preds)
        joblib.dump(en, "models/elasticnet.pkl")

        # Random Forest
        rf_params = self.params.get("random_forest", {"n_estimators": 100})
        rf = RandomForestRegressor(n_estimators=rf_params["n_estimators"], random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        results["RandomForest"] = self.evaluate("RandomForest", y_test, preds)
        joblib.dump(rf, "models/random_forest.pkl")

        # Save metrics
        with open("reports/metrics.yaml", "w") as f:
            yaml.dump(results, f)

        print("✅ Training complete. Metrics saved to reports/metrics.yaml")


if __name__ == "__main__":
    trainer = ModelTraining(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
        params_path="params.yaml"
    )
    trainer.run()
